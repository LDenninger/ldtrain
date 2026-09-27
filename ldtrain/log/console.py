from __future__ import annotations

import atexit
import contextlib
import copy
import functools
import logging
import os
import queue
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from typing_extensions import Literal, get_args

from ldtrain.distributed import dist

STREAM_CLOSE_TIMEOUT_S = 5.0

_async_streams: list[AsyncStream] = []
_logging_pause_depth = 0
_logging_pause_saved: list[tuple[logging.Handler, logging.Formatter | None]] = []

#---------------------------------------------------------------------
# life cycle
#---------------------------------------------------------------------

def initialize(
    output_file: str | None = None,
    only_master_to_console: bool = False,
    color: bool = False,
    log_level: LogLevelName = 'info',
    log_level_file: LogLevelName = 'info',
    reset: bool = True,
) -> logging.Logger:
    """Attach console and file handlers to the root logger.

    Both handlers write through an `AsyncStream`, so a stalled stdout pipe or a slow
    file system never blocks the caller. Rank `r > 0` writes to `<stem>.rank<r><suffix>`. No collective is
    issued, so ranks may call this independently and with different arguments.

    Args:
        output_file: Log file path, `.txt` appended when it has no suffix. None disables
            file logging.
        only_master_to_console: Silence the console on every rank but 0.
        color: Emit ANSI colors on the console.
        log_level: Minimum level written to the console.
        log_level_file: Minimum level written to the file.
        reset: Also remove the handlers third-party loggers attached themselves. The
            root logger is always reset, so repeated calls never duplicate output.

    Returns:
        The root logger, cast to `Logger`.
    """
    assert log_level in get_args(LogLevelName), \
        f'Unrecognized log level: {log_level} (expected one of {get_args(LogLevelName)})'
    assert log_level_file in get_args(LogLevelName), \
        f'Unrecognized log level: {log_level_file} (expected one of {get_args(LogLevelName)})'

    if reset:
        reset_logging()
    reset_root_logger()

    log_level_int = LogLevel[log_level.upper()]
    log_level_file_int = LogLevel[log_level_file.upper()]

    logger = get_logger()
    logger.setLevel(min(log_level_int, log_level_file_int))

    handlers: list[logging.Handler] = []
    if dist.is_master or not only_master_to_console:
        console_handler = logging.StreamHandler(cached_log_stream(None))
        console_handler.setLevel(log_level_int)
        console_handler.addFilter(DowngradeWarningLikeErrorsFilter())
        console_handler.addFilter(HandlerTargetFilter('to_stdout'))
        console_handler.setFormatter(LoggingFormatter(color=color))
        handlers.append(console_handler)

    if output_file is not None:
        output_path = Path(output_file)
        rank_suffix = f'.rank{dist.rank}' if dist.rank > 0 else ''
        file_path = output_path.parent / f'{output_path.stem}{rank_suffix}{output_path.suffix or ".txt"}'
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.StreamHandler(cached_log_stream(str(file_path)))
        file_handler.setLevel(log_level_file_int)
        file_handler.addFilter(DowngradeWarningLikeErrorsFilter())
        file_handler.addFilter(HandlerTargetFilter('to_file'))
        file_handler.setFormatter(LoggingFormatter(color=False))
        handlers.append(file_handler)

    if not handlers:
        handlers.append(logging.NullHandler())
    for handler in handlers:
        logger.addHandler(handler)

    return logger


def reset_logging(name: str | None = None, reset_root: bool = False) -> None:
    """Remove and close the handlers of one named logger, or of all named loggers.

    Levels and logger identities are left untouched, so a library's own
    `getLogger(__name__)` object and a level set on it before `initialize()` survive.

    Args:
        name: Logger to reset. None resets every named logger.
        reset_root: Also clear the root logger.
    """
    if name is not None:
        clear_handlers(logging.getLogger(name))
    else:
        for log in list(logging.Logger.manager.loggerDict.values()):
            if isinstance(log, logging.Logger):
                clear_handlers(log)

    if reset_root:
        reset_root_logger()


def reset_root_logger() -> None:
    clear_handlers(logging.getLogger())
    logging.getLogger().setLevel(logging.NOTSET)


def clear_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

#---------------------------------------------------------------------
# asynchronous streams
#---------------------------------------------------------------------

class AsyncStream:
    """Text stream whose writes are performed on a file descriptor by a daemon thread.

    `write()` only enqueues, so a reader that stops draining a pipe or a hanging file
    system stalls this thread and never the caller. Pending text is held in an
    unbounded queue until the stream recovers. The thread writes with `os.write`,
    never through a Python buffered writer: a daemon thread blocked inside one aborts
    the interpreter at shutdown when finalization cannot take its lock.
    """
    def __init__(self, fd: int) -> None:
        self.fd = fd
        self._queue: queue.SimpleQueue[str | threading.Event | None] = queue.SimpleQueue()
        self._thread: threading.Thread | None = threading.Thread(target=self.drain, name='ldtrain-logging',
                                                                 daemon=True)
        self._thread.start()
        _async_streams.append(self)

    def write(self, text: str) -> None:
        if self._thread is None:
            self.write_now(text)
        else:
            self._queue.put_nowait(text)

    def flush(self) -> None:
        pass  # called by every logging handler after each record, so it must never wait

    def wait_drained(self) -> None:
        """Block until everything enqueued so far is written, within a bounded wait."""
        if self._thread is None:
            return
        drained = threading.Event()
        self._queue.put_nowait(drained)
        drained.wait(timeout=STREAM_CLOSE_TIMEOUT_S)

    def drain(self) -> None:
        while (item := self._queue.get()) is not None:
            if isinstance(item, threading.Event):
                item.set()
            else:
                self.write_now(item)

    def write_now(self, text: str) -> None:
        data = memoryview(text.encode('utf-8', errors='replace'))
        try:
            while data:
                data = data[os.write(self.fd, data):]
        except OSError:
            pass

    def close(self) -> None:
        """Write out pending text within a bounded wait, then write synchronously.

        The join is bounded, so a stream stuck on a stalled reader cannot hang
        interpreter exit. Text enqueued during the join is written afterwards.
        """
        if self._thread is None:
            return
        thread, self._thread = self._thread, None
        self._queue.put_nowait(None)
        thread.join(timeout=STREAM_CLOSE_TIMEOUT_S)
        if thread.is_alive():
            return
        with contextlib.suppress(queue.Empty):
            while True:
                item = self._queue.get_nowait()
                if isinstance(item, str):
                    self.write_now(item)

    def detach_in_child(self) -> None:
        """Write synchronously in a forked child, where the drain thread does not exist.

        Text the parent had queued at fork time is dropped here, the parent writes it.
        """
        self._thread = None
        self._queue = queue.SimpleQueue()


@functools.cache
def cached_log_stream(file_name: str | None) -> AsyncStream:
    """Return one shared stream per log file, None meaning the process stdout.

    Never closed explicitly: records logged by later atexit hooks still need it.
    """
    if file_name is None:
        sys.__stdout__.flush()  # type: ignore[union-attr]
        return AsyncStream(sys.__stdout__.fileno())  # type: ignore[union-attr]
    return AsyncStream(os.open(file_name, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644))


def flush_async_streams() -> None:
    """Wait until every log stream has written what was logged so far."""
    for stream in _async_streams:
        stream.wait_drained()


def close_async_streams() -> None:
    for stream in _async_streams:
        stream.close()


def detach_async_streams_in_child() -> None:
    for stream in _async_streams:
        stream.detach_in_child()


atexit.register(close_async_streams)
os.register_at_fork(after_in_child=detach_async_streams_in_child)

#---------------------------------------------------------------------
# logger access
#---------------------------------------------------------------------

def get_logger(name: str | None = None) -> Logger:
    """ Get the root logger or a specific logger by name. """
    if name is None:
        root = logging.getLogger()
        root.__class__ = Logger
        return root  # type: ignore[return-value]
    named = logging.getLogger(name)
    named.__class__ = Logger
    return named  # type: ignore[return-value]


def is_initialized() -> bool:
    """ Check if the logging is initialized. """
    return logging.getLogger().handlers != []


def pause_logging() -> None:
    """Switch every handler to an uncolored formatter until the matching resume_logging().

    Calls nest, only the outermost pause saves and the outermost resume restores.
    """
    global _logging_pause_depth, _logging_pause_saved
    _logging_pause_depth += 1
    if _logging_pause_depth > 1:
        return
    handlers = list(logging.getLogger().handlers)
    for log in logging.Logger.manager.loggerDict.values():
        if isinstance(log, logging.Logger):
            handlers.extend(log.handlers)
    _logging_pause_saved = [(h, h.formatter) for h in dict.fromkeys(handlers)]
    for h, _ in _logging_pause_saved:
        h.setFormatter(LoggingFormatter(color=False))
    sys.stdout.flush()
    sys.stderr.flush()


def resume_logging() -> None:
    """Restore formatters after the outermost pause_logging(). No-op if logging was not paused."""
    global _logging_pause_depth, _logging_pause_saved
    if _logging_pause_depth == 0:
        return
    _logging_pause_depth -= 1
    if _logging_pause_depth > 0:
        return
    for h, formatter in _logging_pause_saved:
        h.setFormatter(formatter)
    _logging_pause_saved = []


@contextlib.contextmanager
def logging_disabled() -> Iterator[None]:
    """Temporarily disable all loggers."""
    pause_logging()
    try:
        yield
    finally:
        resume_logging()


#---------------------------------------------------------------------
# formatting
#---------------------------------------------------------------------

_ANSI_RESET = "\033[0m"
_ANSI_FG = {
    "black": 30,
    "grey": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "dark_grey": 30,
    "light_grey": 90,
    "light_red": 91,
    "light_green": 92,
    "light_yellow": 93,
    "light_blue": 94,
    "light_magenta": 95,
    "light_cyan": 96,
}
_ANSI_ATTR = {
    "bold": 1,
    "dark": 2,
    "underline": 4,
    "blink": 5,
    "reverse": 7,
    "concealed": 8,
}

class LogLevel(int, Enum):
    DEV = 1
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
logging.addLevelName(LogLevel.DEV, 'DEV')
LogLevelName = Literal["dev", "debug", "info", "warning", "error", "critical"]


@dataclass
class LogLevelStyle:
    log_level: LogLevel
    prefix: str
    color: str
    attrs: list[str]

    def to_tuple(self) -> tuple[str, str, list[str]]:
        return (self.prefix, self.color, self.attrs)

    def format(self, msg: str) -> str:
        return f'{self.format_prefix()} {self.format_message(msg)}'

    def format_message(self, msg: str) -> str:
        return colored(msg, self.color, attrs=self.attrs)

    def format_prefix(self) -> str:
        time_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        return colored(f'[{self.prefix}][{time_str}]', self.color, attrs=['bold'])


LEVEL_STYLES = {
    LogLevel.DEV: LogLevelStyle(LogLevel.DEV, "DEV ", "magenta", ["bold"]),
    LogLevel.DEBUG: LogLevelStyle(LogLevel.DEBUG, "DEBUG ", "white", ["bold"]),
    LogLevel.INFO: LogLevelStyle(LogLevel.INFO, "", "", []),
    LogLevel.WARNING: LogLevelStyle(LogLevel.WARNING, "WARNING ", "yellow", ["bold"]),
    LogLevel.ERROR: LogLevelStyle(LogLevel.ERROR, "ERROR ", "light_red", ["bold"]),
    LogLevel.CRITICAL: LogLevelStyle(LogLevel.CRITICAL, "CRITICAL ", "red", ["bold", "underline"]),
}


def colored(
    text: object,
    color: str | None = None,
    on_color: str | None = None,
    attrs: list[str] | None = None,
    *,
    no_color: bool | None = None,
) -> str:
    """Wrap text in ANSI escape codes. Compatible with termcolor API; always emits codes when color/attrs given (works when stdout is not a TTY)."""
    if no_color:
        return str(text)
    s = str(text)
    if not color and not attrs:
        return s
    codes: list[int] = []
    if color and color in _ANSI_FG:
        codes.append(_ANSI_FG[color])
    if on_color and on_color.startswith("on_") and on_color[3:] in _ANSI_FG:
        codes.append(_ANSI_FG[on_color[3:]] + 10)  # background = fg + 10
    for a in attrs or []:
        if a in _ANSI_ATTR:
            codes.append(_ANSI_ATTR[a])
    if not codes:
        return s
    return "\033[" + ";".join(map(str, codes)) + "m" + s + _ANSI_RESET

class LoggingFormatter(logging.Formatter):
    """Formats log lines with level, timestamp, file:line, rank; suppresses prefix on traceback continuation lines."""
    def __init__(
        self,
        *,
        color: bool = True,
        prefix_color: str = "light_blue",
        datefmt: str = "%d/%m/%Y %H:%M:%S",
    ) -> None:
        super().__init__(fmt="%(message)s", datefmt=datefmt)
        self._color = color
        self._prefix_color = prefix_color
        self._datefmt = datefmt

    def formatMessage(self, record: logging.LogRecord) -> str:
        msg_str = getattr(record, "message", None) or str(getattr(record, "msg", ""))
        if self.is_traceback_continuation(msg_str):
            msg_color = getattr(record, "color", None)
            msg = colored(msg_str, msg_color) if msg_color and self._color else msg_str
            return "    " + msg

        asctime = getattr(record, "asctime", None) or self.formatTime(record, self._datefmt)
        log_prefix = f"[{asctime}][{Path(record.filename).stem}:{record.lineno}]"
        rank = dist.rank
        if rank is not None:
            log_prefix += f"[rank:{rank}]"
        try:
            style = LEVEL_STYLES[LogLevel(record.levelno)]
            text = style.prefix
            color_name = style.color
            attrs = style.attrs
        except (ValueError, KeyError):
            text, color_name, attrs = '', '', []
        level_prefix = colored(text, color_name, attrs=attrs) if self._color and text else text
        if self._color:
            log_prefix = colored(log_prefix, self._prefix_color)

        msg_color = getattr(record, "color", None)
        if msg_color is not None and self._color:
            msg = colored(record.message, msg_color)
        else:
            msg = record.message
        return level_prefix + log_prefix + " " + msg

    def is_traceback_continuation(self, message: str) -> bool:
        """True if this log message is a traceback/stack line (so we avoid repeating the full prefix)."""
        if not message:
            return False
        s = message.strip()
        if not s:
            return True
        if message.startswith(("  ", "\t")):
            return True
        if s.startswith("File ") and "line " in s and ", in " in s:
            return True
        if s == "Traceback (most recent call last):":
            return True
        if s.startswith(("Call stack:", "--- Logging error ---", "Message:", "Arguments:")):
            return True
        return False


#---------------------------------------------------------------------
# filters
#---------------------------------------------------------------------

class HandlerTargetFilter(logging.Filter):
    """Filter that allows a record through only when record.<attr_name> is True."""
    def __init__(self, attr_name: str):
        super().__init__()
        self._attr = attr_name

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, self._attr, True))


class DowngradeWarningLikeErrorsFilter(logging.Filter):
    """Filter that downgrades warnings emitted at error level from third-party libraries.

    Records of the root logger, which `log_*` and `get_logger()` use, and of `ldtrain.*`
    loggers are never touched.
    The downgrade returns a copy, so other handlers still see the original level.
    """
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        if record.levelno != logging.ERROR or record.name == 'root' or record.name.startswith('ldtrain.'):
            return True
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if not self.is_warning(msg):
            return True
        downgraded = copy.copy(record)
        downgraded.levelno = logging.WARNING
        downgraded.levelname = 'WARNING'
        return downgraded

    def is_warning(self, msg: str) -> bool:
        if not msg:
            return False
        lower = msg.lower()
        warning_keywords = (
            'userwarning',
            'deprecationwarning',
            'futurewarning',
            'warning',
            'warn',
            'deprecated',
            'has been deprecated',
            'torch',
            'cuda',
            'device',
        )
        for keyword in warning_keywords:
            if keyword in lower:
                return True
        return False


#---------------------------------------------------------------------
# logger
#---------------------------------------------------------------------

class Logger(logging.Logger):
    """Logger that supports to_stdout/to_file to direct messages to console and/or file handlers."""
    def log_config(self, config: dict | list | DictConfig | ListConfig, msg: str | None = None, to_stdout: bool = False,
                   to_file: bool = True, log_level: LogLevelName = 'info', stacklevel: int = 2) -> None:
        config_yaml = None
        if type(config) in [dict, list]:
            config_yaml = yaml.dump(config)
        elif type(config) in [DictConfig, ListConfig]:
            config_yaml = OmegaConf.to_yaml(config)
        elif type(config) in [str, int, float, bool]:
            config_yaml = str(config)
        else:
            self.warning(f"Configuration of type '{type(config)}' not parsable.", stacklevel=stacklevel + 2)
        if config_yaml is not None:
            if msg is not None:
                msg_full = f"{msg}\n{config_yaml}"
            else:
                msg_full = config_yaml
            self._log_custom(LogLevel[log_level.upper()], msg_full, to_stdout=to_stdout, to_file=to_file,
                             stacklevel=stacklevel + 1)
        return

    def dev(self, msg: str, *args: Any, color: str | None = None, to_stdout: bool = True, to_file: bool = True,
            stacklevel: int = 3, **kwargs: Any) -> None:
        self._log_custom(LogLevel.DEV, msg, *args, color=color, to_stdout=to_stdout, to_file=to_file,
                         stacklevel=stacklevel, **kwargs)

    def info(
            self,
            msg: str,
            *args: Any,
            color: str | None = None,
            to_stdout: bool = True,
            to_file: bool = True,  # type: ignore[override]
            stacklevel: int = 3,
            **kwargs: Any) -> None:
        self._log_custom(LogLevel.INFO, msg, *args, color=color, to_stdout=to_stdout, to_file=to_file,
                         stacklevel=stacklevel, **kwargs)

    def warning(
            self,
            msg: str,
            *args: Any,
            color: str | None = None,
            to_stdout: bool = True,
            to_file: bool = True,  # type: ignore[override]
            stacklevel: int = 3,
            **kwargs: Any) -> None:
        self._log_custom(LogLevel.WARNING, msg, *args, color=color, to_stdout=to_stdout, to_file=to_file,
                         stacklevel=stacklevel, **kwargs)

    def error(
            self,
            msg: str,
            *args: Any,
            color: str | None = None,
            to_stdout: bool = True,
            to_file: bool = True,  # type: ignore[override]
            stacklevel: int = 3,
            **kwargs: Any) -> None:
        self._log_custom(LogLevel.ERROR, msg, *args, color=color, to_stdout=to_stdout, to_file=to_file,
                         stacklevel=stacklevel, **kwargs)

    def debug(
            self,
            msg: str,
            *args: Any,
            color: str | None = None,
            to_stdout: bool = True,
            to_file: bool = True,  # type: ignore[override]
            stacklevel: int = 3,
            **kwargs: Any) -> None:
        self._log_custom(LogLevel.DEBUG, msg, *args, color=color, to_stdout=to_stdout, to_file=to_file,
                         stacklevel=stacklevel, **kwargs)

    def critical(
            self,
            msg: str,
            *args: Any,
            color: str | None = None,
            to_stdout: bool = True,
            to_file: bool = True,  # type: ignore[override]
            stacklevel: int = 3,
            **kwargs: Any) -> None:
        self._log_custom(LogLevel.CRITICAL, msg, *args, color=color, to_stdout=to_stdout, to_file=to_file,
                         stacklevel=stacklevel, **kwargs)

    def _log_custom(self, level: int, msg: str, *args: Any, color: str | None = None, to_stdout: bool = True,
                    to_file: bool = True, stacklevel: int = 3, **kwargs: Any) -> None:

        extra = kwargs.pop("extra", {})
        extra['to_stdout'] = to_stdout
        extra['to_file'] = to_file
        extra['color'] = color
        self.log(level, msg, *args, extra=extra, stacklevel=stacklevel, **kwargs)
