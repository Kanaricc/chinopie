import logging
from logging import CRITICAL
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
from logging import WARNING
import threading
from typing import Optional

import colorlog


__all__ = [
    "CRITICAL",
    "DEBUG",
    "ERROR",
    "FATAL",
    "INFO",
    "WARN",
    "WARNING",
]

# LOGGER_FORMAT = "<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
# LOGGER_FORMAT = "<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"

_lock: threading.Lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None
_file_handler:Optional[logging.Handler] = None


def create_default_formatter(use_color:bool) -> colorlog.ColoredFormatter:
    """Create a default formatter of log messages.

    This function is not supposed to be directly accessed by library users.
    """
    
    return colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)1.1s %(asctime)s.%(msecs)03d]%(reset)s %(message)s",datefmt="%m-%d %H:%M:%S",
        no_color=not use_color,
    )


def _get_library_name() -> str:

    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:

    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.setFormatter(create_default_formatter(use_color=True))

        # Apply our default configuration to the library root logger.
        library_root_logger: logging.Logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.DEBUG)
        library_root_logger.propagate = False

def _reset_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger: logging.Logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def set_logger_file(path:str):
    global _file_handler
    with _lock:
        if _file_handler is not None:
            return
        _file_handler=logging.FileHandler(path)
        _file_handler.setFormatter(create_default_formatter(use_color=False))

        library_root_logger: logging.Logger = _get_library_root_logger()
        library_root_logger.addHandler(_file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the specified name.

    This function is not supposed to be directly accessed by library users.
    """

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def disable_default_handler() -> None:
    """Disable the default handler of the Optuna's root logger.

    Example:

        Stop and then resume logging to :obj:`sys.stderr`.

        .. testsetup::

            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x**2 + y

        .. testcode::

            import optuna

            study = optuna.create_study()

            # There are no logs in sys.stderr.
            optuna.logging.disable_default_handler()
            study.optimize(objective, n_trials=10)

            # There are logs in sys.stderr.
            optuna.logging.enable_default_handler()
            study.optimize(objective, n_trials=10)
            # [I 2020-02-23 17:00:54,314] Trial 10 finished with value: ...
            # [I 2020-02-23 17:00:54,356] Trial 11 finished with value: ...
            # ...

    """

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    """Enable the default handler of the Optuna's root logger.

    Please refer to the example shown in :func:`~optuna.logging.disable_default_handler()`.
    """

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def disable_propagation() -> None:
    """Disable propagation of the library log outputs.

    Note that log propagation is disabled by default. You only need to use this function
    to stop log propagation when you use :func:`~optuna.logging.enable_propagation()`.

    Example:

        Stop propagating logs to the root logger on the second optimize call.

        .. testsetup::

            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x**2 + y

        .. testcode::

            import optuna
            import logging

            optuna.logging.disable_default_handler()  # Disable the default handler.
            logger = logging.getLogger()

            logger.setLevel(logging.INFO)  # Setup the root logger.
            logger.addHandler(logging.FileHandler("foo.log", mode="w"))

            optuna.logging.enable_propagation()  # Propagate logs to the root logger.

            study = optuna.create_study()

            logger.info("Logs from first optimize call")  # The logs are saved in the logs file.
            study.optimize(objective, n_trials=10)

            optuna.logging.disable_propagation()  # Stop propogating logs to the root logger.

            logger.info("Logs from second optimize call")
            # The new logs for second optimize call are not saved.
            study.optimize(objective, n_trials=10)

            with open("foo.log") as f:
                assert f.readline().startswith("A new study created")
                assert f.readline() == "Logs from first optimize call\\n"
                # Check for logs after second optimize call.
                assert f.read().split("Logs from second optimize call\\n")[-1] == ""

    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """Enable propagation of the library log outputs.

    Please disable the Optuna's default handler to prevent double logging if the root logger has
    been configured.

    Example:

        Propagate all log output to the root logger in order to save them to the file.

        .. testsetup::

            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x**2 + y

        .. testcode::

            import optuna
            import logging

            logger = logging.getLogger()

            logger.setLevel(logging.INFO)  # Setup the root logger.
            logger.addHandler(logging.FileHandler("foo.log", mode="w"))

            optuna.logging.enable_propagation()  # Propagate logs to the root logger.
            optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

            study = optuna.create_study()

            logger.info("Start optimization.")
            study.optimize(objective, n_trials=10)

            with open("foo.log") as f:
                assert f.readline().startswith("A new study created")
                assert f.readline() == "Start optimization.\\n"

    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = True
