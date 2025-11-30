
import logging
from datetime import datetime


def setup_logger(name: str = "agent_logger") -> logging.Logger:
    """
    Create and configure a logger for the agents/orchestrator.
    Demonstrates 'Observability: Logging' for the submission.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_event(logger: logging.Logger, agent_name: str, message: str, status: str = "INFO"):
    """
    Convenience wrapper to log an event with a consistent structure.
    """
    text = f"[{agent_name}] [{status}] {message}"
    logger.info(text)


def log_error(logger: logging.Logger, agent_name: str, message: str):
    """
    Convenience wrapper to log errors.
    """
    text = f"[{agent_name}] [ERROR] {message}"
    logger.error(text)
