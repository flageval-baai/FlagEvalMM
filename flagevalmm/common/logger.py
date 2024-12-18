import logging

_DEFAULT_LOGGER_NAME = "flagevalmm"

# Define ANSI color codes
RED = "\033[91m"
RESET = "\033[0m"


# Custom formatter to add colors
class ColorFormatter(logging.Formatter):
    def format(self, record):
        # Add red color for error messages
        if record.levelno == logging.ERROR:
            record.levelname = f"{RED}{record.levelname}{RESET}"
            record.msg = f"{RED}{record.msg}{RESET}"
        return super().format(record)


# Create formatter with colors
formatter = ColorFormatter(
    "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

# Configure the root logger with the color formatter
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Get the root logger and set handler
root_logger = logging.getLogger()
root_logger.handlers = [handler]
root_logger.setLevel(logging.INFO)


def get_logger(name: str = _DEFAULT_LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)
