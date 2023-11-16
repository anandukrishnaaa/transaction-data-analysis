from icecream import ic
import logging


def set_logger(print_to_console=False):
    # Logger config
    logger = logging.getLogger(__name__)

    def log_to_django(s):
        logger.info(s)
        if print_to_console:
            print("print_to_console = True, change it to False to not print to console")
            print(s)

    ic.configureOutput(outputFunction=log_to_django)

    return ic
