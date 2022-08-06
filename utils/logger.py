import logging


def get_logger(logger_name: str,
               format_: str = '%(asctime)s\t%(levelname)s\t%(name)s\t%(filename)s\t%(message)s',
               level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(fmt=format_)

    file_handler = logging.FileHandler(filename=f"{logger_name}.log")
    file_handler.setFormatter(fmt=formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt=formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level=level)
    return logger


if __name__ == '__main__':
    test_logger = get_logger(__name__)
    test_logger.info('Hello INFO World!')
    test_logger.warning('Hello WARNING World!')
    test_logger.error('Hello ERROR World!')