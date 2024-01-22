import logging

def init_logger(name, path, stream_level=logging.DEBUG):
    
    formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
    
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(stream_level)
    streamhandler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(streamhandler)

    if path is not None:

        filehandler = logging.FileHandler(path)
        # In the file, write Info or the other things with higer lever than info: error, warning and stuff.
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    
    return logger
