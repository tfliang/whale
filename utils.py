import logging


logger = logging.getLogger('whale')
logger.setLevel(level = logging.INFO)

handler = logging.FileHandler("log.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s - %(filename)s - %(module)s - %(process)d - %(thread)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s - M:%(module)s, MSG:%(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)


if __name__ == '__main__':
    logger.info("Start print log")
    logger.debug("Do something")
    logger.warning("Something maybe fail.")
    logger.info("Finish")