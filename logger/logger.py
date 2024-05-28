import logging
import logging.config

def get_logger(conf, name='train'):

    logging.config.fileConfig(conf)

    logger = logging.getLogger(name)
    
    # print(logger)
    
    # logger.debug('debug')
    logger.info('GET LOGGER FROM CONF FILE')
    
    return(logger)

# conf = './logger/logging.conf'
# a = get_logger(conf)
