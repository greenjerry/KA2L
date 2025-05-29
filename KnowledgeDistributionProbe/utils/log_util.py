import logging
import os

from KnowledgeDistributionProbe import config

logger = logging.getLogger()


def setup_logging(log_path, log_file='run.log', log_level=logging.INFO):
    global logger
    """设置日志记录到文件和控制台。"""

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_path, log_file), mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # 获取 websocket 库的 logger
    websocket_logger = logging.getLogger("websocket")

    # 设置 websocket 库 logger 的级别为 WARNING 或更高
    websocket_logger.setLevel(logging.WARNING)
    # 获取 httpx 的 logger
    httpx_logger = logging.getLogger("httpx")
    # 设置 httpx logger 的级别为 ERROR
    httpx_logger.setLevel(logging.ERROR)

    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def init_logger():
    global logger
    if config.LOG_LEVEL == 'DEBUG':
        setup_logging(log_path=os.path.join(config.SAVE_PATH, "logs"), log_level=logging.DEBUG)
    elif config.LOG_LEVEL == 'INFO':
        setup_logging(log_path=os.path.join(config.SAVE_PATH, "logs"), log_level=logging.INFO)
    elif config.LOG_LEVEL == 'WARNING':
        setup_logging(log_path=os.path.join(config.SAVE_PATH, "logs"), log_level=logging.WARNING)
    else:
        setup_logging(log_path=os.path.join(config.SAVE_PATH, "logs"), log_level=logging.ERROR)
