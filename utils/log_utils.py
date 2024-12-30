# utils/log_utils.py

import logging
import os
from datetime import datetime


def setup_logging():
    """
    初始化日志记录。创建带时间戳的日志文件，并设置控制台输出。
    """
    os.makedirs('./log/', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'./log/{timestamp}_windspeed_evaluation.log'

    # 配置基础日志设置
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )

    # 创建控制台日志处理器
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)

    # 将控制台处理器添加到根日志记录器
    logging.getLogger('').addHandler(console)

    logging.info('Logging is set up.')


def log(message, level='info'):
    """
    简化的日志记录函数。

    Parameters:
    - message (str): 要记录的消息。
    - level (str): 日志级别（'debug', 'info', 'warning', 'error', 'critical'）。
    """
    if level == 'debug':
        logging.debug(message)
    elif level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)
    elif level == 'critical':
        logging.critical(message)
    else:
        logging.info(message)
