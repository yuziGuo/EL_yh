from constants import log_dir
from datetime import datetime
import logging

def getYHLogger(fn=None):
    fn = datetime.now().strftime('%y%m%d-%H:%M:%S') if fn is None else fn
    logfile = os.path.join(log_dir, fn)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关  # 脚本中没有配置logger.setLevel会使用handler.setLevel
    fh = logging.FileHandler(logfile, mode='w')
    # fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger