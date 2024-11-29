import os
import os.path as osp
import shutil
import platform
import logging
from tensorboardX import SummaryWriter
import datetime
import numpy as np, torch, random
import imageio
import glob
from tqdm import tqdm

def make_video(src_dir, dst_fn):
    import imageio

    print(f"export video to {dst_fn}...")
    # print(os.listdir(src_dir))
    img_fn = [
        f for f in os.listdir(src_dir) if f.endswith(".png") or f.endswith(".jpg")
    ]
    img_fn.sort()
    frames = []
    for fn in tqdm(img_fn):
        frames.append(imageio.imread(osp.join(src_dir, fn)))
    imageio.mimsave(dst_fn, frames)
    return


def seed_everything(seed):
    logging.info(f"seed: {seed}")
    print(f"seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def fig2nparray(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def create_log(log_dir, debug=False):
    os.makedirs(osp.join(log_dir, "viz_step"), exist_ok=True)
    backup_dir = osp.join(log_dir, "backup")
    tb_dir = osp.join(log_dir, "tensorboard")
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    lib_dir = osp.dirname(osp.abspath(__file__))
    shutil.copytree(lib_dir, backup_dir, dirs_exist_ok=True)
    main_path = osp.join(lib_dir, "../run.py")
    shutil.copy(main_path, backup_dir)
    main_path = osp.join(lib_dir, "../prepare.py")
    shutil.copy(main_path, backup_dir)
    writer = SummaryWriter(log_dir=tb_dir)
    configure_logging(osp.join(log_dir, f"{get_timestamp()}.log"), debug=debug)
    return writer


class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True


def configure_logging(log_path, debug=False):
    """
    https://github.com/facebookresearch/DeepSDF
    """
    logging.getLogger().handlers.clear()
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    logger_handler.addFilter(HostnameFilter())
    formatter = logging.Formatter(
        "| %(hostname)s | %(levelname)s | %(asctime)s | %(message)s   [%(filename)s:%(lineno)d]",
        "%b-%d-%H:%M:%S",
    )
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    file_logger_handler = logging.FileHandler(log_path)

    file_logger_handler.setFormatter(formatter)
    logger.addHandler(file_logger_handler)
