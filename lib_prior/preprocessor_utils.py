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
import os, os.path as osp
import imageio, cv2
import logging
import numpy as np
import time
import logging
from scipy.optimize import least_squares
from matplotlib import pyplot as plt


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


def convert_from_mp4(src_fn, target_height=-1, target_width=-1, skip=1, max_T=-1):
    assert src_fn.endswith(".mp4")
    assert osp.exists(src_fn)
    # read all frames from mp4
    video_reader = imageio.get_reader(src_fn)
    # get all
    video_frames = [frame for frame in video_reader.iter_data()]
    assert len(video_frames) > 0
    # resize all frames to a target width
    if target_height > 0:
        if target_width == -1:
            target_width = int(
                video_frames[0].shape[1] * target_height / video_frames[0].shape[0]
            )
        video_frames = [
            cv2.resize(
                frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )
            for frame in video_frames
        ]
    return video_frames


def load_imgs(load_dir, target_height=None):
    img_fns = [
        f for f in os.listdir(load_dir) if f.endswith(".jpg") or f.endswith(".png")
    ]
    img_fns.sort()
    img_list = []
    for img_fn in img_fns:
        img = imageio.imread(osp.join(load_dir, img_fn))
        if target_height is not None:
            target_width = int(img.shape[1] * target_height / img.shape[0])
            img = cv2.resize(
                img, (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )
        img_list.append(img)
    return img_list, img_fns


def residuals(params, x, y):
    a, b = params
    # 1 / x * a + b = y
    return a / (x + 1e-8) + b - y


def residuals_scale(params, x, y):
    a, b = params
    # 1 / x * a + b = y
    return a / (x + 1e-8) - y


def robust_solver(x_data, y_data, bias=True, loss="cauchy", f_scale=0.001):
    start_t = time.time()
    # Initial guess for parameters [a, b]
    initial_guess = [(x_data * y_data).mean(), 0.0]
    logging.info(f"Initial guess: {initial_guess}")
    # Use robust least squares with 'soft_l1' loss function
    residul_fun = residuals if bias else residuals_scale
    result = least_squares(
        residul_fun,
        x0=initial_guess,
        args=(x_data, y_data),
        # loss="soft_l1",
        # loss="huber",
        # f_scale=1.0,
        loss=loss,
        f_scale=f_scale,
        verbose=1,  # Set to 0 to disable output
    )

    # Extract the estimated parameters
    a_est, b_est = result.x
    end_t = time.time()
    # logging.info(f"Robust solver time: {end_t - start_t}")
    return a_est, b_est


def align_disparity_to_metric_depth(
    disp_map,
    metric_depth,
    metric_depth_t,
    max_N=100000,
    quantil=0.5,
    min_dep=0.001,
    max_dep=1000.0,
    bias=True,
    robust_loss="cauchy",
    robust_f_scale=0.001,
):
    logging.info(f"Align at {metric_depth_t}")
    assert (
        disp_map.shape[1:] == metric_depth.shape[1:]
    ), f"{disp_map.shape[1:]} != {metric_depth.shape[1:]}"
    # 1 / disp * a + b = metric_depth
    # 1 / x * a + b = y
    # todo: here the valid check of x (disp) is not ideal, because this might be in another scale with the GT depth
    x = disp_map[metric_depth_t].flatten()
    y = metric_depth.flatten()
    x_valid_mask = (x > 1.0 / max_dep) * (x < 1.0 / min_dep)
    y_valid_mask = (y > min_dep) * (y < max_dep)

    x_sorted = np.sort(x)  # small first
    y_sorted = np.sort(y)
    sel_x_idx = int(x_sorted.shape[0] * quantil)
    sel_y_idx = int(y_sorted.shape[0] * quantil)

    # x is the largest percentage of the data, y is the smallest percentage of the data
    x_th = x_sorted[-sel_x_idx]
    y_th = y_sorted[sel_y_idx]

    x_valid_mask = x_valid_mask * (x > x_th)
    y_valid_mask = y_valid_mask * (y < y_th)
    valid_mask = x_valid_mask * y_valid_mask

    # ! only use the points in the range
    x = x[valid_mask]
    y = y[valid_mask]
    # randomly sample 10k points
    if x.shape[0] > max_N:
        logging.info(f"Randomly sample {max_N} points from {x.shape[0]}")
        idx = np.random.choice(x.shape[0], max_N, replace=False)
        x = x[idx]
        y = y[idx]

    a, b = robust_solver(x, y, bias=bias, loss=robust_loss, f_scale=robust_f_scale)
    logging.info(f"Estimated parameters: a={a}, b={b}")
    if not bias:
        logging.info(f"Not bias, set b=0")
        b = 0
    assert a > 0, f"Invalid a={a}"
    # ret = 1 / (a * disp_map + b)
    ret = a / (disp_map + 1e-8) + b
    ret = np.clip(ret, min_dep, max_dep)

    # * viz
    # plot a scatter plot of 1/x and y, return a nd array plot
    plt.scatter(1 / x, y, s=0.3, alpha=0.03)
    plt.xlabel("1/disp")
    plt.ylabel("metric_depth")
    # also plot the fitted line
    # x_fit = np.linspace(1 / x.max(), 1 / x.min(), 100)
    x_fit = np.linspace(1e-3, 1 / x.min(), 100)
    y_fit = a * x_fit + b
    plt.plot(x_fit, y_fit, color="red")
    plt.title(f"Loss={robust_loss}, f_scale={robust_f_scale}, bias={bias}")
    # plot to ndarray
    fig = plt.gcf()
    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    plt.close()
    # import imageio
    # imageio.imsave("./debug/dbg.jpg", plot)

    return ret, plot


def align_disparity_to_metric_depth_legacy(
    disp_map,
    metric_depth,
    metric_depth_t,
    max_N=100000,
    quantil=0.5,
    min_dep=0.001,
    max_dep=1000.0,
):
    logging.info(f"Align at {metric_depth_t}")
    assert (
        disp_map.shape[1:] == metric_depth.shape[1:]
    ), f"{disp_map.shape[1:]} != {metric_depth.shape[1:]}"
    # (a * disp + b) (metric_dep + c) = 1
    # metric_dep = 1 / (a * disp + b)  - c
    # todo: here the valid check of x (disp) is not ideal, because this might be in another scale with the GT depth
    x = disp_map[metric_depth_t].flatten()
    y = metric_depth.flatten()
    x_valid_mask = (x > 1.0 / max_dep) * (x < 1.0 / min_dep)
    y_valid_mask = (y > min_dep) * (y < max_dep)

    x_sorted = np.sort(x)  # small first
    y_sorted = np.sort(y)
    sel_x_idx = int(x_sorted.shape[0] * quantil)
    sel_y_idx = int(y_sorted.shape[0] * quantil)

    # x is the largest percentage of the data, y is the smallest percentage of the data
    x_th = x_sorted[-sel_x_idx]
    y_th = y_sorted[sel_y_idx]

    x_valid_mask = x_valid_mask * (x > x_th)
    y_valid_mask = y_valid_mask * (y < y_th)
    valid_mask = x_valid_mask * y_valid_mask

    # ! only use the points in the range
    x = x[valid_mask]
    y = y[valid_mask]
    # randomly sample 10k points
    if x.shape[0] > max_N:
        logging.info(f"Randomly sample {max_N} points from {x.shape[0]}")
        idx = np.random.choice(x.shape[0], max_N, replace=False)
        x = x[idx]
        y = y[idx]
    a, b = robust_solver(x, y)
    logging.info(f"Estimated parameters: a={a}, b={b}")
    assert a > 0, f"Invalid a={a}"
    ret = 1 / (a * disp_map + b)
    ret = np.clip(ret, min_dep, max_dep)
    return ret
