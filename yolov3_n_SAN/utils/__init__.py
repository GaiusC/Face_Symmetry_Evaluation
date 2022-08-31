# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
utils/initialization
"""
from .pts_utils import find_all_peaks, find_batch_peaks
from .pts_utils import find_peaks_v1
from .time_utils import time_for_file, time_string, time_string_short, time_print
from .time_utils import AverageMeter, LossRecorderMeter, convert_secs2time, print_log
from .file_utils     import load_list_from_folders, load_txt_file
from .pts_utils      import generate_label_map_gaussian
from .pts_utils      import generate_label_map_laplacian
from .time_utils     import convert_size2str
from .flop_benchmark import get_model_infos, count_parameters_in_MB

from .stn_utils      import crop2affine


def notebook_init():
    # For  notebooks
    print('Checking setup...')
    from IPython import display  # to display images and clear console output

    from utils.general import emojis
    from utils.torch_utils import select_device  # imports

    display.clear_output()
    select_device(newline=False)
    print(emojis('Setup complete ✅'))
    return display
