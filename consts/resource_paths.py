from pathlib import Path


def relative_to_abs_path(relative_path):
    dirname = Path(__file__).parent
    try:
        return str((dirname / relative_path).resolve())
    except FileNotFoundError:
        return None

prefix                = relative_to_abs_path('../resources/')+"/"
device_cmd_fpath      = relative_to_abs_path('../depthai.cmd')
device_usb2_cmd_fpath = relative_to_abs_path('../depthai_usb2.cmd')
boards_dir_path       = relative_to_abs_path('../resources/boards') + "/"
custom_calib_fpath    = relative_to_abs_path('../resources/depthai.calib')
left_mesh_fpath       = relative_to_abs_path('../resources/mesh_left.calib')
right_mesh_fpath      = relative_to_abs_path('../resources/mesh_right.calib')

nn_resource_path      = relative_to_abs_path('../resources/nn')+"/"
blob_fpath            = relative_to_abs_path('../resources/nn/mobilenet-ssd/mobilenet-ssd.blob')
blob_config_fpath     = relative_to_abs_path('../resources/nn/mobilenet-ssd/mobilenet-ssd.json')
tests_functional_path = relative_to_abs_path('../testsFunctional/') + "/"
eeprom_fail_path      = relative_to_abs_path('../resources/images/eeprom_fail.PNG')
calib_fail_path       = relative_to_abs_path('../resources/images/calib_fail.PNG')
pass_path             = relative_to_abs_path('../resources/images/pass.PNG')
usb_3_failed          = relative_to_abs_path('../resources/images/usb_3_Failed.PNG')
requirements_path     = relative_to_abs_path('../requirements.txt')
rgb_camera_not_found  = relative_to_abs_path('../resources/images/rgb_camera_failed.PNG')
mono_camera_not_found = relative_to_abs_path('../resources/images/stereo_camera_failed.PNG')


if custom_calib_fpath is not None and Path(custom_calib_fpath).exists():
    calib_fpath = custom_calib_fpath
    print("Using Custom Calibration File: depthai.calib")
else:
    calib_fpath = ''
    print("No calibration file. Using Calibration Defaults.")