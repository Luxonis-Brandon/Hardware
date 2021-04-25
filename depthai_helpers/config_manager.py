import os
import platform
import subprocess
import urllib.request
from pathlib import Path
import depthai as dai

import blobconverter

from depthai_helpers.cli_utils import cli_print, PrintColors

default_input_dims = {
    # TODO remove once fetching input size from nn blob is possible
    "mobilenet-ssd": "300x300",
    "face-detection-adas-0001": "672x384",
    "face-detection-retail-0004": "300x300",
    "pedestrian-detection-adas-0002": "672x384",
    "person-detection-retail-0013": "544x320",
    "person-vehicle-bike-detection-crossroad-1016": "512x512",
    "vehicle-detection-adas-0002": "672x384",
    "vehicle-license-plate-detection-barrier-0106": "300x300",
    "tiny-yolo-v3": "416x416",
    "yolo-v3": "416x416"
}
DEPTHAI_ZOO = Path(__file__).parent.parent / Path(f"resources/nn/")


class ConfigManager:
    labels = ""
    NN_config = None
    custom_fw_commit = ''

    def __init__(self, args):
        self.args = args

    @property
    def debug(self):
        return not self.args.no_debug

    @property
    def useCamera(self):
        return not self.args.video

    @property
    def useHQ(self):
        return self.args.high_quality

    @property
    def useDepth(self):
        return not self.args.disable_depth

    def getModelSource(self):
        if not self.useCamera:
            return "host"
        if self.args.camera == "left":
            if self.useDepth:
                return "rectified_left"
            return "left"
        if self.args.camera == "right":
            if self.useDepth:
                return "rectified_right"
            return "right"
        if self.args.camera == "color":
            return "rgb"

    def getModelName(self):
        if self.args.cnn_model:
            return self.args.cnn_model
        model_dir = self.getModelDir()
        if model_dir is not None:
            return Path(model_dir).stem

    def getModelDir(self):
        if self.args.cnn_path:
            return self.args.cnn_path
        if self.args.cnn_model is not None and (DEPTHAI_ZOO / self.args.cnn_model).exists():
            return DEPTHAI_ZOO / self.args.cnn_model
        print("dupa")

    def getInputSize(self):
        if self.args.cnn_input_size is None:
            if self.args.cnn_model not in default_input_dims:
                raise RuntimeError(
                    "Unable to determine the nn input size. Please use --cnn_input_size flag to specify it in WxW format: -nn-size <width>x<height>")
            return map(int, default_input_dims[self.args.cnn_model].split('x'))
        else:
            return map(int, self.args.cnn_input_size.split('x'))

    def getRgbResolution(self):
        if self.args.rgb_resolution == 2160:
            return dai.ColorCameraProperties.SensorResolution.THE_4_K
        elif self.args.rgb_resolution == 3040:
            return dai.ColorCameraProperties.SensorResolution.THE_12_MP
        else:
            return dai.ColorCameraProperties.SensorResolution.THE_1080_P

    def getMonoResolution(self):
        if self.args.mono_resolution == 720:
            return dai.MonoCameraProperties.SensorResolution.THE_720_P
        elif self.args.mono_resolution == 800:
            return dai.MonoCameraProperties.SensorResolution.THE_800_P
        else:
            return dai.MonoCameraProperties.SensorResolution.THE_400_P

    def getMedianFilter(self):
        if self.args.stereo_median_size == 3:
            return dai.StereoDepthProperties.MedianFilter.KERNEL_3x3
        elif self.args.stereo_median_size == 5:
            return dai.StereoDepthProperties.MedianFilter.KERNEL_5x5
        elif self.args.stereo_median_size == 7:
            return dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
        else:
            return dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF

    def getUsb2Mode(self):
        usb2_mode = False
        if self.args['force_usb2']:
            cli_print("FORCE USB2 MODE", PrintColors.WARNING)
            usb2_mode = True
        else:
            usb2_mode = False
        return usb2_mode

    def getCustomFirmwarePath(self, commit):
        fwdir = '.fw_cache/'
        if not os.path.exists(fwdir):
            os.mkdir(fwdir)
        fw_variant = ''
        if self.getUsb2Mode():
            fw_variant = 'usb2-'
        fname = 'depthai-' + fw_variant + commit + '.cmd'
        path = fwdir + fname
        if not Path(path).exists():
            url = 'https://artifacts.luxonis.com/artifactory/luxonis-myriad-snapshot-local/depthai-device-side/'
            url += commit + '/' + fname
            print('Downloading custom FW:', url)
            # Need this to avoid "HTTP Error 403: Forbidden"
            class CustomURLopener(urllib.request.FancyURLopener):
                version = "Mozilla/5.0"
                # FancyURLopener doesn't report by default errors like 404
                def http_error_default(self, url, fp, errcode, errmsg, headers):
                    raise ValueError(errcode)
            url_opener = CustomURLopener()
            with url_opener.open(url) as response, open(path, 'wb') as outf:
                outf.write(response.read())
        return path

    def getCommandFile(self):
        debug_mode = False
        cmd_file = ''
        if self.args['firmware'] != None:
            self.custom_fw_commit = self.args['firmware']
        if self.args['dev_debug'] == None:
            # Debug -debug flag NOT present, check first for custom firmware
            if self.custom_fw_commit == '':
                debug_mode = False
            else:
                debug_mode = True
                cmd_file = self.getCustomFirmwarePath(self.custom_fw_commit)
        elif self.args['dev_debug'] == '':
            # If just -debug flag is present -> cmd_file = '' (wait for device to be connected beforehand)
            debug_mode = True
        else:
            debug_mode = True
            cmd_file = self.args['dev_debug']

        return cmd_file, debug_mode

    def adjustParamsToDevice(self):
        if self.args.device is None:
            dev_type = input("Device type (OAK = 1, OAK-D = 2): ")
            if dev_type == "1":
                self.args.device = "OAK"
            elif dev_type == "2":
                self.args.device = "OAK-D"
            else:
                raise ValueError("Incorrect device type id supplied: {}".format(dev_type))

        if self.args.device == "OAK":
            if not self.args.disable_depth:
                print("Disabling depth...")
            self.args.disable_depth = True
            if self.args.spatial_bounding_box:
                print("Disabling spatial bounding boxes...")
            self.args.spatial_bounding_box = False
            if self.args.camera != 'color':
                print("Switching source to RGB camera...")
            self.args.camera = 'color'

    def linuxCheckApplyUsbRules(self):
        if platform.system() == 'Linux':
            ret = subprocess.call(['grep', '-irn', 'ATTRS{idVendor}=="03e7"', '/etc/udev/rules.d'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if(ret != 0):
                cli_print("\nWARNING: Usb rules not found", PrintColors.WARNING)
                cli_print("\nSet rules: \n"
                """echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules \n"""
                "sudo udevadm control --reload-rules && udevadm trigger \n"
                "Disconnect/connect usb cable on host! \n", PrintColors.RED)
                os._exit(1)


class BlobManager:
    def __init__(self, model_name=None, model_dir=None):
        self.model_dir = None
        self.zoo_dir = None
        self.config_file = None
        self.use_zoo = False
        if model_dir is None:
            self.model_name = model_name
            self.use_zoo = True
        else:
            self.model_dir = Path(model_dir)
            self.zoo_dir = self.model_dir.parent
            self.model_name = model_name or self.model_dir.name
            self.config_file = self.model_dir / "model.yml"
            if not self.config_file.exists():
                self.use_zoo = True

        self.blob_path = None

    def compile(self, shaves, target='auto'):
        if self.use_zoo:
            return blobconverter.from_zoo(name=self.model_name, shaves=shaves)
        else:
            return blobconverter.compile_blob(
                blob_name=self.model_name,
                req_data={
                    "name": self.model_name,
                    "use_zoo": True,
                },
                req_files={
                    'config': self.config_file,
                },
                data_type="FP16",
                shaves=shaves,
            )
