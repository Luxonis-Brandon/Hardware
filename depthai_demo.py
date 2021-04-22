#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import random
import cv2
import depthai as dai
import numpy as np

from depthai_helpers.arg_manager import parse_args
from depthai_helpers.config_manager import BlobManager, ConfigManager
from depthai_helpers.utils import frame_norm, to_planar, to_tensor_result

conf = ConfigManager(parse_args())

in_w, in_h = conf.getInputSize()
rgb_res = conf.getRgbResolution()
mono_res = conf.getMonoResolution()
median = conf.getMedianFilter()


class NNetManager:
    source_choices = ("rgb", "left", "right", "host")
    config = None
    nn_family = None
    confidence = None
    metadata = None
    output_format = None
    device = None
    input = None
    output = None
    sbb = False

    def __init__(self, source, use_depth, use_hq, model_dir=None, model_name=None):
        if source not in self.source_choices:
            raise RuntimeError(f"Source {source} is invalid, available {self.source_choices}")

        self.model_name = model_name
        self.model_dir = model_dir
        self.source = source
        self.use_depth = use_depth
        self.use_hq = use_hq
        self.output_name = f"{self.model_name}_out"
        self.input_name = f"{self.model_name}_in"
        self.blob_path = BlobManager(model_dir=self.model_dir, model_name=self.model_name).compile(conf.args.shaves)

        if model_dir is not None:
            config_path = self.model_dir / Path(self.model_name).with_suffix(f".json")
            if config_path.exists():
                with config_path.open() as f:
                    self.config = json.load(f)
                    nn_config = self.config.get("NN_config", {})
                    self.labels = self.config.get("mappings", {}).get("labels", None)
                    self.nn_family = nn_config.get("NN_family", None)
                    self.output_format = nn_config.get("output_format", None)
                    self.metadata = nn_config.get("NN_specific_metadata", {})

                    self.confidence = self.metadata.get("confidence_threshold", nn_config.get("confidence_threshold", None))

                    # Disaply depth roi bounding boxes
                    self.sbb = conf.args.spatial_bounding_box and self.use_depth and self.nn_family in ("YOLO", "mobilenet")

    def addDevice(self, device):
        self.device = device
        self.input = device.getInputQueue(self.input_name, maxSize=1, blocking=False) if self.source == "host" else None
        self.output = device.getOutputQueue(self.output_name, maxSize=1, blocking=False)

        if self.sbb:
            self.sbb_out = device.getOutputQueue("sbb", maxSize=1, blocking=False)
            self.depth_out = device.getOutputQueue("depth", maxSize=1, blocking=False)

    def create_nn_pipeline(self, p, nodes):
        if self.nn_family == "mobilenet":
            nn = p.createMobileNetSpatialDetectionNetwork() if self.use_depth else p.createMobileNetDetectionNetwork()
            nn.setConfidenceThreshold(self.confidence)
        elif self.nn_family == "YOLO":
            nn = p.createYoloSpatialDetectionNetwork() if self.use_depth else p.createYoloDetectionNetwork()
            nn.setConfidenceThreshold(self.confidence)
            nn.setNumClasses(self.metadata["classes"])
            nn.setCoordinateSize(self.metadata["coordinates"])
            nn.setAnchors(self.metadata["anchors"])
            nn.setAnchorMasks(self.metadata["anchor_masks"])
            nn.setIouThreshold(self.metadata["iou_threshold"])
        else:
            # TODO use createSpatialLocationCalculator
            nn = p.createNeuralNetwork()

        nn.setBlobPath(str(self.blob_path))
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        nn.input.setQueueSize(2)
        xout = p.createXLinkOut()
        xout.setStreamName(self.output_name)
        nn.out.link(xout.input)
        setattr(nodes, self.model_name, nn)
        setattr(nodes, self.output_name, xout)
        if self.source == "rgb":
            nodes.cam_rgb.preview.link(nn.input)
        elif self.source == "host":
            xin = p.createXLinkIn()
            xin.setStreamName(self.input_name)
            xin.out.link(nn.input)
            setattr(nodes, self.input_name, xout)
        elif self.source == "right": # Use spatial information
            # Set XLinkOut sources
            if conf.args.sync:
                nn.passthrough.link(nodes.xout_right.input)
            if self.sbb:
                if conf.args.sync:
                    nn.passthroughDepth.link(nodes.xout_depth.input)
                # If we want to display spatial bounding boxes, create XLinkOut node SBBs:
                xout_sbb = p.createXLinkOut()
                xout_sbb.setStreamName("sbb")
                nn.boundingBoxMapping.link(xout_sbb.input)

            # NN inputs
            nodes.manip.out.link(nn.input)
            if conf.useDepth and self.nn_family in ("mobilenet", "YOLO"):
                nodes.stereo.depth.link(nn.inputDepth)
                # Spatial configs
                nn.setBoundingBoxScaleFactor(conf.args.sbb_scale_factor)
                nn.setDepthLowerThreshold(100)
                nn.setDepthUpperThreshold(3000)

    def get_label_text(self, label):
        if self.config is None or self.labels is None:
            return label
        elif int(label) < len(self.labels):
            return self.labels[int(label)]
        else:
            print(f"Label out of bounds (label_index: {label}, available_labels: {len(self.labels)}")
            return label


class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if not conf.useCamera:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.time()
            if delay > 0:
                time.sleep(delay)
        self.timestamp = time.time()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.time()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            time_diff = time.time() - self.ticks[name]
            return self.ticks_cnt[name] / time_diff if time_diff != 0 else 0
        else:
            return 0

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


class PipelineManager:
    def __init__(self, nn_manager):
        self.p = dai.Pipeline()
        self.nodes = SimpleNamespace()
        self.nn_manager = nn_manager

    def create_color_cam(self, use_hq):
        # Define a source - color camera
        self.nodes.cam_rgb = self.p.createColorCamera()
        self.nodes.cam_rgb.setPreviewSize(in_w, in_h)
        self.nodes.cam_rgb.setInterleaved(False)
        self.nodes.cam_rgb.setResolution(rgb_res)
        self.nodes.cam_rgb.setFps(conf.args.rgb_fps)
        xout_rgb = self.p.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        if use_hq:
            self.nodes.cam_rgb.video.link(xout_rgb.input)
        else:
            self.nodes.cam_rgb.preview.link(xout_rgb.input)

    def create_depth(self, dct, median, lr):
        self.nodes.stereo = self.p.createStereoDepth()
        self.nodes.stereo.setOutputDepth(True)
        self.nodes.stereo.setOutputRectified(True)
        self.nodes.stereo.setConfidenceThreshold(dct)
        self.nodes.stereo.setMedianFilter(median)
        self.nodes.stereo.setLeftRightCheck(lr)

        # Create mono left/right cameras if we haven't already
        if not hasattr(self.nodes, 'mono_left'): self.create_left_cam(create_xout=False)
        if not hasattr(self.nodes, 'mono_right'): self.create_right_cam(create_xout=True)

        self.nodes.mono_left.out.link(self.nodes.stereo.left)
        self.nodes.mono_right.out.link(self.nodes.stereo.right)

        if self.nn_manager.sbb: # If we want to displaty SBBs, we need to send depth frames as well (via XLink)
            self.nodes.xout_depth = self.p.createXLinkOut()
            self.nodes.xout_depth.setStreamName("depth")
            # If we don't want camera/NN in sync, set the disparity as the source for the XOut
            if not conf.args.sync:
                self.nodes.stereo.depth.link(self.nodes.xout_depth.input)

    def create_left_cam(self, create_xout):
        self.nodes.mono_left = self.p.createMonoCamera()
        self.nodes.mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.nodes.mono_left.setResolution(mono_res)
        self.nodes.mono_left.setFps(conf.args.mono_fps)        

        if create_xout:
            self.nodes.xout_left = self.p.createXLinkOut()
            self.nodes.xout_left.setStreamName("left")

    def create_right_cam(self, create_xout):
        self.nodes.mono_right = self.p.createMonoCamera()
        self.nodes.mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        self.nodes.mono_right.setResolution(mono_res)
        self.nodes.mono_right.setFps(conf.args.mono_fps)

        if create_xout:
            self.nodes.xout_right = self.p.createXLinkOut()
            self.nodes.xout_right.setStreamName("right")

    def create_nn(self):
        # If we want depth information , create ImageManip that will be the input for the NN
        if conf.useDepth:
            self.nodes.manip = self.p.createImageManip()
            self.nodes.manip.initialConfig.setResize(in_w, in_h)
            # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
            self.nodes.manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
            # Set the stereo node's rectifiedRight stream as the ImageManip's source
            self.nodes.stereo.rectifiedRight.link(self.nodes.manip.inputImage)

            # If we don't want camera/NN in sync, set the rectifiedRight as the source for the XOut
            if not conf.args.sync:
                self.nodes.manip.out.link(self.nodes.xout_right.input)

        if callable(self.nn_manager.create_nn_pipeline):
            self.nn_manager.create_nn_pipeline(self.p, self.nodes)

nn_manager = NNetManager(
    model_name=conf.getModelName(),
    model_dir=conf.getModelDir(),
    source=conf.getModelSource(),
    use_depth=conf.useDepth,
    use_hq=conf.useHQ
)

pm = PipelineManager(nn_manager)

if conf.useDepth:
    pm.create_depth(conf.args.disparity_confidence_threshold, median, conf.args.stereo_lr_check)
elif conf.useCamera:
    pm.create_color_cam(conf.useHQ)

pm.create_nn()


# Pipeline is defined, now we can connect to the device
with dai.Device(pm.p) as device:
    # Start pipeline
    device.startPipeline()
    nn_manager.addDevice(device)
    if conf.useDepth:
        q_right = device.getOutputQueue(name="right", maxSize=1, blocking=False)
        fps = FPSHandler()
    elif conf.useCamera:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        fps = FPSHandler()
    else:
        cap = cv2.VideoCapture(conf.args.video)
        fps = FPSHandler(cap)
        seq_num = 0

    frame = None
    detections = []
    # Spatial bounding box ROIs (region of interests)
    colors = list(np.random.random(size=3) * 256) # Random Colors for bounding boxes 
    color = (255, 255, 255)

    while True:
        fps.next_iter()
        if conf.useDepth:
            in_right = q_right.get()
            frame = in_right.getCvFrame()
            # Since rectified frames are horizontally flipped by default
            # if not hq:
            frame = cv2.flip(frame, 1)
            fps.tick('right')

            if nn_manager.sbb: # Get spatial bounding boxes and depth map
                depth_frame = nn_manager.depth_out.get().getFrame()
                depth_frame = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depth_frame = cv2.equalizeHist(depth_frame)
                depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_TURBO)

                sbb = nn_manager.sbb_out.tryGet()
                sbb_rois = sbb.getConfigData() if sbb is not None else []
                for roi_data in sbb_rois:
                    roi = roi_data.roi
                    roi = roi.denormalize(depth_frame.shape[1], depth_frame.shape[0])
                    top_left = roi.topLeft()
                    bottom_right = roi.bottomRight()
                    # Display SBB on the disparity map
                    cv2.rectangle(depth_frame, (int(top_left.x), int(top_left.y)), (int(bottom_right.x), int(bottom_right.y)), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                cv2.imshow("disparity", depth_frame)

        elif conf.useCamera:
            in_rgb = q_rgb.get()
            if in_rgb is not None:
                if conf.useHQ:
                    yuv = in_rgb.getData().reshape((in_rgb.getHeight() * 3 // 2, in_rgb.getWidth()))
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                else:
                    frame_data = in_rgb.getData().reshape(3, in_rgb.getHeight(), in_rgb.getWidth())
                    frame = np.ascontiguousarray(frame_data.transpose(1, 2, 0))
                fps.tick('rgb')
        else:
            read_correctly, vid_frame = cap.read()
            if not read_correctly:
                break

            scaled_frame = cv2.resize(vid_frame, (in_w, in_h))
            frame_nn = dai.ImgFrame()
            frame_nn.setSequenceNum(seq_num)
            frame_nn.setWidth(in_w)
            frame_nn.setHeight(in_h)
            frame_nn.setData(to_planar(scaled_frame))
            nn_manager.input.send(frame_nn)
            seq_num += 1

            # if high quality, send original frames
            frame = vid_frame if conf.useHQ else scaled_frame
            fps.tick('rgb')

        in_nn = nn_manager.output.tryGetAll()
        if len(in_nn) > 0:
            if nn_manager.output_format == "detection":
                detections = in_nn[-1].detections
            for packet in in_nn:
                if nn_manager.output_format is None:
                    try:
                        print("Received NN packet: ", to_tensor_result(packet))
                    except Exception as ex:
                        print("Received NN packet: <Preview unavailable: {}>".format(ex))
                fps.tick('nn')

        if frame is not None:
            # Scale the frame by --scale factor
            if not conf.args.scale == 1.0:
                h, w, c = frame.shape
                frame = cv2.resize(frame, (int(w * conf.args.scale), int(h * conf.args.scale)), interpolation=cv2.INTER_AREA)

            # if the frame is available, draw bounding boxes on it and show the frame
            for detection in detections:
                if conf.useDepth: # Since rectified frames are horizontally flipped by default
                    swap = detection.xmin
                    detection.xmin = 1 - detection.xmax
                    detection.xmax = 1 - swap
                    
                bbox = frame_norm(frame, [detection.xmin, detection.ymin, detection.xmax, detection.ymax])
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors, 2)
                cv2.rectangle(frame, (bbox[0], (bbox[1] - 28)), ((bbox[0] + 78), bbox[1]), colors, cv2.FILLED)
                cv2.putText(frame, nn_manager.get_label_text(detection.label), (bbox[0] + 5, bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 78, bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                if conf.useDepth: # Display coordinates as well
                    Ztrackbar = (int(detection.spatialCoordinates.z) * 0.05)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (bbox[0] + 10, bbox[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (bbox[0] + 10, bbox[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    cv2.rectangle(frame, ((bbox[0] - 10), (bbox[1] + 35)), ((bbox[0] - 35), (bbox[1] + 150)), (134, 164, 11), 2)
                    cv2.rectangle(frame, ((bbox[0] - 10), (bbox[1] + (150 - int(Ztrackbar)))), ((bbox[0] - 35), (bbox[1] + 150)), (134, 164, 11), cv2.FILLED)


            frame_fps = f"RIGHT FPS: {round(fps.tick_fps('right'), 1)}" if conf.useDepth else f"RGB FPS: {round(fps.tick_fps('rgb'), 1)}"
            cv2.rectangle(frame, (0,0), (120, 40), (255, 255, 255), cv2.FILLED)

            cv2.putText(frame, frame_fps, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
            cv2.putText(frame, f"NN FPS:  {round(fps.tick_fps('nn'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

            cv2.imshow(conf.getModelSource(), frame)
        if cv2.waitKey(1) == ord('q'):
            break
