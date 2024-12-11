AI models
=========

Through the :ref:`NNComponent`, DepthAI SDK abstracts:

1. **AI model sourcing** using `blobconverter <https://github.com/luxonis/blobconverter>`__ from `Open Model Zoo <https://github.com/openvinotoolkit/open_model_zoo>`__ (OMZ) and `DepthAI Model Zoo <https://github.com/luxonis/depthai-model-zoo>`__ (DMZ).
2. **AI result decoding** - currently SDK supports on-device decoding for YOLO and MobileNet based results using `YoloDetectionNetwork <https://docs.luxonis.com/projects/api/en/latest/components/nodes/yolo_detection_network/>`__ and `MobileNetDetectionNetwork <https://docs.luxonis.com/projects/api/en/latest/components/nodes/mobilenet_detection_network/>`__ nodes.
3. **Decoding** of the ``config.json`` which **allows an easy deployment of custom AI models** trained `using our notebooks <https://github.com/luxonis/depthai-ml-training>`__ and converted using https://tools.luxonis.com.
4. Formatting of the AI model input frame - SDK uses **BGR** color order and **Planar / CHW** (Channel, Height, Width) layout conventions. If model accepts color images, it should accept 3 channels (B,G,R), and if it accepts grayscale images, it should accept 1 channel.

.. figure:: https://github.com/luxonis/depthai/assets/18037362/3d2422ca-6ed4-46fe-948e-7860f2ff066f

  Netron app allows you to check model's input layout

5. Integration with 3rd party tools/services (:ref:`Roboflow`).


SDK supported models
####################

With :ref:`NNComponent` you can **easily try out a variety of different pre-trained models** by simply changing the model name:

.. code-block:: diff

    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        color = oak.create_camera('color')
  -     nn = oak.create_nn('mobilenet-ssd', color)
  +     nn = oak.create_nn('vehicle-detection-0202', color)
        oak.visualize([nn], fps=True)
        oak.start(blocking=True)

Both of the models above are supported by this SDK, so they will be downloaded and deployed to the OAK device along with the pipeline.

The following table lists all the models supported by the SDK. The model name is the same as the name used in the :ref:`NNComponent` constructor.

.. list-table::
   :header-rows: 1

   * - Name
     - Model Source
     - FPS*
   * - ``age-gender-recognition-retail-0013``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_age_gender_recognition_retail_0013.html>`__
     - 33
   * - ``emotions-recognition-retail-0003``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_emotions_recognition_retail_0003.html>`__
     - 33
   * - ``face-detection-adas-0001``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_face_detection_adas_0001.html>`__
     - 18
   * - ``face-detection-retail-0004``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_face_detection_retail_0004.html>`__
     - 33
   * - ``facemesh_192x192``
     - `DMZ <https://github.com/luxonis/depthai-model-zoo/tree/main/models/facemesh_192x192>`__
     - 32
   * - ``facial_landmarks_68_160x160``
     - `DMZ <https://github.com/luxonis/depthai-model-zoo/blob/main/models/facial_landmarks_68_160x160>`__
     - 32
   * - ``human-pose-estimation-0001``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_human_pose_estimation_0001.html>`__
     - 8
   * - ``mobilenet-ssd``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_mobilenet_ssd.html>`__
     - 31
   * - ``mobilenetv2_imagenet_embedder_224x224``
     - `DMZ <https://github.com/luxonis/depthai-model-zoo/tree/main/models/mobilenetv2_imagenet_embedder_224x224>`__
     - /
   * - ``pedestrian-detection-adas-0002``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_pedestrian_detection_adas_0002.html>`__
     - 19
   * - ``person-detection-0200``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_person_detection_0200.html>`__
     - 14
   * - ``person-detection-retail-0013``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_person_detection_retail_0013.html>`__
     - 15
   * - ``person-reidentification-retail-0288``
     - `OMZ <https://docs.openvino.ai/cn/2021.4/omz_models_model_person_reidentification_retail_0288.html>`__
     - 33
   * - ``person-vehicle-bike-detection-crossroad-1016``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_person_vehicle_bike_detection_crossroad_1016.html>`__
     - 12
   * - ``sbd_mask_classification_224x224``
     - `DMZ <https://github.com/luxonis/depthai-model-zoo/tree/main/models/sbd_mask_classification_224x224>`__
     - 64+
   * - ``vehicle-detection-0202``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_vehicle_detection_0202.html>`__
     - 14
   * - ``vehicle-detection-adas-0002``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_vehicle_detection_adas_0002.html>`__
     - 14
   * - ``vehicle-license-plate-detection-barrier-0106``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_vehicle_license_plate_detection_barrier_0106.html>`__
     - 29
   * - ``yolo-v3-tf``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_yolo_v3_tf.html>`__
     - 3.5
   * - ``yolo-v3-tiny-tf``
     - `OMZ <https://docs.openvino.ai/2022.1/omz_models_model_yolo_v3_tiny_tf.html>`__
     - 33
   * - ``yolov4_coco_608x608``
     - `DMZ <https://github.com/luxonis/depthai-model-zoo/tree/main/models/yolov4_coco_608x608>`__
     - 1.1
   * - ``yolov4_tiny_coco_416x416``
     - `DMZ <https://github.com/luxonis/depthai-model-zoo/tree/main/models/yolov4_tiny_coco_416x416>`__
     - 32
   * - ``yolov5n_coco_416x416``
     - `DMZ <https://github.com/luxonis/depthai-model-zoo/tree/main/models/yolov5n_coco_416x416>`__
     - 32
   * - ``yolov6n_coco_640x640``
     - `DMZ <https://github.com/luxonis/depthai-model-zoo/tree/main/models/yolov6n_coco_640x640>`__
     - 26
   * - ``yolov6nr3_coco_640x352``
     - `DMZ <https://github.com/luxonis/depthai-model-zoo/tree/main/models/yolov6nr3_coco_640x352>`__
     - 32
   * - ``yolov7tiny_coco_640x352``
     - `DMZ <https://github.com/luxonis/depthai-model-zoo/tree/main/models/yolov7tiny_coco_640x352>`__
     - 23
   * - ``yolov7tiny_coco_416x416``
     - `DMZ <https://github.com/luxonis/depthai-model-zoo/tree/main/models/yolov7tiny_coco_416x416>`__
     - 29
   * - ``yolov8n_coco_640x352``
     - `DMZ <https://github.com/luxonis/depthai-model-zoo/tree/main/models/yolov8n_coco_640x352>`__
     - 22


``*`` - FPS was measured using only color camera (1080P) and 1 NN using callbacks (without visualization)

..
  TODO: add gif for each model

.. include::  ../includes/footer-short.rst
