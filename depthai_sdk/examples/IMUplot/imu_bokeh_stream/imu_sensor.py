import time
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from threading import Event, Lock, Thread
from time import sleep
from typing import TYPE_CHECKING, Dict, List

import depthai as dai
from bokeh_plot import BokehPage, BokehPlot, LayoutDefaults
from stack import RollingStack

if TYPE_CHECKING:
    from bokeh_plot import BokehPlot


@dataclass
class SensorDetails:
    legend: Dict[str, str]
    title: str

    delay_q: RollingStack
    data_q: RollingStack


class SensorTag(Enum):
    ACCELEROMETER = auto()
    GYROSCOPE = auto()
    MAGNETOMETER = auto()


class SensorProducer(Thread):
    def __init__(self, details: SensorDetails, sensor_is_reading: Event) -> None:
        """Init Sensor Producer

        Args:
            details (SensorDetails): Details on how to plot sensor vals and queues
                                     to share data between threads
            sensor_is_reading (Event): Used to stop start plotting WIP
        """
        Thread.__init__(self)
        self.details = details
        self.sensor_is_reading = sensor_is_reading

        self.start_time = self.current_milli_time()
        self.x = self.start_time

        self.data = dict()
        self.details.data_q.append(self.data)

        self.pipeline = self.init_oak_reader()

    def init_oak_reader(self):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        imu = pipeline.create(dai.node.IMU)
        xlinkOut = pipeline.create(dai.node.XLinkOut)

        xlinkOut.setStreamName("imu")

        # enable ACCELEROMETER_RAW at 500 hz rate
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER, 500)
        # enable GYROSCOPE_RAW at 400 hz rate
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, 400)
        # enable MAGNETMOETER_RAW at 400 hz rate
        imu.enableIMUSensor(dai.IMUSensor.MAGNETOMETER_CALIBRATED, 400)
        # it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
        # above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
        imu.setBatchReportThreshold(1)
        # maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
        # if lower or equal to batchReportThreshold then the sending is always blocking on device
        # useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
        imu.setMaxBatchReports(10)

        # Link plugins IMU -> XLINK
        imu.out.link(xlinkOut.input)

        return pipeline

    def run(self):
        # Pipeline is defined, now we can connect to the device and get data
        with dai.Device(self.pipeline) as device:

            def timeDeltaToMilliS(delta) -> float:
                return delta.total_seconds() * 1000

            # Output queue for imu bulk packets
            imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
            baseTs = None

            self.data = None

            while True:
                imuData = (
                    imuQueue.get()
                )  # blocking call, will wait until a new data has arrived

                imuPackets = imuData.packets
                for imuPacket in imuPackets:
                    acceleroValues = imuPacket.acceleroMeter
                    gyroValues = imuPacket.gyroscope
                    magnetValues = imuPacket.magneticField

                    acceleroTs = acceleroValues.getTimestampDevice()
                    gyroTs = gyroValues.getTimestampDevice()
                    magnetTs = magnetValues.getTimestampDevice()

                    if baseTs is None:
                        baseTs = acceleroTs if acceleroTs < gyroTs else gyroTs

                    acceleroTs = timeDeltaToMilliS(acceleroTs - baseTs)
                    gyroTs = timeDeltaToMilliS(gyroTs - baseTs)
                    magnetTs = timeDeltaToMilliS(magnetTs - baseTs)

                    # x,y,z in frame of reference of horizontal cam
                    data = dict()
                    data[SensorTag.ACCELEROMETER] = dict(
                        x=acceleroTs,
                        y=acceleroValues.y,
                        y1=acceleroValues.z,
                        y2=acceleroValues.x,
                    )

                    data[SensorTag.GYROSCOPE] = dict(
                        x=gyroTs,
                        y=gyroValues.y,
                        y1=gyroValues.z,
                        y2=gyroValues.x,
                    )

                    data[SensorTag.MAGNETOMETER] = dict(
                        x=magnetTs,
                        y=magnetValues.y,
                        y1=magnetValues.z,
                        y2=magnetValues.x,
                    )

                    self.details.data_q.append(data)

    def mean(self, vals: List[Dict]) -> Dict:
        """Used to smooth data

        Args:
            vals (List[Dict]): List of sensor history

        Returns:
            Dict: mean values of recorded values
        """
        history_len = len(vals)
        res = {
            SensorTag.ACCELEROMETER: {},
            SensorTag.GYROSCOPE: {},
            SensorTag.MAGNETOMETER: {},
        }
        for key in ["x", "y", "y1", "y2"]:
            for tag in SensorTag:
                mymean = 0
                for i in range(history_len):
                    mymean += vals[i][tag][key]
                res[tag][key] = [mymean / history_len]

        # calculate magnitude
        for tag in SensorTag:
            res[tag]["y3"] = [
                (
                    res[tag]["y"][0] ** 2
                    + res[tag]["y1"][0] ** 2
                    + res[tag]["y2"][0] ** 2
                )
                ** 0.5
            ]

        return res

    def read(self, sensor_tag: Enum) -> Dict:
        """Get latest stored values

        Args:
            sensor_tag (Enum): tag for each plot

        Returns:
            Dict: mean sensor values in Bokeh format
        """
        vals = self.details.data_q.all()

        if vals[0]:
            return self.mean(vals)[sensor_tag]
        return {}

    def current_milli_time(self, start_time=0):
        return round(time.time() * 1000) - start_time


class SensorConsumer(Thread):
    def __init__(
        self,
        plt: "BokehPlot",
        sensor: SensorProducer,
        sensor_is_reading: Event,
        sensor_tag: str,
    ):
        """_summary_

        Args:
            plt (BokehPlot): plot to display the data
            sensor (SensorProducer):  class that supplies sensor data
            sensor_is_reading (Event): is plotting state
            sensor_tag (str): identifies plot
        """
        Thread.__init__(self)

        self.sensor_tag = sensor_tag
        self.sensor = sensor
        self.sensor_is_reading = sensor_is_reading
        self.threadLock = Lock()

        self.sensor_callback = plt.update
        self.bokeh_callback = plt.doc.add_next_tick_callback

    def run(self):
        """Generate data"""
        while True:
            time.sleep(self.sensor.details.delay_q.latest())

            if self.sensor_is_reading.is_set():
                with self.threadLock:
                    latest = self.sensor.read(self.sensor_tag)

                    if latest:
                        self.bokeh_callback(partial(self.sensor_callback, latest))
                    else:
                        sleep(1)


def init_oak_imu():
    """Create live plots"""
    n_plots = 3
    rolling_mean = 1
    sensor_speed_slider_value = 0.005 * n_plots
    sensor_is_reading = Event()
    sensor_is_reading.set()

    delay_queue = RollingStack(1, sensor_speed_slider_value)
    data_q = RollingStack(rolling_mean)

    accel_deets = SensorDetails(
        {"y": "Accel(x)", "y1": "Accel(y)", "y2": "Accel(z)", "y3": "Magnitude"},
        "Accelerometer",
        delay_queue,
        data_q,
    )

    gyro_deets = SensorDetails(
        {"y": "Gyro(x)", "y1": "Gyro(y)", "y2": "Gyro(z)", "y3": "Magnitude"},
        "Gyroscope",
        delay_queue,
        data_q,
    )

    magnet_deets = SensorDetails(
        {"y": "Magnet(x)", "y1": "Magnet(y)", "y2": "Magnet(z)", "y3": "Magnitude"},
        "Magnetometer",
        delay_queue,
        data_q,
    )

    plots = []

    main_page = BokehPage(
        LayoutDefaults(
            delay_queue, sensor_speed_slider_value=sensor_speed_slider_value
        ),
        sensor_is_reading,
    )

    producer = SensorProducer(accel_deets, sensor_is_reading)
    producer.start()

    for deets, tag in [
        (magnet_deets, SensorTag.MAGNETOMETER),
        (gyro_deets, SensorTag.GYROSCOPE),
        (accel_deets, SensorTag.ACCELEROMETER),
    ]:
        plt = BokehPlot(main_page, deets)
        consumer = SensorConsumer(plt, producer, sensor_is_reading, tag)

        plots.append(plt)
        consumer.start()

    main_page.add_plots(plots)
