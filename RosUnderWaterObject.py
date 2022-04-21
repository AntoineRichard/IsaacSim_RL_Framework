from os import posix_fadvise
import omni
import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from omni.isaac.imu_sensor import _imu_sensor
import carb

class ROSHeronWrapper():
    def __init__(self, stage, PhysXIFace, DCIFace, IMUIFace, HydroSettings, ThrusterSettings, namespace="/robot"):
        # Load plugins after kit is loaded
        from UnderWaterObject import UnderWaterObject
        from Thruster import ThrusterPlugin
        
 
        # Hydrodynamics simulation with python plugin
        self.UWO = UnderWaterObject(stage, PhysXIFace, DCIFace)
        self.UWO.Load(HydroSettings)
        self.THR = []
        for i, settings in enumerate(ThrusterSettings):
            self.THR.append(ThrusterPlugin(stage, PhysXIFace, DCIFace))
            self.THR[i].Load(settings)
        # Add IMU
        self.IMUIFace = IMUIFace
        self.makeIMU()
        self.imu_pub = rospy.Publisher(namespace+"/imu", Imu, queue_size=1)
        # Add Perfect Pose Sensor
        self.makePerfectPoseSensor()
        self.pp_pub = rospy.Publisher(namespace+"/pose_gt", Odometry, queue_size=1)
        # Add Noisy GPS
        self.makeNavSatFixSensor()
        self.nfs_pub = rospy.Publisher(namespace+"/NavSatFix", NavSatFix, queue_size=1)
        # ROS callbacks
        rospy.Subscriber(namespace+"/left",  Float32, self.leftCallback)
        rospy.Subscriber(namespace+"/right", Float32, self.rightCallback)
        rospy.Subscriber(namespace+"/flow_vel", Vector3, self.flowVelCallback)
    
    def Update(self, dt):
        self.UWO.Update(dt)
        for i in self.THR:
            i.Update(dt)
        self.publishIMUMessage()
        self.publishPerfectPoseMessage()
    
    def leftCallback(self, data):
        self.THR[1].UpdateCommand(data.data)

    def rightCallback(self, data):
        self.THR[0].UpdateCommand(data.data)
    
    def flowVelCallback(self, data):
        self.UWO.UpdateFlowVelocity(data.data)
    
    def makeIMU(self):
        #TODO make class, get data from user, add noise
        props = _imu_sensor.SensorProperties()
        props.position = carb.Float3(0, 0, 0)
        props.orientation = carb.Float4(0, 0, 0, 1)
        props.sensorPeriod = 0
        self.imu_handle = self.IMUIFace.add_sensor_on_body("/heron/imu_link", props)
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = 'heron/imu_link'

    def makePerfectPoseSensor(self):
        #TODO make class, get data from user, add noise
        self.pp_msg = Odometry()
        self.pp_msg.header.frame_id = 'heron/base_link'

    def makeNavSatFixSensor(self):
        #TODO make class, get data from user, add noise
        self.nsf_msg = NavSatFix()
        self.nsf_msg.header.frame_id = 'heron/base_link'
        status = NavSatStatus()
        status.status = 0
        status.service = 1
        self.nsf_msg.status = status
        self.nsf_msg.position_covariance_type = 2
        self.nsf_msg.position_covariance = [0.05,0,0,0,0.05,0,0,0,0.05]
 
    def publishIMUMessage(self):
        #TODO make class, get data from user, add noise
        data = self.IMUIFace.get_sensor_readings(self.imu_handle)
        print(data)
        self.imu_msg.header.stamp = rospy.Time.from_seconds(data[0][0])
        self.imu_msg.angular_velocity.x = data[0][4]
        self.imu_msg.angular_velocity.y = data[0][5]
        self.imu_msg.angular_velocity.z = data[0][6]
        self.imu_msg.linear_acceleration.x = data[0][1]
        self.imu_msg.linear_acceleration.y = data[0][2]
        self.imu_msg.linear_acceleration.z = data[0][3]
        self.imu_pub.publish(self.imu_msg)