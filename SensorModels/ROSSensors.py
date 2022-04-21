from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus, Imu
from geometry_msgs.msg import Vector3Stamped

import PhysxUtils as utils
import numpy as np
import UsdPhysics
import rospy
import math
import os

from standalone_examples.python_samples.Buoyancy.TEST_usv_thruster_hydrodynamics import PhysXIFace

class GenericSensor:
    def __init__(self, stage, simulation_handle, settings, namespace=""):
        self.sim_hdl = simulation_handle
        self.settings = settings
        self.namespace = namespace
        self.path = os.path.join("/"+namespace,settings["frame_id"])
        self.msg = None
        self.pub = None

    def getTimeStamp(self):
        return self.sim_hdl.current_time
    
    def publish(self):
        self.pub.publish(self.msg)

class PerfectPoseSensor(GenericSensor):
    def __init__(self, stage, simulation_handle, settings, PhysXIFace, namespace=""):
        super().__init__(self, simulation_handle, settings, namespace=namespace)
        self.PhysXIFace = PhysXIFace
        self.msg = Odometry()
        self.msg.header.frame_id = os.path.join(namespace,settings["frame_id"])
        self.rigidBodyAPI = UsdPhysics.RigidBodyAPI.Get(stage, self.path)
        self.pub = rospy.Publisher(os.path.join(namespace,settings["topic_name"]), Odometry, queue_size=1)
    
    def makeMessage(self):
        self.msg.header.stamp = self.getTimeStamp()

        lin_vel = utils.getLinearVel(self.rigidBodyAPI)
        ang_vel = utils.getAngularVel(self.rigidBodyAPI)
        pose, quat = utils.getPose(self.PhysXIFace,self.path)

        self.msg.pose.pose.position.x = pose[0]
        self.msg.pose.pose.position.y = pose[1]
        self.msg.pose.pose.position.z = pose[2]
        self.msg.pose.pose.orientation.x = quat[0]
        self.msg.pose.pose.orientation.y = quat[1]
        self.msg.pose.pose.orientation.z = quat[2]
        self.msg.pose.pose.orientation.w = quat[3]
        self.msg.twist.twist.linear.x = lin_vel[0]
        self.msg.twist.twist.linear.y = lin_vel[1]
        self.msg.twist.twist.linear.z = lin_vel[2]
        self.msg.twist.twist.angular.x = ang_vel[0]
        self.msg.twist.twist.angular.y = ang_vel[1]
        self.msg.twist.twist.angular.z = ang_vel[2]

class GPSSensor(GenericSensor):
    def __init__(self, stage, simulation_handle, settings, PhysXIFace, namespace=""):
        super().__init__(self, simulation_handle, settings, namespace=namespace)
        self.PhysXIFace = PhysXIFace
        self.sat_msg = NavSatFix()
        self.vel_msg = Vector3Stamped()
        self.sat_msg.header.frame_id = os.path.join(namespace,settings["frame_id"])
        self.vel_msg.header.frame_id = os.path.join(namespace,settings["frame_id"])
        self.rigidBodyAPI = UsdPhysics.RigidBodyAPI.Get(stage, self.path)
        self.sat_pub = rospy.Publisher(os.path.join(namespace,settings["topic_name"]), NavSatFix, queue_size=1)
        self.vel_pub = rospy.Publisher(os.path.join(namespace,settings["topic_name"]), Vector3Stamped, queue_size=1)
        # WGS84 constants
        self.equatorial_radius = 6378137.0
        self.flattening = 1.0/298.257223563
        self.excentrity2 = 2*self.flattening - self.flattening*self.flattening
        self.initializeNavSatFixSensor()

    def initializeNavSatFixSensor(self):
        status = NavSatStatus()
        status.status = 0
        status.service = 1
        self.ref_lat = self.settings["reference_lattitude"]
        self.ref_long = self.settings["reference_longitude"]
        self.ref_alt = self.settings["reference_altitude"]
        self.ref_head = self.settings["reference_heading"]
        self.sat_msg.status = status
        self.sat_msg.position_covariance_type = 2
        self.epsilon = self.settings["uncertainty"]
        self.sat_msg.position_covariance = [self.epsilon,0,0,0,self.epsilon,0,0,0,self.epsilon]
        tmp = 1.0 / (1.0 - self.excentrity2 * math.sin(self.ref_lat * math.pi/180.0) * math.sin(self.ref_lat * math.pi/180.0))
        prime_vertical_radius = self.equatorial_radius * math.sqrt(tmp)
        self.rad_north = prime_vertical_radius * (1 - self.excentrity2) * tmp
        self.rad_east  = prime_vertical_radius * math.cos(self.ref_lat * math.pi/180.0)
    
    def makeMessage(self):
        self.sat_msg.header.stamp = rospy.Time.from_sec(self.getTimeStamp())
        self.vel_msg.header.stamp = self.sat_msg.header.stamp

        pose, quat = utils.getPose(self.PhysXIFace,self.path)
        lin_vel = utils.getLinearVel(self.rigidBodyAPI)

        pose += np.random.normal(0, self.epsilon, [3])
        self.sat_msg.latitude  = self.ref_lat+(math.cos(self.ref_head)*pose[0]+math.sin(self.ref_head)*pose[1])/self.rad_north*180.0/math.pi
        self.sat_msg.longitude = self.ref_long-(-math.sin(self.ref_head)*pose[0]+math.cos(self.ref_head)*pose[1])/self.rad_east*180.0/math.pi
        self.sat_msg.altitude  = self.ref_alt + pose[-1]
        self.vel_msg.vector.x = lin_vel[0]*math.cos(self.ref_head) + lin_vel[1]*math.sin(self.ref_head)
        self.vel_msg.vector.y = - lin_vel[0]*math.sin(self.ref_head) + lin_vel[1]*math.cos(self.ref_head)
        self.vel_msg.vector.y = lin_vel[2]
    
    def publish(self):
        self.sat_pub.publish(self.sat_msg)
        self.vel_pub.publish(self.vel_msg)

        
class IMUSensor(GenericSensor): 
    def __init__(self, stage, simulation_handle, settings, PhysXIFace, namespace=""):
        super().__init__(self, simulation_handle, settings, namespace=namespace)
        self.PhysXIFace = PhysXIFace
        self.msg = Imu()

    def publishIMUMessage(self):
        data = self.IMUIFace.get_sensor_readings(self.imu_handle)
        self.imu_msg.header.stamp = rospy.Time.from_seconds(data[0][0])
        self.imu_msg.angular_velocity.x = data[0][4]
        self.imu_msg.angular_velocity.y = data[0][5]
        self.imu_msg.angular_velocity.z = data[0][6]
        self.imu_msg.linear_acceleration.x = data[0][1]
        self.imu_msg.linear_acceleration.y = data[0][2]
        self.imu_msg.linear_acceleration.z = data[0][3]
        self.imu_pub.publish(self.imu_msg)

    def publishNavSatFixMessage(self):
        self.nsf_msg.header.stamp = self.imu_msg.header.stamp
        pose = self.UWO.getPose()
        self.nsf_msg.altitude = pose[-1]