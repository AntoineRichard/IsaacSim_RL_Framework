import os
import omni
import carb
from pxr import Gf, UsdGeom
from omni.isaac.python_app import OmniKitHelper
import time
import numpy as np

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": False,
}

if __name__ == "__main__":

    omniverse_kit = OmniKitHelper(CONFIG)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    omniverse_kit.set_setting("/app/window/drawMouse", True)
    omniverse_kit.set_setting("/app/livestream/proto", "ws")
    ext_manager.set_extension_enabled_immediate("omni.physx.bundle", True)
    ext_manager.set_extension_enabled_immediate("omni.syntheticdata", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.livestream.core", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.livestream.native", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.window.stage", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.property.bundle", True)
    import utils
    from omni.isaac.dynamic_control import _dynamic_control
    #from omni.physx import get_physx_scene_quety_interface
    from python_samples.Buoyancy.PhysxUtils import Quaternion2RotationMatrix, getRelativeLinearVel, AddRelativeTorque, getRelativeAngularVel
    from omni.physx import get_physx_interface, get_physx_authoring_interface
    from pxr import UsdPhysics
    nucleus_server = utils.get_nucleus_server()
    asset_path = nucleus_server + "/Isaac/Props/Blocks/nvidia_cube.usd"
    #asset_path = nucleus_server + "/LakeSimulation/heron3.usd"
    #asset_path = nucleus_server + "/Isaac/Robots/Jetbot/jetbot.usd"
    stage = omni.usd.get_context().get_stage()
    utils.setup_cpu_physics(stage, "/World/physicsScene")
    #utils.create_ground_plane(stage, "/World/ground_plane", size=100)

    dc = _dynamic_control.acquire_dynamic_control_interface()
    physxIFace = get_physx_interface()
    PhysXAuthIFace = get_physx_authoring_interface()
    global t
    global it
    t = 0
    it = 0

    def physics_update(dt: float):
        global t, it
        #prims[-1] = '/heron/dummy_link'
        #prims[-1] = '/heron/chassis'
        #prim1 = '/heron/Group/Cube_01'
        #prim2 = '/heron/Group/Cube_02'
        #height = 19.5
        dt = PhysXAuthIFace.get_elapsed_time()
        #mass = 1.0
        #volume = height*height*height
        #physxIFace = get_physx_interface()
        transform1 = physxIFace.get_rigidbody_transformation(prims[-1])
        rot = Quaternion2RotationMatrix(transform1['rotation'])
        rot_mat_inv = np.linalg.inv(rot)
        print(it, it*1/120.0, t, getRelativeAngularVel(rigidbody,rot_mat_inv))
        
        t += dt
        it += 1
        #transform2 = physxIFace.get_rigidbody_transformation(prim2)
        #com = np.array([-5.06677653e-01,1.70636084e-06,-1.17833001e+01])
        #com = np.array([0,0,-13])
        #com = np.matmul(np.linalg.inv(rot),com)
        #com = np.matmul(rot,com)
        #print(transform1['position'], com)
        #com = np.array([0,0,0])
        #z = transform['position'][2]
        #x = transform['position'][0]
        #y = transform['position'][1]
        #vol = 0

        #if (z + height/2) < 0:
        #    vol = volume
        #elif (z - height/2) < 0:
        #    vol = volume * ((height/2) - z)/ height

        #force = 981.1*29.307
        #force = 10*981.1*1/2
        #print(z, force/981.1)
        AddRelativeTorque(physxIFace,prims[-1],np.array([100,0,0]),dist=1)
        #physxIFace.apply_force_at_pos(prims[-1], carb._carb.Float3([0,0,100]), np.array(transform1['position']))
        #physxIFace.apply_force_at_pos(prim1, carb._carb.Float3([0,0,force]),np.array(transform1['position']))
        #physxIFace.apply_force_at_pos(prim2, carb._carb.Float3([0,0,force]),np.array(transform2['position']))
        #physxIFace.apply_force_at_pos(prims[-1], carb._carb.Float3([0,0,100]),carb._carb.Float3([0,0,0]))
        #physxIFace.apply_force_at_pos(prims[-1], carb._carb.Float3([1000,0,0]),carb._carb.Float3([0,100,0]))
        #physxIFace.apply_force_at_pos(prims[-1], carb._carb.Float3([-1000,0,0]),carb._carb.Float3([0,-100,0]))
        #rb = dc.get_rigid_body(prims[-1])
        #dc.apply_body_force(rb, carb._carb.Float3([0,0,29.302*981.1]),[0,0,0])#transform1['position'])

    physics_sub = omni.physx.acquire_physx_interface().subscribe_physics_step_events(physics_update)

    prims = []

    position = Gf.Vec3d(0, 0, 0)
    #prims = utils.createObject('/heron',stage,asset_path,False,position=position, group=prims, density=1, scale=Gf.Vec3d(3,3,3), allow_physics=False)
    prims = utils.createObject('/World/nvidia_cube',stage,asset_path,False,position=position, group=prims, density=1, scale=Gf.Vec3d(3,3,3), allow_physics=True)
    rigidbody = UsdPhysics.RigidBodyAPI.Get(stage, prims[-1])
    #prim = stage.GetPrimAtPath(prims[-1])
    #print(UsdGeom.Boundable(prim).ComputeLocalBound(0,"default").GetRange().GetSize())

    #time.sleep(10)
    while omniverse_kit.app.is_running():
        omniverse_kit.update(1.0/30, physics_dt=1/120.0)
    
    omniverse_kit.stop()
    omni.usd.get_context().save_as_stage(nucleus_server + "/Users/test/saved.usd", None)
    time.sleep(10)
    omniverse_kit.shutdown()

""" TEST SCRIPT THRUST HERON
import omni
from pxr import UsdPhysics
from omni.physx import get_physx_interface
from scipy.spatial.transform import Rotation as SSTR
import numpy as np

stage = omni.usd.get_context().get_stage()
#massAPI = UsdPhysics.MassAPI.Get(stage,'/heron/dummy_link')
#print(massAPI.GetMassAttr().Get())
#print(massAPI.GetCenterOfMassAttr().Get())
#print(massAPI.GetDiagonalInertiaAttr().Get())
rigidBodyAPI = UsdPhysics.RigidBodyAPI.Get(stage, '/heron/dummy_link')
pose_base = PhysXIFace.get_rigidbody_transformation('/heron/dummy_link')
vlin_world = rigidBodyAPI.GetVelocityAttr().Get()
vang_world = rigidBodyAPI.GetAngularVelocityAttr().Get()
R_base = SSTR.from_quat(pose_base["rotation"])
rot_base = R_base.as_matrix()
vrob = np.matmul(np.linalg.inv(rot_base),np.array(vlin_world))
print(vrob)
pose0 = PhysXIFace.get_rigidbody_transformation('/heron/thruster_0')
pose1 = PhysXIFace.get_rigidbody_transformation('/heron/thruster_1')
p0 = pose0['position']
q0 = pose0['rotation']
p1 = pose1['position']
q1 = pose1['rotation']
R0 = SSTR.from_quat(q0)
R1 = SSTR.from_quat(q1)
rot0 = R0.as_matrix()
rot1 = R1.as_matrix()
F = np.array([10000000, 0, 0])
F0R = np.matmul(rot0, F)
F1R = np.matmul(rot1, F)

PhysXIFace = get_physx_interface()
PhysXIFace.apply_force_at_pos("/heron/thruster_1",F1R,p1)
PhysXIFace.apply_force_at_pos("/heron/thruster_0",F0R,p0)
"""
"""
import omni
from pxr import UsdPhysics
from omni.isaac.dynamic_control import _dynamic_control
dc = _dynamic_control.acquire_dynamic_control_interface()

# Get a handle to the Franka articulation
# This handle will automatically update if simulation is stopped and restarted
art = dc.get_articulation("/heron")
stage = omni.usd.get_context().get_stage()
# Get information about the structure of the articulation
num_joints = dc.get_articulation_joint_count(art)
num_dofs = dc.get_articulation_dof_count(art)
num_bodies = dc.get_articulation_body_count(art)

print(num_joints, num_dofs, num_bodies)
dof_ptr = dc.find_articulation_dof(art, "thruster_0_joint")
#dof_ptr2 = dc.find_articulation_dof(art, "thruster_1_joint")
#dof_props = dc.get_articulation_dof_properties(art)
#print(str("DOF properties:\n") + str(dof_props) +"\n")
print(dc.get_dof_type(dof_ptr))
joint =  dc.get_dof_joint(dof_ptr)
print(dc.get_joint_type(joint))
joint_type = dc.get_joint_type(joint)

dof_state = dc.get_dof_state(dof_ptr)
# print position for the degree of freedom
print(dof_state.pos)


# This should be called each frame of simulation if state on the articulation is being changed.
dc.wake_up_articulation(art)
dc.set_dof_velocity_target(dof_ptr, 0)


joint = UsdPhysics.RevoluteJoint.Get(stage, "/heron/base_link/thruster_0_joint")
print(joint.GetAxisAttr().Get())
#dc.set_dof_velocity_target(dof_ptr2, 0)
"""