import mujoco
import numpy as np
from mujoco import viewer
import time
from itertools import product, permutations
mj_model_cheetah = mujoco.MjModel.from_xml_path("assets/legs/robot_5.urdf")
leg_model = mujoco.MjSpec.from_file("assets/legs/robot_5.urdf")

surface_name = "floor"
# Disable self-collision by adding exclude pairs for all body combinations
combinations = list(permutations(leg_model.bodies, 2))
for body1, body2 in combinations:
    if body1.name != surface_name or body2.name != surface_name:
        leg_model.add_exclude(bodyname1=body1.name, bodyname2=body2.name)

leg_model.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY, name1 = "Main_connection_1_L5Pseudo", name2 = "Main_connection_1_L4Pseudo")
leg_model.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY, name1 = "Main_connection_2_L6Pseudo", name2 = "Main_connection_2_L7Pseudo")
 
# Compile the modified model
mj_model_cheetah = leg_model.compile()
mj_data_cheetah = mujoco.MjData(mj_model_cheetah)

#mj_viewer = viewer.launch(mj_model_cheetah, mj_data_cheetah)
mj_viewer_for_control = viewer.launch_passive(mj_model_cheetah, mj_data_cheetah)

action_seq = np.zeros((10000, mj_model_cheetah.nu))

for action_i in action_seq:
    #mj_data_cheetah.ctrl = action_i
    mujoco.mj_step(mj_model_cheetah, mj_data_cheetah)
    mj_viewer_for_control.sync()
    time.sleep(0.01)  # Adjust the sleep time as needed for your simulation speed
