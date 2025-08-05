import mujoco
import numpy as np
from mujoco import viewer
import time

mj_model_cheetah = mujoco.MjModel.from_xml_path("assets/half_cheetah_modified.xml")
mj_data_cheetah = mujoco.MjData(mj_model_cheetah)

mj_viewer = viewer.launch(mj_model_cheetah, mj_data_cheetah)
mj_viewer_for_control = viewer.launch_passive(mj_model_cheetah, mj_data_cheetah)

action_seq = np.zeros((100, mj_model_cheetah.nu))

for action_i in action_seq:
    mj_data_cheetah.ctrl = action_i
    mujoco.mj_step(mj_model_cheetah, mj_data_cheetah)
    mj_viewer.sync()
    time.sleep(0.01)  # Adjust the sleep time as needed for your simulation speed
