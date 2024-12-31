import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    # action = np.zeros(num_actions, dtype=np.float32)
    # obs = np.zeros(num_obs, dtype=np.float32)
    # counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    

    # load policy
    # policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            # Generate sinusoidal target positions
            target_dof_pos = np.zeros(29)
            target_dof_pos[0] = default_angles[0] + 0.6 * np.sin(1 * np.pi * 0.5 * time.time()) # l hip pitch
            # target_dof_pos[0] = default_angles[0] + 0
            target_dof_pos[1] = default_angles[1] + 0 # l hip roll
            target_dof_pos[2] = default_angles[2] + 0 # l hip yaw

            target_dof_pos[3] = default_angles[3] + 0.4 + 0.6 * np.sin(2 * np.pi * 0.5 * time.time()) # l knee pitch
            target_dof_pos[4] = default_angles[4] + 0 # l ankle pitch
            target_dof_pos[5] = default_angles[5] + 0 # l ankle roll

            target_dof_pos[6] = default_angles[6] + 0 # r hip pitch
            target_dof_pos[7] = default_angles[7] + 0 # r hip roll
            target_dof_pos[8] = default_angles[8] + 0 # r hip yaw

            target_dof_pos[9] = default_angles[9] + 0.4 + 0.6 * np.sin(2 * np.pi * 0.5 * time.time()) # r knee pitch
            target_dof_pos[10] = default_angles[10] + 0 # r ankle pitch
            target_dof_pos[11] = default_angles[11] + 0 # r ankle roll

            target_dof_pos[12] = default_angles[12] + 0 # waist yaw
            target_dof_pos[13] = default_angles[13] + 0 # waist roll
            target_dof_pos[14] = default_angles[14] + 0 # waist pitch
            
            target_dof_pos[15] = default_angles[15] + 0 # l shoulder pitch
            target_dof_pos[16] = default_angles[16] + 0 # l shoulder roll
            target_dof_pos[17] = default_angles[17] + 0 # l shoulder yaw
            target_dof_pos[18] = default_angles[18] + 0 # l elbow
            target_dof_pos[19] = default_angles[19] + 0.3 * np.sin(1 * np.pi * 0.5 * time.time()) # l wrist roll
            target_dof_pos[20] = default_angles[20] + 0.3 * np.sin(1 * np.pi * 0.5 * time.time()) # l wrist roll # l wrist pitch
            target_dof_pos[21] = default_angles[21] + 0.3 * np.sin(1 * np.pi * 0.5 * time.time()) # l wrist roll # l wrist yaw

            target_dof_pos[22] = default_angles[22] + 0 # r shoulder pitch
            target_dof_pos[23] = default_angles[23] + 0 # r shoulder roll
            target_dof_pos[24] = default_angles[24] + 0 # r shoulder yaw
            target_dof_pos[25] = default_angles[25] + 0 # r elbow
            target_dof_pos[26] = default_angles[26] + 0 # r wrist roll
            target_dof_pos[27] = default_angles[27] + 0 # r wrist pitch
            target_dof_pos[28] = default_angles[28] + 0 # r wrist yaw

                        # 在 pd_control 函数调用之前添加打印语句
            # target_dof_pos = default_angles + 0.4 * np.sin(2 * np.pi * 0.5 * time.time())
            # tau = pd_control(target_dof_pos, d.qpos, kps, np.zeros_like(kds), d.qvel, kds)
            tau = pd_control(target_dof_pos, d.qpos, kps, np.zeros_like(kds), d.qvel, kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            # counter += 1
            # if counter % control_decimation == 0:
            #     # Apply control signal here.
            #     qj = d.qpos[7:]
            #     dqj = d.qvel[6:]
            #     quat = d.qpos[3:7]
            #     omega = d.qvel[3:6]
            #     qj = (qj - default_angles) * dof_pos_scale
            #     dqj = dqj * dof_vel_scale
            #     gravity_orientation = get_gravity_orientation(quat)
            #     omega = omega * ang_vel_scale
            #     period = 0.8
            #     count = counter * simulation_dt
            #     phase = count % period / period
            #     sin_phase = np.sin(2 * np.pi * phase)
            #     cos_phase = np.cos(2 * np.pi * phase)
            #     obs[:3] = omega
            #     obs[3:6] = gravity_orientation
            #     obs[6:9] = cmd * cmd_scale
            #     obs[9 : 9 + num_actions] = qj
            #     obs[9 + num_actions : 9 + 2 * num_actions] = dqj
            #     obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            #     obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
            #     obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            #     action = policy(obs_tensor).detach().numpy().squeeze()
            #     target_dof_pos = action * action_scale + default_angles

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
