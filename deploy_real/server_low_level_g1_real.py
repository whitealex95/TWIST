#!/usr/bin/env python3
# server_low_level_policy_real.py

import argparse
import time
import json
import numpy as np
import torch
import redis
from collections import deque
from tqdm import tqdm
from robot_control.common.remote_controller import KeyMap

from robot_control.g1_wrapper import G1RealWorldEnv
from robot_control.config import Config
import os
from data_utils.rot_utils import quatToEuler

# from robot_control.dex_hand_wrapper import Dex3_1_Controller
from data_utils.params import DEX31_QPOS_OPEN, DEX31_QPOS_CLOSE

# from robot_control.speaker import Speaker

def extract_mimic_obs_to_body_and_wrist(mimic_obs):
    total_degrees = 33
    wrist_ids = [27, 32]
    other_ids = [f for f in range(total_degrees) if f not in wrist_ids]
    policy_target = mimic_obs[other_ids]
    wrist_dof_pos = mimic_obs[wrist_ids]
    
    return policy_target, wrist_dof_pos

class RealTimePolicyControllerReal(object):
    """
    参考 server_low_level_policy_sim.py 结构，但内部针对真实机器人环境做适配。
    """
    def __init__(self, 
                 policy_path,
                 config_path,
                 device='cuda',
                 net='eno1'):
        self.redis_client = None
   
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        except Exception as e:
            print(f"Error connecting to Redis: {e}")
            self.redis_client = None

       
        self.config = Config(config_path)
        self.env = G1RealWorldEnv(net=net, config=self.config)

        self.device = device
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()
        print(f"Policy loaded from {policy_path}")

        # 4. 定义一些和动作/观测相关的参数（示例）
        self.num_actions = 23
        self.default_dof_pos = np.concatenate([self.config.default_angles,
                                               self.config.arm_waist_target], axis=0)
        # 若需要的话，可进一步调节观测中对关节位置、速度等做 scale
        self.ang_vel_scale = 0.25
        self.dof_vel_scale = 0.05
        self.dof_pos_scale = 1.0

        self.ankle_idx = [4, 5, 10, 11]

        # 这里给定一个简化的观测长度(与 sim2real_mimic.py 中一致)
        # 假设我们只拼接 mimic_obs + proprio + 历史
        # 实际中需根据你自己的模型观测来修改
        self.n_mimic_obs = 8 + self.num_actions  # 参考 sim2real_mimic.py
        self.n_proprio = self.n_mimic_obs + 3 + 2 + 3*self.num_actions
        self.history_len = 10  # 跟 sim 中一样维护10帧历史
        self.proprio_history_buf = deque(maxlen=self.history_len)

        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio, dtype=np.float32))


        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        self.control_dt = self.config.control_dt
        self.action_scale = self.config.action_scale
        
    def reset_robot(self):
        """
        一些真实机器人启动前的流程，比如进入零力矩模式，然后转到默认姿态。
        仅示例化，具体请参考你的硬件及安全流程。
        """
        print("Entering zero torque state, waiting for user to enter `start` ...")
        self.env.zero_torque_state()

        print("Press START on remote to move to default position ...")
        self.env.move_to_default_pos()

        print("Now in default position, press A to continue ...")
        self.env.default_pos_state()

        print("Robot will hold default pos. If needed, do other checks here.")

        # 如果需要再移动到某个"运动起始位姿"，可以自行在此实现
        # 比如在 sim2real_mimic.py 中是 get_motion_start_pos() 等
    
    def run(self):
        """
        主循环：从 Redis 或本地构造 mimic_obs，获取机器人状态 -> 前向策略 -> 发送动作
        """
        self.reset_robot()
        print("Begin main policy loop. Press [Select] on remote to exit.")

        # 主循环
        try:
            while True:
                t_start = time.time()

                # Press the select key to exit
                if self.env.remote_controller.button[KeyMap.select] == 1:
                    print("Select pressed, exiting main loop.")
                    break
                
                # get robot state
                dof_pos, dof_vel, quat, ang_vel = self.env.get_robot_state()
                
                rpy = quatToEuler(quat)

                obs_dof_vel = dof_vel.copy()
                obs_dof_vel[self.ankle_idx] = 0.0  # 根据你的机器人具体情况决定

                obs_proprio = np.concatenate([
                    ang_vel * self.ang_vel_scale,
                    rpy[:2],
                    (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                    obs_dof_vel * self.dof_vel_scale,
                    self.last_action
                ])
                
                # send proprio to redis
                proprio_json = json.dumps(obs_proprio.tolist())
                self.redis_client.set("state_body_g1", proprio_json)
                
                
                # receive mimic_obs from redis
                try:
                    action_mimic_json = self.redis_client.get("action_mimic_g1")
                    if action_mimic_json is not None:
                        action_mimic_list = json.loads(action_mimic_json)
                        action_mimic = np.array(action_mimic_list, dtype=np.float32)
                        action_mimic, wrist_dof_pos = extract_mimic_obs_to_body_and_wrist(action_mimic)
                    else:
                        raise Exception("cannot get action_mimic from redis")
                except:
                    raise Exception("cannot get action_mimic from redis")

                obs_full = np.concatenate([action_mimic, obs_proprio])

                obs_hist = np.array(self.proprio_history_buf).flatten()
                obs_buf = np.concatenate([obs_full, obs_hist])
                self.proprio_history_buf.append(obs_full)

                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                self.last_action = raw_action.copy()

                raw_action = np.clip(raw_action, -10.0, 10.0)
                target_dof_pos = self.default_dof_pos + raw_action * self.action_scale

                kp_scale = 1.0
                kd_scale = 1.0
                self.env.send_robot_action(target_dof_pos, kp_scale, kd_scale,
                                           left_wrist_roll=wrist_dof_pos[0], right_wrist_roll=wrist_dof_pos[1])
                
                elapsed = time.time() - t_start
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.env.zero_torque_state()


def main_low_level_real(args):
    controller = RealTimePolicyControllerReal(
        policy_path=args.policy_path,
        config_path=args.config_path,
        device=args.device,
        net=args.net,
    )
    controller.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path",  help="Path to the policy",
                        default="../assets/twist_general_motion_tracker.pt"
                        )
                        
    HERE = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config_path", type=str, default=os.path.join(HERE, "robot_control/configs/g1.yaml"),
                        help="Robot config file path.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device for inference: cpu or cuda.")
    parser.add_argument("--net", type=str, default='eno1',
                        help="Network interface used by G1RealWorldEnv.")


    args = parser.parse_args()

    main_low_level_real(args)
