import socket, struct
import time
import json

UDP_IP = "127.0.0.1"
UDP_PORT = 54010
BIG_NUMBER = 100000

import numpy as np
import threading

class Switch4EAIController:
    def __init__(self, sleep_interval=0.01, default_stand=0):
        command = {}
        command["root_pos"]      = np.array([3.0, 0.0, 0.75])
        command["root_rot"]      = np.array([0, 0, 0, 1])
        command["root_vel"]      = np.array([0, 0, 0])
        command["root_ang_vel"]  = np.array([0, 0, 0])
        command["dof_pos"]       = np.zeros(23)  # 6+6+3+4+4
        self.command = command
        self.sleep_interval = sleep_interval

        # UDP setting
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))

        def receive_loop():
            self.last_receive_time = time.time()
            while True:
                # num_elements = 36  # 3 + 4 + 3 + 3 + 23
                # msg, valid = server.receive_message(num_elements=num_elements, data_type='f')  # 3 + 4 + 29(dof_pos)
                data, addr = sock.recvfrom(65535)
                msg = json.loads(data.decode("utf-8"))
                receive_time = time.time()
                valid = True
                if valid and msg is not None:
                    print(f"Received full message: {len(msg)} elements, dt: {receive_time - self.last_receive_time:.3f} s")
                    self.update_command(msg)
                else:
                    time.sleep(self.sleep_interval)  # Avoid tight loop if no data
                self.last_receive_time = receive_time

        # Start background thread for receiving
        receiver_thread = threading.Thread(target=receive_loop, daemon=True)
        receiver_thread.start()

    def extract_dof_positions(self, dof_pos):
        """
        Extract and validate DOF positions for the robot.
        
        Args:
            dof_pos (np.array): Full DOF positions
            
        Returns:
            np.array: 23 DOF positions
        """
        # TODO: Make it correct.
        if len(dof_pos) == 23:
            return dof_pos
        elif len(dof_pos) == 29:
            # Convert 29 dof to 23 dof
            dof_idx_twist_from_gmr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25]
            dof_pos = dof_pos[..., dof_idx_twist_from_gmr]
            return dof_pos
        else:
            raise ValueError(f"Unexpected DOF positions length: {len(dof_pos)}. Expected 23 or 29.")

    def update_command(self, msg):
        """Update the command with new values from the motion streamer."""
        if msg is not None:
            # Extract root position (first 3 elements)
            self.command["root_pos"] = msg['root_pos']
            
            # Extract root rotation (next 4 elements - quaternion xyzw)
            if 'root_rot_xyzw' in msg.keys():
                self.command["root_rot"] = np.array(msg['root_rot_xyzw'])
            else:
                self.command["root_rot"] = np.array(msg['root_rot'])
            
            # Extract DOF positions (remaining 23 elements)
            # dof_pos_raw = np.array(msg[7:])
            dof_pos_raw = np.array(msg['dof_pos'])
            dof_pos = self.extract_dof_positions(dof_pos_raw)
            self.command["dof_pos"] = dof_pos
            
            
            print(f"Updated command - root_pos: {self.command['root_pos']}, "
                  f"root_rot: {self.command['root_rot']}, "
                  f"dof_pos shape: {self.command['dof_pos'].shape}")
        else:
            print(f"Invalid message: expected 36 elements, got {len(msg) if msg else 0}")

    def get_command(self):
        return self.command

#!/usr/bin/env python
import argparse
import time
import redis
import json
import numpy as np
# import isaacgym
import torch
from rich import print
import os
import mujoco
from mujoco.viewer import launch_passive
# ------------------------------------  ---------------------------------
# Example imports: adapt to your actual file structure
# ---------------------------------------------------------------------
from pose.utils.motion_lib_pkl import MotionLib
from data_utils.rot_utils import euler_from_quaternion, quat_rotate_inverse, quat_rotate_inverse_torch

from data_utils.params import DEFAULT_MIMIC_OBS, DEFAULT_ACTION_HAND

# ---------------------------------------------------------------------
# A small helper to replicate "mimic obs" logic from your code
# ---------------------------------------------------------------------
def build_mimic_obs_switch4eai(
    switch4eai_controller: Switch4EAIController,
    robot_type: str = "g1"
):
    """
    Build the mimic_obs at time-step t_step, referencing the code in MimicRunner.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get command (numpy arrays) and convert to batched torch tensors
    command = switch4eai_controller.get_command()

    # root_pos: (3,) -> (1,1,3)
    root_pos = torch.tensor(command['root_pos'], device=device, dtype=torch.float32).reshape(1, 1, 3)
    # root_rot: (4,) -> (1,4)
    root_rot = torch.tensor(command['root_rot'], device=device, dtype=torch.float32).reshape(1, 4)
    # velocities: (3,) -> (1,3)
    root_vel = torch.tensor(command['root_vel'], device=device, dtype=torch.float32).reshape(1, 3)
    root_ang_vel = torch.tensor(command['root_ang_vel'], device=device, dtype=torch.float32).reshape(1, 3)
    # dof_pos: (N,) -> (1,1,N)
    dof_pos = torch.tensor(command['dof_pos'], device=device, dtype=torch.float32).reshape(1, 1, -1)
    body_pos = None

    # euler_from_quaternion and quat_rotate_inverse_torch expect batched torch tensors
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    # roll/pitch/yaw are (batch,) -> reshape to (1,1,1)
    roll = roll.reshape(1, 1, 1)
    pitch = pitch.reshape(1, 1, 1)
    yaw = yaw.reshape(1, 1, 1)

    # Transform velocities to root frame. Output shape (batch,3) -> reshape to (1,1,3)
    root_vel = quat_rotate_inverse_torch(root_rot, root_vel).reshape(1, 1, 3)
    root_ang_vel = quat_rotate_inverse_torch(root_rot, root_ang_vel).reshape(1, 1, 3)
    
    if robot_type == "g1":
        dof_pos_with_wrist = torch.zeros(25, device=device).reshape(1, 1, 25)
        wrist_ids = [19, 24]
        other_ids = [f for f in range(25) if f not in wrist_ids]
        dof_pos_with_wrist[..., other_ids] = dof_pos
        dof_pos = dof_pos_with_wrist
    
    adjust_root_height = True
    if adjust_root_height:
        root_pos[..., 2] -= 0.1 # adjust root height down a bit

    ignore_root_xy_vel = True
    if ignore_root_xy_vel:
        root_vel[..., :2] *= 0.0
    
    ignore_lower_body = False
    if ignore_lower_body:
        dof_pos[..., :12] *= 0.0
    
    ignore_waist = False
    if ignore_waist:
        dof_pos[..., 12:15] *= 0.0
        # roll *= 0.0
        # pitch *= 0.0

    mimic_obs_buf = torch.cat((
                root_pos[..., 2:3],
                roll, pitch, yaw,
                root_vel,
                root_ang_vel[..., 2:3],
                dof_pos
            ), dim=-1)[:, 0:1]  # shape (1, 1, ?)
    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
    
    return mimic_obs_buf.detach().cpu().numpy().squeeze(), root_pos.detach().cpu().numpy().squeeze(), \
        root_rot.detach().cpu().numpy().squeeze(), dof_pos.detach().cpu().numpy().squeeze(), \
            root_vel.detach().cpu().numpy().squeeze(), root_ang_vel.detach().cpu().numpy().squeeze()


def main(args, xml_file, robot_base):

    if args.vis:
        sim_model = mujoco.MjModel.from_xml_path(xml_file)
        sim_data = mujoco.MjData(sim_model)
        viewer = launch_passive(model=sim_model, data=sim_data, show_left_ui=False, show_right_ui=False)
        
        # Print DoF names in order
        print("Degrees of Freedom (DoF) names and their order:")
        for i in range(sim_model.nv):  # 'nv' is the number of DoFs
            dof_name = mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_JOINT, sim_model.dof_jntid[i])
            print(f"DoF {i}: {dof_name}")

        # print("Body names and their IDs:")
        # for i in range(self.model.nbody):  # 'nbody' is the number of bodies
        #     body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        #     print(f"Body ID {i}: {body_name}")
        
        print("Motor (Actuator) names and their IDs:")
        for i in range(sim_model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"Motor ID {i}: {motor_name}")
            
    # 1. Connect to Redis
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    # 2. Load motion library
    device = "cuda" if torch.cuda.is_available() else "cpu"
    switch4eai_controller = Switch4EAIController(sleep_interval=0.001)
    
    # 3. Prepare the steps array
    tar_obs_steps = [int(x.strip()) for x in args.steps.split(",")]
    tar_obs_steps_tensor = torch.tensor(tar_obs_steps, device=device, dtype=torch.int)

    # 4. Loop over time steps and publish mimic obs
    control_dt = 0.02
    # compute num_steps based on motion length
    motion_id = None
    motion_length = BIG_NUMBER
    num_steps = BIG_NUMBER
    
    print(f"[Motion Server] Streaming for {num_steps} steps at dt={control_dt:.3f} seconds...")

    last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
    vis_root_vel = False
    vis_root_ang_vel = False
    if vis_root_vel:
        root_vel_list = []
    if vis_root_ang_vel:
        root_ang_vel_list = []
        
    try:
        for t_step in range(num_steps):
            t0 = time.time()

            # Build a mimic obs from the motion library
            # mimic_obs, root_pos, root_rot, dof_pos, root_vel, root_ang_vel = build_mimic_obs(
            #     motion_lib=motion_lib,
            #     t_step=t_step,
            #     control_dt=control_dt,
            #     tar_obs_steps=tar_obs_steps_tensor,
            #     robot_type=args.robot
            # )
            mimic_obs, root_pos, root_rot, dof_pos, root_vel, root_ang_vel = build_mimic_obs_switch4eai(
                switch4eai_controller=switch4eai_controller,
                robot_type=args.robot
            )
            if vis_root_vel:
                root_vel_list.append(root_vel)
            if vis_root_ang_vel:
                root_ang_vel_list.append(root_ang_vel)

            # Convert to JSON (list) to put into Redis
            mimic_obs_list = mimic_obs.tolist() if mimic_obs.ndim == 1 else mimic_obs.flatten().tolist()
            redis_client.set(f"action_mimic_{args.robot}", json.dumps(mimic_obs_list))
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            last_mimic_obs = mimic_obs
            # Print or log it
            print(f"Step {t_step:4d} => mimic_obs shape = {mimic_obs.shape} published...", end="\r")

            if args.vis:
                sim_data.qpos[:3] = root_pos
                # filp rot
                # root_rot = root_rot[[1,2,3,0]]
                root_rot = root_rot[[3,0,1,2]]
                sim_data.qpos[3:7] = root_rot
                sim_data.qpos[7:] = dof_pos
                mujoco.mj_forward(sim_model, sim_data)
                robot_base_pos = sim_data.xpos[sim_model.body(robot_base).id]
                viewer.cam.lookat = robot_base_pos
                # set distance to pelvis
                viewer.cam.distance = 2.5
                viewer.sync()
                
            # Sleep to maintain real-time pace
            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
        
    except KeyboardInterrupt:
        print("[Motion Server] Keyboard interrupt. Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (DEFAULT_MIMIC_OBS[args.robot] - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.set(f"action_mimic_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_mimic_{args.robot}", json.dumps(DEFAULT_MIMIC_OBS[args.robot].tolist()))
        redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
        last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
        exit()
    finally:
        print("[Motion Server] Exiting...Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (DEFAULT_MIMIC_OBS[args.robot] - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.set(f"action_mimic_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_mimic_{args.robot}", json.dumps(DEFAULT_MIMIC_OBS[args.robot].tolist()))
        redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
        last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
        exit()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", help="Path to your *.pkl motion file for MotionLib", 
                        default="/home/yanjieze/projects/g1_wbc/humanoid-motion-imitation/track_dataset/twist_motion_dataset/mocap/0.pkl")
    parser.add_argument("--robot", type=str, default="g1", choices=["g1"])
    parser.add_argument("--steps", type=str,
                        default="1",
                        help="Comma-separated steps for future frames (tar_obs_steps)")
    parser.add_argument("--vis", action="store_true", help="Visualize the motion")
    args = parser.parse_args()

    args.vis = True
    
    print("Robot type: ", args.robot)
    print("Motion file: ", args.motion_file)
    print("Steps: ", args.steps)
    
    HERE = os.path.dirname(os.path.abspath(__file__))
    
    if args.robot == "g1":
        xml_file = f"{HERE}/../assets/g1/g1_mocap_with_wrist_roll.xml"
        robot_base = "pelvis"
    else:
        raise ValueError(f"robot type {args.robot} not supported")
    
    
    main(args, xml_file, robot_base)
