import argparse
import json
import time
import numpy as np
import redis
import mujoco
import torch
from rich import print
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
from data_utils.params import DEFAULT_MIMIC_OBS
import os
from data_utils.rot_utils import quatToEuler

import datetime
import csv
import atexit

def draw_root_velocity(mujoco_model, mujoco_data, mujoco_viewer, tgt_root_vel, init_geom_id, root_name, rgba_velocity=[1, 1, 0, 1]):
    """
    Draws an arrow representing velocity, for debug/visualization.
    """
    mujoco_viewer.user_scn.ngeom = init_geom_id
    root_body_id = mujoco_model.body(root_name).id
    root_pos = mujoco_data.xpos[root_body_id]
    root_vel = tgt_root_vel
    vel_scale = 1.0

    mujoco.mjv_initGeom(
        mujoco_viewer.user_scn.geoms[mujoco_viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.zeros(9),
        rgba=rgba_velocity,
    )
    mujoco.mjv_connector(
        mujoco_viewer.user_scn.geoms[mujoco_viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        width=0.01,
        from_=root_pos,
        to=root_pos + vel_scale * np.array(root_vel),
    )
    mujoco_viewer.user_scn.ngeom += 1
    return mujoco_viewer.user_scn.ngeom


# -------------------------------------------------------------------
# Main low-level policy controller that:
#   - reads mimic obs from Redis
#   - feeds into policy
#   - runs the sim
# -------------------------------------------------------------------
def extract_mimic_obs_to_body_and_wrist(mimic_obs):
    total_degrees = 33
    wrist_ids = [27, 32]
    other_ids = [f for f in range(total_degrees) if f not in wrist_ids]
    policy_target = mimic_obs[other_ids]
    wrist_dof_pos = mimic_obs[wrist_ids]
    
    return policy_target, wrist_dof_pos

def aggregate_wrist_dof_pos(body_dof_pos, wrist_dof_pos):
    total_degrees = 25
    wrist_ids = [19, 24]
    other_ids = [f for f in range(total_degrees) if f not in wrist_ids]
    whole_body_pd_target = np.zeros(total_degrees)
    whole_body_pd_target[other_ids] = body_dof_pos
    whole_body_pd_target[wrist_ids] = wrist_dof_pos
    
    return whole_body_pd_target
    
class RealTimePolicyController:
    def __init__(self, 
                 xml_file, 
                 policy_path, 
                 device='cuda', 
                 record_video=False):
        
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        except Exception as e:
            print(f"Error connecting to Redis: {e}")

        self.device = device

        # Load policy
        self.policy = torch.jit.load(policy_path, map_location=device)
        print(f"Policy loaded from {policy_path}")

        # Create MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        
        # Print DoF names in order
        print("Degrees of Freedom (DoF) names and their order:")
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            print(f"DoF {i}: {dof_name}")

        # print("Body names and their IDs:")
        # for i in range(self.model.nbody):  # 'nbody' is the number of bodies
        #     body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        #     print(f"Body ID {i}: {body_name}")
        
        print("Motor (Actuator) names and their IDs:")
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"Motor ID {i}: {motor_name}")
            

        self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
        self.viewer.cam.distance = 2.5
        
        # Create offscreen renderer for video recording
        if record_video:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            # Create a camera for rendering
            self.render_camera = mujoco.MjvCamera()
            self.render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.render_camera.distance = 2.5

        # Example defaults & placeholders
        self.num_actions = 23
        self.sim_duration = 100000.0
        self.sim_dt = 0.001
        self.sim_decimation = 20

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # PD Gains, etc. (adapt as needed)
        self.default_dof_pos = np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (1)
                0.0, 0.4, 0.0, 1.2,
                0.0, -0.4, 0.0, 1.2,
            ])
        """
        mimic_obs = np.concatenate([
        root_pos[2:3],      # just the z for height
        rpy,                # roll, pitch, yaw
        root_vel_relative,  # local root vel
        dof_pos])
        """
        self.default_mimic_obs = DEFAULT_MIMIC_OBS["g1"]
        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]),
            np.array([0, 0, 0, 1]),
             np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (1)
                0.0, 0.2, 0.0, 1.2, 0.0, # left arm (4)
                0.0, -0.2, 0.0, 1.2, 0.0, # right arm (4)
                ])
        ])
        self.stiffness = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40, 20,
                40, 40, 40, 40, 20,
            ])
        self.damping = np.array([
                2, 2, 2, 4, 2, 2,
                2, 2, 2, 4, 2, 2,
                4, 4, 4,
                5, 5, 5, 5, 1,
                5, 5, 5, 5, 1,
            ])
        self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25, 25,
                25, 25, 25, 25, 25,
            ])
        
        self.action_scale = 0.5

        
        self.ankle_idx = [4, 5, 10, 11]
        
        # For multi-step history
        self.n_mimic_obs = 31
        self.n_proprio = self.n_mimic_obs + 3 + 2 + 3*self.num_actions
        self.proprio_history_buf = deque(maxlen=10)
        for _ in range(10):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))

        self.record_video = record_video

        # Setup logging
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = f"{self.log_dir}/sim_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_file = open(self.log_path, "a", newline="")
        self.csv_writer = csv.writer(self.log_file)
        atexit.register(self.log_file.close)
    
    def extract_data(self):
        qpos = self.data.qpos.astype(np.float32)
        qvel = self.data.qvel.astype(np.float32)
        
        body_ids = [0,1,2,3,4,5,
                    6,7,8,9,10,11,
                    12,13,14,
                    15,16,17,18,# 19
                    20,21,22,23, # 24
                    ]
        wrist_ids = [19, 24]
        
        whole_body_dof = qpos[7:]
        whole_body_dof_vel = qvel[6:]
        body_dof_pos = qpos[[f+7 for f in body_ids]]
        body_dof_vel = qvel[[f+6 for f in body_ids]]
        wrist_dof_pos = qpos[[f+7 for f in wrist_ids]]
        wrist_dof_vel = qvel[[f+6 for f in wrist_ids]]

        quat = self.data.sensor('orientation').data.astype(np.float32)
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        return whole_body_dof, whole_body_dof_vel, body_dof_pos, body_dof_vel, wrist_dof_pos, wrist_dof_vel, quat, ang_vel

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, mujoco_dof_pos=None):
        # body & hand
        self.data.qpos[:] = mujoco_dof_pos
        mujoco.mj_forward(self.model, self.data)
       
    def run(self):
        # Optionally record video
        if self.record_video:
            import imageio
            video_name = "debug_sim.mp4"
            print(f"Saving video to {video_name}")
            mp4_writer = imageio.get_writer(video_name, fps=50)
        else:
            mp4_writer = None

        self.reset_sim()
        self.reset(self.mujoco_default_dof_pos)

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating...")

        # send initial proprio to redis
        proprio_json = json.dumps(self.proprio_history_buf[0].tolist())
        self.redis_client.set("state_body_g1", proprio_json)
        self.redis_client.set("state_hand_g1", json.dumps(np.zeros(14).tolist()))
        try:
            for i in pbar:
                
                t_start = time.time()
                whole_body_dof, whole_body_dof_vel, body_dof_pos, body_dof_vel, wrist_dof_pos, wrist_dof_vel, quat, ang_vel = self.extract_data()
                
                if i % self.sim_decimation == 0:
                    
                    # Build a "proprio" vector for your policy, e.g.:
                    rpy = quatToEuler(quat)
                    obs_body_dof_vel = body_dof_vel.copy()
                    obs_body_dof_vel[self.ankle_idx] = 0.
                    obs_proprio = np.concatenate([
                        ang_vel * 0.25,
                        rpy[:2],
                        (body_dof_pos - self.default_dof_pos),
                        obs_body_dof_vel * 0.05,
                        self.last_action
                    ])
                    # send proprio to redis
                    self.redis_client.set("state_body_g1", json.dumps(obs_proprio.tolist()))
                    self.redis_client.set("state_hand_g1", json.dumps(np.zeros(14).tolist()))

                    # Try to get the latest mimic obs from Redis
                    try:
                        action_mimic_json = self.redis_client.get("action_mimic_g1")
                        if action_mimic_json is not None:
                            action_mimic_list = json.loads(action_mimic_json)
                            action_mimic = np.array(action_mimic_list, dtype=np.float32)
                            action_mimic, wrist_dof_pos = extract_mimic_obs_to_body_and_wrist(action_mimic)
                        else:
                            raise Exception("cannot get action mimic from redis")
                    except:
                        raise Exception("cannot get action mimic from redis")

                    obs_full = np.concatenate([action_mimic, obs_proprio])
                    obs_hist = np.array(self.proprio_history_buf).flatten()
                    obs_buf = np.concatenate([obs_full, obs_hist])
                    self.proprio_history_buf.append(obs_full)

                    obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                    
                    self.last_action = raw_action
                    raw_action = np.clip(raw_action, -10., 10.)
                    scaled_actions = raw_action * self.action_scale
                    pd_target = scaled_actions + self.default_dof_pos
                    pd_target = aggregate_wrist_dof_pos(pd_target, wrist_dof_pos)
                    # debug draw velocity arrow if you want
                    self.viewer.user_scn.ngeom = 0
                    draw_root_velocity(self.model, self.data, self.viewer, [0,0,0], 0, "pelvis", [1,0,0,1])
                    
                    # Make camera face the robot from the front
                    torso_pos = self.data.xpos[self.model.body("torso_link").id]
                    
                    # Get robot's orientation (yaw angle)
                    pelvis_quat = self.data.sensor('orientation').data.astype(np.float32)
                    rpy = quatToEuler(pelvis_quat)
                    yaw = rpy[2]
                    
                    camera_distance = 2.5
                    
                    # Point camera at the torso (face)
                    self.viewer.cam.lookat = torso_pos
                    self.viewer.cam.azimuth = np.rad2deg(yaw) + 180
                    self.viewer.cam.elevation = 0
                    self.viewer.cam.distance = camera_distance
                    
                    self.viewer.sync()
                    if mp4_writer is not None:
                        # Set camera to match viewer camera settings
                        self.render_camera.lookat[:] = torso_pos
                        self.render_camera.azimuth = np.rad2deg(yaw) + 180
                        self.render_camera.elevation = 0
                        self.render_camera.distance = camera_distance
                        
                        # Update renderer with current state and camera
                        self.renderer.update_scene(self.data, camera=self.render_camera)
                        
                        # Render the image
                        img = self.renderer.render()
                        mp4_writer.append_data(img)

                # PD control
                torque = (pd_target - whole_body_dof) * self.stiffness - whole_body_dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                
                self.data.ctrl[:] = torque
                
                mujoco.mj_step(self.model, self.data)
                
                # Log data
                # 1(timestamp) + 23(dof_pos) + 4(quat: wxyz mujoco convension) = 268 columns
                # We assume dof_pos of order: 6 left leg, 6 right leg, 3 torso, 4 left arm, 4 right arm without wrist
                timestamp = time.time()
                row = [timestamp] + body_dof_pos.tolist() + quat.tolist()
                self.csv_writer.writerow(row)
                self.log_file.flush()  # make sure it’s written immediately
                
                # sleep to maintain real-time pace
                elapsed = time.time() - t_start
                if elapsed < self.sim_dt:
                    time.sleep(self.sim_dt - elapsed)
        except Exception as e:
            print(f"Error in run: {e}")
            pass
        finally:
            if mp4_writer is not None:
                mp4_writer.close()
                print("Video saved")

            self.viewer.close()


def main_low_level_sim(args):
    controller = RealTimePolicyController(
        xml_file=args.xml_file,
        policy_path=args.policy_path,
        device='cuda',
        record_video=args.record_video,
    )
    controller.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    HERE = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--xml_file", default=os.path.join(HERE, "../assets/g1/g1_sim2sim_with_wrist_roll.xml"), help="Mujoco XML file")
    
    parser.add_argument("--policy_path",  help="Path to the policy",
                        default="../assets/twist_general_motion_tracker.pt"
                        )
                        
    parser.add_argument("--record_video", action="store_true", help="Record a video")
    args = parser.parse_args()

    args.record_proprio = True
    
    main_low_level_sim(args)
