from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC


from robot_control.common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from robot_control.common.rotation_helper import get_gravity_orientation, transform_imu_data
from robot_control.common.remote_controller import RemoteController, KeyMap

from termcolor import cprint

class G1RealWorldEnv:
    def __init__(self, net, config):
        
        self.config = config
        
        self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25,
                25, 25, 25, 25,
            ])
        
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.tauj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.counter = 0
        
        # Initialize DDS communication
        ChannelFactoryInitialize(0, net)
        
        self.remote_controller = RemoteController()
        
        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)
        else:
            raise ValueError("Invalid msg_type")
        
        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)
        
    
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx + self.config.wrist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps + self.config.wrist_kps
        kds = self.config.kds + self.config.arm_waist_kds + self.config.wrist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target, self.config.wrist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)


    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.wrist_joint2motor_idx)):
                motor_idx = self.config.wrist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.wrist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.wrist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.wrist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
    
    def get_robot_state(self):
        
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq
            self.tauj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].tau_est
        start_idx = len(self.config.leg_joint2motor_idx)
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            self.qj[start_idx + i] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[i]].q
            self.dqj[start_idx + i] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[i]].dq
            self.tauj[start_idx + i] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[i]].tau_est
        # TODO: add wrist
            
        # imu_state quaternion: w, x, y, z
        quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32)
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
       
        
        dof_pos = self.qj.copy()
        dof_vel = self.dqj.copy()
        
        return (dof_pos, dof_vel, quat, ang_vel)
    

    def send_robot_action(self, target_dof_pos, kp_scale=1.0, kd_scale=1.0, left_wrist_roll=0.0, right_wrist_roll=0.0):
        
        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i] * kp_scale
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i] * kd_scale
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # arm & waist
        start_idx = len(self.config.leg_joint2motor_idx)
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[start_idx + i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i] * kp_scale
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i] * kd_scale
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        
        # wrist use default pos now. not actuated by the policy
        for i in range(len(self.config.wrist_joint2motor_idx)):
            motor_idx = self.config.wrist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.wrist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.wrist_kps[i] * kp_scale
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.wrist_kds[i] * kd_scale
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        
        # set left wrist roll
        motor_idx = 19
        self.low_cmd.motor_cmd[motor_idx].q = left_wrist_roll
        # set right wrist roll
        motor_idx = 26
        self.low_cmd.motor_cmd[motor_idx].q = right_wrist_roll
        
        self.send_cmd(self.low_cmd)

    
    def slowly_move_to_target_dof_pos(self, target_dof_pos, total_time=4.0):
        # use send_robot_action to execute the action
        # but with a slower speed
        num_step = int(total_time / self.config.control_dt)
        current_dof_pos = self.get_robot_state()[0]
        # move to target pos
        for i in range(num_step):
            alpha = i / num_step
            target_dof_pos_i = current_dof_pos * (1 - alpha) + target_dof_pos * alpha
            self.send_robot_action(target_dof_pos_i)
            time.sleep(self.config.control_dt)
        
    def keep_in_current_pos(self):
        current_dof_pos = self.get_robot_state()[0]
        print("Enter keep in current pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            self.send_robot_action(current_dof_pos)
            time.sleep(self.config.control_dt)
