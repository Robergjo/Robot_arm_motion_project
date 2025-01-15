import numpy as np
import pybullet as p
import pybullet_data
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random

"""
    Set starting position of the robot




"""


# Building the env
class RoboticArmEnv:
    def __init__(self):
        p.connect(p.GUI)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)

        # This is for loading the models and were in the env to place them, also made some changes to the env
        plane_id = p.loadURDF("plane.urdf", [0, 0, 0])  #[0, 0, 0, 1])
        p.changeDynamics(plane_id, -1, lateralFriction=1.0)
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.8, 0.8, 0.8, 1])
        # Set end effector index
        self.end_effector_index = 11
        
        self.table_id = p.loadURDF("table/table.urdf", [0, 0, 0], useFixedBase=True)
        table_aabb = p.getAABB(self.table_id)
        self.table_height = table_aabb[1][2]
        
        wall_scaling = 5.0
        self.wall = p.loadURDF("cube.urdf",
                               [0, 3.5, 0.5],
                               [0, 0, 0, 1],
                               globalScaling=wall_scaling,
                               useFixedBase=True)
        p.changeVisualShape(self.wall, -1, rgbaColor=[0.9, 0.9, 0.9, 1])

        self.wall_2 = p.loadURDF("cube.urdf",
                               [-5.0, 0, 0.5],
                               [0, 0, 0, 1],
                               globalScaling=wall_scaling,
                               useFixedBase=True)
        p.changeVisualShape(self.wall_2, -1, rgbaColor=[0.9, 0.9, 0.9, 1])

        self.wall_3 = p.loadURDF("cube.urdf",
                               [5.0, 0, 0.5],
                               [0, 0, 0, 1],
                               globalScaling=wall_scaling,
                               useFixedBase=True)
        p.changeVisualShape(self.wall_3, -1, rgbaColor=[0.9, 0.9, 0.9, 1])
        
        
        # Targets for the robot
        self.target_one = [0.5, -0.3, self.table_height + 0.1]
        self.target_two = [-0.5, 0.3, self.table_height + 0.1]
        self.current_target = self.target_one


        # Loading the arm and changing size of a sphere that the arm will grab
        self.sphere_radius = 0.10 
        self.sphere = p.loadURDF("sphere2.urdf", [-0.2, -0.3, self.table_height], [0, 0, 0, 1], globalScaling= self.sphere_radius)

        #self.current_position = [-0.3, 0.3, self.table_height]
        #"franka_panda/panda.urdf"
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0.2, self.table_height], [0, 0, -1, 1], useFixedBase = True) # useFixedBase makes it so the arm does not fall when moving, if set to True
        self.obj_of_focus = self.robot
        
        p.addUserDebugLine(self.current_target,
                          [self.current_target[0], self.current_target[1],
                           self.current_target[2] + 0.2],
                           [0, 1, 0], 2)

        # Joint control
        jointid = 4
        self.jlower = p.getJointInfo(self.robot, jointid)[8]
        self.jupper = p.getJointInfo(self.robot, jointid)[9]
        print(p.getNumJoints(self.robot))
        
        # Set starting position of the robot
        # p.resetJointState(self.robot, 0, 3.14)
        p.resetJointState(self.robot, 1, 0.0)
        p.resetJointState(self.robot, 3, 1.57)
        p.resetJointState(self.robot, 4, 3.14)
        p.resetJointState(self.robot, 5, 1.57)   

        # Camera angle
        p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])


        # Action and state space
        self.action_size = 6  # Discrete actions: move +/- in x/y/z
        self.state_size = len(self.get_state())

    def get_state(self):
        joint_positions = [p.getJointState(self.robot, i)[0] for i in range(p.getNumJoints(self.robot))]
        end_effector_pos = p.getLinkState(self.robot, self.end_effector_index)[0]
        distance_to_goal = np.linalg.norm(np.array(end_effector_pos) - np.array(self.current_target))
        state = np.array(list(joint_positions) + [distance_to_goal])
        # print("state:", state, "joint_positions:", joint_positions, "end_effector_pos:", end_effector_pos, "distance_to_goal:", distance_to_goal)
        return state

    def step(self, action):
        # Converting discrete action to continuous movement
        action_map = {
            0: [0.1, 0, 0],   # +x
            1: [-0.1, 0, 0],  # -x
            2: [0, 0.1, 0],   # +y
            3: [0, -0.1, 0]  # -y  
            # 4: [0, 0, 0.1],   # +z
            # 5: [0, 0, -0.1]   # -z
        }
        delta = action_map[action]
        # random_pos = np.random.uniform(-0.1, 0.1, 3)
        random_positions = [
            np.clip(np.random.uniform(-1, 1), 0.2, 0.6),  # X-axis
            np.clip(np.random.uniform(-1, 1), -0.3, 0.3), # Y-axis
            np.clip(np.random.uniform(-1, 1), 0.1, 0.5)   # Z-axis
        ]
        print("Random pos", random_positions)

        # Gets the current position and applies action
        current_pos = p.getLinkState(self.robot, self.end_effector_index)[0]
        print("current_pos:", current_pos)
        # new_pos = [current_pos[i] + random_pos[i] for i in range(3)]
        new_pos = [i + j for i, j in zip(current_pos, random_positions)]
        print("new_pos:", new_pos)

        # Using inverse kinematics to move the arm
        joint_poses = p.calculateInverseKinematics(self.robot, self.end_effector_index, new_pos)
        for i in range(len(joint_poses)):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, joint_poses[i])
        
        p.stepSimulation()
        # HEJ
        # Getting a new state and compute reward
        new_state = self.get_state()
        distance = np.linalg.norm(np.array(p.getLinkState(self.robot, self.end_effector_index)[0]) - np.array(self.current_target))
        reward = -distance
        done = distance < 0.05  # ends when close enough to target

        # Switch targets when reaching current target
        if done:
            if self.current_target == self.target_one:
                self.current_target = self.target_two
                # Update the visual target indicator
                p.addUserDebugLine(self.current_target, 
                             [self.current_target[0], self.current_target[1], 
                              self.current_target[2] + 0.2], 
                             [0, 1, 0], 2)
            else:
                self.current_target = self.target_one
                # Update the visual target indicator
                p.addUserDebugLine(self.current_target, 
                                [self.current_target[0], self.current_target[1], 
                                self.current_target[2] + 0.2], 
                                [0, 1, 0], 2)

        return new_state, reward, done, {}

    def reset(self):
        p.resetSimulation()
        self.__init__()
        return self.get_state()

if __name__ == "__main__":

    env = RoboticArmEnv()
    # observation = env.reset()
    # print(observation)
    observation = env.get_state()
    print(observation)

    print("Starting simulation test...")
    print("Taking random actions for 2000 steps...")

    for i in range(200):
        time.sleep(0.5)
        action = random.randint(0, 3)
        print(action)
        state, reward, done, _ = env.step(action)

        # joint_positions = state[:-1]  # All elements except the last one
        # distance = state[-1] 
    
        # if i % 10 == 0:
            # print(f"\nStep {i}")
            # print("Joint States:")
            # for j, pos in enumerate(state[:-1]):
            #     print(f"Joint {j}: {pos:.3f}")
            # print(f"Distance to target: {state[-1]:.3f}")
            # print(f"Reward: {reward:.3f}")
            # print(f"Action taken: {action}")
            # print(f"Step {i}, Reward: {reward:.3f}")
            

        time.sleep(0.01)
        # done=False

        if done:
            print("Goal reached!")
            env.reset()
    
    print("Test complete! Keeping window open...")

    while True:
        time.sleep(1)