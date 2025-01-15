import numpy as np
import pybullet as p
import pybullet_data
import time
import random

class RoboticArmEnv:
    def __init__(self, mode="GUI"):
        if mode == "GUI":
            p.connect(p.GUI)
        elif mode == "DIRECT":
            p.connect(p.DIRECT)
        else:
            raise ValueError("Invalid mode. Use 'GUI' or 'DIRECT'.")

        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)

        # Ladda miljöobjekt
        plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
        if plane_id < 0:
            raise RuntimeError("Failed to load plane.")

        self.table_id = p.loadURDF("table/table.urdf", [0, 0, 0], useFixedBase=True)
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        self.current_target = [0.5, -0.3, 0.8]  # Default target position

        # Action och state space
        self.action_size = 4
        self.state_size = len(self.get_state())

        # Kamera
        p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])

    def get_state(self):
        joint_positions = [p.getJointState(self.robot, i)[0] for i in range(p.getNumJoints(self.robot))]
        end_effector_pos = p.getLinkState(self.robot, 6)[0]
        distance_to_goal = np.linalg.norm(np.array(end_effector_pos) - np.array(self.current_target))
        return np.array(list(joint_positions) + [distance_to_goal])

    def step(self, action):
        action_map = {0: [0.1, 0, 0], 1: [-0.1, 0, 0], 2: [0, 0.1, 0], 3: [0, -0.1, 0]}
        delta = action_map[action]

        current_pos = p.getLinkState(self.robot, 6)[0]
        new_pos = [current_pos[i] + delta[i] for i in range(3)]

        joint_poses = p.calculateInverseKinematics(self.robot, 6, new_pos)
        for i, joint_pos in enumerate(joint_poses):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, joint_pos)

        p.stepSimulation()

        state = self.get_state()
        distance = np.linalg.norm(np.array(p.getLinkState(self.robot, 6)[0]) - np.array(self.current_target))
        reward = -distance
        done = distance < 0.05
        return state, reward, done, {}

    def reset(self):
        p.resetSimulation()
        self.__init__()
        return self.get_state()

if __name__ == "__main__":
    env = RoboticArmEnv()
    for _ in range(2000):
        action = random.randint(0, 3)
        state, reward, done, _ = env.step(action)
        if done:
            env.reset()

