from train_model.training_script import train_model
from train_model.fine_tuning import fine_tune
import gymnasium as gym
import panda_gym
import time

def main():
    """
    Main function to manage the training, fine-tuning, and running the environment.
    """
    print("Starting the Robot Arm Motion Planning Project!")

    # Step 1: Train the model from scratch
    print("\nStep 1: Training the model...")
    try:
        train_model(env_id="PandaReachDense-v3", total_timesteps=100000)
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Step 2: Fine-tune the pre-trained model
    print("\nStep 2: Fine-tuning the model...")
    try:
        fine_tune(env_id="PandaReachDense-v3", total_timesteps=50000)
        print("Model fine-tuning completed successfully!")
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return

    # Step 3: Run the environment using the fine-tuned model
    print("\nStep 3: Running the environment with the fine-tuned model...")
    try:
        env = gym.make("PandaReachDense-v3", render_mode="human")
        obs, _ = env.reset()

        from stable_baselines3 import PPO
        model = PPO.load("ppo_robot_arm_fine_tuned.zip", env=env)

        for _ in range(1000):
            time.sleep(0.3)
            action, _ = model.predict(obs)
            obs, reward, done, info, _ = env.step(action)
            env.render()
            if done:
                obs, _ = env.reset()

        print("Environment run completed successfully!")
    except Exception as e:
        print(f"Error during environment run: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()

