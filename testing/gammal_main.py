from training import train
from robot_arm_env import RoboticArmEnv
from network import ActorCriticNetwork
import time
import random
import pybullet as p


def test_random_actions(env, steps=2000):
    """
    Testa robotarmens miljö med slumpmässiga handlingar för att säkerställa att den fungerar korrekt.
    """
    print("Startar test med slumpmässiga handlingar...")
    for step in range(steps):
        action = random.randint(0, 3)
        state, reward, done, _ = env.step(action)

        if step % 100 == 0:
            print(f"Steg {step}: Reward: {reward:.3f}, Done: {done}")

        # Återställ om målet är nått
        if done:
            print("Målet nått! Återställer miljön...")
            env.reset()

        time.sleep(0.01)

    print("Test med slumpmässiga handlingar avslutat!")


def main():
    """
    Huvudfunktionen för att starta RL-träningen och valfritt testa slumpmässiga handlingar.
    """
    print("Välkommen till Robotarmens Rörelseplanering!")
    print("Startar miljön...")

    env = RoboticArmEnv()

    try:
        test_random_actions(env, steps=2000)

        print("Startar RL-träningen...")
        train(env)
        print("Träning slutförd! Kontrollera resultaten och visualiseringarna.")

    except Exception as e:
        print(f"Ett fel inträffade under träningen: {e}")

    finally:
        p.disconnect()
        print("PyBullet-anslutningen avslutad.")


if __name__ == "__main__":
    main()