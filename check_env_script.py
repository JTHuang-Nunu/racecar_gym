from time import sleep
import sys

import gymnasium
from matplotlib import pyplot as plt

import racecar_gym.envs.gym_api  # Necessary!!!Cannot be deleted!!!
from agent.Agent import get_valid_agent


if 'racecar_gym.envs.gym_api' not in sys.modules:
    raise RuntimeError('Please run: pip install -e . and import racecar_gym.envs.gym_api first!')


def main():
    # ======================================================================
    # Create the environment
    # ======================================================================
    render_mode = 'human'  # 'human', 'rgb_array_birds_eye' and 'rgb_array_follow'
    env = gymnasium.make(
        'SingleAgentAustria-v0',
        render_mode=render_mode,
        scenario='scenarios/circle_cw.yml',  # change the scenario here (change map)
        # scenario='scenarios/validation.yml',  # change the scenario here (change map), ONLY USE THIS FOR VALIDATION
        # scenario='scenarios/validation2.yml',   # Use this during the midterm competition, ONLY USE THIS FOR VALIDATION
    )
    done = False
    # agent = get_valid_agent("PPO")

    # ======================================================================
    # Run the environment
    # ======================================================================
    obs, info = env.reset(options=dict(mode='grid'))
    t = 0
    while not done:
        # ==================================
        # Execute RL model to obtain action
        # ==================================
        # action = agent.get_action(obs)
        action = {"motor": 1.0, "steering": 0.0}
        obs, rewards, done, truncated, states = env.step(action)

        if t % 30 == 0 and "rgb" in render_mode:
            # ==================================
            # Render the environment
            # ==================================

            image = env.render()
            plt.clf()
            plt.title("Pose")
            plt.imshow(image)
            plt.pause(0.01)
            plt.ioff()

        t += 1
        if done or truncated:
            break

    env.close()


if __name__ == '__main__':

    main()
