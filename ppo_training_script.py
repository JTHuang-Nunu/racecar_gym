from time import sleep
import sys

import gymnasium
import numpy as np
import torch
from matplotlib import pyplot as plt

from task import MixTask
import pybullet as p

import racecar_gym.envs.gym_api  # Necessary!!!Cannot be deleted!!!
from agent.Agent import get_training_agent
if 'racecar_gym.envs.gym_api' not in sys.modules:
    raise RuntimeError('Please run: pip install -e . and import racecar_gym.envs.gym_api first!')


# ======================================================================
# Set the environment hyperparameter
# ======================================================================
scenarios = [
    'scenarios/torino.yml',
    'scenarios/austria.yml',
    'scenarios/circle_cw.yml',
    'scenarios/barcelona.yml',
]
render_mode = 'human'  # 'human', 'rgb_array_birds_eye' and 'rgb_array_follow'


def make_env(env_id):
    # ======================================================================
    # Make the environment
    # ======================================================================
    if not env_id: 
        scenario = np.random.choice(scenarios)
    else:
        scenario = scenarios[env_id]
    env = gymnasium.make(
        'SingleAgentAustria-v0',
        render_mode=render_mode,
        # scenario='scenarios/circle_cw.yml',  # change the scenario here (change map)
        scenario=scenario
        # scenario='scenarios/validation.yml',  # change the scenario here (change map), ONLY USE THIS FOR VALIDATION
        # scenario='scenarios/validation2.yml',   # Use this during the midterm competition, ONLY USE THIS FOR VALIDATION
    )
    return env
# def worker_process(epoch, total_timesteps):
#     agent = get_training_agent(agent_name='PPO_LOAD')
#     env = gymnasium.make(
#         'SingleAgentAustria-v0',
#         render_mode=render_mode,
#         # scenario='scenarios/circle_cw.yml',  # change the scenario here (change map)
#         scenario='scenarios/austria.yml',  # change the scenario here (change map)
#         # scenario='scenarios/validation.yml',  # change the scenario here (change map), ONLY USE THIS FOR VALIDATION
#         # scenario='scenarios/validation2.yml',   # Use this during the midterm competition, ONLY USE THIS FOR VALIDATION
#     )
#     for e in range(epoch):
#         obs,_  = env.reset(options=dict(mode='grid'))
#         task = MixTask(task_weights={'progress': 0.7, 'tracking': 0.3, 'collision': 1.0})
#         t = 0
#         total_reward = 0
#         old_progress = 0
#         done = False
        
#         while not done and t < total_timesteps - 1:
#             # ==================================
#             # Execute RL model to obtain action
#             # ==================================
#             action, a_logp, value = agent.get_action(obs)

#             next_obs, _, done, truncated, states = env.step(
#                 {'motor': np.clip(action[0], -1, 1),
#                     'steering': np.clip(action[1], -1, 1)}
#             )

#             # Calculate reward
#             reward = 0
#             reward += task.reward(states, action)
            
#             done = task.done(states)
            
#             # reward += np.linalg.norm(states['velocity'][:3])
#             # reward += states['progress'] - old_progress
#             # old_progress = states['progress']

#             # if states['wall_collision']:
#             #     reward = -10
#             #     done = True

#             total_reward += reward
#             agent.store_trajectory(obs, action, value, a_logp, reward)

#             t += 1
#             obs = next_obs

#             if done:
#                 agent.store_trajectory(obs, action, value, a_logp, reward)
#                 break
#         env.close()
#         agent.learn()

#         if total_reward > best_reward:
#             best_reward = total_reward
#             agent.save_model()
#         print(f"Epoch: {e}, Total reward: {total_reward:.3f}, Best reward: {best_reward:.3f}")

# def create_parallel_envs(epoch, total_timesteps, num_envs):
#     processes = []
#     parent_conns = []
#     for i in range(num_envs):
#         parent_conn, child_conn = mp.Pipe()
#         process = mp.Process(target=worker_process, args=([epoch, total_timesteps]))
#         process.start()
#         processes.append(process)
#         parent_conns.append(parent_conn)
#     return processes, parent_conns

# def train_ppo(num_envs, total_timesteps, epoch):
    processes, conns = create_parallel_envs(epoch, total_timesteps, num_envs)
    for process in processes:
        process.join()

def main():
    # ======================================================================
    # Set the Training hyperparameter
    # ======================================================================
    EPOCHS = 10000
    MAX_STEP = 6000
    best_reward = -np.inf
    agent = get_training_agent(agent_name='PPO_LOAD', max_step=MAX_STEP, model_path='./agent/PPO/weight_41.pth')
    env = make_env(env_id=3)
    set
    console_speed:bool = False
    console:bool = True

    for e in range(EPOCHS):
        obs, info = env.reset(options=dict(mode='random')) # mode='grid', 'random', 'random_ball'
        # p.resetDebugVisualizerCamera(cameraDistance=30, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[0,0,0])
        task = MixTask(task_weights={'progress': 5, 'tracking': 0.0, 'collision': 1.0}, obs=obs, info=info)
        t = 0
        total_reward = 0
        done = False
        
        while not done and t < MAX_STEP - 1:
            # ==================================
            # Execute RL model to obtain action
            # ==================================
            action, a_logp, value = agent.get_action(obs)
            next_obs, _, done, truncated, states = env.step(
                {'motor': np.clip(action[0], -1, 1),
                 'steering': np.clip(action[1], -1, 1)}
            )

            p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=180, cameraPitch=-70, cameraTargetPosition=states['pose'][:3])

            states['lidar'] = obs['lidar']

            # Calculate reward
            reward = float(0.0)
            task_reward = task.reward(states, action)
            reward += task_reward
            # reward += check_lidar(obs, states, action)
            done = task.done(states)


            total_reward += reward
            if total_reward <= -10 or done:
                total_reward = -10
                done = True
            # Print data
            if t%100 == 0:
                if console_speed:
                    print(f'progress_reward: {r_progress:.3f}')
                    print(f'max_speed: {max_speed:.3f}, max_steer: {max_steer:.3f}')
            if console:
                if t%100 == 0:
                    print('------------------------------------------------------------------')
                    print(f't: {t}, total_reward: {total_reward:.3f}, task_reward: {task_reward:.3f}')
                if t%10 == 0:
                    task.console(action, mode=['cum_range'])
            if t == 1000:
                pass
            agent.store_trajectory(obs, action, value, a_logp, reward)

            if t % 1 == 0 and "rgb" in render_mode:
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
            obs = next_obs

            if done:
                agent.store_trajectory(obs, action, value, a_logp, reward)
                break

        env.close()
        agent.learn()

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_model()

        print(f"Epoch: {e}, Total reward: {total_reward:.3f}, Best reward: {best_reward:.3f}")



if __name__ == '__main__':
    main()