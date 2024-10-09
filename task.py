from racecar_gym.tasks import Task
import numpy as np
import matplotlib.pyplot as plt


class MixTask:
    def __init__(self, task_weights: dict, obs, info):
        self.task_weights = task_weights or {'progress': 5, 'tracking': 0.3, 'collision': 1.0}
        self.ProgressTask = MaximizeProgressTask(obs, info)
        self.TrackingTask = TrackingTask(obs, info)
        self.CollisionTask = CollisionDetectionTask(obs, info)
        self.progress_reward = -np.inf
        self.tracking_reward = -np.inf
        self.collision_reward = -np.inf

        self.progress_cum_reward = []
        self.tracking_cum_reward = []
        self.collision_cum_reward = []
    
    def reward(self, states, action) -> tuple[float, bool]: 
        self.progress_reward = self.task_weights['progress'] * self.ProgressTask.reward(states, action)
        self.tracking_reward = self.task_weights['tracking'] * self.TrackingTask.reward(states, action)
        self.collision_reward = self.task_weights['collision'] * self.CollisionTask.reward(states, action)

        self.progress_cum_reward.append(self.progress_reward)
        self.tracking_cum_reward.append(self.tracking_reward)
        self.collision_cum_reward.append(self.collision_reward)

        total_reward = self.progress_reward + self.tracking_reward + self.collision_reward
        done = self.done(states)
        if done: total_reward = -10
        
        return total_reward, done
        
    def done(self, states) -> None:
        done = self.ProgressTask.done(states) or self.CollisionTask.done(states) \
            or self.TrackingTask.done(states)
        return done
    
    def reset(self):
        self.ProgressTask.reset()
        self.TrackingTask.reset()
        self.CollisionTask.reset()

    def console(self, action, mode:list=['all'], num=10):
        console_list = [f'Progress Reward: {self.progress_reward:.3}, Tracking Reward: {self.tracking_reward:.3f}, Collision Reward: {self.collision_reward:.3f}',
                        f'Progress Cumulative : {sum(self.progress_cum_reward):.3f}, Tracking Cumulative: {sum(self.tracking_cum_reward):.3f}, Collision Cumulative: {sum(self.collision_cum_reward):.3f}',
                        f'motor: {action[0]}, steer: {action[1]}']
        if 'all' in mode:
            print(console_list)
            return
        if 'reward' in mode:
            print(console_list[0])
        if 'cum' in mode:
            print(console_list[1])
        if 'cum_range' in mode:
            print(f'Progress Range: {sum(self.progress_cum_reward[-num:]):.3f}, Tracking Range: {sum(self.tracking_cum_reward[-num:]):.3f}, Collision Range: {sum(self.collision_cum_reward[-num:]):.3f}')
        if 'action' in mode:
            print(console_list[2])
        

class MaximizeProgressTask:
    def __init__(self, obs, info, progress_reward: float = 100.0):
        self._last_stored_progress = info['lap'] + info['progress']
        self._progress_reward = progress_reward
        self._t_low_motor = int(0)
        self._opitmal_speed = 3.0
        self._max_speed = 5.0
        self._min_speed = 0.5
        
        self._t_low_upper = 1000
    def reward(self, states, action) -> float:
        progress = states['lap'] + states['progress']
        delta = progress - self._last_stored_progress
        progress_reward = delta * self._progress_reward

        # Avoiding negative progress
        if progress_reward < 0.0:
            progress_reward *= 10
        curr_reward = progress_reward

        # Control Speed
        velocity = np.linalg.norm(states['velocity'])
        speed_reward = 0.0
        if self._min_speed < velocity < self._opitmal_speed:
            speed_reward = 0.003 * velocity
        elif velocity >= self._opitmal_speed:
            speed_reward = 0.005 * self._opitmal_speed - 0.001 * (velocity - self._opitmal_speed)
        else:
            speed_reward = -0.1 *  (self._min_speed - velocity)
        # if velocity < 1:
        #     self._t_low_motor += 1
        #     if self._t_low_motor > 100:
        #         curr_reward += -0.003*self._t_low_motor
        # elif velocity > 1:
        #     self._t_low_motor = 0
        #     curr_reward += 0.001 * velocity
        curr_reward += speed_reward

        self._last_stored_progress = progress
        return curr_reward
    def done(self, states)->bool:
        if self._t_low_motor > self._t_low_upper:
            return True
        return False
    def reset(self):
        self._last_stored_progress = None

class CollisionDetectionTask:
    def __init__(self, obs, info, collision_penalty=-100, obstacle_margin=0.3, lidar_margin=1.7,
                 scan_range=150):
        # Initialize some parameters of the task
        self._collided = False  # Flag to indicate whether a collision occurred
        self._collision_penalty = collision_penalty  # Penalty for collisions
        self._obstacle_margin = obstacle_margin  # Safe distance from obstacles
        self._lidar_high_margin = lidar_margin
        self._lidar_low_margin = 0.5
        self._lidar_mid_index = len(obs['lidar'])//2
        self.last_lidar_scans = obs['lidar']
        self.last_num_obstacle = len(np.where(self.last_lidar_scans < self._lidar_high_margin)[0])
        self._t_error = 0
        self._t_max = 4 # 7->4
        self._cum_penalty = 0
        # Scan scope
        self._scan_range = scan_range
        # plt.ion()
        # self._fig = plt.figure(figsize=(10, 5))

        # Speed
        self._opitmal_speed = 5.0
        self._max_speed = 7.0
        self._min_speed = 0.3
        
    def reward(self, states, action) -> float:
        # If a collision occurs, return the collision penalty
        curr_reward = float(0.0)
        curr_reward += self._check_collision(states)
        if curr_reward == self._collision_penalty: # Collision
            return curr_reward
        else:
            curr_reward += self._check_lidar3(states, action) #TODO
            return curr_reward

    def done(self, states) -> bool:
        """
        Check if the agent collides with the wall or is too close to obstacles
        """
        if hasattr(self, '_fig'):
            plt.close(self._fig)
            del self._fig  # Optional: clean up the attribute
        return self._collided
    
    def reset(self):
        self._collided = False

    def _check_lidar2(self, states, action) -> float: # 1009 0952
        self._lidar_high_margin = 4
        
        # range = 100
        # lidar_scans = states['lidar'][range:-range]  # Assuming lidar data is stored in state['lidar']
        
        
        # curr_reward = float(0.0)
        # velocity = np.linalg.norm(states['velocity'])
        # steer = action[1]

        # # Look far way
        # high_lidar_indices = np.where(lidar_scans > self._lidar_high_margin)[0]
        # len_high_lidar = len(high_lidar_indices)
        # lidar_mid_index = len(lidar_scans)//2
        # if abs(np.where(high_lidar_indices > lidar_mid_index)[0].size - np.where(high_lidar_indices < lidar_mid_index)[0].size) > 10:
        #     # Deviation Range
        #     high_lidar_mid = high_lidar_indices[len_high_lidar//2]
        #     deviation_range = 50 # 100->75

        #     # self._display_lidar(states)
        #     if high_lidar_mid < lidar_mid_index + deviation_range and high_lidar_mid > lidar_mid_index - deviation_range:
        #         if self._min_speed < velocity < self._opitmal_speed:
        #             curr_reward += 0.01 * velocity
        #     # if high_lidar_mid < lidar_mid_index + deviation_range and high_lidar_mid > lidar_mid_index - deviation_range and velocity >= 2:
        #     #     curr_reward += 0.03
        #     #     if velocity > 3:
        #     #         curr_reward += 0.03
        #     elif high_lidar_mid > lidar_mid_index + deviation_range :
        #         if self._min_speed < velocity < self._opitmal_speed and steer > 0:
        #             curr_reward += 0.02 * velocity
        #         elif steer < 0:
        #             curr_reward -= 0.2
        #         # self._display_lidar(states)
        #     elif high_lidar_mid < lidar_mid_index - deviation_range:
        #         if self._min_speed < velocity < self._opitmal_speed and steer < 0:
        #             curr_reward += 0.02 * velocity
        #         elif steer > 0:
        #             curr_reward -= 0.2
        #         # self._display_lidar(states)

        # # Check obstacles
        # range = 300
        # self._lidar_low_margin = 0.8 # 0.8->1
        # self._mid_lidar_low_margin = 2.5
        # # num_obstacles = len(np.where(lidar_scans < self._lidar_low_margin)[0]) # Count detected obstacles
        # left_obstacles = len(np.where(lidar_scans[:range] < self._lidar_low_margin)[0])
        # right_obstacles = len(np.where(lidar_scans[-range:] < self._lidar_low_margin)[0])
        # num_obstacles = left_obstacles + right_obstacles
        # s_range = 60
        # mid_obstacles = len(np.where(lidar_scans[lidar_mid_index-s_range:lidar_mid_index+s_range] < self._mid_lidar_low_margin)[0])
        # ml_obstacles = len(np.where(lidar_scans[lidar_mid_index-s_range:lidar_mid_index] < self._mid_lidar_low_margin)[0])
        # mr_obstacles = len(np.where(lidar_scans[lidar_mid_index:lidar_mid_index+s_range] < self._mid_lidar_low_margin)[0])
    
        # baseline_obstacles = 200
        
        # if (mid_obstacles) > 80 and (-0.2<steer<0.2):
        #     self._t_error +=1
        #     self._cum_penalty -= 0.01 * (ml_obstacles + mr_obstacles) * velocity
        #     print(f'Obstacle Detected: {mid_obstacles}, mid_obstacles')
        # # Turn right
        # elif ml_obstacles > 50 and ml_obstacles > mr_obstacles and steer <  -0.2:
        #     self._t_error +=1
        #     self._cum_penalty -= 0.015 * ml_obstacles * velocity * abs(steer)
        #     print(f'Obstacle Detected: {ml_obstacles}, ml_obstacles')
        # # Turn left
        # elif mr_obstacles > 50 and mr_obstacles > ml_obstacles and steer > 0.2:
        #     self._t_error +=1
        #     self._cum_penalty -= 0.015 * mr_obstacles * velocity * abs(steer)
        #     print(f'Obstacle Detected: {mr_obstacles}, mr_obstacles')
        
        # # if mid_obstacles > 50:
        # #     if left_obstacles > right_obstacles and steer < 0:
        # #         curr_reward -= 0.05 * (num_obstacles - 50) * velocity
        # #     elif left_obstacles < right_obstacles and steer > 0:
        # #         curr_reward -= 0.05 * (num_obstacles - 50) * velocity
        # #     elif left_obstacles == right_obstacles:
        # #         curr_reward -= 0.05 * (num_obstacles - 50) * velocity
        # #     print(f'Obstacle Detected: {mid_obstacles}, mid_obstacles')
        # elif num_obstacles>baseline_obstacles and left_obstacles > right_obstacles:
        #     self._t_error +=1
        #     self._cum_penalty -= 0.03 * (num_obstacles - baseline_obstacles)
        #     # curr_reward -= 0.003 * (num_obstacles)
        #     print(f'Obstacle Detected: {left_obstacles}, left_obstacles')
        # elif num_obstacles>baseline_obstacles and left_obstacles > right_obstacles:
        #     self._t_error +=1
        #     self._cum_penalty -= 0.03 * (num_obstacles - baseline_obstacles)
        #     # curr_reward -= 0.003 * (num_obstacles)
        #     print(f'Obstacle Detected: {right_obstacles}, right_obstacles')
        # else:
        #     if self._t_error > 0:
        #         self._t_error -=1
        #     else:
        #         self._cum_penalty = 0
        #     curr_reward += 0.01

        # if self._t_error > self._t_max:
        #     curr_reward = self._cum_penalty * 0.1 * (np.log2((self._t_error - self._t_max))+1)
        #     print(f'get penlaty: {curr_reward}')

        # return curr_reward
            
    def _check_lidar3(self, states, action) -> float:
        self._lidar_high_margin = 4
        
        range = 100
        lidar_scans = states['lidar'][range:-range]  # Assuming lidar data is stored in state['lidar']
        
        
        curr_reward = float(0.0)
        velocity = np.linalg.norm(states['velocity'])
        steer = action[1]

        # Look far way
        high_lidar_indices = np.where(lidar_scans > self._lidar_high_margin)[0]
        len_high_lidar = len(high_lidar_indices)
        lidar_mid_index = len(lidar_scans)//2
        if abs(np.where(high_lidar_indices > lidar_mid_index)[0].size - np.where(high_lidar_indices < lidar_mid_index)[0].size) > 10:
            # Deviation Range
            high_lidar_mid = high_lidar_indices[len_high_lidar//2]
            deviation_range = 50 # 100->75

            # self._display_lidar(states)
            if high_lidar_mid < lidar_mid_index + deviation_range and high_lidar_mid > lidar_mid_index - deviation_range:
                if self._min_speed < velocity < self._opitmal_speed:
                    curr_reward += 0.01 * velocity
            # if high_lidar_mid < lidar_mid_index + deviation_range and high_lidar_mid > lidar_mid_index - deviation_range and velocity >= 2:
            #     curr_reward += 0.03
            #     if velocity > 3:
            #         curr_reward += 0.03
            elif high_lidar_mid > lidar_mid_index + deviation_range :
                if self._min_speed < velocity < self._opitmal_speed and steer > 0:
                    curr_reward += 0.02 * velocity
                elif steer < 0:
                    curr_reward -= 0.2
                # self._display_lidar(states)
            elif high_lidar_mid < lidar_mid_index - deviation_range:
                if self._min_speed < velocity < self._opitmal_speed and steer < 0:
                    curr_reward += 0.02 * velocity
                elif steer > 0:
                    curr_reward -= 0.2
                # self._display_lidar(states)

        # Check obstacles
        range = 300
        self._lidar_low_margin = 0.6 # 0.8->1
        self._mid_lidar_low_margin = 3.5 # 2.5->3
        # num_obstacles = len(np.where(lidar_scans < self._lidar_low_margin)[0]) # Count detected obstacles
        left_obstacles = len(np.where(lidar_scans[:range] < self._lidar_low_margin)[0])
        right_obstacles = len(np.where(lidar_scans[-range:] < self._lidar_low_margin)[0])
        num_obstacles = left_obstacles + right_obstacles
        s_range = 60
        mid_obstacles = len(np.where(lidar_scans[lidar_mid_index-s_range:lidar_mid_index+s_range] < self._mid_lidar_low_margin)[0])
        ml_obstacles = len(np.where(lidar_scans[lidar_mid_index-s_range:lidar_mid_index] < self._mid_lidar_low_margin)[0])
        mr_obstacles = len(np.where(lidar_scans[lidar_mid_index:lidar_mid_index+s_range] < self._mid_lidar_low_margin)[0])
    
        baseline_obstacles = 200
        
        if (mid_obstacles) > 80 and (-0.2<steer<0.2):
            self._t_error +=1
            self._cum_penalty -= 0.01 * (ml_obstacles + mr_obstacles) * velocity
            print(f'Obstacle Detected: {mid_obstacles}, mid_obstacles')
            local_margin = 1.5
            right_ob = len(np.where(lidar_scans[-range:] < local_margin)[0])
            left_ob = len(np.where(lidar_scans[:range] < local_margin)[0])
            num_ob = right_ob + left_ob
            if num_ob >= 250:
                self._t_error +=1
                print(f'Obstacle Detected: {num_ob}, num_ob')
                curr_reward -= 0.003 * num_ob * velocity
        # Turn right
        elif ml_obstacles > 50 and ml_obstacles > mr_obstacles and steer <  -0.2:
            self._t_error +=1
            self._cum_penalty -= 0.015 * ml_obstacles * velocity * abs(steer)
            print(f'Obstacle Detected: {ml_obstacles}, ml_obstacles')
        # Turn left
        elif mr_obstacles > 50 and mr_obstacles > ml_obstacles and steer > 0.2:
            self._t_error +=1
            self._cum_penalty -= 0.015 * mr_obstacles * velocity * abs(steer)
            print(f'Obstacle Detected: {mr_obstacles}, mr_obstacles')
        
        # if mid_obstacles > 50:
        #     if left_obstacles > right_obstacles and steer < 0:
        #         curr_reward -= 0.05 * (num_obstacles - 50) * velocity
        #     elif left_obstacles < right_obstacles and steer > 0:
        #         curr_reward -= 0.05 * (num_obstacles - 50) * velocity
        #     elif left_obstacles == right_obstacles:
        #         curr_reward -= 0.05 * (num_obstacles - 50) * velocity
        #     print(f'Obstacle Detected: {mid_obstacles}, mid_obstacles')
        elif num_obstacles>baseline_obstacles and left_obstacles > right_obstacles:
            self._t_error +=1
            self._cum_penalty -= 0.005 * (num_obstacles - baseline_obstacles)
            # curr_reward -= 0.003 * (num_obstacles)
            print(f'Obstacle Detected: {left_obstacles}, left_obstacles')
        elif num_obstacles>baseline_obstacles and right_obstacles > left_obstacles:
            self._t_error +=1
            self._cum_penalty -= 0.005 * (num_obstacles - baseline_obstacles)
            # curr_reward -= 0.003 * (num_obstacles)
            print(f'Obstacle Detected: {right_obstacles}, right_obstacles')
        else:
            if self._t_error > 0:
                self._t_error -=1
            else:
                self._cum_penalty = 0
            curr_reward += 0.01 * abs(steer)

        if self._t_error > self._t_max:
            curr_reward = self._cum_penalty * 0.05 * (np.log2((self._t_error - self._t_max))+1)
            print(f'get penlaty: {curr_reward}')

        return curr_reward
    def _check_lidar(self, states, action) -> float:
        lidar_scans = states['lidar']  # Assuming lidar data is stored in state['lidar']
        curr_reward = float(0.0)
        motor = action[0]
        steer = action[1]

        num_obstacles = len(np.where(lidar_scans < self._lidar_high_margin)[0]) # Count detected obstacles

        # Compare with previous lidar scans to penalize increase in detected obstacles
        if 'lidar' in states:
            range = 200
            left_scans = states['lidar'][:self._lidar_mid_index+range]
            right_scans = states['lidar'][self._lidar_mid_index-range:]

            if num_obstacles > 500:
                # print(f'Obstacle Detected: {num_obstacles}, reward: {curr_reward}')
                left_obstacle = len(np.where(left_scans < self._lidar_high_margin)[0])
                right_obstacle = len(np.where(right_scans < self._lidar_high_margin)[0])
                max_scan = np.max(lidar_scans)
                
                if max_scan > 10:
                    # go right
                    if np.argmax(lidar_scans)>self._lidar_mid_index and steer >= 1 and motor > 0:
                        curr_reward += 0.5
                    # go left
                    elif np.argmax(lidar_scans)<self._lidar_mid_index and steer <= 1 and motor > 0:
                        curr_reward += 0.5
                    # if max_left_scan < max_right_scan and steer > 0 and motor > 0:
                    #     curr_reward += 0.5
                    # # go left
                    # elif max_left_scan > max_right_scan and steer < 0 and motor > 0:
                    #     curr_reward += 0.5
                else:
                    # go right
                    if left_obstacle > right_obstacle and steer > 0 and motor>0:
                        curr_reward += 0.5
                    # go left
                    elif left_obstacle < right_obstacle and steer < 0 and motor>0:
                        curr_reward += 0.5
                self._display_lidar(states)

            if num_obstacles > 800:
                curr_reward -= (num_obstacles - self.last_num_obstacle) * num_obstacles * 0.0001 + 1
                if curr_reward <= -10:
                    curr_reward = -10
                print(f'Obstacle Detected: {num_obstacles}, reward: {curr_reward}')
                # self._display_lidar(states)
        self.last_lidar_scans = lidar_scans
        self.last_num_obstacle = num_obstacles

        return curr_reward

    def _display_lidar(self, states):
        lidar_scans = states['lidar']
        plt.clf()
        # if not hasattr(self, '_fig'):
        #     self._fig = plt.figure(figsize=(10, 5))
        plt.plot(lidar_scans, label='Lidar Scan Distances')
        plt.xlabel('Lidar Scan Index')
        plt.ylabel('Distance (meters)')
        plt.title('1D Lidar Scan Visualization')
        plt.grid(True)
        plt.legend()
        plt.pause(0.001)
        # plt.ioff()
    def _check_collision(self, states) -> float:
        """
        Assume a simple collision detection logic
        Return: 
            - collision_penalty: float, penalty for collision
            - is_done: bool, True if the agent collides with the wall
        """
        # Check if the agent touches the wall
        if states['wall_collision']:
            self._collided = True
            return self._collision_penalty
        # Check if the agent is too close to obstacles
        if states['obstacle'] < self._obstacle_margin:
            # return (1+np.exp(((states['obstacle']-self.safe_margin))**2)) * self.collision_penalty
            # return (10*(self.safe_margin - states['obstacle']))**1.5
            return 2*((5*(self._obstacle_margin - states['obstacle']))**3)
        return float(0.0)

class TrackingTask:
    def __init__(self, obs, info, position:list = [0,0,0], action:list[float,float] = [0,0], state_gain: float = 0.1,
                 action_gain: float = 0.5):
        self._init_position = position # init position
        self._init_action = action
        self._last_position = self._init_position
        self._last_action = self._init_action
        self._state_gain = state_gain
        self._action_gain = action_gain
                 
    def reward(self, states, action) -> float:
        """
        Idea: def. a quadratic cost by weighting the deviation from a target state (waypoint) and from the prev action.
        However, aiming to have a positive reward, the change the sign (i.e. reward=-cost) lead to cumulative penalties
        which encourage the agent to terminate the episode asap.
        For this reason, the resulting negative cost is passed through an exponential function,
        obtaining the desired behaviour:
            1. exp(- small cost) -> 1
            2. exp(- big cost) -> 0
        Optionally, we can add a negative reward in case of collision.
        """
        # LQR
        position = states['pose'][:3]
        waypoint = self._last_position
        Q = self._state_gain * np.identity(len(position))
        R = self._action_gain * np.identity(len(action))
        delta_pos = waypoint - position
        delta_act = np.array(action) - np.array(self._last_action)
        cost = (np.matmul(delta_pos, np.matmul(Q, delta_pos)) + np.matmul(delta_act, np.matmul(R, delta_act)))
        curr_reward = np.exp(-cost)
        
        self._last_action = action
        self._last_position = position

        # Speed Control
        # print(f'Current Speed: {states["velocity"]}, Target Speed: {action[1]}')

        return curr_reward
    def done(self, states) -> bool:
        return False
    def reset(self, states):
        self._last_position = self._init_position
        self._last_action = self._init_action
# Deprecated
class MaximizeProgressRegularizeAction(MaximizeProgressTask):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool, delta_progress=0.0,
                 collision_reward=0, frame_reward=0, progress_reward=100, action_reg=0.25):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward)
        self._action_reg = action_reg
        self._last_action = None

    def reset(self):
        super(MaximizeProgressRegularizeAction, self).reset()
        self._last_action = None

    def reward(self, agent_id, state, action) -> float:
        """ Progress-based with action regularization: penalize sharp change in control"""
        reward = super().reward(agent_id, state, action)
        action = np.array(list(action.values()))
        if self._last_action is not None:
            reward -= self._action_reg * np.linalg.norm(action - self._last_action)
        self._last_action = action
        return reward

# Deprecated
class RankDiscountedMaximizeProgressTask(MaximizeProgressTask):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool, delta_progress=0.001,
                 collision_reward=-100, frame_reward=-0.1, progress_reward=1):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward)

    def reward(self, agent_id, state, action) -> float:
        rank = state[agent_id]['rank']
        reward = super().reward(agent_id, state, action)
        reward = reward / float(rank)
        return reward
