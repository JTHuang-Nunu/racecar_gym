def main():
    # ======================================================================
    # Create the environment
    # ======================================================================
    render_mode = 'human'  # 'human', 'rgb_array_birds_eye' and 'rgb_array_follow'
    env = gymnasium.make(
        'SingleAgentAustria-v0',
        render_mode=render_mode,
        scenario='scenarios/austria.yml',  # change the scenario here (change map)
        # scenario='scenarios/validation.yml',  # change the scenario here (change map), ONLY USE THIS FOR VALIDATION
        # scenario='scenarios/validation2.yml',   # Use this during the midterm competition, ONLY USE THIS FOR VALIDATION
    )

    EPOCHS = 1000
    MAX_STEP = 12000
    #MAX_STEP = 6000
    best_reward = -np.inf
    agent = get_training_agent(agent_name='PPO')

    # ======================================================================
    # Run the environment
    # ======================================================================
    for e in range(EPOCHS):
        obs, info = env.reset(options=dict(mode='grid'))
        t = 0
        total_reward = 0
        old_progress = 0
        old_checkpoint = 0
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
            """
            # Calculate reward
            
            reward = 0
            reward += np.linalg.norm(states['velocity'][:3])
            reward += states['progress'] - old_progress
            old_progress = states['progress']

            if states['wall_collision']:
                reward = -10
                done = True

            total_reward += reward
            """

            # Lidar-based obstacle detection
            lidar_scans = obs['lidar']  # Assuming lidar data is stored in obs['lidar']
            distance_threshold = -0.4  # Define a threshold for obstacle detection (adjust as needed)

            # Detect obstacles by checking if any lidar scans are below the distance threshold
            obstacle_indices = np.where(lidar_scans < distance_threshold)[0]  # Get indices of obstacles
            num_obstacles = len(obstacle_indices)  # Count detected obstacles

            # Compare with previous lidar scans to penalize increase in detected obstacles
            if 'old_lidar_scans' in states:
                # Compare the number of obstacles between current and previous lidar scans
                previous_obstacle_indices = np.where(states['old_lidar_scans'] < distance_threshold)[0]
                num_previous_obstacles = len(previous_obstacle_indices)
                
                if num_obstacles > num_previous_obstacles:
                    # Penalize if more obstacles are detected compared to the previous step
                    obstacle_penalty = -(num_obstacles - num_previous_obstacles) * 0.5  # Adjust penalty weight as needed
                else:
                    obstacle_penalty = 0
            else:
                # No penalty if this is the first lidar scan
                obstacle_penalty = 0

            # Update previous lidar scan for the next step
            states['old_lidar_scans'] = lidar_scans

            # Initialize time variables
            previous_time = states['time']

            # Calculate rewards
            speed_reward = np.linalg.norm(states['velocity'][:3]) * 1.0  # Scaled speed reward
            progress_reward = states['progress'] - old_progress
            checkpoint_reward = 50 if states['checkpoint'] > old_checkpoint else 0  # New checkpoint reward

            # Time penalty
            current_time = states['time']
            time_penalty = -0.1 * (current_time - previous_time)  # Encourage faster completion
            previous_time = current_time  # Update for next step

            # Penalty for being stuck (no progress for 3 seconds)
            if states['progress'] <= old_progress and (current_time - previous_time) > 3:
                time_penalty -= 30  # Significant penalty for being stuck

            # Collision handling
            if states['wall_collision']: