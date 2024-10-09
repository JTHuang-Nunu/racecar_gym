from agent.DQN.dqn_agent import DQNAgent
from agent.PPO.ppo_agent import PPOAgent


def get_training_agent(agent_name: str = 'DQN', max_step: int = 6000, model_path: str = None):
    if agent_name == 'DQN':
        agent = DQNAgent(
            state_dim=1098,
            lr=1e-3,
            gamma=0.99,
            epsilon=0.8,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            memory_size=10000,
            batch_size=64,
            target_update=100,
            is_training=True
        )
    elif agent_name == 'PPO':
        agent = PPOAgent(
            obs_dim=1098,
            action_dim=2,
            max_step=max_step,
            gamma=0.99,
            lamb=0.95,
            lr=1e-4,
            clip_val=0.2,
            max_grad_norm=0.5,
            ent_weight=0.01,
            sample_n_epoch=4,
            sample_mb_size=64,
            is_training=True
        )
    elif agent_name == 'PPO_LOAD':
        agent = PPOAgent(
            obs_dim=1098,
            action_dim=2,
            max_step=max_step,
            gamma=0.99,
            lamb=0.95,
            lr=1e-4,
            clip_val=0.2,
            max_grad_norm=0.5,
            ent_weight=0.01,
            sample_n_epoch=10, # 4->10
            sample_mb_size=32, # 64->128
            is_training=True,
            is_resume_training=True,
            # model_path='./agent/PPO/weight.pth'
            model_path='./agent/PPO/weight_23.pth' if model_path is None else model_path
        )
    
    else:
        raise NotImplementedError

    return agent


def get_valid_agent(agent_name: str = 'DQN', model_path: str = None):
    if agent_name == 'DQN':
        agent = DQNAgent(
            state_dim=1098,
            lr=1e-3,
            gamma=0.99,
            epsilon=0.8,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            memory_size=10000,
            batch_size=64,
            target_update=100,
            is_training=False
        )
    elif agent_name == 'PPO':
        agent = PPOAgent(
            obs_dim=1098,
            action_dim=2,
            max_step=6000,
            gamma=0.99,
            lamb=0.95,
            lr=1e-4,
            clip_val=0.2,
            max_grad_norm=0.5,
            ent_weight=0.01,
            sample_n_epoch=4,
            sample_mb_size=64,
            is_training=False
        )
    elif agent_name == 'PPO_VALID':
        agent = PPOAgent(
            obs_dim=1098,
            action_dim=2,
            max_step=6000,
            gamma=0.99,
            lamb=0.95,
            lr=1e-4,
            clip_val=0.2,
            max_grad_norm=0.5,
            ent_weight=0.01,
            sample_n_epoch=4,
            sample_mb_size=64,
            is_training=False,
            model_path='./agent/PPO/weight_41.pth' if model_path is None else model_path
        )
    else:
        raise NotImplementedError

    return agent
