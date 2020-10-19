import argparse
import random
import time
import datetime

import os
import gym
import numpy as np
import itertools
import torch
from sac import SAC
import wandb
from replay_memory import ReplayMemory

import dmc2gym


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="hopper", choices=['hopper', 'ant', 'walker', 'cheetah', 'humanoid', 'finger', 'cartpole', 'dog', 'quad', 'reacher', 'ball', 'pick', 'push', 'reach', 'slide', 'bring_ball', 'bring_peg', 'insert_ball', 'insert_peg', 'lift', 'site', 'place'],
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')


# OUR ARGS
parser.add_argument('--num_q_layers', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--dmc', action='store_true')
parser.add_argument('--manip', action='store_true')
parser.add_argument('--rw', action='store_true')
parser.add_argument('--challenge', default='easy', type=str)
args = parser.parse_args()

gym_env = 'dmc' if args.dmc else ''

writer = wandb.init(
        project='d2rl',
        config=args,
        dir='wandb_logs',
        group='{}_{}{}'.format(args.env_name, gym_env),
)

assert args.challenge in ['easy', 'hard', 'medium']
if args.rw:
    assert args.env_name in ['cartpole', 'walker']


# Environment
envs = {
        'hopper': 'Hopper-v2',
        'walker': 'Walker2d-v2',
        'cheetah': 'HalfCheetah-v2', 
        'ant': 'Ant-v2',
        'humanoid': 'Humanoid-v2',
        'pick': 'FetchPickAndPlace-v1',
        'push': 'FetchPush-v1',
        'reach': 'FetchReach-v1',
        'slide': 'FetchSlide-v1',
}

args.gcp = args.env_name in ['pick', 'push', 'reach', 'slide'] and not args.manip

dm_envs = {
        'finger': ['finger', 'spin'],
        'dog': ['dog', 'walk'],
        'quad': ['quadruped', 'walk'],
        'cartpole': ['cartpole', 'swingup'],
        'reacher': ['reacher', 'easy'],
        'cheetah': ['cheetah', 'run'],
        'walker': ['walker', 'walk'],
        'ball': ['ball_in_cup', 'catch'],
        'humanoid': ['humanoid', 'stand'],
        'bring_ball': ['manipulator', 'bring_ball'],
        'bring_peg': ['manipulator', 'bring_peg'],
        'insert_ball': ['manipulator', 'insert_ball'],
        'insert_peg': ['manipulator', 'insert_peg'],
}

manip_envs = {
        'site': 'reach_site_features',
        'lift': 'lift_brick_features',
        'place': 'place_cradle_features',
}

if args.dmc:
    if args.env_name in ['dog', 'quad']:
        args.num_steps = int(args.num_steps * 3) 
    random.seed(time.time())
    seed = random.randint(0, 1000)
    env_details = dm_envs[args.env_name]
    env = dmc2gym.make(
            domain_name=env_details[0],
            task_name=env_details[1],
            seed=seed,
    )
    env.seed(seed)
    if env_details[0] == 'cheetah':
        args.batch_size = 512
    else:
        args.batch_size = 128 
    args.lr = 1e-4
    args.hidden_size = 1024
    args.alpha = 0.1
    args.target_update_interval = 2
    test_env = env
elif args.manip:
#    args.num_steps = int(args.num_steps) if args.env_name == 'site' else int(args.num_steps*3)
    args.num_steps = int(args.num_steps*5)
    random.seed(time.time())
    seed = random.randint(0, 1000)
    env = manip_envs[args.env_name]
    env = dmc2gym.make(
            domain_name=env,
            seed=seed,
    )
    env.seed(seed)
    args.lr = 1e-4
    args.hidden_size = 1024
    args.alpha = 0.1
    args.target_update_interval = 2
    test_env = env
elif args.rw:
    args.num_steps = int(args.num_steps)
    random.seed(time.time())
    seed = random.randint(0, 1000)
    env_details = dm_envs[args.env_name]
    if env_details[0] == 'cartpole':
        env_details[1] = 'realworld_swingup'
    else:
        env_details[1] = 'realworld_walk'
    env = dmc2gym.make(
            domain_name=env_details[0],
            task_name=env_details[1],
            challenge=args.challenge,
    )
    env.seed(seed)
    args.lr = 1e-4
    args.hidden_size = 1024
    args.alpha = 0.1
    args.target_update_interval = 2
    test_env = env
else:
    args.env_name = envs[args.env_name]
    if args.gcp:
        env = gym.make(args.env_name, reward_type='dense')
    else:
        env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

args.cuda = True if torch.cuda.is_available() else False

# Agent
if args.gcp:
    obs_space = env.observation_space['desired_goal'].shape[0] + \
            env.observation_space['observation'].shape[0]
else:
    obs_space = env.observation_space.shape[0] 

agent = SAC(obs_space, env.action_space, args)
args.automatic_entropy_tuning = True

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    if args.gcp:
        goal = state['desired_goal']

    while not done:
        state = state['observation'] if args.gcp else state
        state =  np.concatenate([state, goal]) if args.gcp else state

        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)


                updates += 1

        next_state, reward, done, info = env.step(action) # Step

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
#        mask = 1 if episode_steps == 2000 else float(not done)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        next_state_push = next_state['observation'] if args.gcp else next_state
        next_state_push =  np.concatenate([next_state_push, goal]) if args.gcp else next_state

        memory.push(state, action, reward, next_state_push, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    train_reward = episode_reward

    with torch.no_grad():
        if i_episode % 1 == 0:
            avg_reward = 0.
            episodes = 5
            for _  in range(episodes):
                state = test_env.reset()
                if args.gcp:
                    goal = state['desired_goal']
                episode_reward, episode_success  = 0, False
                done = False
                while not done:
                    state = state['observation'] if args.gcp else state
                    state = np.concatenate([state, goal]) if args.gcp else state
        
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, info = test_env.step(action)

                    if args.gcp and reward != -1.0:
                        done = True
                        episode_success = True
                    else:
                        episode_reward += reward

                    state = next_state
                if args.gcp:
                    avg_reward += 1 if episode_success else 0
                else:
                    avg_reward += episode_reward
            avg_reward /= episodes


            writer.log({
                    'reward': avg_reward,
                    'train_reward': train_reward, 
                    'env_steps': total_numsteps,
                }
            )

env.close()

