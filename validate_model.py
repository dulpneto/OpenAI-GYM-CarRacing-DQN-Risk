import argparse
from collections import deque

import numpy as np
import math

from CarRacingDQNAgent import CarRacingDQNAgent
from CarRacingEnv import CarRacingEnv
from common_functions import generate_state_frame_stack_from_queue

if __name__ == '__main__':

    env = CarRacingEnv(render=False, frames_to_run=200)

    all_rewards = {}
    all_rewards_utility = {}
    gamma = 0.95

    for l in range(-100, 101, 25):
        lamb = l/100

        all_rewards[lamb] = []
        all_rewards_utility[lamb] = {}

        train_model = './save_data/slippery/trial_{}_10000.h5'.format(lamb)

        # Set epsilon to 0 to ensure all actions are instructed by the agent
        agent = CarRacingDQNAgent(epsilon=0, lamb=lamb)
        agent.load(train_model)

        play_episodes = 1000


        for e in range(play_episodes):
            init_state, info = env.reset()

            total_reward = 0
            punishment_counter = 0
            state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
            time_frame_counter = 1

            rewards = []

            while True:
                env.render()

                current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
                action = agent.act(current_state_frame_stack)
                next_state, reward, terminated, truncated, info = env.step(action)

                done = (terminated or truncated)

                init_state = next_state

                total_reward += reward
                rewards.append(reward)

                state_frame_stack_queue.append(next_state)

                # print('{} REWARD {} TILES {}'.format(environment.frames, reward, environment.tiles_visited))

                if done:
                    print(
                        'Risk: {}, Episode: {}/{}, Total Frames: {}, Tiles Visited: {}, Total Rewards: {}'.format(lamb,
                                                                                                                  e + 1,
                                                                                                                  play_episodes,
                                                                                                                  env.frames,
                                                                                                                  env.tiles_visited,
                                                                                                                  env.total_reward))

                    for l2 in range(-100, 101, 25):
                        lamb2 = l2 / 100
                        all_rewards_utility[lamb][lamb2] = []

                        t = 0
                        try:
                            for i in range(len(rewards)-1, -1, -1):
                                u = np.sign(lamb2) * math.exp(lamb2 * rewards[i])
                                t = u + gamma * t

                            all_rewards_utility[lamb][lamb2].append(t)
                        except:
                            print("EXP error for {} {}".format(lamb, lamb2))

                    break

            all_rewards[lamb].append(total_reward)

    print('\n\n*** FINAL ****\n')
    print('RISK\tMEAN\tVAR')
    for l in range(-100, 101, 25):
        lamb = l/100
        print('{}\t{}\t{}'.format(lamb, round(np.mean(all_rewards[lamb]), 2), round(np.var(all_rewards[lamb]), 2)))

    print('\n\n*** UTILITY ****\n')

    v = 'XXXX'.format(lamb)
    for l in range(-100, 101, 25):
        lamb = l / 100
        v = v+'\t{}'.format(lamb)
    print(v)
    for l in range(-100, 101, 25):
        lamb = l / 100
        v = 'LAMB {}'.format(lamb)
        for l2 in range(-100, 101, 25):
            lamb2 = l2 / 100
            v = v + '\t{}'.format(round(np.mean(all_rewards_utility[lamb][lamb2]), 2))
        print(v)
