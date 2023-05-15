
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from car_river_crossing import CarRiverCrossing
from common_functions import generate_state_frame_stack_from_queue
from common_functions import process_state_image
import numpy as np
from os.path import exists

SKIP_FRAMES = 3
RENDER = True

FILE_NAME = './rewards_r_bound.csv'

def log(txt):
    with open('./find_r_bound_300.log', 'a') as f:
        f.write(txt + '\n')
    print(txt)

if __name__ == '__main__':
    play_area = 300
    zoom = 1.8

    if RENDER:
        env = CarRiverCrossing(render_mode='human', play_field_area=play_area, zoom=zoom)
    else:
        env = CarRiverCrossing(play_field_area=play_area, zoom=zoom)


    all_rewards = {}

    policies = [-0.5, 0.0, 0.5, 1.0]

    for policy_id in policies:

        all_rewards[policy_id] = []

        # Set epsilon to 0 to ensure all actions are instructed by the agent
        agent = CarRacingDQNAgent(epsilon=0, lamb=0.0)

        play_episodes = 10

        episode = 25

        while True:
            train_model = './r_bound_300_v2/trial_{}_{}.h5'.format(policy_id, episode)

            episode+=25
            if not exists(train_model):
                break

            agent.load(train_model)

            init_state, info = env.reset()
            init_state = process_state_image(init_state)

            total_reward = 0
            punishment_counter = 0
            state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
            time_frame_counter = 1

            rewards = []

            while True:
                current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
                action = agent.act(current_state_frame_stack)
                reward = 0
                for _ in range(SKIP_FRAMES + 1):
                    next_state, r, terminated, truncated, info = env.step(action)
                    reward += r
                    if terminated or truncated:
                        break

                time_frame_counter +=1

                next_state = process_state_image(next_state)
                done = terminated

                init_state = next_state

                total_reward += reward
                rewards.append(reward)

                if time_frame_counter > 200:
                    log('BAD {}'.format(train_model))
                    break

                if truncated:
                    total_reward = 0
                    rewards = []

                state_frame_stack_queue.append(next_state)

                if done:
                    log('GOOD {}'.format(train_model))

                    break

            all_rewards[policy_id].append(total_reward)


