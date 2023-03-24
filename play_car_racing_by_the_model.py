import argparse
import cv2
import numpy as np
import gymnasium as gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-m', '--model', required=True, help='The `.h5` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes should the model plays.')
    parser.add_argument('-l', '--lamb', type=float, default=0.0, help='The risk param, default to 0.0.')
    args = parser.parse_args()
    train_model = args.model
    play_episodes = args.episodes

    env = gym.make('CarRacing-v2', render_mode="human")
    # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent = CarRacingDQNAgent(epsilon=0, lamb=args.lamb)
    agent.load(train_model)

    for e in range(play_episodes):
        init_state, info = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        
        while True:
            env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            next_state, reward, terminated, truncated, info = env.step((0, 0,   0))


            cv2.imwrite('frames/frame_{}.jpeg'.format(time_frame_counter), next_state)
            cropped = next_state[0:80, 8:88]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray[gray > 150] = 255
            gray[gray <= 150] = 0
            cv2.imwrite('frames/frame_{}_crop.jpeg'.format(time_frame_counter), gray)

            done = (terminated or truncated)

            similar = 'DIFF'
            if time_frame_counter > 1 and all((np.array(init_state).flatten()) == (np.array(next_state).flatten())):
                similar = 'SIMILAR'

            print(time_frame_counter, 'REWARD', reward, "TOTAL", total_reward, similar)

            init_state = next_state

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)

            if done or time_frame_counter > 50:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {}'.format(e+1, play_episodes, time_frame_counter, total_reward))
                break
            time_frame_counter += 1
