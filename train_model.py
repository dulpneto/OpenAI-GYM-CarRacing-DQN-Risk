import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import argparse
import gymnasium as gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue

RENDER                        = False
STARTING_EPISODE              = 1
ENDING_EPISODE                = 10000
SKIP_FRAMES                   = 3
TRAINING_BATCH_SIZE           = 64
TRAINING_MODEL_FREQUENCY      = 1
SAVE_TRAINING_FREQUENCY       = 10
UPDATE_TARGET_MODEL_FREQUENCY = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-s', '--start', type=int, help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int, help='The ending episode, default to 1000.')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='The starting epsilon of the agent, default to 1.0.')
    parser.add_argument('-l', '--lamb', type=float, default=0.0, help='The risk param, default to 0.0.')
    parser.add_argument('-f', '--progress_file', type=bool, default=False, help='save progress files, default to False.')
    args = parser.parse_args()

    print('Training with risk factor', args.lamb)

    if RENDER:
        env = gym.make('CarRacing-v2', render_mode="human")
    else:
        env = gym.make('CarRacing-v2')

    agent = CarRacingDQNAgent(epsilon=args.epsilon, lamb=args.lamb)
    if args.model:
        agent.load(args.model)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end

    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):
        init_state, info = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False
        
        while True:
            if RENDER:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            reward = 0
            for _ in range(SKIP_FRAMES+1):
                next_state, r, terminated, truncated, info = env.step(action)
                done = (terminated or truncated)
                reward += r
                if done:
                    break

            # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            # Extra bonus for the model if it uses full gas
            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {}, Epsilon: {:.2}'.format(e, ENDING_EPISODE, time_frame_counter, total_reward, float(agent.epsilon)))
                break

            time_frame_counter += 1

            if len(agent.memory) > TRAINING_BATCH_SIZE and time_frame_counter % TRAINING_MODEL_FREQUENCY == 0:
                agent.replay_batch(TRAINING_BATCH_SIZE)

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            agent.save('./save/trial_{}_{}.h5'.format(args.lamb, e))

            if args.progress_file:
                with open('./CURRENT_MODEL.txt', 'w') as f:
                    f.write('./save/trial_{}_{}.h5'.format(args.lamb, e))
                with open('./NEXT_EPISODE.txt', 'w') as f:
                    f.write('{}'.format(e+1))
                with open('./CURRENT_EPSILON.txt', 'w') as f:
                    f.write('{}'.format(float(agent.epsilon)))

    env.close()
