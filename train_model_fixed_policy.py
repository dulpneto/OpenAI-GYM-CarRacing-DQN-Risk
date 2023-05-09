import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import argparse
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from car_river_crossing import CarRiverCrossing

from common_functions import generate_state_frame_stack_from_queue
from common_functions import process_state_image

from datetime import datetime

STARTING_EPISODE              = 1
ENDING_EPISODE                = 10000
TRAINING_BATCH_SIZE           = 64
TRAINING_MODEL_FREQUENCY      = 4
SAVE_TRAINING_FREQUENCY       = 25
UPDATE_TARGET_MODEL_FREQUENCY = 1
RESETS_BEFORE_FIXED_POLICY    = 2
SKIP_FRAMES                   = 3
MAXIMUM_FRAMES                = 150

def log(txt, lamb):
    with open('./save/result_train_fixed_{}.log'.format(lamb), 'a') as f:
        f.write(txt + '\n')
    print(txt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-s', '--start', type=int, help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int, help='The ending episode, default to 1000.')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='The starting epsilon of the agent, default to 1.0.')
    parser.add_argument('-l', '--lamb', type=float, default=0.0, help='The risk param, default to 0.0.')
    parser.add_argument('-r', '--render', type=bool, default=False, help='Render while training, default to False.')
    args = parser.parse_args()

    print('Training with risk factor', args.lamb)

    play_area = 300
    zoom = 1.8

    if args.render:
        env = CarRiverCrossing(render_mode='human', play_field_area=play_area, zoom=zoom)
    else:
        env = CarRiverCrossing(play_field_area=play_area, zoom=zoom)

    agent = CarRacingDQNAgent(epsilon=args.epsilon, lamb=args.lamb)
    if args.model:
        agent.load(args.model)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end

    run_fixed_policy = 0

    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):
        current_state, info = env.reset()
        current_state = process_state_image(current_state)
        state_frame_stack_queue = deque([current_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)

        time_frame_counter = 1
        time_frame_counter_without_reset = 1
        total_reward = 0
        done = False

        run_fixed_policy += 1

        if run_fixed_policy > 5:
            run_fixed_policy = 1

        truncated_count = 0

        maximum_frames_reached = 0

        while True:
            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            action = agent.get_fixed_policy(run_fixed_policy, time_frame_counter_without_reset)

            reward = 0
            for _ in range(SKIP_FRAMES + 1):
                next_state, r, terminated, truncated, info = env.step(action)
                reward += r
                if terminated or truncated:
                    break

            next_state = process_state_image(next_state)
            done = terminated

            total_reward += reward
            time_frame_counter += 1

            if time_frame_counter_without_reset > MAXIMUM_FRAMES:
                maximum_frames_reached += 1
                next_state, info = env.reset(reward=env.reward)
                next_state = process_state_image(next_state)
                truncated = True

            if truncated:
                time_frame_counter_without_reset = 0
                truncated_count += 1
            time_frame_counter_without_reset += 1

            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            current_state = next_state

            if done:
                log('{} - Episode: {}/{}, Total Frames: {}, Tiles Visited: {}, Total Rewards: {}, Epsilon: {:.2}, Policy: {}, Max Famres: {}'.format(datetime.now(), e, ENDING_EPISODE, time_frame_counter, env.tile_visited_count, total_reward, float(agent.epsilon), run_fixed_policy, maximum_frames_reached), args.lamb)
                break

            if len(agent.memory) > TRAINING_BATCH_SIZE and time_frame_counter % TRAINING_MODEL_FREQUENCY == 0:
                agent.replay_batch(TRAINING_BATCH_SIZE)

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            agent.save('./save/trial_fixed_{}_{}.h5'.format(args.lamb, e))

    env.close()
