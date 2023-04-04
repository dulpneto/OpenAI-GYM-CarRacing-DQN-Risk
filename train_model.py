import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import argparse
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from CarRacingEnv import CarRacingEnv

from common_functions import generate_state_frame_stack_from_queue

STARTING_EPISODE              = 1
ENDING_EPISODE                = 1_000_000
TRAINING_BATCH_SIZE           = 64
TRAINING_MODEL_FREQUENCY      = 4
SAVE_TRAINING_FREQUENCY       = 25
UPDATE_TARGET_MODEL_FREQUENCY = 1
EPISODES_TO_INIT = 100

def log(txt, lamb):
    with open('./save/result_train_{}.log'.format(lamb), 'a') as f:
        f.write(txt + '\n')
    print(txt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-s', '--start', type=int, help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int, help='The ending episode, default to 1000.')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='The starting epsilon of the agent, default to 1.0.')
    parser.add_argument('-l', '--lamb', type=float, default=0.0, help='The risk param, default to 0.0.')
    parser.add_argument('-r', '--render', type=bool, default=False, help='Render, default to False.')
    args = parser.parse_args()

    print('Training with risk factor', args.lamb)

    env = CarRacingEnv(render=args.render)

    agent = CarRacingDQNAgent(epsilon=args.epsilon, lamb=args.lamb)
    if args.model:
        agent.load(args.model)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end

    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):
        current_state, info = env.reset()
        state_frame_stack_queue = deque([current_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        negative_reward_counter = 0
        done = False

        run_fixed_policy = (e < EPISODES_TO_INIT or e % 7 == 0)
        if run_fixed_policy:
            log('FIXED POLICY {}'.format(e), args.lamb)
        
        while True:
            if args.render:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            # forcing an initial policy
            if run_fixed_policy:
                road_run = (e % 2 == 0)
                if road_run:
                    if 245 < env.env.step_count < 273:
                        action = 2
                    else:
                        action = 0
                else:
                    if 100 < env.env.step_count < 110:
                        action = 2
                    else:
                        action = 0

            next_state, reward, terminated, truncated, info = env.step(action)
            done = (terminated or truncated)

            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            current_state = next_state

            if done:
                log('Episode: {}/{}, Total Frames: {}, Tiles Visited: {}, Total Rewards: {}, Epsilon: {:.2}'.format(e,
                                                                                                                ENDING_EPISODE,
                                                                                                                env.frames,
                                                                                                                env.tiles_visited,
                                                                                                                env.total_reward,
                                                                                                                float(
                                                                                                                    agent.epsilon))
                    , args.lamb)
                break

            if len(agent.memory) > TRAINING_BATCH_SIZE and env.frames % TRAINING_MODEL_FREQUENCY == 0:
                agent.replay_batch(TRAINING_BATCH_SIZE)

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            agent.save('./save/trial_{}_{}.h5'.format(args.lamb, e))


    env.close()
