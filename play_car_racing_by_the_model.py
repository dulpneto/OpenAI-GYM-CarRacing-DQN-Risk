import argparse
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from CarRacingEnv import CarRacingEnv
from common_functions import generate_state_frame_stack_from_queue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-m', '--model', required=True, help='The `.h5` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes should the model plays.')
    parser.add_argument('-l', '--lamb', type=float, default=0.0, help='The risk param, default to 0.0.')
    args = parser.parse_args()
    train_model = args.model
    play_episodes = args.episodes

    env = CarRacingEnv(render=True)
    # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent = CarRacingDQNAgent(epsilon=0.0, lamb=args.lamb)
    agent.load(train_model)

    for e in range(play_episodes):
        init_state, info = env.reset()

        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        
        while True:
            env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            print(action)
            next_state, reward, terminated, truncated, info = env.step(action)

            done = (terminated or truncated)

            init_state = next_state

            total_reward += reward

            state_frame_stack_queue.append(next_state)

            # print('{} REWARD {} TILES {}'.format(environment.frames, reward, environment.tiles_visited))

            if done:
                print('Episode: {}/{}, Total Frames: {}, Tiles Visited: {}, Total Rewards: {}'.format(e+1, play_episodes,
                                                                                                                      env.frames,
                                                                                                                      env.tiles_visited,
                                                                                                                      env.total_reward))
                break
