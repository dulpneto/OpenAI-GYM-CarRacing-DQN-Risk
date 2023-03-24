import gymnasium as gym
import cv2


def convert_image_to_state(state):
    # crop to remove low bar
    cropped = state[0:80, 8:88]
    # change to gray scale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # removing green part and normalize
    gray[gray <= 150] = 0
    gray[gray > 150] = 1
    return gray


class CarRacingEnv:
    def __init__(
        self,
        render          = False,
        frames_to_skip  = 50,  # number of frames environment will skip at the beginning
        frames_to_run   = 200,  # number of frames to reach the goal
        frames_to_die   = 50,  # number of frames it allows agent to get negative reward in a row
        skip_frames     = 3,
    ):
        self.human_render   = render
        self.frames_to_skip = frames_to_skip
        self.frames_to_run  = frames_to_run
        self.frames_to_die  = frames_to_die
        self.skip_frames    = skip_frames
        if self.human_render:
            # 5 actions: [do nothing, left, right, gas, brake]
            self.env = gym.make('CarRacing-v2', continuous=False, domain_randomize=False, render_mode="human")
        else:
            self.env = gym.make('CarRacing-v2', continuous=False, domain_randomize=False)

        self.frames = 0
        self.tiles_visited = 0
        self.negative_rewards = 0
        self.total_reward = 0
        self.init_env()

    def init_env(self):
        init_state, info = self.env.reset()
        # skipping first N frames to avoid getting initial images
        for i in range(self.frames_to_skip+1):
            next_state_img, reward, terminated, truncated, info = self.env.step(0)
        self.frames = 0
        self.tiles_visited = 0
        self.negative_rewards = 0
        self.total_reward = 0
        return next_state_img, info

    def step(self, action):
        self.frames += 1

        reward = 0
        for _ in range(self.skip_frames + 1):
            next_state_img, r, terminated, truncated, info = self.env.step(action)
            done = (terminated or truncated)
            reward += r
            if done:
                break

        if reward > 0:
            self.tiles_visited += int(reward / 3)  # we get a reward a little greater than 3 whe we visit a new tile
            self.negative_rewards = 0
        else:
            self.negative_rewards += 1

        if self.frames >= self.frames_to_run:
            terminated = True
            if self.tiles_visited > self.frames_to_run * 0.8:
                reward += 10
        elif self.negative_rewards >= self.frames_to_die:
            truncated = True
            reward -= 1

        self.total_reward += reward
        return convert_image_to_state(next_state_img), reward, terminated, truncated, info

    def reset(self):
        # run init to skip initial frames
        next_state_img, info = self.init_env()
        return convert_image_to_state(next_state_img), info

    def render(self):
        if self.human_render:
            self.env.render()

