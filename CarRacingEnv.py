import cv2
from bug_racing_env import CarRacing

def convert_image_to_state(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state


class CarRacingEnv:
    def __init__(
        self,
        render          = False,
        frames_to_skip  = 0,  # number of frames environment will skip at the beginning
        skip_frames     = 2
    ):
        self.human_render   = render
        self.frames_to_skip = frames_to_skip
        self.skip_frames    = skip_frames
        if self.human_render:
            # 5 actions: [do nothing, left, right, gas, brake]
            self.env = CarRacing(continuous=False, render_mode="human")
        else:
            self.env = CarRacing(continuous=False)

        self.total_reward = 0
        self.tiles_visited = 0
        self.frames = 0

    def init_env(self):
        init_state, info = self.env.reset()
        if self.frames_to_skip <= 0:
            return init_state, info
        # skipping first N frames to avoid getting initial images
        for i in range(self.frames_to_skip+1):
            next_state_img, reward, terminated, truncated, info = self.env.step(0)
        self.frames = 0
        return next_state_img, info

    def step(self, action):
        self.frames += 1

        reward = 0
        for _ in range(self.skip_frames + 1):
            next_state_img, r, terminated, truncated, info = self.env.step(action)
            done = (terminated or truncated)
            reward += r
            if done:
                reward = r
                break

        self.total_reward += reward
        self.tiles_visited = self.env.tile_visited_count
        return convert_image_to_state(next_state_img), reward, terminated, truncated, info

    def reset(self):
        # run init to skip initial frames
        self.total_reward = 0
        self.tiles_visited = 0
        next_state_img, info = self.init_env()
        return convert_image_to_state(next_state_img), info

    def render(self):
        if self.human_render:
            self.env.render()

    def close(self):
        self.env.close()

