import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import math

class CarRacingDQNAgent:
    def __init__(
        self,
        action_space    = 5,
        frame_stack_num = 3,
        memory_size     = 5000,
        gamma           = 0.95,  # discount rate
        epsilon         = 1.0,   # exploration rate
        epsilon_min     = 0.1,
        epsilon_decay   = 0.9999,
        learning_rate   = 0.001,
        lamb            = 0.0,
        q_learning_alpha= 0.1
    ):
        self.action_space    = action_space
        self.frame_stack_num = frame_stack_num
        self.memory          = deque(maxlen=memory_size)
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.learning_rate   = learning_rate
        self.model           = self.build_model()
        self.target_model    = self.build_model()
        self.lamb            = lamb
        self.q_learning_alpha= q_learning_alpha
        self.log_sum_exp = True
        self.update_target_model()

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, self.frame_stack_num)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(self.action_space, activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7))
        return model

    def update_target_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() > self.epsilon:
            act_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            action = np.argmax(act_values[0])
        else:
            action = random.randrange(self.action_space)
        return action

    def replay_batch(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states, verbose=0)
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        train_state = []
        train_target = []
        for index, (current_state, action, reward, next_state, done) in enumerate(minibatch):
            future_qs = future_qs_list[index]
            current_qs = current_qs_list[index]

            if done:
                current_qs[action] = reward
            else:
                t = future_qs
                if self.lamb == 0:
                    # print('NEUTRAL')
                    current_qs[action] = reward + self.gamma * np.amax(t)
                    # target = reward + self.gamma * np.amax(t)
                    # td = target - current_qs[action]
                    # current_qs[action] + (self.q_learning_alpha * td)
                elif not self.log_sum_exp:
                    # print('RISK', self.lamb)
                    # q_rev = self.reverse_utility(np.amax(t))
                    # current_qs[action] = self.utility(reward + self.gamma * q_rev)
                    target = reward + self.gamma * np.amax(t)
                    u = self.utility(target)
                    u_q = self.utility(current_qs[action])
                    current_qs[action] = self.reverse_utility(u_q + (self.q_learning_alpha * (u - u_q)))
                else:
                    target = reward + self.gamma * np.amax(t)

                    exp_1 = self.lamb * current_qs[action] + np.log(1 - self.q_learning_alpha)
                    exp_2 = self.lamb * target + np.log(self.q_learning_alpha)

                    a = max([exp_1, exp_2])
                    b = min([exp_1, exp_2])

                    current_qs[action] = (a + math.log(1 + math.exp(b - a))) / self.lamb

            train_state.append(current_state)
            train_target.append(current_qs)

        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)

    def utility(self, x):
        return np.sign(self.lamb) * math.exp(self.lamb * x)

    def reverse_utility(self, x):
        return math.log(np.sign(self.lamb) * x) / self.lamb

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)

    @staticmethod
    def get_fixed_policy(policy_id, time_frame_counter_without_reset):
        if policy_id == 1:  # risk
            if 17 < time_frame_counter_without_reset < 23:
                action = 1
            else:
                action = 3
        elif policy_id == 2:  # risk
            if 18 < time_frame_counter_without_reset < 24:
                action = 1
            else:
                action = 3
        elif policy_id == 3:  # middle 2
            if 25 <= time_frame_counter_without_reset < 30:
                action = 1
            elif 39 < time_frame_counter_without_reset < 44:
                action = 1
            else:
                action = 3
        elif policy_id == 4:  # middle
            if 25 <= time_frame_counter_without_reset < 31:
                action = 1
            elif 40 < time_frame_counter_without_reset < 43:
                action = 2
            else:
                action = 3
        elif policy_id == 5:  # safe
            if 25 <= time_frame_counter_without_reset < 29:
                action = 1
            elif 37 < time_frame_counter_without_reset < 43:
                action = 1
            else:
                action = 3
        else:
            raise NotImplementedError('No policy found for {}.'.format(policy_id))

        return action
