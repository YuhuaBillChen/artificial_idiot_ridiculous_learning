import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import random
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
import os
import glob
from keras.backend.tensorflow_backend import set_session

SAVE_PER_EPISODE = 250
RENDER_PER_EPISODE = 50
NB_EPISDOE = 500
LEN_EPISODE = 200
TRAINING_SUCCESS_CNT = 5
FN_PREFIX = "DDQN"
GYM_ENVIRON = 'MountainCar-v0'


def set_up_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    set_session(sess)


class DDQNAgent(object):
    """
        DQN Agent
    """
    MEMORY_LENGTH = 20000
    BATCH_SIZE = 32
    NB_TARGET_MODEL_UPDATE_STEP = 200

    def __init__(
            self,
            env,
            memory=deque(maxlen=MEMORY_LENGTH),
            gamma=0.99,
            learning_rate=1e-3,
            epsilon_init=1.0,
            epsilon_min=0.001,
            epsilon_decay=0.9995
    ):
        self.env = env
        self.time_step = 0
        self.memory = memory
        self.gamma = gamma
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.train_model = self.nn_Q_model()
        self.target_model = self.nn_Q_model()

    def nn_Q_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape

        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model  # Q_value

    def remember(self, state, action, reward, new_state, done):
        self.time_step += 1
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        """ Experience Replay"""
        if len(self.memory) < self.BATCH_SIZE:
            return

        samples = np.array(random.sample(self.memory, self.BATCH_SIZE))

        states, actions, rewards, new_states, dones = np.hsplit(samples, 5)

        states = np.concatenate(np.squeeze(states), 0)
        actions = np.squeeze(actions)
        rewards = np.squeeze(rewards)
        new_states = np.concatenate(np.squeeze(new_states), 0)
        dones = np.squeeze(dones)

        targets = self.train_model.predict(states)
        q_futures = self.target_model.predict(new_states)       

        action_futures = self.train_model.predict(new_states)

        # DQN target
        # Y_k = r + gamma max Q(s', a'; theta_k^-) (4.3)
        # targets[(range(self.BATCH_SIZE), actions[:].astype(int))] = rewards + \
        #     (self.gamma*q_futures.max(axis=1))*np.logical_not(dones)

        # DDQN target
        # Y_k = r + gamma Q(s', argmax Q(s', a; thetha_k); theta_k^-) (4.3)
        targets[range(self.BATCH_SIZE), actions[:].astype(int)] = rewards + \
            (self.gamma*q_futures[range(self.BATCH_SIZE), np.argmax(action_futures, -1)])*np.logical_not(dones)

        self.train_model.fit(states, targets, epochs=1, verbose=0)

    def preceive(self, state, action, reward, new_state, done):
        self.remember(state, action, reward, new_state, done)
        self.replay()
        if self.time_step % self.NB_TARGET_MODEL_UPDATE_STEP == 0:
            self.update_target()        

    def update_target(self):
        self.target_model.set_weights(self.train_model.get_weights())

    def take_action(self, state):
        return np.argmax(self.train_model.predict(state)[0])

    def egreedy_action(self, state):
        self.epsilon *= self.epsilon_decay

        self.epsilon = max(self.epsilon_min, self.epsilon)

        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        return self.take_action(state)

    def save_model(self, fn):
        self.train_model.save(fn)

    def load_model(self, fn):
        import os
        if os.path.exists(fn):
            from tensorflow.keras.models import load_model
            self.train_model = load_model(fn)
            self.target_model = load_model(fn)


def train():
    env = gym.make(GYM_ENVIRON)

    nb_episode = NB_EPISDOE
    len_episode = LEN_EPISODE
    success_count = 0

    dqn_agent = DDQNAgent(env=env)

    chkpts = sorted(glob.glob("episode-*.model"), key=os.path.basename, reverse=True)
    if len(chkpts) > 0:
        dqn_agent.load_model(chkpts[0])

    steps = []

    for i_episode in range(nb_episode):
        max_distance = -1.5 # minimum -1.2
        cur_state = env.reset().reshape(1, -1)
        for step in range(len_episode):
            action = dqn_agent.egreedy_action(cur_state)
            if i_episode % RENDER_PER_EPISODE == 0:
                env.render()
            new_state, reward, done, _ = env.step(action)

            new_state = new_state.reshape(1, -1)

            reward = 1.0 if new_state[0][0] >= 0.5 else -1.0

            dqn_agent.preceive(cur_state, action, reward, new_state, done)
            
            max_distance = max(new_state[0][0], max_distance)

            cur_state = new_state
            if done:
                break

        if step < len_episode - 1:
            success_count += 1
            print("Completed in %d episode, in %d steps, reward: %d." % (i_episode, step, reward))
        else:
            success_count = 0
            print("Failed in %d episode, max_distance: %.4f, epsilon: %.4f."%(i_episode, max_distance, dqn_agent.epsilon))

        if success_count > TRAINING_SUCCESS_CNT:
            break
            
        if i_episode % SAVE_PER_EPISODE == 0:
            dqn_agent.save_model("{}-episode-{}.model".format(FN_PREFIX, i_episode))

    dqn_agent.save_model("{}-episode-{}.model".format(FN_PREFIX, i_episode))            
    env.close()


def test():
    env = gym.make(GYM_ENVIRON)
    nb_episode = 10
    len_episode = 200

    dqn_agent = DDQNAgent(env=env)
    chkpts = sorted(glob.glob("%s-episode-*.model"%FN_PREFIX), key=os.path.basename, reverse=True)
    dqn_agent.load_model(chkpts[0])
    print("loaded checkpoint:%s" % chkpts[0])

    for i_episode in range(nb_episode):
        cur_state = env.reset().reshape(1, -1)
        for step in range(len_episode):
            action = dqn_agent.take_action(cur_state)
            env.render()
            new_state, reward, done, _ = env.step(action)
            cur_state = new_state.reshape(1, -1)
            if done:
                break
        if step < len_episode - 1:
            print("Completed in {} steps in {} episode, reward: {}.".format(step, i_episode, reward))
        else:
            print("Failed in {} episode, reward：{}.".format(i_episode, reward))
            pass

    env.close()


if __name__ == "__main__":
    set_up_session()
    train()
    test()
