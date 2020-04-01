"""
DDPG Continuous agent

March 26th 2020
"""
import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, Input
import tensorflow.keras.backend as kb
import os
import glob

SAVE_PER_EPISODE = 100
NB_EPISDOE = 500
LEN_EPISODE = 999
TRAINING_SUCCESS_CNT = 5
TEST_PER_EPISODE = 10
FN_PREFIX = "DDPG-MCC"
GYM_ENVIRON = 'MountainCarContinuous-v0'
WIN_REWARD = 90


class OUNoise:
    """
        OU Noise
        Credit from https://github.com/floodsung/DDPG/blob/master/ou_noise.py
    """

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class DDPGAgent(object):
    """
        DQN Agent
    """
    MEMORY_LENGTH = 20000
    BATCH_SIZE = 32
    TARGET_UPDATE_TAU = 0.01
    ACTION_BOUNDARY = 2.0

    def __init__(
            self,
            env,
            memory=deque(maxlen=MEMORY_LENGTH),
            gamma=0.99,
            learning_rate=1e-3,
            epsilon_init=1.0,
            epsilon_min=0.1,
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
        self.exploration_noise = OUNoise(self.env.action_space.shape[0])

        self.actor_model = self.nn_actor_model()
        self.actor_target = self.nn_actor_model()

        self.critic_model = self.nn_critic_model(self.actor_model)
        self.critic_target = self.nn_critic_model(self.actor_target)

        self.actor_target.set_weights(self.actor_model.get_weights())
        self.critic_target.set_weights(self.critic_model.get_weights())

        # Merged two models for passing gradient from critic to actor
        self.merged_model = Model(
            inputs=[self.actor_model.input],
            outputs=[self.critic_model.output]
        )

        self.merged_target = Model(
            inputs=[self.actor_target.input],
            outputs=[self.critic_target.output]
        )

        self.actor_train_fn = self.make_actor_train_fn()
        self.critic_train_fn = self.make_critic_train_fn()

    def update_target(self):
        actor_weights = self.TARGET_UPDATE_TAU * np.array(self.actor_model.get_weights()) + \
                        (1 - self.TARGET_UPDATE_TAU) * np.array(self.actor_target.get_weights())
        self.actor_target.set_weights(actor_weights)
        critic_weights = self.TARGET_UPDATE_TAU * np.array(self.critic_model.get_weights()) + \
                         (1 - self.TARGET_UPDATE_TAU) * np.array(self.critic_target.get_weights())

        self.critic_target.set_weights(critic_weights)

    def nn_actor_model(self):
        # Actor
        state_shape = self.env.observation_space.shape
        input_layer = Input(shape=state_shape)
        hidden_layer1 = Dense(24, activation="relu")(input_layer)
        hidden_layer2 = Dense(48, activation="relu")(hidden_layer1)
        results = Dense(
            self.env.action_space.shape[0],
            activation='tanh',
            kernel_initializer=keras.initializers.RandomUniform(minval=-0.03, maxval=0.03)
        )(hidden_layer2)
        results = self.ACTION_BOUNDARY*results
        model = Model(inputs=input_layer, outputs=results)
        return model  # Probs of action

    def nn_critic_model(self, actor_model):
        # Critic
        state_layer = actor_model.input
        actor_layer = actor_model.output
        input_layer = Concatenate(axis=-1)([state_layer, actor_layer])
        hidden_layer1 = Dense(24, activation="relu")(input_layer)
        hidden_layer2 = Dense(48, activation="relu")(hidden_layer1)
        results = Dense(1, activation='linear')(hidden_layer2)
        model = Model(inputs=[state_layer], outputs=results)
        return model  # value estimation

    def make_actor_train_fn(self):
        upd_grad = tf.gradients(
            -self.merged_model.output,
            self.actor_model.trainable_weights
        )

        adam = Adam(lr=self.learning_rate)

        update_op = [adam.apply_gradients(zip(upd_grad, self.actor_model.trainable_weights))]

        train_fn = kb.function(
            inputs=[
                self.actor_model.input
            ],
            outputs=[self.actor_model.output, upd_grad],
            updates=update_op
        )
        return train_fn

    def make_critic_train_fn(self):
        rewards_pl = kb.placeholder(shape=(None,))
        dones_pl = kb.placeholder(shape=(None,))

        q_hats = self.merged_model.output
        new_q_hats = self.merged_target.output

        # Mean squared error of the prediction
        loss = kb.mean(rewards_pl + (1.0 - dones_pl) * self.gamma * new_q_hats - q_hats)
        loss = kb.mean(loss ** 2)

        adam = Adam(lr=self.learning_rate)

        update_op = adam.get_updates(
            loss=loss,
            params=self.critic_model.trainable_weights
        )

        train_fn = kb.function(
            inputs=[
                self.merged_model.input,
                self.merged_target.input,
                rewards_pl,
                dones_pl
            ],
            outputs=[self.merged_model.output, loss],
            updates=update_op
        )
        return train_fn

    def remember(self, state, action, reward, new_state, done):
        self.time_step += 1
        self.memory.append([state, action, reward, new_state, done])

    def preceive(self, state, action, reward, new_state, done):
        self.remember(state, action, reward, new_state, done)
        return self.actor_critic_replay()

    def get_memory(self):
        np_memory = np.array(random.sample(self.memory, self.BATCH_SIZE))

        states, actions, rewards, new_states, dones = np.hsplit(np_memory, 5)

        states = np.array([_[0][0] for _ in states])
        new_states = np.array([_[0][0] for _ in new_states])

        return states, actions, rewards, new_states, dones

    def actor_critic_replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return None

        states, actions, rewards, new_states, dones = self.get_memory()

        c_output, crt_loss = self.critic_train_fn(
            [states, new_states, rewards, dones])

        _, upd_grad = self.actor_train_fn([states])

        return crt_loss

    def policy_action(self, state):
        action_val = self.take_action(state)
        action_val += self.exploration_noise.noise()
        return action_val

    def take_action(self, state):
        action_val = self.actor_model.predict(state)
        return action_val

    def save_model(self, fn):
        self.actor_model.save(fn.replace('.model', '-actor.model'))
        self.critic_model.save(fn.replace('.model', '-critic.model'))

    def load_model(self, fn):
        from tensorflow.keras.models import load_model
        self.actor_model = load_model(fn.replace('.model', '-actor.model'))
        self.critic_model = load_model(fn.replace('.model', '-critic.model'))


def evaluation(env, agent):
    # Test during training
    for _ in range(TRAINING_SUCCESS_CNT):
        print("Testing.. %d/%d" % (_, TRAINING_SUCCESS_CNT))
        cur_state = env.reset().reshape(1, -1)
        max_distance = -1.5
        for step in range(LEN_EPISODE):
            action = agent.take_action(cur_state)
            new_state, reward, done, _ = env.step(action)
            env.render()
            new_state = new_state.reshape(1, -1)
            cur_state = new_state
            max_distance = max(new_state[0][0], max_distance)
            if done:
                break
        if reward > WIN_REWARD:
            print("Testing completed, in %d steps, reward: %f." % (step, reward))
        else:
            print("Testing failed, reward: %.2f, maxdist: %.2f" % (reward, max_distance))
            return False
    return True


def train():
    env = gym.make(GYM_ENVIRON)

    nb_episode = NB_EPISDOE
    len_episode = LEN_EPISODE

    agent = DDPGAgent(env=env)
    losses = [0.0, 0.0]

    for i_episode in range(nb_episode):
        max_distance = -1.5
        cur_state = env.reset().reshape(1, -1)
        agent.exploration_noise.reset()

        for step in range(len_episode):
            action = agent.policy_action(cur_state)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, -1)
            # reward = 1.0 if new_state[0][0] >= 0.5 else -1.0

            crt_loss = agent.preceive(cur_state, action, reward, new_state, done)
            if crt_loss is not None:
                losses[0] = crt_loss
                losses[1] = 0.99 * losses[1] + 0.01 * crt_loss

            agent.update_target()

            max_distance = max(new_state[0][0], max_distance)

            if done:
                break

            cur_state = new_state

        if reward > WIN_REWARD:
            print("Completed in %d episode, in %d steps, reward: %f." % (i_episode, step, reward))
        else:
            print("Failed in %d eps, rewards: %.2f, maxdist: %.2f, avg_loss: %.4f" % (
                i_episode, reward, max_distance, losses[1]))

        if i_episode % SAVE_PER_EPISODE == 0:
            agent.save_model("{}-episode-{}.model".format(FN_PREFIX, i_episode))

        if i_episode % TEST_PER_EPISODE == 0:
            if evaluation(agent=agent, env=env):
                break

    agent.save_model("{}-episode-{}.model".format(FN_PREFIX, i_episode))
    env.close()


def test():
    env = gym.make(GYM_ENVIRON)
    nb_episode = 10
    len_episode = LEN_EPISODE

    agent = DDPGAgent(env=env)
    chkpts = sorted(glob.glob("%s-episode-*.model" % FN_PREFIX), key=os.path.basename, reverse=True)
    mdl_file = chkpts[0].replace("-actor.model", ".model").replace("-critic.model", ".model")
    print("loading checkpoint:%s" % mdl_file)

    agent.load_model(mdl_file)

    for i_episode in range(nb_episode):
        cur_state = env.reset().reshape(1, -1)
        for step in range(len_episode):
            action = agent.take_action(cur_state)
            env.render()
            new_state, reward, done, _ = env.step(action)
            cur_state = new_state.reshape(1, -1)
            if done:
                break
        if reward > WIN_REWARD:
            print("Completed in %d episode, in %d steps, reward: %f." % (i_episode, step, reward))
        else:
            print("Failed in %d eps, rewards: %.2f" % (i_episode, reward))

    env.close()


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    train()
    test()
