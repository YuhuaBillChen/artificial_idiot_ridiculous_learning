"""
Q Actor-critic agent

March 26th 2020
"""
import tensorflow as tf
import numpy as np
import gym
from collections import deque
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
import tensorflow.keras.backend as kb
from tensorflow.keras.utils import to_categorical
import os
import glob

SAVE_PER_EPISODE = 250
RENDER_PER_EPISODE = 100
NB_EPISDOE = 1000
LEN_EPISODE = 200
TRAINING_SUCCESS_CNT = 5
FN_PREFIX = "A2C-CP"
GYM_ENVIRON = 'CartPole-v0'


def set_up_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    kb.set_session(sess)


class A2CAgent(object):
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

        self.actor_model = self.nn_actor_model()
        self.actor_train_fn = self.make_actor_train_fn()
        self.critic_model = self.nn_critic_model()
        self.critic_train_fn = self.make_critic_train_fn()

    def nn_actor_model(self):
        # Actor
        state_shape = self.env.observation_space.shape
        input_layer = Input(shape=state_shape)
        hidden_layer1 = Dense(24, activation="relu")(input_layer)
        hidden_layer2 = Dense(48, activation="relu")(hidden_layer1)
        results = Dense(self.env.action_space.n, activation='softmax')(hidden_layer2)
        model = Model(inputs=input_layer, outputs=results)
        return model    # Probs of action

    def nn_critic_model(self):
        # Critic
        state_shape = self.env.observation_space.shape[0]
        actor_shape = self.env.action_space.n
        input_layer = Input(shape=(state_shape+actor_shape))
        hidden_layer1 = Dense(24, activation="relu")(input_layer)
        hidden_layer2 = Dense(48, activation="relu")(hidden_layer1)
        results = Dense(1, activation='linear')(hidden_layer2)
        model = Model(inputs=input_layer, outputs=results)
        return model  # value estimation

    def make_actor_train_fn(self):
        action_oh_pl = kb.placeholder(shape=(None, self.env.action_space.n))
        adv_pl = kb.placeholder(shape=(None,))

        action_prob = kb.sum(action_oh_pl * self.actor_model.output, axis=-1)
        log_action_prob = kb.log(action_prob)

        loss = kb.mean(- log_action_prob * adv_pl)

        adam = Adam(lr=self.learning_rate)

        update_op = adam.get_updates(
            loss=loss,
            params=self.actor_model.trainable_weights
        )

        train_fn = kb.function(
            inputs=[
                self.actor_model.input,
                action_oh_pl,
                adv_pl
            ],
            outputs=[self.actor_model.output, loss],
            updates=update_op
        )
        return train_fn

    def make_critic_train_fn(self):
        action_oh_pl = kb.placeholder(shape=(None, self.env.action_space.n))
        newaction_oh_pl = kb.placeholder(shape=(None,self.env.action_space.n))
        rewards_pl = kb.placeholder(shape=(None,))
        dones_pl = kb.placeholder(shape=(None,))

        critic_results = self.critic_model.output
        q_hat, new_q_hat = critic_results[0], critic_results[1]

        # TD loss
        val = rewards_pl + (1.0 - dones_pl) * self.gamma * new_q_hat - q_hat

        # Mean squared error of the prediction
        loss = kb.mean(val**2)

        adam = Adam(lr=self.learning_rate)

        update_op = adam.get_updates(
            loss=loss,
            params=self.critic_model.trainable_weights
        )

        train_fn = kb.function(
            inputs=[
                self.critic_model.input,
                action_oh_pl,
                newaction_oh_pl,
                rewards_pl,
                dones_pl
            ],
            outputs=[self.critic_model.output, val, loss],
            updates=update_op
        )
        return train_fn

    def remember(self, state, action, reward, new_state, done):
        self.time_step += 1
        self.memory.append([state, action, reward, new_state, done])

    def preceive(self, state, action, reward, new_state, done):
        self.remember(state, action, reward, new_state, done)

    def get_memory(self):
        np_memory = np.array(self.memory[-1])

        states, actions, rewards, new_states, dones = np.hsplit(np_memory, 5)

        states = np.array([_[0] for _ in states])
        new_states = np.array([_[0] for _ in new_states])

        return states, actions, rewards, new_states, dones

    def reset_memory(self):
        self.memory.clear()

    def update_actor_critic(self):
        states, actions, rewards, new_states, dones = self.get_memory()

        actions_oh = to_categorical(actions, self.env.action_space.n)

        new_actions_oh = to_categorical(
            self.take_action(new_states), self.env.action_space.n
        )[np.newaxis, :]

        crt_inputs = np.concatenate([
            np.concatenate([states, actions_oh], axis=-1),          # cur_state
            np.concatenate([new_states, new_actions_oh], axis=-1),  # next_state
        ], axis=0
        )
        c_output, adv_val, crt_loss = self.critic_train_fn(
            [crt_inputs, actions_oh, new_actions_oh, rewards, dones])

        _, act_loss = self.actor_train_fn([states, actions_oh, adv_val])

        return crt_loss, act_loss

    def policy_action(self, state):
        probs = np.squeeze(self.actor_model.predict(state))
        highest_prob_action = np.random.choice(self.env.action_space.n, p=probs)
        log_prob = np.log(probs[highest_prob_action])
        return highest_prob_action, log_prob

    def take_action(self, state):
        action, _ = self.policy_action(state)
        return action

    def save_model(self, fn):
        self.actor_model.save(fn.replace('.model', '-actor.model'))
        self.critic_model.save(fn.replace('.model', '-critic.model'))

    def load_model(self, fn):
        from tensorflow.keras.models import load_model
        self.actor_model = load_model(fn.replace('.model', '-actor.model'))
        self.critic_model = load_model(fn.replace('.model', '-critic.model'))


def train():
    env = gym.make(GYM_ENVIRON)

    nb_episode = NB_EPISDOE
    len_episode = LEN_EPISODE

    agent = A2CAgent(env=env)
    losses = [0.0, 0.0, 0.0, 0.0]
    success_count = 0

    for i_episode in range(nb_episode):
        max_distance = -1.5
        cur_state = env.reset().reshape(1, -1)

        for step in range(len_episode):
            action, log_prob = agent.policy_action(cur_state)
            if i_episode % RENDER_PER_EPISODE == 0:
                env.render()
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, -1)
            # reward = 1.0 if new_state[0][0] >= 0.5 else -1.0

            agent.preceive(cur_state, action, reward, new_state, done)

            crt_loss, act_loss = agent.update_actor_critic()
            losses[0] = act_loss
            losses[2] = 0.9 * losses[2] + 0.1 * act_loss
            losses[1] = crt_loss
            losses[3] = 0.9 * losses[3] + 0.1 * crt_loss

            max_distance = max(new_state[0][0], max_distance)

            if done:
                agent.reset_memory()
                break

            cur_state = new_state

        if step >= len_episode - 1:
            success_count += 1
            print("Completed in %d episode, in %d steps, reward: %d." % (i_episode, step, reward))
        else:
            success_count = 0
            print("Failed in %d episode, steps: %d, loss: %.4f, %.4f avg_loss: %.4f, %.4f" % (
                i_episode, step, losses[0], losses[1],  losses[2], losses[3]))

        if success_count > TRAINING_SUCCESS_CNT:
            break

        if i_episode % SAVE_PER_EPISODE == 0:
            agent.save_model("{}-episode-{}.model".format(FN_PREFIX, i_episode))

    agent.save_model("{}-episode-{}.model".format(FN_PREFIX, i_episode))
    env.close()


def test():
    env = gym.make(GYM_ENVIRON)
    nb_episode = 10
    len_episode = 200

    agent = A2CAgent(env=env)
    chkpts = sorted(glob.glob("%s-episode-*.model"%FN_PREFIX), key=os.path.basename, reverse=True)
    agent.load_model(chkpts[0].replace("-actor.model", ".model").replace("-critic.model", ".model"))
    print("loaded checkpoint:%s" % chkpts[0])

    for i_episode in range(nb_episode):
        cur_state = env.reset().reshape(1, -1)
        for step in range(len_episode):
            action = agent.take_action(cur_state)
            env.render()
            new_state, reward, done, _ = env.step(action)
            cur_state = new_state.reshape(1, -1)
            if done:
                break
        if step >= len_episode - 1:
            print("Completed in %d episode, in %d steps, reward: %d." % (i_episode, step, reward))
        else:
            print("Failed in %d episode, steps: %d" % (i_episode, step))

    env.close()


if __name__ == "__main__":
    set_up_session()
    train()
    test()
