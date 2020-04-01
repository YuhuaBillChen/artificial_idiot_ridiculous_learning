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

SAVE_PER_EPISODE = 200
RENDER_PER_EPISODE = 100
NB_EPISDOE = 800
LEN_EPISODE = 200
FN_PREFIX = "ActorCritic"
GYM_ENVIRON = 'CartPole-v0'


class ActorCriticAgent(object):
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
        discounted_qw_pl = kb.placeholder(shape=(None,))

        action_prob = kb.sum(action_oh_pl * self.actor_model.output, axis=-1)
        log_action_prob = kb.log(action_prob)

        loss = kb.mean(- log_action_prob * discounted_qw_pl)

        adam = Adam(lr=self.learning_rate)

        update_op = adam.get_updates(
            loss=loss,
            params=self.actor_model.trainable_weights
        )

        train_fn = kb.function(
            inputs=[
                self.actor_model.input,
                action_oh_pl,
                discounted_qw_pl
            ],
            outputs=[self.actor_model.output, loss],
            updates=update_op
        )
        return train_fn

    def make_critic_train_fn(self):
        action_oh_pl = kb.placeholder(shape=(None, self.env.action_space.n))
        discounted_rw_pl = kb.placeholder(shape=(None,))

        critic_results = self.critic_model.output

        # Mean squared error of the prediction
        loss = kb.mean(mean_absolute_error(discounted_rw_pl, critic_results))

        adam = Adam(lr=self.learning_rate)

        update_op = adam.get_updates(
            loss=loss,
            params=self.critic_model.trainable_weights
        )

        train_fn = kb.function(
            inputs=[
                self.critic_model.input,
                action_oh_pl,
                discounted_rw_pl
            ],
            outputs=[self.critic_model.output, loss],
            updates=update_op
        )
        return train_fn

    def remember(self, state, action, reward, new_state, done):
        self.time_step += 1
        self.memory.append([state, action, reward, new_state, done])

    def preceive(self, state, action, reward, new_state, done):
        self.remember(state, action, reward, new_state, done)

    def compute_discounted_q(self, states, actions_oh):
        """
            Compute the discounted q-value

        """
        discounted_rewards = []
        critics_value = np.squeeze(
            self.critic_model.predict(
                np.squeeze(np.concatenate([states, actions_oh], axis=-1))
            )
        )
        for t in range(len(states)):
            Gt = 0
            pw = 0
            for c in critics_value[t:]:
                Gt = Gt + self.gamma ** pw * c
                pw = pw + 1
            discounted_rewards.append(Gt)

        return np.squeeze(np.array(discounted_rewards))

    def compute_discounted_rewards(self, rewards):
        """
            Compute the discounted rewards

        """
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        # Normalization of the discounted rewards
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-10)

        return discounted_rewards

    def get_memory(self):
        np_memory = np.array(self.memory)
        states, actions, rewards, new_states, dones = np.hsplit(np_memory, 5)

        states = np.squeeze(np.array([_[0][0] for _ in states]))
        actions = np.squeeze(actions)
        rewards = np.squeeze(rewards)
        new_states = np.squeeze(new_states)
        dones = np.squeeze(dones)

        return states, actions, rewards, new_states, dones

    def reset_memory(self):
        self.memory.clear()

    def update_actor(self):
        states, actions, _, _, _ = self.get_memory()
        if len(self.memory) > 1:
            actions_oh = to_categorical(actions, self.env.action_space.n)
            qw = self.compute_discounted_q(states, actions_oh)
            _, loss = self.actor_train_fn([states, actions_oh, qw])
            return loss

    def update_critic(self):
        states, actions, rewards, _, _ = self.get_memory()
        loss = 0.0

        if len(self.memory) > 1:
            actions_oh = to_categorical(actions, self.env.action_space.n)
            rw = self.compute_discounted_rewards(rewards)
            c_output, loss = self.critic_train_fn(
                [np.concatenate([states, actions_oh], axis=-1), actions_oh, rw])

        return loss

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

    agent = ActorCriticAgent(env=env)
    losses = [0.0, 0.0, 0.0, 0.0]

    for i_episode in range(nb_episode):
        max_distance = -1.5
        cur_state = env.reset().reshape(1, -1)
        loss = np.array([0.0, 0.0])

        for step in range(len_episode):
            action, log_prob = agent.policy_action(cur_state)
            if i_episode % RENDER_PER_EPISODE == 0:
                env.render()
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, -1)
            # reward = 1.0 if new_state[0][0] >= 0.5 else -1.0

            agent.preceive(cur_state, action, reward, new_state, done)

            if done:
                crt_loss = agent.update_critic()
                losses[1] = crt_loss
                losses[3] = 0.999*losses[3] + 0.001*crt_loss
                act_loss = agent.update_actor()
                losses[0] = act_loss
                losses[2] = 0.999 * losses[2] + 0.001 * act_loss
                agent.reset_memory()
                break

            max_distance = max(new_state[0][0], max_distance)
            cur_state = new_state

        if step >= len_episode - 1:
            print("Completed in %d episode, in %d steps, reward: %d." % (i_episode, step, reward))
        else:
            print("Failed in %d episode, steps: %d, loss: %.4f, %.4f avg_loss: %.4f, %.4f" % (
                i_episode, step, losses[0], losses[1],  losses[2], losses[3]))

        if i_episode % SAVE_PER_EPISODE == 0:
            agent.save_model("{}-episode-{}.model".format(FN_PREFIX, i_episode))

    agent.save_model("{}-episode-{}.model".format(FN_PREFIX, i_episode))
    env.close()


def test():
    env = gym.make(GYM_ENVIRON)
    nb_episode = 10
    len_episode = 200

    agent = ActorCriticAgent(env=env)
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
    tf.compat.v1.disable_eager_execution()
    train()
    test()
