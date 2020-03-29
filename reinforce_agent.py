"""
Policy gradient, REINFORCE algorithm

"""
import tensorflow as tf
import numpy as np
import gym
from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as kb
import os
import glob

SAVE_PER_EPISODE = 500
RENDER_PER_EPISODE = 100
NB_EPISDOE = 1000
LEN_EPISODE = 200
FN_PREFIX = "Reinforce"
GYM_ENVIRON = "CartPole-v0"


def set_up_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    kb.set_session(sess)


class ReinforceAgent(object):
    """
    REINFORCE Algorithm

    """

    def __init__(
            self,
            env,
            memory=deque(),
            gamma=0.99,
            learning_rate=1e-3,
            epsilon_init=None,
            epsilon_min=None,
            epsilon_decay=None
    ):
        self.env = env
        self.gamma = gamma
        self.time_step = 0
        self.memory = memory
        self.learning_rate = learning_rate

        self.train_model = self.policy_nn()     # Policy network

        self.train_fn = self.make_train_fn()

    def compute_discounted_rewards(self, rewards):
        """
        compute the discounted rewards

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

    def make_train_fn(self):
        action_oh_pl = kb.placeholder(shape=(None, self.env.action_space.n))
        discounted_rw_pl = kb.placeholder(shape=(None,))

        action_prob = kb.sum(action_oh_pl * self.train_model.output, axis=-1)
        log_action_prob = kb.log(action_prob)

        loss = kb.mean(- log_action_prob * discounted_rw_pl)

        adam = Adam(lr=self.learning_rate)

        update_op = adam.get_updates(
            loss=loss,
            params=self.train_model.trainable_weights
        )

        train_fn = kb.function(
            inputs=[
                self.train_model.input,
                action_oh_pl,
                discounted_rw_pl
            ],
            outputs=[self.train_model.output, loss],
            updates=update_op
        )
        return train_fn

    def update_policy(self):
        np_memory = np.array(self.memory)
        states, actions, rewards, new_states, dones = np.hsplit(np_memory, 5)

        states = np.array([_[0][0] for _ in states])
        actions = np.squeeze(actions)
        rewards = np.squeeze(rewards)

        actions_oh = kb.one_hot(actions, self.env.action_space.n)
        discounted_rw = self.compute_discounted_rewards(rewards)

        _, loss = self.train_fn([states, actions_oh, discounted_rw])
        return loss

    def preceive(self, state, action, reward, new_state, done):
        self.remember(state, action, reward, new_state, done)

    def reset_memory(self):
        self.memory.clear()

    def policy_nn(self):
        """
        Create policy network

        """
        state_shape = self.env.observation_space.shape
        input_layer = Input(shape=state_shape)
        hidden_layer1 = Dense(24, activation="relu")(input_layer)
        hidden_layer2 = Dense(48, activation="relu")(hidden_layer1)
        results = Dense(self.env.action_space.n, activation='softmax')(hidden_layer2)
        model = Model(inputs=input_layer, outputs=results)
        return model

    def remember(self, state, action, reward, new_state, done):
        self.time_step += 1
        self.memory.append([state, action, reward, new_state, done])

    def policy_action(self, state):
        probs = np.squeeze(self.train_model.predict(state))
        highest_prob_action = np.random.choice(self.env.action_space.n, p=probs)
        log_prob = np.log(probs[highest_prob_action])
        return highest_prob_action, log_prob

    def take_action(self, state):
        action, _ = self.policy_action(state)
        return action

    def save_model(self, fn):
        self.train_model.save(fn)

    def load_model(self, fn):
        import os
        if os.path.exists(fn):
            from tensorflow.keras.models import load_model
            self.train_model = load_model(fn)


def train():
    env = gym.make(GYM_ENVIRON)

    nb_episode = NB_EPISDOE
    len_episode = LEN_EPISODE

    agent = ReinforceAgent(env=env)
    avg_loss = 0.0

    for i_episode in range(nb_episode):
        max_distance = -1.5
        cur_state = env.reset().reshape(1, -1)
        loss = 0.0

        for step in range(len_episode):
            action, log_prob = agent.policy_action(cur_state)
            if i_episode % RENDER_PER_EPISODE == 0:
                env.render()
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, -1)
            # reward = 1.0 if new_state[0][0] >= 0.5 else -1.0

            agent.preceive(cur_state, action, reward, new_state, done)

            if done:
                loss = agent.update_policy()
                avg_loss = 0.999*avg_loss + 0.001*loss
                agent.reset_memory()
                break

            max_distance = max(new_state[0][0], max_distance)
            cur_state = new_state

        if step >= len_episode - 1:
            print("Completed in %d episode, in %d steps, reward: %d." % (i_episode, step, reward))
        else:
            print("Failed in %d episode, steps: %d, avg_loss: %f, loss: %f" % (
                i_episode, step, avg_loss, loss))

        if i_episode % SAVE_PER_EPISODE == 0:
            agent.save_model("{}-episode-{}.model".format(FN_PREFIX, i_episode))

    agent.save_model("{}-episode-{}.model".format(FN_PREFIX, i_episode))
    env.close()


def test():
    env = gym.make(GYM_ENVIRON)
    nb_episode = 10
    len_episode = 200

    agent = ReinforceAgent(env=env)
    chkpts = sorted(glob.glob("%s-episode-*.model"%FN_PREFIX), key=os.path.basename, reverse=True)
    agent.load_model(chkpts[0])
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
    # train()
    test()







