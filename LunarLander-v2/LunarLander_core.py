import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from threading import Thread
import numpy as np
from tf_agents.environments import suite_gym, tf_py_environment, batched_py_environment
from tf_agents.agents import DqnAgent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.agents.ddpg import ddpg_agent, actor_network, critic_network
from tf_agents.policies import random_tf_policy, policy_saver, py_tf_eager_policy, greedy_policy, ou_noise_policy
from tf_agents.networks import sequential, actor_distribution_network, value_network
from tf_agents.utils import common
from tf_agents.typing.types import TensorSpec
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.drivers import dynamic_episode_driver, dynamic_step_driver
from tf_agents.specs import tensor_spec
from tf_agents.metrics import tf_metrics
import functools
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu
import shutil

class my_AI:
    def __init__(self, chk_dir):
        env_name = 'LunarLander-v2'
        self.checkpoint_dir = chk_dir
        self.num_parallel_envs = 8

        self.strategy = tf.distribute.experimental.CentralStorageStrategy()

        test_env = suite_gym.load(env_name)

        parallel_train_env = batched_py_environment.BatchedPyEnvironment([suite_gym.load(env_name) for _ in range(self.num_parallel_envs)], multithreading=True)

        self.env_tf = tf_py_environment.TFPyEnvironment(parallel_train_env)
        self.test_env_tf = tf_py_environment.TFPyEnvironment(test_env)

        #self.create_dqn_agent(0.0003)
        self.create_ppo_agent(learning_rate=0.001)
        #self.create_ddpg_agent()
        #self.create_sac_agent(act_lr=0.0003, crit_lr=0.0003, alph_lr=0.0003)

        self.total_episode = 0
        self.total_step = 0
        # INFO PARAMETERS
        self.replay_buffer_max_length = 100000  # PARAMETER
        self.initial_steps_collected = 100  # PARAMETER
        self.batch_size = 256  # PARAMETER

        self.create_replay_buffer()
        #self.collect_steps(self.initial_steps_collected)

    def create_dqn_model(self):
        action_tensor_spec = tensor_spec.from_spec(self.env_tf.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        X1 = Dense(64, activation=relu, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'))
        X2 = Dense(64, activation=relu, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'))
        X3 = Dense(num_actions, activation='linear', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03), bias_initializer=tf.keras.initializers.Constant(0))

        model = sequential.Sequential([X1,X2,X3])

        return model

    def create_dqn_agent(self, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model = self.create_dqn_model()

        agent = dqn_agent.DqnAgent(
            self.env_tf.time_step_spec(),
            self.env_tf.action_spec(),
            q_network=self.model,
            optimizer=optimizer,
            gamma=0.8, epsilon_greedy=0.1,
            td_errors_loss_fn=common.element_wise_squared_loss)

        agent.initialize()
        self.agent = agent

    def create_ppo_actor(self):
        model = actor_distribution_network.ActorDistributionNetwork(self.test_env_tf.observation_spec(), self.test_env_tf.action_spec(), fc_layer_params=(256, 256))
        return model

    def create_ppo_value(self):
        model = value_network.ValueNetwork(self.test_env_tf.observation_spec(), fc_layer_params=(256, 256))
        return model

    def create_ppo_agent(self, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        actor_model = self.create_ppo_actor()
        value_model = self.create_ppo_value()

        agent = ppo_agent.PPOAgent(
            self.test_env_tf.time_step_spec(),
            self.test_env_tf.action_spec(),
            optimizer=optimizer,
            actor_net=actor_model,
            value_net=value_model,
            num_epochs=3,
            adaptive_kl_target=0.2,
            entropy_regularization=0.02)

        agent.initialize()
        self.agent = agent

    def create_ddpg_model(self):
        act_net = actor_network.ActorNetwork(self.env_tf.observation_spec(), self.env_tf.action_spec())
        cri_net = critic_network.CriticNetwork((self.env_tf.observation_spec(), self.env_tf.action_spec()))

        return act_net, cri_net

    def create_ddpg_agent(self):
        act_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        cri_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        act_net, cri_net = self.create_ddpg_model()

        agent = ddpg_agent.DdpgAgent(
            self.env_tf.time_step_spec(),
            self.env_tf.action_spec(),
            actor_network=act_net,
            critic_network=cri_net,
            actor_optimizer=act_optimizer,
            critic_optimizer=cri_optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            target_update_tau=0.001,
            target_update_period=100)

        agent.initialize()
        self.agent = agent

    def create_sac_agent(self, act_lr, crit_lr, alph_lr):
        crit = critic_network.CriticNetwork(
            (self.env_tf.observation_spec(), self.env_tf.action_spec()),
            joint_fc_layer_params=(256, 256), kernel_initializer='glorot_uniform', last_kernel_initializer='glorot_uniform')
        acto = actor_distribution_network.ActorDistributionNetwork(
            self.env_tf.observation_spec(),
            self.env_tf.action_spec(),
            fc_layer_params=(256, 256),
            continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork
        )
        act_opt = tf.keras.optimizers.Adam(learning_rate=act_lr)
        crit_opt = tf.keras.optimizers.Adam(learning_rate=crit_lr)
        alph_opt = tf.keras.optimizers.Adam(learning_rate=alph_lr)

        agent = sac_agent.SacAgent(
            self.env_tf.time_step_spec(),
            self.env_tf.action_spec(),
            actor_network=acto,
            critic_network=crit,
            actor_optimizer=act_opt,
            critic_optimizer=crit_opt,
            alpha_optimizer=alph_opt,
            target_update_tau=0.005,
            target_update_period=1,
            target_entropy=-40,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=0.99,
            reward_scale_factor=1.0,
            train_step_counter=tf.Variable(0))

        agent.initialize()
        crit.summary()
        acto.summary()
        self.agent = agent

    def test_agent(self, tries=100, display_freq=10):
        reward = 0
        tot_reward = 0
        for _tr in range(1, tries + 1):
            time_step = self.test_env_tf.reset()
            while not time_step.is_last()[0]:
                if _tr % display_freq == 0:
                    self.test_env_tf.render()
                action_step = self.agent.policy.action(time_step)
                time_step = self.test_env_tf.step(action_step.action)
                reward += time_step.reward.numpy()[0]
            tot_reward += reward
            awg = tot_reward / _tr
            print('|Episode {0} | Episode reward {1} |Average reward {2}|'.format(_tr, reward, awg))
            reward = 0

    def collect_steps_old(self, steps, policy=None):
        if policy == None:
            policy = self.agent.collect_policy
        tot_reward = 0
        for _ in range(steps):
            time_step = self.env_tf.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = self.env_tf.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            # Add trajectory to the replay buffer
            self.replay_buffer.add_batch(traj)
            for ts in time_step.is_last():
                self.total_episode += int(ts)
            tot_reward += sum(time_step.reward.numpy())
            self.total_step += self.num_parallel_envs
        return tot_reward

    def collect_episodes_old(self, episodes=1, policy=None):
        if policy == None:
            policy = self.agent.collect_policy
        tot_reward = 0
        for _ in range(episodes):
            while True:
                time_step = self.env_tf.current_time_step()
                action_step = policy.action(time_step)
                next_time_step = self.env_tf.step(action_step.action)
                traj = trajectory.from_transition(time_step, action_step, next_time_step)
                # Add trajectory to the replay buffer
                self.replay_buffer.add_batch(traj)
                self.total_step += 1
                tot_reward += time_step.reward
                if next_time_step.is_last():
                    self.env_tf.reset()
                    break
            self.total_episode += 1
        return tot_reward / episodes

    def create_replay_buffer(self):
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.env_tf.batch_size,
            max_length=self.replay_buffer_max_length)

        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=4,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(3)

        self.iterator = iter(dataset)

    def step_driver_def(self, number_of_steps=1, policy=None):
        if policy is None:
            policy = self.agent.collect_policy
        self.driver_episodes = tf_metrics.NumberOfEpisodes()
        self.driver_steps = tf_metrics.EnvironmentSteps()
        self.driver_reward = tf_metrics.AverageReturnMetric(batch_size=1)
        replay_observer = [self.replay_buffer.add_batch, self.driver_episodes, self.driver_steps, self.driver_reward]
        #replay_observer = [num_episodes, env_steps]
        self.driver = dynamic_step_driver.DynamicStepDriver(
            self.env_tf,
            policy,
            observers=replay_observer,
            num_steps=number_of_steps)

    def save_policy(self):
        policy_dir = os.path.join('/', 'Brains')
        tf_policy_saver = policy_saver.PolicySaver(self.agent.policy)
        tf_policy_saver.save(policy_dir)
        print('Policy saved to {0}'.format(policy_dir))

    def load_policy(self):
        policy_dir = os.path.join('/', 'Brains')
        model = tf.saved_model.load(policy_dir)
        self.loaded = model

        #print(list(model.signatures.keys()))
        #print(model.model_variables)

    def checkpointer_save(self, steps=0):
        cwd = os.getcwd()
        save_dir = os.path.join(cwd, self.checkpoint_dir)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        self.checkpointer = common.Checkpointer(
            ckpt_dir=save_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer
        )
        self.checkpointer.save(steps)

    def checkpointer_load(self):
        self.checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer
        )

    def train(self):
        self.agent.train = common.function(self.agent.train)
        for _ in range(0):
            final_time_step, policy_state = self.driver.run()
            self.total_episode = self.driver_episodes.result().numpy()
            self.total_step = self.driver_steps.result().numpy()

            iterations = 16
            loss = 0
            for _ in range(iterations):
                experience, unused_info = next(self.iterator)
                loss += self.agent.train(experience).loss
            print('Episode {0} - Total Steps {1} - Average Reward {2} - Loss {3}'.format(self.total_episode, self.total_step, self.driver_reward.result().numpy(), loss / iterations))
            self.driver_reward.result()
            if self.total_episode % 2000 == 0:
                self.test_agent(tries=1, display_freq=1)
                print('Brain saved.')
                self.checkpointer_save()
        self.env_tf.reset()
        for _ in range(5000):
            train_reward = self.collect_steps_old(25)

            iterations = 16
            loss = 0
            for _ in range(iterations):
                experience, unused_info = next(self.iterator)
                loss += self.agent.train(experience).loss
            print('Episode {0} - Total Steps {1} - Average Reward {2} - Loss {3}'.format(self.total_episode, self.total_step, train_reward, loss / iterations))
            #self.test_agent(tries=1, display_freq=100)
            if self.total_step % 10000 == 0:
                self.test_agent(tries=1, display_freq=10)
                print('Brain saved.')
                self.checkpointer_save()
                #self.replay_buffer.clear()

one = my_AI(chk_dir='Models/')
one.checkpointer_load()
#one.train()

one.test_agent(tries=100, display_freq=1)
