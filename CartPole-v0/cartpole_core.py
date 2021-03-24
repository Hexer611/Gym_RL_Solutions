import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.networks import sequential, actor_distribution_network, value_network
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver, dynamic_step_driver
from tf_agents.specs import tensor_spec
from tf_agents.metrics import tf_metrics
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu
import shutil

class my_AI:
    def __init__(self, chk_dir):
        env_name = 'CartPole-v0'
        self.checkpoint_dir = chk_dir

        train_env = suite_gym.load(env_name)
        test_env = suite_gym.load(env_name)

        self.total_episode = 0
        self.total_step = 0

        self.env_tf = tf_py_environment.TFPyEnvironment(train_env)
        self.test_env_tf = tf_py_environment.TFPyEnvironment(test_env)

        #self.create_dqn_agent(learning_rate=0.002)
        self.create_ppo_agent(learning_rate=0.001)

        # INFO PARAMETERS
        self.replay_buffer_max_length = 100000  # PARAMETER
        self.batch_size = 32  # PARAMETER

        self.create_replay_buffer()

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
            td_errors_loss_fn=common.element_wise_squared_loss)

        agent.initialize()
        self.agent = agent

    def create_ppo_actor(self):
        model = actor_distribution_network.ActorDistributionNetwork(self.env_tf.observation_spec(), self.env_tf.action_spec(), fc_layer_params=(64, 64))
        return model

    def create_ppo_value(self):
        model = value_network.ValueNetwork(self.env_tf.observation_spec(), fc_layer_params=(64, 64))
        return model

    def create_ppo_agent(self, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        actor_model = self.create_ppo_actor()
        value_model = self.create_ppo_value()

        agent = ppo_agent.PPOAgent(
            self.env_tf.time_step_spec(),
            self.env_tf.action_spec(),
            optimizer=optimizer,
            actor_net=actor_model,
            value_net=value_model,
            num_epochs=3,
            adaptive_kl_target=0.2,
            normalize_observations=True)

        agent.initialize()
        actor_model.summary()
        value_model.summary()
        self.agent = agent

    def test_agent(self, tries=100, display_freq=10):
        reward = 0
        tot_reward = 0
        for _tr in range(1, tries+1):
            time_step = self.test_env_tf.reset()
            while not time_step.is_last():
                if _tr % display_freq == 0:
                    self.test_env_tf.render()
                action_step = self.agent.policy.action(time_step)
                time_step = self.test_env_tf.step(action_step.action)
                reward += time_step.reward[0]
            tot_reward += reward
            awg = tot_reward / _tr
            print('|Episode-{0}|Episode reward-{1} |Average reward-{2}|'.format(_tr, reward, awg))
            if awg < 180 and _tr > 5:
                break
            reward = 0
        return tot_reward/tries

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

    def collect_episodes(self, number_of_episodes=1, policy=None):
        if policy is None:
            policy = self.agent.collect_policy
        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()
        replay_observer = [self.replay_buffer.add_batch, num_episodes, env_steps]
        #replay_observer = [num_episodes, env_steps]
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.env_tf,
            policy,
            observers=replay_observer,
            num_episodes=number_of_episodes)
        final_time_step, policy_state = driver.run()
        self.total_episode += num_episodes.result().numpy()
        self.total_step += env_steps.result().numpy()

    def collect_steps(self, number_of_steps=1, policy=None):
        if policy is None:
            policy = self.agent.collect_policy
        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()
        replay_observer = [self.replay_buffer.add_batch, num_episodes, env_steps]
        #replay_observer = [num_episodes, env_steps]
        driver = dynamic_step_driver.DynamicStepDriver(
            self.env_tf,
            policy,
            observers=replay_observer,
            num_steps=number_of_steps)
        final_time_step, policy_state = driver.run()
        self.total_episode += num_episodes.result().numpy()
        self.total_step += env_steps.result().numpy()

    def step_driver_define(self, num_steps=1, policy=None):
        if policy is None:
            policy = self.agent.collect_policy
        self.num_episodes_driver = tf_metrics.NumberOfEpisodes()
        self.env_steps_driver = tf_metrics.EnvironmentSteps()
        self.avg_return_driver = tf_metrics.AverageReturnMetric()
        replay_observer = [self.replay_buffer.add_batch, self.num_episodes_driver, self.env_steps_driver, self.avg_return_driver]
        self.driver = dynamic_step_driver.DynamicStepDriver(
            self.env_tf,
            policy,
            observers=replay_observer,
            num_steps=num_steps)

    def save_policy(self):
        policy_dir = os.path.join('/', 'Brains')
        tf_policy_saver = policy_saver.PolicySaver(self.agent.policy)
        tf_policy_saver.save(policy_dir)
        print('Policy saved to {0}'.format(policy_dir))

    def load_policy(self):
        policy_dir = os.path.join('/', 'Brains')
        model = tf.saved_model.load(policy_dir)
        self.loaded = model

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

        for _ in range(500):
            self.collect_episodes(1)

            iterations = 32
            loss = 0
            for _ in range(iterations):
                experience, unused_info = next(self.iterator)
                loss += self.agent.train(experience).loss

            tot_reward = self.test_agent(tries=100, display_freq=200)
            print('Episode {0} - Total Steps {1} - Average Reward {2} - Loss {3}'.format(self.total_episode, self.total_step, tot_reward, loss / iterations))
            if tot_reward >= 195.0:
                print('The game is solved.')
                self.checkpointer_save()
                break

one = my_AI(chk_dir='Models/')
#one.train()

one.checkpointer_load()
one.test_agent(tries=100, display_freq=1)
