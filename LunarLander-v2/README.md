This is a solution to gym/LunarLander-v2 environment using PPO agent from tf_agents library in TensorFlow.

# Dependencies

    tensorflow==2.4.1
    tf-agents==0.7.1
    gym
    shutil
    os
 
# Usage

The class 'my_AI' has all the necessary functions from defining the agent to training and testing it.
Initialize the class with a checkpoint path. That path will be used to save/load a trained model.

    Call 'train' function to train the model.
    Call 'checkpointer_load' to load from a checkpoint. This loads agent, a policy and replay buffer.
    Call 'test_agent' to see what the trained agent can do. 'tries' parameter is total number of episodes to play, 'display_freq' is rendering frequency of the test.

# Results

The PPO agent with the parameters included in the python file, solves the 'LunarLander-v2' environment in about 287 episodes.
