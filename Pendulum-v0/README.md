This is a solution to gym/Pendulum-v0 environment using SAC agent from tf_agents library in TensorFlow.

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

The SAC agent with the parameters included in the python file, it takes about 500-1000 episodes before solving 'Pendulum-v0' environment.