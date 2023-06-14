import gym
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow import keras
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
import cv2

# Create the Atari Breakout environment
env = gym.make('Breakout-v0', render_mode='human')
n_outputs = env.action_space.n

# Preprocessing function to resize and convert the image to grayscale
def preprocess_observation(obs):
    obs_processed = cv2.resize(obs, (80, 88))
    obs_processed = cv2.cvtColor(obs_processed, cv2.COLOR_BGR2GRAY)
    obs_processed = np.expand_dims(obs_processed, axis=-1)
    return obs_processed

# Define the Q-network
def q_network(X, name):
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,seed=42)

    # Convolutional layers
    conv1 = Conv2D(32, (8, 8), strides=4, activation='relu', kernel_initializer=initializer)(X)
    conv2 = Conv2D(64, (4, 4), strides=2, activation='relu', kernel_initializer=initializer)(conv1)
    conv3 = Conv2D(64, (3, 3), strides=1, activation='relu', kernel_initializer=initializer)(conv2)
    flat = Flatten()(conv3)

    # Fully-connected layers
    fc1 = Dense(128, activation='relu', kernel_initializer=initializer)(flat)
    output = Dense(n_outputs, kernel_initializer=initializer)(fc1)
    return output

# Set hyperparameters
learning_rate = 0.0005
discount_factor = 0.99
num_episodes = 10 #1000
batch_size = 32
replay_buffer_size = 10000
epsilon_initial = 1.0
epsilon_final = 0.1
epsilon_decay_steps = 500000
steps_train = 4
start_steps = 1000
copy_steps = 10000

# Define the epsilon-greedy policy
def epsilon_greedy(action, step):
    epsilon = max(epsilon_final, epsilon_initial - (epsilon_initial - epsilon_final) * step / epsilon_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return action

# Create the replay buffer
exp_buffer = deque(maxlen=replay_buffer_size)

# Initialize the global step
global_step = 0

# Define the input shape
X_shape = (88, 80, 1)

# Define the input layer
X = tf.keras.Input(shape=X_shape, dtype=tf.float32)

# Define the main Q-network
mainQ_outputs = q_network(X, 'mainQ')
mainQ = tf.keras.Model(inputs=X, outputs=mainQ_outputs)

# Define the target Q-network
targetQ_outputs = q_network(X, 'targetQ')
targetQ = tf.keras.Model(inputs=X, outputs=targetQ_outputs)

# Define the placeholder for the action values
X_action = tf.keras.Input(shape=(), dtype=tf.int32)
Q_action = tf.reduce_sum(tf.gather(targetQ_outputs, X_action, axis=1), axis=-1, keepdims=True)

# Define the loss function
y = tf.keras.Input(shape=(1,), dtype=tf.float32)
loss = tf.reduce_mean(tf.square(y - Q_action))

# Create the optimizer
optimizer = Adam(learning_rate=learning_rate)

# Define the training step function
@tf.function
def train_step(X, X_action, y):
    with tf.GradientTape() as tape:
        Q_action = tf.reduce_sum(tf.gather(mainQ_outputs, X_action, axis=1), axis=-1, keepdims=True)
        loss = tf.reduce_mean(tf.square(y - Q_action))
    grads = tape.gradient(loss, mainQ.trainable_variables)
    optimizer.apply_gradients(zip(grads, mainQ.trainable_variables))
    return loss

# Function to copy the weights of the main Q-network to the target Q-network
def copy_target_to_main():
    targetQ.set_weights(mainQ.get_weights())

# Initialize the target Q-network with the weights of the main Q-network
copy_target_to_main()

# Training loop

result={}
for episode in range(num_episodes):
    obs = env.reset()
    episode_reward = 0
    loss = 0

    while True:

        # Preprocess the observation
        obs_processed = preprocess_observation(obs)
        obs_processed = np.expand_dims(obs_processed, axis=-1)

        # Choose an action
        Q_values = mainQ.predict(np.array([obs_processed]))[0]
        action = np.argmax(Q_values)
        action = epsilon_greedy(action, global_step)

        # Take the action
        next_obs, reward, done, _ = env.step(action)

        # Store the transition in the replay buffer
        exp_buffer.append((obs_processed, action, reward, preprocess_observation(next_obs), done))

        episode_reward += reward

        # Update the global step
        global_step += 1

        if global_step > start_steps and global_step % steps_train == 0:
            # Sample a minibatch from the replay buffer
            minibatch = np.array(exp_buffer)[np.random.choice(len(exp_buffer), batch_size, replace=False), :]
            o_obs = np.array([elem[0] for elem in minibatch])
            o_action = np.array([elem[1] for elem in minibatch])
            o_reward = np.array([elem[2] for elem in minibatch])
            o_next_obs = np.array([elem[3] for elem in minibatch])
            o_done = np.array([elem[4] for elem in minibatch])

            # Compute the target Q-values
            q_values_next = targetQ.predict(o_next_obs)
            max_q_values_next = np.max(q_values_next, axis=1)
            targetQ_values = o_reward + (1 - o_done) * discount_factor * max_q_values_next

            # Train the network and calculate loss
            loss = train_step(o_obs, o_action, np.expand_dims(targetQ_values, axis=-1))

            # After some interval, copy our main Q network weights to target Q network
            if global_step % copy_steps == 0:
                copy_target_to_main()

        obs = next_obs

        if done:
            break
    result[episode+1]=reward

    print('Episode:', episode, 'Reward:', episode_reward, 'Loss:', loss)

env.close()
print(result)
