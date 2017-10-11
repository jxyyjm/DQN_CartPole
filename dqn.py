# -------------------------------
# DQN for CartPole in OpenAI Gym
# Date: 2017-10-11
# All rights reserved
# -------------------------------
# reference : https://zhuanlan.zhihu.com/p/21477488 #
# reference : http://www.cnblogs.com/mandalalala/p/6227201.html #

import gym
import tensorflow as tf 
import numpy as np 
import random
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q 
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size ## experiece-replay pool size ##
BATCH_SIZE = 32 # size of minibatch

class DQN():
	# DQN Agent
	def __init__(self, env):
		# init experience replay
		self.replay_buffer = deque()
		# init some parameters
		self.time_step = 0
		self.epsilon = INITIAL_EPSILON
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n
		
		print 'self.state_dim : ', self.state_dim
		print 'self.action_dim: ', self.action_dim
		print 'env.observation_space : ', env.observation_space
		print 'env.action_space      : ', env.action_space

		self.create_Q_network()
		self.create_training_method()

		# Init session # notice : here complete session initialization #
		self.session = tf.InteractiveSession()
		#self.session.run(tf.initialize_all_variables())
		self.session.run(tf.global_variables_initializer())

		# loading networks and check checkpoint #
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
				print "Could not find old network weights"

		global summary_writer
		#summary_writer = tf.train.SummaryWriter('~/logs',graph=self.session.graph) # notice : here tf-Version-0.X #
		summary_writer = tf.summary.FileWriter('~/logs',graph=self.session.graph)

	def create_Q_network(self):
		# network weights
		W1 = self.weight_variable([self.state_dim,20])
		b1 = self.bias_variable([20])
		W2 = self.weight_variable([20,self.action_dim])
		b2 = self.bias_variable([self.action_dim])
		# input layer
		self.state_input = tf.placeholder("float",[None,self.state_dim])
		# hidden layers
		h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
		# Q Value layer
		self.Q_value = tf.matmul(h_layer,W2) + b2
		# notice : Q-network constains weight and state-input and Qvalue-output #

	def create_training_method(self):
		self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
		self.y_input = tf.placeholder("float",[None])
		#Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input),reduction_indices = 1)
		Q_action = tf.reduce_sum(tf.matmul(self.Q_value,self.action_input,transpose_b=True),reduction_indices = 1)
		# notice : above line compute the Q-value and Action_input 
		self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
		#tf.scalar_summary("loss",self.cost)
		tf.summary.scalar("loss",self.cost)
		global merged_summary_op
		#merged_summary_op = tf.merge_all_summaries()
		merged_summary_op = tf.summary.merge_all()
		self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

	def perceive(self,state,action,reward,next_state,done):
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
		# notice : here store the [st, at, rt, st+1)
		if len(self.replay_buffer) > REPLAY_SIZE:
			self.replay_buffer.popleft()
			# remove and return the leftmost element #
			# only reduce the expence #

		if len(self.replay_buffer) > BATCH_SIZE:
			self.train_Q_network()

	def train_Q_network(self):
		self.time_step += 1
		# Step 1: sample random minibatch from replay memory
		minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		next_state_batch = [data[3] for data in minibatch]

		# Step 2: calculate y , this is target-Q value #
		y_batch = []
		Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
		for i in range(0,BATCH_SIZE):
			done = minibatch[i][4]
			if done:
				y_batch.append(reward_batch[i])
			else :
				y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

		self.optimizer.run(feed_dict={
			self.y_input:y_batch,
			self.action_input:action_batch,
			self.state_input:state_batch
			})
		summary_str = self.session.run(merged_summary_op,feed_dict={
				self.y_input : y_batch,
				self.action_input : action_batch,
				self.state_input : state_batch
				})
		summary_writer.add_summary(summary_str,self.time_step)

		# save network every 1000 iteration
		if self.time_step % 1000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.time_step)

	def egreedy_action(self,state):
		Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
		# compute the Q-value so that it is easy to max(Q) #
		if random.random() <= self.epsilon:
			return random.randint(0,self.action_dim - 1)
		else:
			return np.argmax(Q_value)
		# notice: in DQN Algorithm #
		#   with probability \epsilon select a random action
		#   othewise select action = max{ Q(state, a; \theta) }
		self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000
		# notice: here epsilon is becoming smaller and smaller #

	def action(self,state):
		return np.argmax(self.Q_value.eval(feed_dict = {
			self.state_input:[state]
			})[0])

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 5000 # Episode limitation
#EPISODE = 100 # Episode limitation
STEP = 300 # Step limitation in an episode #
		   # notice : we know one episode contains a sequence of states, actions, rewards,
		   #          which ends with terminal state.
TEST = 10 # The number of experiment test every 100 episode

def main():
	# initialize OpenAI Gym env and dqn agent
	env = gym.make(ENV_NAME)
	agent = DQN(env)

	for episode in xrange(EPISODE):
		# initialize task # === env.reset() === #
		#   Resets the state of the environment,
		#   returns an initial observation.
		state = env.reset()
		# Train 
		for step in xrange(STEP): # in one episode # contains STEP (state, action, reward) #
			action = agent.egreedy_action(state)
			# notice : above line, e-greedy action for train #
			next_state,reward,done,_ = env.step(action)
			# notice : env.step(action)
			#   run one timestep of the env's dynamics. 
			#   return, 1) observation(here next_state): after run one time-step
			#   2) reward: amount of reward returned after previous action
			#   3) done  : whether the episode has ended. if true, futher step will return undefined results 
			# Define reward for agent
			reward_agent = -1 if done else 0.1
			agent.perceive(state,action,reward,next_state,done)
			# notice : above line, perceive the infomation #
			# do threee things: 1) store {current and step-res } in D #
			# 2) random sample minibatch from D #
			# 3) train Q-network
			state = next_state
			if done:
				break
		# Test every 100 episodes
		if episode % 100 == 0:
			total_reward = 0
			for i in xrange(TEST):
				state = env.reset()
				for j in xrange(STEP):
					env.render()
					action = agent.action(state) # direct action with Q-network for test
					state,reward,done,_ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward/TEST
			print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
			if ave_reward >= 200:
				break

	# save results for uploading
	#env.monitor.start('gym_results/CartPole-v0-experiment-1',force = True)
	env = gym.wrappers.Monitor(env, 'gym_results/CartPole-v0-experiment-1', force=True)
	for i in xrange(100):
	#for i in xrange(10):
		state = env.reset()
		for j in xrange(200):
		#for j in xrange(20):
			env.render()
			action = agent.action(state) # direct action for test
			state,reward,done,_ = env.step(action)
			total_reward += reward
			if done:
				break
	#env.monitor.close()
	gym.wrappers.Monitor.close(env)
	env.close()
	#gym.upload( \
	#		"./gym_results/CartPole-v0-experiment-1/", \
	#		api_key=' sk_FYp0Gc1dQU69epifs7ZE6w')
	#		#api_key="JMXoHnlRtm86Fif6FUw4Qop1DwDkYHy0")
	# notice: has been qianged by dang #

if __name__ == '__main__':
	main()
