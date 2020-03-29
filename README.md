# Artificial Idiot for Ridiculous Learning
AI of RL: something dumb, something plaid,  something artificial. 

## Mountaincar

DQN sample application, key points:
- Experience replay
- Async *target* network
- Epsilon greedy policy

Double DQN:
- Use different value estimators in selecting and evaluating

Dueling network DQN/DDQN:
- Decoupled Value and Advantage function estimator
- Video: https://youtu.be/6fB8kJ-v-7c

## Cartpole

Policy Gradients, REINFORCE algorithm
- On-policy MCMC
- Discounted rewards
- Customized model update

Actor-Critic, on-policy
- The Critic provide states-actor value
- Simply replaces REINFORCE's reward with the Critic's value in the actor (policy)

A2C
- Critic estimates Advantage function instead of direct q-value 
- TD updates




