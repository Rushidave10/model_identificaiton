# model_identificaiton
Various methods of model identification are proposed to be used in classical control and/or reinforcement learning setting.


Reinforcement Learning is adaptive in a way that it can learn from data and not necessarily requires the model of underlying system. However, in real-world scenario, safety is also very important. In order to incooporate safety, Model predictive control(MPC) could be used that optimize an objective function and handle hard state and input constraints. In order to ensure safety and keep RL method relatively model-free, in this repository model identification is performed. This repo. mostly works with model from electrical domain such as power grids and electrical drives. 
