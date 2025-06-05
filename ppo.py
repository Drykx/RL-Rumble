import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam

import gymnasium as gym

class Feedforward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):

        super(Feedforward, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        #self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, obs):
        # Convert obs to tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype = torch.float)
        # Forward pass
        activation_1 = F.relu(self.fc1(obs))
        #activation_1 = self.leaky_relu(self.fc1(obs))
        activation_2 = self.fc2(activation_1)
        return activation_2
    
    def vizualisation(self,obs):
        activation = F.relu(self.fc1(obs))
        #activation = self.leaky_relu(self.fc1(obs))

        return(activation)

class PPO():
    def __init__(self, env, device = "cpu"):
        
        # Device
        self.device = torch.device(device)
        # Hyperparameters
        self.init_parameters = self._init_hyperparameters()
        # Environment
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        # NN
        self.actor = Feedforward(self.obs_dim, self.hidden_dim, self.act_dim).to(self.device) # Policy
        self.critic = Feedforward(self.obs_dim,self.hidden_dim, 1).to(self.device)            # Value function
        # Optimizer
        self.actor_opti = Adam(self.actor.parameters(), lr = self.lr)
        self.critic_opti = Adam(self.critic.parameters(), lr = self.lr)
        # Extra Instances
        self.cov_vect = torch.full((self.act_dim, ), fill_value = self.var_matrix ,device = self.device)
        self.cov_mat = torch.diag(self.cov_vect).to(self.device) # [act_dim, act_dim]

    def _init_hyperparameters(self):

        self.timesteps_per_batch       = 6000
        self.max_timesteps_per_episode = 2000
        self.gamma                     = 0.95
        self.n_updates_per_iteration   = 5
        self.clip                      = 0.2
        self.lr                        = 0.005
        self.hidden_dim                = 32
        self.num_minibatches           = 5
        self.entropy_coef              = 0.1
        self.max_grad_norm             = 0.5
        self.lambd                     = 1
        self.var_matrix                = 0.5

    def rollout(self):
        """Generate time_steps_per_batch in multiple episodes each 
        of maximum length max_timesteps_per_episode
        """
        # Batch data
        batch_obs       = []
        batch_acts      = []
        batch_log_probs = []
        batch_rews      = []
        batch_rgts      = [] # Batch rewards-to-go
        batch_lens      = [] # Episiodic length in batch
        batch_vals      = []
        batch_dones     = []

        current_timesteps = 0
        while current_timesteps < self.timesteps_per_batch:
            
            # Initialization
            ep_rews = [] # special format for rgts
            ep_vals = []
            ep_dones = []

            obs, _ = self.env.reset() # Type= (np.ndarray, dict)
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # Collect observations
                batch_obs.append(obs)
                action, log_probs = self.get_action(torch.tensor(obs, dtype = torch.float, device = self.device))
                val = self.critic(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Collect reward, action and log_prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_probs)
                ep_vals.append(val)
                ep_dones.append(done)

                current_timesteps += 1
                if done : break # If the agent completed the task finish

            # Collect episodic length and rewards
            current_timesteps += ep_t
            batch_rews.append(ep_rews)            
            batch_lens.append(ep_t + 1)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

        # Transform into desired format (Torch)
        batch_obs       = torch.tensor(np.array(batch_obs), dtype = torch.float, device = self.device)
        batch_acts      = torch.tensor(np.array(batch_acts), dtype = torch.float, device = self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype = torch.float, device = self.device)

        batch_rgts = self.compute_rgts(batch_rews)
        batch_gae  = self.compute_gae(batch_rews, batch_vals, batch_dones)

        return batch_obs,batch_acts, batch_log_probs, batch_rgts, batch_lens, batch_gae

    def compute_rgts(self, batch_rews: list[list[float]]) -> torch.Tensor:
        "Compute RTG for each episode in a batch"

        batch_rgts = []

        for ep_rews in batch_rews: # iteration: number episode
            ep_rgts = []
            discounted_sum = 0
            for rew in reversed(ep_rews): # iteration: steps of the episode
                discounted_sum = rew + self.gamma * discounted_sum
                ep_rgts.insert(0,discounted_sum) # We need the history of rgts at each time steps
            batch_rgts.extend(ep_rgts)
        
        return torch.tensor(batch_rgts, dtype = torch.float, device = self.device)

    def compute_gae(self, rewards, values, done):
        
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, done): # loop in parallel
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else: 
                    delta = ep_rews[t] - ep_vals[t]
                
                advantage = delta + self.gamma * self.lambd * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype = torch.float, device = self.device)

    def get_action(self, obs: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        "Generate an action as a sample of a Multivariate normal distribution"
        # Query actor network for an action
        obs = obs.to(self.device)
        mean = self.actor(obs) # Call NN (self.actor.forward(obs))
        dist = MultivariateNormal(mean, self.cov_mat)

        # Generate sample from the distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy(), log_prob.detach()

    def learn(self, total_timesteps: int): 

        print(f"Learning Process:\n"
              f"Total timesteps {total_timesteps}\n"
              f"Timesteps per batch {self.timesteps_per_batch}\n"
              f"Max timesteps per episode {self.max_timesteps_per_episode}\n"
              f"Number of backward passes per batch {self.n_updates_per_iteration}\n"
              f"Number of mini batches per pass {self.num_minibatches} \n \n"
            )
             
        current_timestep = 0
        number_rollout   = 0

        history_actor_loss  = []
        history_critic_loss = []

        while current_timestep < total_timesteps:
            # Rollout 
            batch_obs, batch_acts, batch_log_probs, batch_rgts, batch_lens, batch_gae  = self.rollout()
            # Update number steps
            current_timestep += np.sum(batch_lens)
            # Update number of rollout
            number_rollout += 1

            # Calculate Advantage
            # V_k, _ , _  = self.evaluate(batch_obs, batch_acts)  # V_(k,phi)
            # A_k = batch_rgts - V_k.detach()                # Remove computational graph
            A_k = batch_gae
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # Normalize

            # Mini Batch Hyperparameters
            step = batch_obs.size(0)
            minibatch_size = step // self.num_minibatches
            inds = np.arange(step)

            history_actor_loss_batch  = []
            history_critic_loss_batch = []

            for _ in range(self.n_updates_per_iteration):

                # Update learning rate
                self.learning_rate_annealing(current_timestep,total_timesteps)
                
                np.random.shuffle(inds) # Random sampling
                for start in range(0, step, minibatch_size):
                    
                    end = start + minibatch_size
                    idx = inds[start:end]
                    # Take mini batches
                    mini_obs        = batch_obs[idx]
                    mini_acts       = batch_acts[idx]
                    mini_log_prob   = batch_log_probs[idx]
                    mini_advantage  = A_k[idx]
                    mini_rgts       = batch_rgts[idx]

                    V, cur_log_probs, entropy = self.evaluate(mini_obs, mini_acts)
                    ratios = torch.exp(cur_log_probs - mini_log_prob)
                    # Calculate surrogate losses
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage
                    # Calculate losses
                    actor_loss   = (-torch.min(surr1,surr2)).mean()
                    critic_loss  = nn.MSELoss()(V, mini_rgts)
                    entropy_loss = entropy.mean()

                    history_actor_loss_batch.append(actor_loss.item())
                    history_critic_loss_batch.append(critic_loss.item())

                    # Calculate gradient and perform backward propagation
                    self.actor_opti.zero_grad()
                    actor_loss.backward(retain_graph = True) # Add this parameter as the computations graphs overlap
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_opti.step()

                    self.critic_opti.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_opti.step()

            meam_batch_actor_loss  = np.mean(history_actor_loss_batch)
            mean_batch_critic_loss = np.mean(history_critic_loss_batch)

            history_actor_loss.append(meam_batch_actor_loss)
            history_critic_loss.append(mean_batch_critic_loss)
            
            print(f"{number_rollout} Rollout - Timestep {current_timestep} || "
                  f"Actor loss: {meam_batch_actor_loss:.4f} | Critic loss: {mean_batch_critic_loss:.4f}")
        
        return(history_actor_loss, history_critic_loss)

    def evaluate(self, batch_obs: torch.Tensor, batch_action: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """
        Value function at batch k: V_(k,phi) and log probs
        We compute the prob of taking an action w.r.t current policy defined by the NN
        """
        # We need the value at each state given the obs sequence of the batch
        V = self.critic(batch_obs).squeeze()
        # Compute log probs in the same fashion to get_action
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_action)

        return V, log_probs, dist.entropy()
    
    def learning_rate_annealing(self, current_timestep, total_timestep):

        frac = current_timestep / total_timestep
        new_lr = self.lr * (1 - frac)
        self.actor_opti.param_groups[0]["lr"] = new_lr
        self.critic_opti.param_groups[0]["lr"] = new_lr
