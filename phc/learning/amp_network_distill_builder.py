from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_builder import AMPBuilder
from phc.learning.network_builder import init_mlp
import torch
import torch.nn as nn
import numpy as np
from phc.utils.torch_utils import project_to_norm
from phc.learning.vq_quantizer import EMAVectorQuantizer, Quantizer
from phc.utils.flags import flags
DISC_LOGIT_INIT_SCALE = 1.0

class AMPZBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPZBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.self_obs_size = kwargs['self_obs_size']
            self.task_obs_size = kwargs['task_obs_size']

            self.embedding_size         = params['vae']['emebedding_size']
            self.use_vae_prior          = params['vae'].get("use_vae_prior", False)
            self.use_vae_fixed_prior    = params['vae'].get("use_vae_fixed_prior", False)
            self.use_vae_clamped_prior  = params['vae'].get("use_vae_clamped_prior", False)
            self.vae_var_clamp_max      = params['vae'].get("vae_var_clamp_max", 2.)
            self.use_vae_sphere_posterior = params['vae'].get("use_vae_sphere_posterior", False)
            self.vae_prior_fixed_logvar = params['vae'].get("vae_prior_fixed_logvar", 0.)
            self.vae_units              = params['vae']['units']
            self.vae_activation         = params['vae']['activation']
            self.vae_initializer        = params['vae']['initializer']
            # self.
            self.task_obs_size_detail = kwargs['task_obs_size_detail']

            # self.proj_norm = self.task_obs_size_detail["proj_norm"]
            # self.embedding_norm = self.task_obs_size_detail['embedding_norm']
            # self.z_type = self.task_obs_size_detail.get("z_type", "sphere")
            
            kwargs['input_shape'] = (kwargs['self_obs_size'] + self.embedding_size,)  # Task embedding size + self_obs

            super().__init__(params, **kwargs)
            self.running_mean = kwargs['mean_std'].running_mean
            self.running_var = kwargs['mean_std'].running_var

            self._build_z_mlp()
                
            self.actor_mlp

        
        def form_embedding(self, task_out_z):
            extra_dict = {}
            B, N = task_out_z.shape
            self.vae_mu = vae_mu = self.z_mu(task_out_z)
            self.vae_log_var = vae_log_var = self.z_logvar(task_out_z)
                
            if self.use_vae_clamped_prior:
                self.vae_log_var = vae_log_var = torch.clamp(vae_log_var, min = -5, max = self.vae_var_clamp_max)
                
            task_out_proj, self.z_noise = self.reparameterize(vae_mu, vae_log_var)
                    
            if flags.test:
                task_out_proj = vae_mu
                    
            if flags.trigger_input:
                flags.trigger_input = False
                flags.debug = not flags.debug
                    
            if self.use_vae_sphere_posterior:
                task_out_proj = project_to_norm(task_out_proj, norm=self.embedding_norm, z_type="sphere")
                
            extra_dict = {"vae_mu": vae_mu, "vae_log_var": vae_log_var, "noise": self.z_noise}
            return task_out_proj, extra_dict
        
        
        def compute_prior(self, obs_dict):
            obs = obs_dict['obs']
            self_obs = obs[:, :self.self_obs_size]
            
            prior_latent = self.z_prior(self_obs)
            prior_mu = self.z_prior_mu(prior_latent)
            if self.use_vae_prior:
                prior_logvar = self.z_prior_logvar(prior_latent)
                if self.use_vae_clamped_prior:
                    prior_logvar = torch.clamp(prior_logvar, min = -5, max = self.vae_var_clamp_max)
                return prior_mu, prior_logvar
            elif self.use_vae_fixed_prior:
                if self.use_vae_sphere_prior:
                    return project_to_norm(prior_mu, z_type="sphere", norm = self.embedding_norm), torch.ones_like(prior_mu) * self.vae_prior_fixed_logvar
                else:
                    return prior_mu, torch.ones_like(prior_mu) * self.vae_prior_fixed_logvar
                    
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps * std, eps

        def eval_z(self, obs_dict):
            obs = obs_dict['obs']
            z_out = self.z_mlp(obs)
            z_out, extra_dict = self.form_embedding(z_out)
            return z_out, extra_dict
        
        def eval_critic(self, obs_dict):
            
            obs = obs_dict['obs']
            z_out = self.z_mlp(obs)
            z_out, extra_dict = self.form_embedding(z_out)

            obs = obs_dict['obs']

            self_obs = obs[:, :self.self_obs_size]
            assert (obs.shape[-1] == self.self_obs_size + self.task_obs_size)
            #### ZL: add CNN here
            
            if self.has_rnn:
                raise NotImplementedError
            else:   
                c_input = torch.cat([self_obs, z_out], dim=-1)
                c_out = self.critic_mlp(c_input)
                value = self.value_act(self.value(c_out))
                return value
            
        def eval_actor(self, obs_dict):
            obs = obs_dict['obs']
            z_out = self.z_mlp(obs)
            z_out, extra_dict = self.form_embedding(z_out)
            
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)

            self_obs = obs[:, :self.self_obs_size]
            assert (obs.shape[-1] == self.self_obs_size + self.task_obs_size)
            
            if self.has_rnn:
                raise NotImplementedError
            else:
                actor_input = torch.cat([self_obs, z_out], dim=-1)
                a_out = self.actor_mlp(actor_input)
                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))
                        
                    return mu, sigma

        def forward(self, obs_dict):
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs_dict)
            value_outputs = self.eval_critic(obs_dict)
            
            if self.has_rnn:
                raise NotImplementedError
            else:
                output = actor_outputs + (value_outputs, states)

            return output

        def _build_z_mlp(self):
            self_obs_size, task_obs_size = self.self_obs_size, self.task_obs_size
            
            out_size = self.embedding_size * 5

            mlp_input_shape = self_obs_size + task_obs_size  # target
            mlp_args = {'input_size': mlp_input_shape, 'units': self.vae_units, 'activation': self.vae_activation, 'dense_func': torch.nn.Linear}
            self.z_mlp = self._build_mlp(**mlp_args)
            
            if not self.has_rnn:
                self.z_mlp.append(nn.Linear(in_features=self.vae_units[-1], out_features=out_size))
            else:
                self.z_proj_linear = nn.Linear(in_features=self.rnn_units, out_features=out_size)
            
            mlp_init = self.init_factory.create(**self.vae_initializer)
            init_mlp(self.z_mlp, mlp_init)

            self.z_mu = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size)
            self.z_logvar = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size)
                
            init_mlp(self.z_mu, mlp_init)
            init_mlp(self.z_logvar, mlp_init)
                
            if self.use_vae_prior:
                mlp_args = {'input_size': self_obs_size, 'units': self.vae_units, 'activation': self.vae_activation, 'dense_func': torch.nn.Linear}
                self.z_prior = self._build_mlp(**mlp_args)
                self.z_prior_mu = nn.Linear(in_features=self.vae_units[-1], out_features=self.embedding_size)
                self.z_prior_logvar = nn.Linear(in_features=self.vae_units[-1], out_features=self.embedding_size)
                init_mlp(self.z_prior, mlp_init)
                init_mlp(self.z_prior_mu, mlp_init)
                init_mlp(self.z_prior_logvar, mlp_init)
            elif self.use_vae_fixed_prior:
                mlp_args = {'input_size': self_obs_size, 'units': self.vae_units, 'activation': self.vae_activation, 'dense_func': torch.nn.Linear}
                self.z_prior = self._build_mlp(**mlp_args)
                self.z_prior_mu = nn.Linear(in_features=self.vae_units[-1], out_features=self.embedding_size)
                init_mlp(self.z_prior, mlp_init)
                init_mlp(self.z_prior_mu, mlp_init)
            return
