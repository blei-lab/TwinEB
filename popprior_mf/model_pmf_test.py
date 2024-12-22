""""
    The Probablistic Matrix Factorization class with amortized variational infernce.
"""


import numpy as np
import torch
import os
from model_encoders import Encoder_full, Encoder_full_transformed
from model_pmf import PMF
from model_ppca_amortized import AmortizedVariationalFamily_full


class AmortizedPMFTest(PMF):
    """A poisson matrix factorization class with amortized variational inference."""

    def __init__(
        self,
        device,
        num_datapoints,
        data_dim,
        latent_dim,
        num_samples,
        print_steps,
        summary_writer,
        annealing_factor=1.0,
        targetLibSize=1e3,
    ):
        self.targetLibSize = targetLibSize
        super().__init__(
            device=device,
            num_datapoints=num_datapoints,
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_samples=num_samples,
            print_steps=print_steps,
            summary_writer=summary_writer,
            annealing_factor=annealing_factor,
        )
        
        
    def _setup_column_vars(self, init_loc=None):
        #self.V = torch.nn.Parameter(torch.ones((self.data_dim, self.latent_dim)))
        self.V = torch.nn.init.xavier_uniform_(torch.nn.Parameter(torch.ones((self.data_dim, self.latent_dim))))
        # loc = torch.from_numpy(np.loadtxt('/Users/De-identified Authors/projects/rnaseq-pfm/results/rna_small_XOIQD_202208-22-161551.254303PMF/qv_loc'))
        # scale = torch.from_numpy(np.loadtxt('/Users/De-identified Authors/projects/rnaseq-pfm/results/rna_small_XOIQD_202208-22-161551.254303PMF/qv_scale'))
        # scale = torch.clamp(torch.nn.functional.softplus(scale), min=1e-8)
        # self.V = torch.distributions.LogNormal(loc, scale).sample()
        

    def _setup_row_vars(self, init_loc=None):
        self.qu_distribution = AmortizedVariationalFamily_full(
            device=self.device,
            family="lognormal",
            shape=[self.latent_dim, self.num_datapoints],
            encoder=Encoder_full_transformed.get_simple_encoder(D=self.data_dim, L=self.latent_dim, targetLibSize=self.targetLibSize),
            nugget=0.0,
        )

    def _get_v(self, min_v=1e-6):
        # If it is coming from the log normal distribution
        #return torch.clamp_min(torch.nn.functional.softplus(self.V), min_v)
        return torch.exp(self.V)
        #return torch.nn.functional.softplus(self.V)
        return self.V

    def get_samples(self, x, datapoints_indices):
        """Return samples from variational distributions."""
        #v_samples = self.qv_distribution.sample(self.num_samples)
        u_samples = self.qu_distribution.sample(self.num_samples, x)
        
        #assert v_samples.min() > 0.0, "v_samples must be positive."
        assert u_samples.min() > 0.0, "u_samples must be positive."
        assert u_samples.shape[2] == datapoints_indices.shape[0]

        # u_samples = torch.clamp_min(u_samples, 1e-6)
        #v_samples = torch.clamp_min(v_samples, 1e-6)

        #return [v_samples, u_samples]
        return [None, u_samples]

    def get_entropy(self, samples, x, datapoints_indices):
        """Calculate entropy of variational samples.

        Args:
            samples: A list of samples. The length of the list is the number of variables being sampled.

        Returns:
          log_prior: A Monte-Carlo approximation of the variational entropy,
            summed across latent dimensions and averaged over the number of
            samples.
        """
        (v_samples, u_samples) = samples
        v_entropy = self.qv_distribution.get_entropy(v_samples)
        u_entropy = self.qu_distribution.get_entropy(u_samples, x)
        # Adjust for minibatch
        batch_size = u_samples.shape[-1]
        u_entropy = u_entropy * (self.num_datapoints / batch_size)
        entropy = v_entropy + u_entropy
        return torch.mean(entropy)

    def get_data_log_likelihood(self, samples, counts, holdout_mask):
        """Approximate log-likelihood term of ELBO using Monte Carlo samples.

        Args:
          samples: A list of samples. The length of the list is the number of variables being sampled.
          counts: A float-tensor with shape [batch_size, num_words].
          holdout_mask: A binary tensor with shape [batch_size, data_dim]. 1=valid, 0=train


        Returns:
          data_log_likelihood: A Monte-Carlo approximation of the count
            log-likelihood, summed across latent dimensions and averaged over the
            number of samples.

        """
        #(v_samples, u_samples) = samples
        (_, u_samples) = samples
        rate = torch.transpose(torch.matmul(self._get_v(), u_samples), 1, 2)
        data_distribution = self.distribution(rate=rate)
        data_log_likelihood = data_distribution.log_prob(counts)
        data_log_likelihood = torch.sum(torch.mul(data_log_likelihood, 1 - holdout_mask), dim=tuple(range(1, data_log_likelihood.dim())))
        # Adjust for minibatch
        batch_size = len(counts)
        data_log_likelihood = data_log_likelihood * (self.num_datapoints / batch_size)
        return torch.mean(data_log_likelihood)

    def forward(self, datapoints_indices, data, holdout_mask, step):
        """Approximate variational Lognormal ELBO using reparameterization.

        Args:
            datapoints_indices: An int-vector with shape [batch_size].
            data: A matrix with shape `[batch_size, num_words]`.
            step: The training step, used to log summaries to Tensorboard.

        Returns:
            elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value
              is averaged across samples and summed across batches.
        """
        samples = self.get_samples(data, datapoints_indices)
        #log_prior = self.get_log_prior(samples)
        data_log_likelihood = self.get_data_log_likelihood(samples, data, holdout_mask)
        elbo = data_log_likelihood
        #entropy = self.get_entropy(samples, data, datapoints_indices)
        #neg_kl_divergence = log_prior + entropy
        #elbo = data_log_likelihood + neg_kl_divergence * self.annealing_factor
        #self.log_elbo(step, elbo, entropy, log_prior, data_log_likelihood, neg_kl_divergence)
        if step % self.print_steps == 0:
            self.summary_writer.add_scalar("elbo/elbo", elbo, step)

        return elbo

    def generate_data(self, n_samples=1, x=None, datapoints_indices=None):
        """
        Generate observations from the model.
        """
        obs = []
        for j in range(n_samples):
            #v_samples = self.qv_distribution.sample(1) 
            u_samples = torch.squeeze(self.qu_distribution.sample(1, x), 0)
            mean_sample = torch.transpose(
                #torch.squeeze(torch.matmul(v_samples, u_samples)), 0, 1
                torch.squeeze(torch.matmul(self._get_v(), u_samples)), 0, 1
            )
            x_gen = self.distribution(mean_sample).sample()
            obs.append(x_gen)

        return obs

    @torch.no_grad()
    def save_txt(self, param_save_dir, **kwargs):
        """Save the model params to file

        Args:
            param_save_dir: The validation data matrix with shape `[num_datapoints, data_dim]`.

        Returns:
            None
        """
        # np.savetxt(
        #     os.path.join(param_save_dir, "qv_loc"),
        #     self.qv_distribution.location.cpu().detach(),
        # )
        np.savetxt(os.path.join(param_save_dir, "qv_loc"), self.V)
        # np.savetxt(
        #     os.path.join(param_save_dir, "qv_scale"),
        #     self.qv_distribution.scale().cpu().detach(),
        # )

        # Save u by encoding the train data
        qu_loc, qu_scale = self.qu_distribution.encoder.encode(
            torch.from_numpy(kwargs["train_data"].astype(np.double))
        )
        np.savetxt(os.path.join(param_save_dir, "qu_loc"), qu_loc.cpu().detach())
        np.savetxt(os.path.join(param_save_dir, "qu_scale"), qu_scale.cpu().detach())

    def compute_llhood(self, x_vad, holdout_mask):
        """
        Compute loglikelihood for the given data, taking the mask into account.
        logP(X_vad | Z@W)

        Args:
            x_vad: A matrix with shape [num_datapoints, data_dim].
            holdout_mask: A binary tensor with shape [num_datapoints, data_dim]. 1=valid, 0=train

        Returns:
            log_ll: A tensor of shape [num_datapoints]
        """
        #v_samples = self.qv_distribution.sample(1)
        u_samples = torch.squeeze(self.qu_distribution.sample(1), 0)
        holdoutmean_sample = torch.transpose(
            torch.squeeze(torch.matmul(self._get_v(), u_samples)), 0, 1
        )
        holdoutmean_sample = holdoutmean_sample * holdout_mask
        # log_ll is [num_datapoints, data_dim]
        log_ll = self.distribution(holdoutmean_sample).log_prob(x_vad)
        # Remove the heldin llhoods
        log_ll = log_ll * holdout_mask
        log_ll = torch.sum(log_ll, dim=1)
        return log_ll

