"""
    Neural Network Encoders. 
    
    Defines the following classes:
        - Encoder: the encoder network, a feedforward neural network that given the input X, outputs MEAN for the latent distibution.
        - Encoder_full: as encoder, but also outputs STDDEV for the latent distribution.

    Author: De-identified Author
"""

import torch


class Encoder(torch.nn.Module):
    """
    Adopted from https://jmtomczak.github.io/blog/4/4_VAE.html
    """

    def __init__(self, encoder_net, inDim, outDim):
        """
        A constructor for the encoder network.
        Args:
            encoder_net: the encoder network (torch.nn.seuqential)
            inDim: the input dimension of the encoder network
            outDim: the output dimension of the encoder network
        """
        super(Encoder, self).__init__()
        self.encoder = encoder_net
        self.inDim = inDim
        self.outDim = outDim

    def get_simple_encoder(D, L):
        """
        encoder: X \in R^{D} -> Z \in R^{L}
        where L is the latent dimension and we want 2*L, one for the mean and one for the (log_variance)
        NB: assuming a fixed variance... (so the last layer is L by L)
        """
        encoder_net = torch.nn.Sequential(
            torch.nn.Linear(D, L),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(L, L),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(L, L),
        )

        return Encoder(encoder_net=encoder_net, inDim=D, outDim=L)

    @staticmethod
    def reparameterization(mu, log_var):
        "The reparameterization trick for Gaussians."
        # z = mu + std * epsilon
        # epsilon ~ Normal(0,1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + std * eps

    def encode(self, x):
        """This function implements the output of the encoder network (i.e., parameters of a Gaussian)."""
        # First, we calculate the output of the encoder network of size 2M.
        h_e = self.encoder(x)
        # # Second, we must divide the output to the mean and the log-variance.
        # mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)

        # return mu_e, log_var_e
        return h_e

    # Sampling procedure.
    def sample(self, x=None, mu_e=None, log_var_e=None):
        raise NotImplementedError("This function must be implemented in the child class!")
        # If we don't provide a mean and a log-variance, we must first calcuate it:
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.encode(x)
        # Or the final sample
        else:
            # Otherwise, we can simply apply the reparameterization trick!
            if (mu_e is None) or (log_var_e is None):
                raise ValueError("mu and log-var can`t be None!")
        z = self.reparameterization(mu_e, log_var_e)
        return z

    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        """This function calculates the log-probability that is later used for calculating the ELBO."""
        raise NotImplementedError("This function must be implemented in the child class!")
        def log_Normal_diag(x, mean, log_var, average=False, dim=None):
            log_normal = -0.5 * (
                log_var + torch.pow(x - mean, 2) * torch.pow(torch.exp(log_var), -1)
            )
            if average:
                return torch.mean(log_normal, dim)
            else:
                return torch.sum(log_normal, dim)

        # If we provide x alone, then we can calculate a corresponsing sample:
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        else:
            # Otherwise, we should provide mu, log-var and z!
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError("mu, log-var and z can`t be None!")

        return log_normal_diag(z, mu_e, log_var_e)

    def forward(self, x, type="log_prob"):
        """
        PyTorch forward pass: it is either log-probability (by default) or sampling.
        """
        assert type in ["encode", "log_prob"], "Type could be either encode or log_prob"
        if type == "log_prob":
            return self.log_prob(x)
        else:
            return self.sample(x)

    def size(self):
        """Returns the tupple (inDim, outDim)"""
        return self.inDim, self.outDim


class Encoder_full(Encoder):
    """
    Adopted from https://jmtomczak.github.io/blog/4/4_VAE.html
    and here: https://github.com/jmtomczak/intro_dgm/blob/5715618f75db120229c3a9b57f964900c0210fc7/ddgms/ddgm_example.ipynb
    """

    @classmethod
    def get_simple_encoder(cls, D, L):
        """
        encoder: X \in R^{D} -> Z \in R^{L}\times R^{L}
        where L is the latent dimension and we want 2*L, one for the mean and one for the log_variance

        Args:
            D: the input dimension (data_dim)
            L: the output dimension (latent_dim)

        Returns:
            encoder: An initialized Encoder class
        """
        encoder_net = torch.nn.Sequential(
            torch.nn.Linear(D, L),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(L, L),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(L, 2 * L),
        )

        return cls(encoder_net=encoder_net, inDim=D, outDim=L)

    def encode(self, x):
        """This function implements the output of the encoder network (i.e., parameters of a Gaussian).
        Args:
            x: the input data (torch.tensor) [batch_size, inDim]

        Returns:
            mu_e:  (torch.tensor) [batch_size, outDim]
            log_var_e: softplus(log_var_e) (torch.tensor) [batch_size, outDim]
        """
        # First, we calculate the output of the encoder network of size 2M.
        h_e = self.encoder(x)

        # Second, we must divide the output to the mean and the log-variance.
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        var_e = torch.nn.functional.softplus(log_var_e)
        return mu_e, var_e




class Encoder_full_transformed(Encoder_full):
    """
    Additionally batch and log normalizes its inputs
    """

    def weights_init(self, m):
        pass
        print(m.__class__.__name__)
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # print(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            # elif isinstance(m, torch.nn.BatchNorm1d):
            #     torch.nn.init.ones_(m.weight)
            #     torch.nn.init.zeros_(m.bias)


    def __init__(self, encoder_net, inDim, outDim, targetLibSize, is_natural=False, init_params=False, device=None):
        super().__init__(encoder_net=encoder_net, inDim=inDim, outDim=outDim)
        self.targetLibSize = targetLibSize
        self.is_natural = is_natural
        self.device = device
        if init_params:
            encoder_net.apply(self.weights_init)
        

    @classmethod
    def get_simple_encoder(cls, D, L, targetLibSize, is_natural=False, init_params=False, device=None):
        """
        encoder: X \in R^{D} -> Z \in R^{L}\times R^{L}
        where L is the latent dimension and we want 2*L, one for the mean and one for the log_variance

        Args:
            D: the input dimension (data_dim)
            L: the output dimension (latent_dim)
            targetLibSize: the size of the target library (used for batch normalization)

        Returns:
            encoder: An initialized Encoder class
        """
        dropout_rate = .1
        # adding a dropout layer
        encoder_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(D, track_running_stats=False),
            torch.nn.Linear(D, L),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(L, L),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(L, 2*L),
        )
        # encoder_net = torch.nn.Sequential(
        #     torch.nn.BatchNorm1d(D, track_running_stats=False),
        #     torch.nn.Linear(D, L),
        #     #torch.nn.LeakyReLU(),
        #     torch.nn.SELU(),
        #     #torch.nn.BatchNorm1d(L, track_running_stats=False),
        #     torch.nn.Linear(L, L),
        #     #torch.nn.LeakyReLU(),
        #     torch.nn.SELU(),
        #     #torch.nn.BatchNorm1d(L, track_running_stats=False),
        #     torch.nn.Linear(L, 2*L),
        # encoder_net = torch.nn.Sequential(
        #     torch.nn.Linear(D, L),
        #     torch.nn.BatchNorm1d(L),
        #     torch.nn.LeakyReLU(),
        #     #torch.nn.Dropout(dropout_rate),
        #     torch.nn.Linear(L, L),
        #     #torch.nn.BatchNorm1d(L),
        #     torch.nn.LeakyReLU(),
        #     #torch.nn.Dropout(dropout_rate),
        #     torch.nn.Linear(L, 2 * L),
        #     # torch.nn.BatchNorm1d(2 * L)
        #     # If you need an activation function after the last layer, add it here
        # )
        # add regularization
    
        return cls(encoder_net=encoder_net, inDim=D, outDim=L, targetLibSize=targetLibSize, is_natural=is_natural, init_params=init_params, device=device)

    def encode(self, x):
        """This function implements the output of the encoder network (i.e., parameters of a Gaussian).
        Args:
            x: the input data (torch.tensor) [batch_size, inDim]

        Returns:
            mu_e:  (torch.tensor) [batch_size, outDim]
            log_var_e: softplus(log_var_e) (torch.tensor) [batch_size, outDim]
        """
        # First, we calculate the output of the encoder network of size 2M.
        # Transforms the input to log-space
        # TODO: this is not memory efficient
        #x_in = x / 1e4 # SOMEWHAT WORKING SOLUTION
        #x_in = torch.clamp_max(x, 400)
        #x_in = 10*(x / x.sum(dim=1, keepdim=True))
        #x_in = (x / x.sum(dim=1, keepdim=True))
        #x_in = torch.log1p(x/500)
        # Normalize the features
        # x_in.std(axis=0).sum()
        # x_std = x_in.std(axis=0)
        # x_std[x_std == 0] = 1.
        # x_in = (x_in - x_in.mean(axis=0)) / x_std
        # assert torch.isnan(x_in).sum() == 0, "x_in contains NaNs"

        # Something that was being done:
        # x_in = torch.log1p(self.targetLibSize*(x / (1 + x.sum(dim=1, keepdim=True))))

        # New idea: just log1p normalize the input
        x_in = torch.log1p(x)
        #x_in = x

        h_e = self.encoder(x_in.double().to(self.device))

        # Second, we must divide the output to the mean and the log-variance.
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)

        if self.is_natural is False:
            var_e = torch.nn.functional.softplus(log_var_e)
        else:
            var_e = log_var_e
        return mu_e, var_e
