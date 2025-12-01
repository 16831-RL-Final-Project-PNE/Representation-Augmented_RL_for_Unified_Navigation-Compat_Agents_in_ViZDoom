import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as td

class DenseModel(nn.Module):
    def __init__(
            self, 
            output_shape,
            input_size,
            layers = 3, 
            node_size = 100,
            dist = 'normal',
        ):
        """
        :param output_shape: tuple containing shape of expected output
        :param input_size: size of input features
        :param layers: number of hidden layers
        :param node_size: size of hidden layers
        :param dist: output distribution
        """
        super().__init__()
        self._output_shape = output_shape
        self._input_size = input_size
        self._layers = layers
        self._node_size = node_size
        self.dist = dist
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._input_size, self._node_size)]
        model += [nn.ELU()]
        for i in range(self._layers-1):
            model += [nn.Linear(self._node_size, self._node_size)]
            model += [nn.ELU()]
        out_dim = int(np.prod(self._output_shape))
        if self.dist == "normal":
            model += [nn.Linear(self._node_size, out_dim * 2)]
        else:
            model += [nn.Linear(self._node_size, out_dim)]
        return nn.Sequential(*model)

    def forward(self, input):
        dist_inputs = self.model(input)
        if self.dist == 'normal':
            mean, log_std = torch.chunk(dist_inputs, 2, dim=-1)
            std = F.softplus(log_std) + 1e-4
            return td.Independent(td.Normal(mean, std), len(self._output_shape))
        if self.dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=dist_inputs), len(self._output_shape))
        if self.dist == None:
            return dist_inputs

        raise NotImplementedError(self._dist)

class ObsEncoder(nn.Module):
    def __init__(self, input_shape, embedding_size, depth = 4, kernel = 8):
        """
        :param input_shape: tuple containing shape of input
        :param embedding_size: Supposed length of encoded vector
        """
        super(ObsEncoder, self).__init__()
        self.shape = input_shape
        self.kernel = kernel
        self.depth = depth
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_shape[0], self.depth, self.kernel),
            nn.ELU(),
            nn.Conv2d(self.depth, 2*self.depth, self.kernel),
            nn.ELU(),
            nn.Conv2d(2*self.depth, 4*self.depth, self.kernel),
            nn.ELU(),
        )
        
        if embedding_size == self.embed_size:
            self.fc_1 = nn.Identity()
        else:
            self.fc_1 = nn.Linear(self.embed_size, embedding_size)

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        embed = self.fc_1(embed)
        return embed

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 0, self.kernel, 1)
        conv2_shape = conv_out_shape(conv1_shape, 0, self.kernel, 1)
        conv3_shape = conv_out_shape(conv2_shape, 0, self.kernel, 1)
        embed_size = int(4*self.depth*np.prod(conv3_shape).item())
        return embed_size

class ObsDecoder(nn.Module):
    def __init__(self, output_shape, embed_size, depth = 4, kernel = 8):
        """
        :param output_shape: tuple containing shape of output obs (c, h, w)
        :param embed_size: the size of input vector, for dreamerv2 : modelstate 
        """
        super(ObsDecoder, self).__init__()
        c, h, w = output_shape
        conv1_shape = conv_out_shape(output_shape[1:], 0, kernel, 1)
        conv2_shape = conv_out_shape(conv1_shape, 0, kernel, 1)
        conv3_shape = conv_out_shape(conv2_shape, 0, kernel, 1)
        self.conv_shape = (4*depth, *conv3_shape)
        self.output_shape = output_shape
        if embed_size == np.prod(self.conv_shape).item():
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(embed_size, np.prod(self.conv_shape).item())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4*depth, 2*depth, kernel, 1),
            nn.ELU(),
            nn.ConvTranspose2d(2*depth, depth, kernel, 1),
            nn.ELU(),
            nn.ConvTranspose2d(depth, 2*c, kernel, 1),
        )

    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.decoder(x)

        # Split channels into mean and log_std
        c = self.output_shape[0]
        mean, log_std = torch.split(x, c, dim=1)

        # Mean in [0,1]
        mean = torch.sigmoid(mean)

        # std = softplus(log_std) + epsilon
        std = F.softplus(log_std) + 1e-4

        # reshape
        mean = mean.reshape(*batch_shape, *self.output_shape)
        std  = std.reshape(*batch_shape, *self.output_shape)

        obs_dist = td.Independent(td.Normal(mean, std), len(self.output_shape))
        return obs_dist
    
def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))
