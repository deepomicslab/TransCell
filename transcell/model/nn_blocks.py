"""Neural building blocks for TransCell expression translation."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


ACTIVATIONS = {
    "relu": nn.ReLU(),
    "rrelu": nn.RReLU(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
    "tanh": nn.Tanh(),
    "softmax": nn.Softmax(dim=1),
    "softplus": nn.Softplus(),
    "": None,
}


class SpeciesBatchNorm(nn.Module):
    """Batch normalization with separate statistics for each species/domain."""

    def __init__(self, num_features, n_species, eps=1e-5, momentum=0.1):
        super().__init__()
        self.n_species = n_species
        self.num_features = num_features
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for _ in range(n_species)]
        )

    def reset_running_stats(self):
        for norm in self.norms:
            norm.reset_running_stats()

    def reset_parameters(self):
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, species_id):
        output = torch.zeros(x.size(0), self.num_features, device=x.device)
        species_np = species_id.detach().cpu().numpy()
        for species in range(self.n_species):
            indices = np.where(species_np == species)[0]
            if len(indices) > 1:
                output[indices] = self.norms[species](x[indices])
            elif len(indices) == 1:
                self.norms[species].training = False
                output[indices] = self.norms[species](x[indices])
                self.norms[species].training = True
        return output


class DenseExpressionBlock(nn.Module):
    """Linear layer with optional species-aware normalization, activation, and dropout."""

    def __init__(self, input_dim, output_dim, norm=None, activation="", dropout=0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        if isinstance(norm, int):
            self.norm = nn.BatchNorm1d(output_dim) if norm == 1 else SpeciesBatchNorm(output_dim, norm)
        else:
            self.norm = None
        self.activation = ACTIVATIONS[activation]
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, species_id=None):
        h = self.linear(x)
        if self.norm and len(x) > 1:
            if isinstance(self.norm, SpeciesBatchNorm):
                h = self.norm(h, species_id)
            else:
                h = self.norm(h)
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        return h


class ExpressionMLP(nn.Module):
    """Configurable MLP used by TransCell encoders and decoders."""

    def __init__(self, input_dim, config):
        super().__init__()
        layers = []
        current_dim = input_dim
        for layer in config:
            if layer[0] == "fc":
                layers.append(DenseExpressionBlock(current_dim, *layer[1:]))
            current_dim = layer[1]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, species_id=None):
        for layer in self.layers:
            x = layer(x, species_id)
        return x


class SharedCellStateEncoder(nn.Module):
    """Encode expression into a shared latent cell-state distribution."""

    def __init__(self, input_dim_by_decoder, config, translation_mode="h"):
        super().__init__()
        encoders = []
        mu_heads = []
        var_heads = []
        hidden_dim = config[-2][1]

        if translation_mode == "d":
            keys = input_dim_by_decoder.keys()
        else:
            keys = [0]

        for key in keys:
            encoders.append(ExpressionMLP(input_dim_by_decoder[key], config[:-1]))
            mu_heads.append(ExpressionMLP(hidden_dim, config[-1:]))
            var_heads.append(ExpressionMLP(hidden_dim, config[-1:]))

        self.encoders = nn.ModuleList(encoders)
        self.mu_heads = nn.ModuleList(mu_heads)
        self.var_heads = nn.ModuleList(var_heads)

    @staticmethod
    def reparameterize(mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, expression, encoder_id=0, species_id=None):
        hidden = self.encoders[encoder_id](expression, species_id)
        mu = self.mu_heads[encoder_id](hidden, species_id)
        var = self.var_heads[encoder_id](hidden, species_id)
        var = torch.clamp(var, min=-2000000, max=15)
        var = torch.exp(var) + 0.0001
        z = self.reparameterize(mu, var)
        return z, mu, var


class ZINBExpressionDecoder(nn.Module):
    """Decode latent cell states into ZINB expression parameters."""

    def __init__(self, latent_dim, decoder_config):
        super().__init__()
        trunk = []
        px_scale = []
        px_dropout = []
        px_r = []
        for decoder_id in decoder_config.keys():
            trunk.append(ExpressionMLP(latent_dim, decoder_config[decoder_id][:-1]))
            output_dim = decoder_config[decoder_id][-1][1]
            norm = decoder_config[decoder_id][0][2]
            px_scale.append(ExpressionMLP(256, [["fc", output_dim, norm, "softmax", 0]]))
            px_dropout.append(ExpressionMLP(256, [["fc", output_dim, norm, "", 0]]))
            px_r.append(ExpressionMLP(256, [["fc", output_dim, norm, "", 0]]))

        self.trunk = nn.ModuleList(trunk)
        self.px_scale = nn.ModuleList(px_scale)
        self.px_dropout = nn.ModuleList(px_dropout)
        self.px_r = nn.ModuleList(px_r)

    def forward(self, latent_state, decoder_id, species_id=None):
        hidden = self.trunk[decoder_id](latent_state, species_id)
        scale = self.px_scale[decoder_id](hidden, species_id)
        dropout = self.px_dropout[decoder_id](hidden, species_id)
        dispersion = self.px_r[decoder_id](hidden, species_id)
        return scale, dropout, dispersion


ExpressionEncoder = SharedCellStateEncoder
SpeciesSpecificDecoderHead = ZINBExpressionDecoder
MLP = ExpressionMLP
DenseBlock = DenseExpressionBlock

__all__ = [
    "DenseBlock",
    "DenseExpressionBlock",
    "ExpressionEncoder",
    "ExpressionMLP",
    "MLP",
    "SharedCellStateEncoder",
    "SpeciesBatchNorm",
    "SpeciesSpecificDecoderHead",
    "ZINBExpressionDecoder",
]
