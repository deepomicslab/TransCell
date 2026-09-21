"""TransCell model interface for cross-species expression translation."""

import numpy as np
import torch
from torch import nn

from .nn_blocks import SharedCellStateEncoder, ZINBExpressionDecoder


class TransCellModel(nn.Module):
    """Ortholog-aware cross-species expression translator."""

    def __init__(self, encoder_config, decoder_config, reference_species_id, n_species):
        super().__init__()
        input_dim_by_decoder = {
            decoder_id: decoder_config[decoder_id][-1][1] for decoder_id in decoder_config.keys()
        }
        self.latent_dim = encoder_config[-1][1]
        self.encoder = SharedCellStateEncoder(
            input_dim_by_decoder,
            encoder_config,
            translation_mode="h",
        )
        self.decoder = ZINBExpressionDecoder(self.latent_dim, decoder_config)
        self.reference_species_id = reference_species_id
        self.n_species = n_species

    def load_model(self, path):
        pretrained = torch.load(path, map_location=lambda storage, loc: storage)
        current = self.state_dict()
        pretrained = {key: value for key, value in pretrained.items() if key in current}
        current.update(pretrained)
        self.load_state_dict(current)

    def encode_cell_state(self, expression, species_id=None):
        return self.encoder(expression, 0, species_id)

    def decode_expression(self, latent_state, decoder_id, species_id=None):
        return self.decoder(latent_state, decoder_id, species_id)

    @staticmethod
    def expected_expression(px_scale, px_r, expression):
        libsize = torch.sum(expression, dim=1)
        px_r = torch.clamp(px_r, min=-2000000, max=15)
        px_r = torch.exp(px_r)
        reconstructed = torch.t(px_scale.t() * libsize)
        return reconstructed, px_r

    def embed(self, dataloader, num_gene, device="cuda", eval_mode=False):
        self.to(device)
        self.eval() if eval_mode else self.train()
        output = np.zeros((dataloader.dataset.shape[0], self.latent_dim))
        common_gene_count = num_gene[self.n_species]
        for batch in dataloader:
            x, species_id, index, _, _ = batch
            x_common = x[:, 0:common_gene_count].float().to(device)
            _, mu, _ = self.encode_cell_state(x_common)
            output[index] = mu.detach().cpu().numpy()
        return output

    def predict_target_expression(
        self,
        dataloader,
        num_gene,
        target_decoder_id=1,
        device="cuda",
        eval_mode=True,
    ):
        self.to(device)
        self.eval() if eval_mode else self.train()
        output = np.zeros((dataloader.dataset.shape[0], num_gene[target_decoder_id]))
        common_gene_count = num_gene[self.n_species]
        for batch in dataloader:
            x, species_id, index, _, _ = batch
            x_common = x[:, 0:common_gene_count].float().to(device)
            _, mu, _ = self.encode_cell_state(x_common)
            px_scale, _, px_r = self.decode_expression(mu, target_decoder_id + 1)
            reconstruction, _ = self.expected_expression(px_scale, px_r, x_common)
            output[index] = reconstruction.detach().cpu().numpy()
        return output

