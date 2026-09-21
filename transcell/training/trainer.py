"""Trainer for TransCell cross-species translation."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from .objectives import (
    HierarchicalCellStateLoss,
    LatentKLFidelityLoss,
    UOTAlignmentObjective,
    ZINBReconstructionLoss,
)


@dataclass
class TransCellTrainer:
    """Train TransCell in the source-to-target common-ortholog setting."""

    model: object
    config: object

    def __post_init__(self):
        self.zinb_loss = ZINBReconstructionLoss()
        self.kl_loss = LatentKLFidelityLoss()
        self.uot_objective = UOTAlignmentObjective(
            entropy=self.config.ot_entropy,
            marginal_relaxation=self.config.ot_marginal_relaxation,
        )
        self.hierarchy_loss = HierarchicalCellStateLoss()

    def fit(
        self,
        dataloader,
        transport_plan,
        num_cell,
        num_gene,
        prior=None,
        label_weight=None,
        early_stopping=None,
        device="cpu",
        save_transport=False,
        verbose=False,
        max_gene=2000,
    ):
        self.model.to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=5e-4,
        )
        n_epoch = int(np.ceil(self.config.iteration / len(dataloader)))
        with tqdm(range(n_epoch), total=n_epoch, desc="Epochs") as epoch_bar:
            for epoch in epoch_bar:
                epoch_loss = self.train_epoch(
                    dataloader=dataloader,
                    optimizer=optimizer,
                    transport_plan=transport_plan,
                    num_cell=num_cell,
                    num_gene=num_gene,
                    prior=prior,
                    label_weight=label_weight,
                    device=device,
                    save_transport=save_transport,
                    verbose=verbose,
                    max_gene=max_gene,
                )
                epoch_info = ",".join(
                    ["{}={:.2f}".format(key, value) for key, value in epoch_loss.items()]
                )
                epoch_bar.set_postfix_str(epoch_info)
                if early_stopping is not None:
                    early_stopping(sum(epoch_loss.values()), self.model, metrics=epoch_loss)
                    if early_stopping.early_stop:
                        print("EarlyStopping: run {} epoch".format(epoch + 1))
                        break

    def train_epoch(
        self,
        dataloader,
        optimizer,
        transport_plan,
        num_cell,
        num_gene,
        prior=None,
        label_weight=None,
        device="cpu",
        save_transport=False,
        verbose=False,
        max_gene=2000,
    ):
        self.model.train()
        n_species = self.model.n_species
        reference_id = self.model.reference_species_id
        common_gene_count = num_gene[n_species]
        progress = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            leave=False,
            desc="Iterations",
            disable=(not verbose),
        )
        epoch_loss = defaultdict(float)
        n_batches = 0

        for batch_id, batch in progress:
            x, species_id, sample_index, cell_type_primary, cell_state = batch
            x_common = x[:, 0:common_gene_count].float().to(device)
            species_id = species_id.long().to(device)
            sample_index = sample_index.to(device)
            cell_type_primary = cell_type_primary.int().to(device)
            cell_state = cell_state.int().to(device)

            loc_ref = torch.where(species_id == reference_id)[0]
            idx_ref = sample_index[loc_ref] - sum(num_cell[0:reference_id])
            query_ids = list(range(n_species))
            query_ids.remove(reference_id)

            loc_by_species = {reference_id: loc_ref}
            idx_by_species = {reference_id: idx_ref}
            transport_batch = {}
            prior_batch = None
            if len(loc_ref) > 0:
                for query_id in query_ids:
                    loc_query = torch.where(species_id == query_id)[0]
                    idx_query = sample_index[loc_query] - sum(num_cell[0:query_id])
                    loc_by_species[query_id] = loc_query
                    idx_by_species[query_id] = idx_query
                    if save_transport and len(idx_query) > 0:
                        base_plan = torch.from_numpy(transport_plan[query_id]).to(device)
                        transport_batch[query_id] = base_plan[idx_query][:, idx_ref]
                    else:
                        transport_batch[query_id] = None
                    if prior is not None:
                        prior_matrix = prior[query_id] if isinstance(prior, (dict, list, tuple)) else prior
                        prior_batch = prior_matrix[
                            idx_query.detach().cpu()
                        ][:, idx_ref.detach().cpu()].to(device)

            z, mu, var = self.model.encode_cell_state(x_common)
            px_scale, px_dropout, px_r_raw = self.model.decode_expression(z, 0, species_id)
            reconstruction, px_r = self.model.expected_expression(px_scale, px_r_raw, x_common)

            reconstruction_loss = self.zinb_loss(
                px_dropout,
                px_r,
                px_scale,
                x_common,
                reconstruction,
            ) * 2000
            kl_loss = self.kl_loss(mu, var)
            ot_loss = torch.tensor(0.0, device=device)
            hierarchy_loss = torch.tensor(0.0, device=device)

            if max_gene is not None:
                x_specific = x[:, common_gene_count:].float().to(device)
                hierarchy_input = torch.zeros((len(cell_type_primary), max_gene), device=device)
                for decoder_species in range(n_species):
                    loc = loc_by_species.get(decoder_species)
                    if loc is None or len(loc) == 0:
                        continue
                    px_scale_s, px_dropout_s, px_r_s_raw = self.model.decode_expression(
                        z[loc],
                        decoder_species + 1,
                    )
                    lib_x_s = x_specific[loc][:, 0:num_gene[decoder_species]]
                    recon_s, px_r_s = self.model.expected_expression(
                        px_scale_s,
                        px_r_s_raw,
                        lib_x_s,
                    )
                    px_scale_s_mean, _, px_r_s_mean_raw = self.model.decode_expression(
                        mu[loc],
                        decoder_species + 1,
                    )
                    recon_s_mean, _ = self.model.expected_expression(
                        px_scale_s_mean,
                        px_r_s_mean_raw,
                        lib_x_s,
                    )
                    if recon_s_mean.shape[1] < max_gene:
                        padding = torch.zeros(
                            (recon_s_mean.shape[0], max_gene - recon_s_mean.shape[1]),
                            device=device,
                        )
                        recon_s_mean = torch.cat([recon_s_mean, padding], dim=1)
                    hierarchy_input[loc] = recon_s_mean
                    reconstruction_loss = reconstruction_loss + (
                        self.config.species_specific_reconstruction
                        * self.zinb_loss(px_dropout_s, px_r_s, px_scale_s, lib_x_s, recon_s)
                        * 2000
                    )
                hierarchy_loss = self.hierarchy_loss(
                    hierarchy_input,
                    cell_type_primary,
                    cell_state,
                )

            if len(torch.unique(species_id)) > 1 and len(loc_ref) != 0:
                mu_by_species = {}
                var_by_species = {}
                for species in range(n_species):
                    loc = loc_by_species.get(species)
                    if loc is not None and len(loc) > 0:
                        mu_by_species[species] = mu[loc]
                        var_by_species[species] = var[loc]

                for query_id in query_ids:
                    loc = loc_by_species.get(query_id)
                    if loc is None or len(loc) == 0:
                        continue
                    ot_tmp, transport_batch[query_id] = self.uot_objective(
                        transport_batch[query_id],
                        mu_by_species[query_id],
                        var_by_species[query_id],
                        mu_by_species[reference_id].detach(),
                        var_by_species[reference_id].detach(),
                        prior=prior_batch,
                        device=device,
                        idx_q=idx_by_species[query_id],
                        idx_r=idx_ref,
                        query_weight=label_weight[query_id] if label_weight is not None else None,
                        ref_weight=label_weight[reference_id] if label_weight is not None else None,
                    )
                    if save_transport:
                        t0 = np.repeat(
                            idx_by_species[query_id].detach().cpu().numpy(),
                            len(idx_ref),
                        ).reshape(len(idx_by_species[query_id]), len(idx_ref))
                        t1 = np.tile(idx_ref.detach().cpu().numpy(), (len(idx_by_species[query_id]), 1))
                        transport_plan[query_id][t0, t1] = transport_batch[query_id].cpu().numpy()
                    ot_loss = ot_loss + ot_tmp

            losses = {
                "recloss": self.config.reconstruction_fidelity * reconstruction_loss,
                "klloss": self.config.kl_fidelity * kl_loss,
                "otloss": self.config.ot_alignment * ot_loss,
                "loss_hierarchy": 0.7 * hierarchy_loss,
            }

            optimizer.zero_grad()
            sum(losses.values()).backward()
            optimizer.step()

            for key, value in losses.items():
                epoch_loss[key] += value.item()
            n_batches += 1
            progress.set_postfix_str(
                ",".join(["{}={:.2f}".format(key, value) for key, value in losses.items()])
            )

        if n_batches == 0:
            return {key: 0.0 for key in ("recloss", "klloss", "otloss", "loss_hierarchy")}
        return {key: value / n_batches for key, value in epoch_loss.items()}
