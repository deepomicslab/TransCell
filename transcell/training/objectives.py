"""Training objectives used by TransCell."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class LatentKLFidelityLoss(nn.Module):
    """KL regularization for latent cell-state fidelity."""

    def forward(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return kl_divergence(
            Normal(mu, var.sqrt()),
            Normal(torch.zeros_like(mu), torch.ones_like(var)),
        ).sum(dim=1).mean()


class GaussianWassersteinCost(nn.Module):
    """Pairwise diagonal-Gaussian Wasserstein cost."""

    def forward(
        self,
        mu_src: torch.Tensor,
        var_src: torch.Tensor,
        mu_tgt: torch.Tensor,
        var_tgt: torch.Tensor,
    ) -> torch.Tensor:
        mean_cost = self._distance_matrix(mu_src, mu_tgt)
        var_cost = self._distance_matrix(var_src.sqrt(), var_tgt.sqrt())
        return mean_cost + var_cost + 1e-6

    @staticmethod
    def _distance_matrix(src: torch.Tensor, tgt: torch.Tensor, p: int = 2) -> torch.Tensor:
        return torch.sum(torch.abs(src.unsqueeze(1) - tgt.unsqueeze(0)) ** p, dim=2)


class UOTAlignmentObjective(nn.Module):
    """Unbalanced optimal transport alignment with reusable minibatch plans."""

    def __init__(self, entropy: float = 0.1, marginal_relaxation: float = 1.0):
        super().__init__()
        self.entropy = entropy
        self.marginal_relaxation = marginal_relaxation
        self.cost = GaussianWassersteinCost()

    def forward(
        self,
        plan,
        mu_src,
        var_src,
        mu_tgt,
        var_tgt,
        prior=None,
        device="cpu",
        idx_q=None,
        idx_r=None,
        query_weight=None,
        ref_weight=None,
    ):
        ns = mu_src.size(0)
        nt = mu_tgt.size(0)
        cost = self.cost(mu_src, var_src, mu_tgt, var_tgt)
        p_s = self._mass(ns, query_weight, idx_q, device)
        p_t = self._mass(nt, ref_weight, idx_r, device)
        if plan is None:
            plan = torch.ones(ns, nt, device=device) / (ns * nt)

        dual = (torch.ones(ns, 1, device=device) / ns)
        exponent = self.marginal_relaxation / (self.marginal_relaxation + self.entropy)
        for _ in range(10):
            adjusted_cost = cost * prior if prior is not None else cost
            kernel = torch.exp(
                -adjusted_cost / (self.entropy * torch.max(torch.abs(adjusted_cost)))
            ) * plan
            b = p_t / (torch.t(kernel) @ dual)
            for _ in range(10):
                dual = (p_s / (kernel @ b)) ** exponent
                b = (p_t / (torch.t(kernel) @ dual)) ** exponent
            plan = (dual @ torch.t(b)) * kernel

        if torch.isnan(plan).sum() > 0:
            plan = torch.ones(ns, nt, device=device) / (ns * nt)
        return (cost * plan.detach().data).sum(), plan.detach()

    @staticmethod
    def _mass(size, weights, idx, device):
        if weights is None:
            return (torch.ones(size, 1) / size).to(device)
        batch_weight = weights[idx]
        return (batch_weight / torch.sum(batch_weight)).to(device)


class ZINBReconstructionLoss(nn.Module):
    """Zero-inflated negative binomial expression reconstruction loss."""

    def forward(self, px_dropout, px_r, px_scale, input_x, reconstr_x):
        softplus_pi = F.softplus(-px_dropout)
        log_theta_eps = torch.log(px_r + 1e-8)
        log_theta_mu_eps = torch.log(px_r + reconstr_x + 1e-8)
        pi_theta_log = -px_dropout + torch.mul(px_r, (log_theta_eps - log_theta_mu_eps))

        case_zero = F.softplus(pi_theta_log) - softplus_pi
        zero_mask = (input_x < 1e-8).float()
        case_non_zero = (
            -softplus_pi
            + pi_theta_log
            + torch.mul(input_x, (torch.log(reconstr_x + 1e-8) - log_theta_mu_eps))
            + torch.lgamma(input_x + px_r)
            - torch.lgamma(px_r)
            - torch.lgamma(input_x + 1)
        )
        non_zero_mask = (input_x > 1e-8).float()
        log_prob = torch.mul(zero_mask, case_zero) + torch.mul(non_zero_mask, case_non_zero)
        return -torch.mean(torch.sum(log_prob, dim=1))


class HierarchicalCellStateLoss(nn.Module):
    """Encourage primary cell-type and substate consistency in predicted expression."""

    def forward(self, representation, primary_type, substate):
        n_obs = representation.size(0)
        if n_obs < 2:
            return torch.tensor(0.0, device=representation.device)
        distances = torch.cdist(representation, representation, p=2)
        mask = torch.triu(torch.ones(n_obs, n_obs, dtype=torch.bool, device=representation.device), diagonal=1)
        same_primary = (primary_type.view(-1, 1) == primary_type.view(1, -1)) & mask
        diff_primary = (~same_primary) & mask
        loss = distances[same_primary].pow(2).sum()
        loss = loss + torch.clamp(1 - distances[diff_primary], min=0).pow(2).sum()

        if len(substate) and substate[0] >= 0:
            base = (primary_type.view(-1, 1) == primary_type.view(1, -1)) & mask
            same_substate = base & (substate.view(-1, 1) == substate.view(1, -1))
            diff_substate = base & (substate.view(-1, 1) != substate.view(1, -1))
            loss = loss + distances[same_substate].pow(2).sum()
            loss = loss + torch.clamp(1 - distances[diff_substate], min=0).pow(2).sum()

        valid_pairs = mask.sum().float()
        return loss / valid_pairs if valid_pairs > 0 else loss


class TransCellObjective(nn.Module):
    """Weighted container matching the TransCell paper-level objectives."""

    def __init__(
        self,
        reconstruction_fidelity=1.0,
        kl_fidelity=5.0,
        ot_alignment=1.0,
        hierarchy_weight=0.7,
    ):
        super().__init__()
        self.reconstruction_fidelity = reconstruction_fidelity
        self.kl_fidelity = kl_fidelity
        self.ot_alignment = ot_alignment
        self.hierarchy_weight = hierarchy_weight
        self.zinb = ZINBReconstructionLoss()
        self.kl = LatentKLFidelityLoss()
        self.hierarchy = HierarchicalCellStateLoss()

    def combine(self, reconstruction, kl, ot, hierarchy):
        return {
            "reconstruction": self.reconstruction_fidelity * reconstruction,
            "kl_fidelity": self.kl_fidelity * kl,
            "uot_alignment": self.ot_alignment * ot,
            "hierarchy": self.hierarchy_weight * hierarchy,
        }

