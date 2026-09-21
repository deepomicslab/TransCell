#!/usr/bin/env python3
"""Loss functions for TransCell, including unbalanced optimal transport."""

from sklearn.metrics import pairwise_distances
import torch
import numpy as np
from torch.distributions import Normal, kl_divergence
from scipy.sparse import issparse
import scipy
import torch
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def kl_div(mu, var, weight=None):
    loss = kl_divergence(Normal(mu, var.sqrt()), Normal(torch.zeros_like(mu),torch.ones_like(var))).sum(dim=1)
    
    # if weight is not None:
    #     loss = loss * weight.squeeze(dim=1)
    return loss.mean()
 
# def balanced_binary_cross_entropy(recon_x, x):

#     return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)

def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.

    Parameters
    ----------
    pts_src
        [R, D] matrix
    pts_dst
        C, D] matrix
    p
        p-norm
    
    Return
    ------
    [R, C] matrix
        distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance

def distance_gmm(mu_src: torch.Tensor, mu_dst: torch.Tensor, var_src: torch.Tensor, var_dst: torch.Tensor):
    """
    Calculate a Wasserstein distance matrix between the gmm distributions with diagonal variances

    Parameters
    ----------
    mu_src
        [R, D] matrix, the means of R Gaussian distributions
    mu_dst
        [C, D] matrix, the means of C Gaussian distributions
    logvar_src
        [R, D] matrix, the log(variance) of R Gaussian distributions
    logvar_dst
        [C, D] matrix, the log(variance) of C Gaussian distributions
    
    Return
    ------
    [R, C] matrix 
        distance matrix
    """
    std_src = var_src.sqrt()
    std_dst = var_dst.sqrt()
    distance_mean = distance_matrix(mu_src, mu_dst, p=2)
    distance_var = distance_matrix(std_src, std_dst, p=2)

    # distance_var = torch.sum(sum_matrix(std_src, std_dst) - 2 * (prod_matrix(std_src, std_dst) ** 0.5), 2)
    
    return distance_mean + distance_var + 1e-6

def unbalanced_ot(tran, mu1, var1, mu2, var2, reg=0.1, reg_m=1.0, Couple=None, device='cpu', \
    idx_q=None, idx_r=None, query_weight=None, ref_weight=None):
    '''
    Calculate a unbalanced optimal transport matrix between mini batches.

    Parameters
    ----------
    tran
        transport matrix between the two batches sampling from the global OT matrix. 
    mu1
        mean vector of batch 1 from the encoder
    var1
        standard deviation vector of batch 1 from the encoder
    mu2
        mean vector of batch 2 from the encoder
    var2
        standard deviation vector of batch 2 from the encoder
    reg:
        Entropy regularization parameter in OT. Default: 0.1
    reg_m:
        Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        training device
    idx_q
        domain_id of query batch
    idx_r
        domain_id of reference batch
    query_weight
        reweighted vectors of query batch
    ref_weight
        reweighted vectors of reference batch

    Returns
    -------
    float
        minibatch unbalanced optimal transport loss
    matrix
        minibatch unbalanced optimal transport matrix
    '''

    ns = mu1.size(0)
    nt = mu2.size(0)
    

    cost_pp = distance_gmm(mu1, mu2, var1, var2)

    if query_weight is None: 
        p_s = torch.ones(ns, 1) / ns
    else:
        query_batch_weight = query_weight[idx_q]
        p_s = query_batch_weight/torch.sum(query_batch_weight)

    if ref_weight is None: 
        p_t = torch.ones(nt, 1) / nt
    else:
        ref_batch_weight = ref_weight[idx_r]
        p_t = ref_batch_weight/torch.sum(ref_batch_weight)

    p_s = p_s.to(device)
    p_t = p_t.to(device)

    if tran is None:
        tran = torch.ones(ns, nt) / (ns * nt)
        tran = tran.to(device)

    dual = (torch.ones(ns, 1) / ns).to(device)
    f = reg_m / (reg_m + reg)

    for m in range(10):
        if Couple is not None:
            cost = cost_pp*Couple
        else:
            cost = cost_pp

        kernel = torch.exp(-cost / (reg*torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        # dual = p_s / (kernel @ b)
        for i in range(10):
            dual =( p_s / (kernel @ b) )**f
            b = ( p_t / (torch.t(kernel) @ dual) )**f
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    # pho = tran.mean()
    # h_func = 1 - 0.5 * ( 1 + torch.sign(pho - tran) )
    # hat_tran = tran * h_func
    # d_fgw1 = (cost_pp * hat_tran.detach().data).sum()
    # d_fgw2 = ((tran.detach().data - hat_tran.detach().data) * torch.log(1 + torch.exp(-cost_pp))).sum()
    # d_fgw = d_fgw1 + d_fgw2

    d_fgw = (cost_pp * tran.detach().data).sum()

    return d_fgw, tran.detach()

def construct_sample_pairs(
    transport_plan=None,
    ns=None,
    nt=None,
    source_adata=None, 
    target_adata=None,
    source_type=None,
    target_type=None,
    device='cpu',
    confidence_threshold=0.8,
    negative_sample_ratio=1.0,
    random_seed=42
):
    """
    构建用于细胞类型匹配的正负样本对
    
    参数:
    -----
    source_adata : AnnData
        源数据集（如小鼠数据）
    target_adata : AnnData
        目标数据集（如人类数据）
    transport_plan : np.ndarray, optional
        从最优传输算法获得的传输矩阵，如果为None，将使用KNN方法构建
    confidence_threshold : float
        正样本选择的置信度阈值，范围[0,1]
    negative_sample_ratio : float
        负样本与正样本的比例
    embedding_key : str
        存储细胞嵌入的键名
    cell_type_key : str
        存储细胞类型标签的键名
    random_seed : int
        随机种子，确保结果可重现
    
    返回:
    -----
    positive_pairs : pd.DataFrame
        正样本对
    negative_pairs : pd.DataFrame
        负样本对
    """
    print("构建正负样本对...")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 确保数据在同一设备上
    source_embed = source_adata.to(device) if isinstance(source_adata, torch.Tensor) else torch.tensor(source_adata, device=device)
    target_embed = target_adata.to(device) if isinstance(target_adata, torch.Tensor) else torch.tensor(target_adata, device=device)
    source_type = torch.tensor(source_type, device=device) if not isinstance(source_type, torch.Tensor) else source_type.to(device)
    target_type = torch.tensor(target_type, device=device) if not isinstance(target_type, torch.Tensor) else target_type.to(device)
    
    # 1. 处理传输计划
    if transport_plan is not None:
        transport_plan = torch.tensor(transport_plan, device=device) if not isinstance(transport_plan, torch.Tensor) else transport_plan.to(device)
        # 归一化传输计划
        row_max, _ = torch.max(transport_plan, dim=1, keepdim=True)
        normalized_plan = transport_plan / (row_max + 1e-10)

    # 2. 构建正样本对 - 使用矩阵操作
    positive_pairs = []
    if transport_plan is not None:
        # 创建掩码矩阵：对应位置的类型相同
        type_mask = source_type.unsqueeze(1) == target_type.unsqueeze(0)
        
        # 找到高置信度匹配且类型相同的位置
        confident_mask = (normalized_plan >= confidence_threshold) & (type_mask)
        
        # 获取所有符合条件的索引及置信度
        pos_indices = torch.nonzero(confident_mask, as_tuple=False)

        if len(pos_indices) > 0:
            confidences = normalized_plan[pos_indices[:, 0], pos_indices[:, 1]]
            
            # 按置信度排序
            _, sorted_indices = torch.sort(confidences, descending=True)
            pos_indices = pos_indices[sorted_indices]
            
            # 限制数量
            max_pos_pairs=2000
            num_pos = min(len(pos_indices), max_pos_pairs)
            pos_indices = pos_indices[:num_pos]
            
            # 转换为列表格式 [i, j, confidence]
            for idx in range(num_pos):
                i, j = pos_indices[idx]
                if j < nt:  # 确保目标索引在范围内
                    positive_pairs.append([i.item(), j.item(), normalized_plan[i, j].item()])

    print("构建了", len(positive_pairs), "个正样本对")
    # 3. 构建负样本对 - 使用批处理计算相似度
    negative_pairs = []
    if len(positive_pairs) > 0:
        max_neg_pairs=2000
        num_negative_samples = min(int(len(positive_pairs) * negative_sample_ratio), max_neg_pairs)
        
        # 快速获取所有不同类型的细胞对
        diff_type_mask = source_type.unsqueeze(1) != target_type.unsqueeze(0)
        diff_indices = torch.nonzero(diff_type_mask, as_tuple=False)
        
        if len(diff_indices) > 0:
            # 批量计算余弦相似度
            batch_size = 1000  # 调整批处理大小以适应内存
            candidate_negatives = []
            
            for start_idx in range(0, len(diff_indices), batch_size):
                end_idx = min(start_idx + batch_size, len(diff_indices))
                batch_indices = diff_indices[start_idx:end_idx]
                
                source_batch = source_embed[batch_indices[:, 0]]
                target_batch = target_embed[batch_indices[:, 1]]
                
                # 计算余弦相似度 (批处理)
                source_norm = torch.norm(source_batch, dim=1, keepdim=True)
                target_norm = torch.norm(target_batch, dim=1, keepdim=True)
                similarities = torch.sum(source_batch * target_batch, dim=1) / (source_norm * target_norm).squeeze() + 1e-8
                
                for k, (i, j) in enumerate(batch_indices):
                    candidate_negatives.append([i.item(), j.item(), similarities[k].item()])
            # 按相似度排序并选择前N个
            candidate_negatives.sort(key=lambda x: x[2], reverse=True)
            
            # 限制每个源细胞的负样本数量
            k_per_source = 3
            source_counts = {}
            
            for i, j, _ in candidate_negatives:
                if source_counts.get(i, 0) < k_per_source:
                    negative_pairs.append([i, j])
                    source_counts[i] = source_counts.get(i, 0) + 1
                    
                if len(negative_pairs) >= num_negative_samples:
                    break
    print("构建了", len(negative_pairs), "个负样本对")
    return positive_pairs, negative_pairs
    

def get_hierarchy_loss(z, celltype1, celltype2, device='cpu'):
    z = z.to(device)
    celltype1 = celltype1.to(device)
    celltype2 = celltype2.to(device)
    
    n = z.size(0)
    dist_matrix = torch.cdist(z, z, p=2)
    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)

    # ===== 第一部分：基础celltype1损失 =====
    # 生成同类样本掩码 [n, n]
    same_mask = (celltype1.view(-1, 1) == celltype1.view(1, -1)) & triu_mask
    loss_same = (dist_matrix[same_mask].pow(2)).sum()
    
    # 异类样本损失
    diff_mask = (~same_mask) & triu_mask
    loss_diff = (torch.clamp(1 - dist_matrix[diff_mask], min=0).pow(2)).sum()
    total_loss = loss_same + loss_diff

    # ===== 第二部分：有条件celltype2损失 =====
    if celltype2[0] >= 0:
        # 生成celltype1相同的基础掩码 [n, n]
        base_mask = (celltype1.view(-1, 1) == celltype1.view(1, -1)) & triu_mask
        
        # 在celltype1相同的基础上生成celltype2相同掩码
        cross_same_mask = base_mask & (celltype2.view(-1, 1) == celltype2.view(1, -1))
        cross_loss_same = (dist_matrix[cross_same_mask].pow(2)).sum()
        
        # 在celltype1相同但celltype2不同的掩码
        cross_diff_mask = base_mask & (celltype2.view(-1, 1) != celltype2.view(1, -1))
        cross_loss_diff = (torch.clamp(1 - dist_matrix[cross_diff_mask], min=0).pow(2)).sum()
        
        total_loss += cross_loss_same + cross_loss_diff

    # 归一化处理
    valid_pairs = triu_mask.sum().float()
    return total_loss / valid_pairs if valid_pairs > 0 else total_loss

def get_ContrastiveLoss( transport_plan=None,
    ns=None,
    nt=None,
    source_adata=None, 
    target_adata=None,
    source_type=None,
    target_type=None,
    device='cpu',
    confidence_threshold=0.9,
    negative_sample_ratio=1.0,
    random_seed=42
    ):
    
    loss = torch.tensor(0.0).to(device)
    margin=torch.tensor(1.0).to(device)

    positive_pairs, negative_pairs = construct_sample_pairs(transport_plan=transport_plan,
    ns=ns,
    nt=nt,
    source_adata=source_adata, 
    target_adata=target_adata,
    source_type=source_type,
    target_type=target_type,
    device=device,
    confidence_threshold=confidence_threshold,
    negative_sample_ratio=negative_sample_ratio,
    random_seed=random_seed
    )

    sum_length=len(positive_pairs)+len(negative_pairs)

    # 没有样本对时返回零损失
    if len(positive_pairs) == 0:
        return torch.tensor(0.0, device=device)
    
    positive_pairs = torch.tensor(positive_pairs)
    negative_pairs = torch.tensor(negative_pairs)
    positive_pairs.to(device)
    negative_pairs.to(device)

    source_emb=source_adata
    target_emb=target_adata
    source_emb.to(device)
    target_emb.to(device)

    # 批量处理正样本对
    if len(positive_pairs) > 0:
        pos_source_idx = torch.tensor([int(p[0]) for p in positive_pairs], dtype=torch.long, device=device)
        pos_target_idx = torch.tensor([int(p[1]) for p in positive_pairs], dtype=torch.long, device=device)
        
        pos_source_emb = source_emb[pos_source_idx]
        pos_target_emb = target_emb[pos_target_idx]
        
        # 批量计算正样本对距离
        pos_dists = torch.norm(pos_source_emb - pos_target_emb, p=2, dim=1)
        pos_loss = torch.sum(pos_dists.pow(2))
    else:
        pos_loss = torch.tensor(0.0, device=device)
        
    
    # 批量处理负样本对
    if len(negative_pairs) > 0:
        neg_source_idx = torch.tensor([int(p[0]) for p in negative_pairs], dtype=torch.long, device=device)
        neg_target_idx = torch.tensor([int(p[1]) for p in negative_pairs], dtype=torch.long, device=device)
        
        neg_source_emb = source_emb[neg_source_idx]
        neg_target_emb = target_emb[neg_target_idx]
        
        # 批量计算负样本对距离
        neg_dists = torch.norm(neg_source_emb - neg_target_emb, p=2, dim=1)
        neg_loss = torch.sum(torch.clamp(margin - neg_dists, min=0).pow(2))
    else:
        neg_loss = torch.tensor(0.0, device=device)

    loss=0.5*pos_loss+0.5*neg_loss
    if sum_length>0:
        loss=loss/sum_length
    return loss

import torch
import torch.nn.functional as F

def calc_zinb_loss(px_dropout, px_r, px_scale, input_x, reconstr_x):
    # Calculate softplus terms
    # reconstr_x = torch.clamp(reconstr_x, min=1e-6)
    softplus_pi = F.softplus(-px_dropout)  # log(sigmoid(x)) = -softplus(-x)
    # Log terms with epsilon for numerical stability
    log_theta_eps = torch.log(px_r + 1e-8)
    log_theta_mu_eps = torch.log(px_r + reconstr_x + 1e-8)
    
    # Core ZINB calculations
    pi_theta_log = -px_dropout + torch.mul(px_r, (log_theta_eps - log_theta_mu_eps))
    
    # Case 1: Zero counts
    case_zero = F.softplus(pi_theta_log) - softplus_pi
    zero_mask = (input_x < 1e-8).float()
    mul_case_zero = torch.mul(zero_mask, case_zero)
    
    # Case 2: Non-zero counts
    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + torch.mul(input_x, (torch.log(reconstr_x + 1e-8) - log_theta_mu_eps))
        + torch.lgamma(input_x + px_r)
        - torch.lgamma(px_r)
        - torch.lgamma(input_x + 1)
    )
    non_zero_mask = (input_x > 1e-8).float()
    mul_case_non_zero = torch.mul(non_zero_mask, case_non_zero)
    
    # Combine both cases
    res = mul_case_zero + mul_case_non_zero
    
    # Calculate final loss
    translator_loss_x = -torch.mean(torch.sum(res, dim=1))
    
    return translator_loss_x
