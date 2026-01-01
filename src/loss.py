import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class Loss(nn.Module):
    def __init__(
            self, 
            w_im_sample_loss: float = 1.0, 
            w_equilibrium_loss: float = 0.0001,
            w_sample_loss: float = 1.0,
            w_bbx_loss: float = 1.0,
            w_center_loss: float = 0.1,
            w_overlap_loss: float = 0.1,
            w_balance_loss: float = 0.1,
            w_containment_loss: float=0.001,
            n_levels: int = 3,
            n_parts: List[int] = [4, 16, 32],
        ) -> None:
        super(Loss, self).__init__()

        self._w_im_sample = w_im_sample_loss
        self._w_equilibrium = w_equilibrium_loss
        self._w_sample = w_sample_loss
        self._w_bbx_loss = w_bbx_loss
        self._w_center = w_center_loss
        self._w_overlap = w_overlap_loss
        self._w_balance = w_balance_loss
        self._w_containment = w_containment_loss

        self.Lmax = len(n_parts)
        self.n_parts = n_parts

    def _compute_relation_loss(self, relations: Dict[str, torch.Tensor], max_levels: int) -> torch.Tensor:
        """Compute relation loss."""
        total_loss = 0.0 
        for i in range(1, max_levels):
            B = relations[f'level_{i}'].shape[0]
            pc_relation = relations[f'level_{i}'] 
            wk = torch.sum(pc_relation, dim=1, keepdim=True).reshape(B, -1)
            if wk.shape[-1] > 1: 
                total_loss += torch.mean(torch.var(input=wk, dim=-1, keepdim=True))
            else:
                continue
        return total_loss

    
    def _compute_sample_loss(self, gt: torch.Tensor, pred: torch.Tensor, Lmax: int) -> torch.Tensor:

        num_ones = torch.sum(gt == 1).item()
        num_zeros = torch.sum(gt == 0).item()

        total_loss = 0.0

        for i in range(Lmax):
            level_pred = pred[:, :gt.shape[1], i:i+1]

            total_elements = gt.numel()
            weight_1 = num_zeros / total_elements
            weight_0 = num_ones / total_elements
            weights = gt * weight_1 + (1 - gt) * weight_0
            total_loss += torch.mean(weights * (gt - level_pred) ** 2)


        return total_loss
    

    def _compute_overlap_loss(self, sum_indica: torch.Tensor, trans_sum_indica: Dict[str, torch.Tensor], max_levels: int) -> torch.Tensor:
        total_loss = 0.0

        for indx_ in range(max_levels):
            indx_shape_sum_indicator = sum_indica[...,indx_:indx_+1]
            indx_trans_sum = trans_sum_indica[f'level_{indx_}']
            imgsum = torch.cat([indx_shape_sum_indicator, indx_trans_sum], dim=1) 
            total_loss += torch.mean((F.relu(imgsum - 2.0))**2)

        return total_loss
    

    def _compute_mahalanobis_distance(self, pts: torch.Tensor, rotations: torch.Tensor, translations: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance.
        Args:
            :param pts (torch.Tensor): Points.
            :param rotations (torch.Tensor): Rotation matrix. Euler rotation matrix
            :param translations (torch.Tensor): Translation vector.
            :param scales (torch.Tensor): Scale factor. [0, inf]
        Returns:
            :return: Mahalanobis distance.
        """
        # Compute the Mahalanobis distance
        diff = pts[:, None, :, :] - translations[:, :, None, :]
        x, y, z = diff.unbind(-1)
        DIV_EPSILON = 1e-6
        scales = repeat(scales, 'b p c -> b p (repeat c)', repeat=pts.shape[-1]) 

        d = 1 / (scales + DIV_EPSILON)
        diag = torch.diag_embed(d)
        rotation = rearrange(rotations, 'b p (i j) -> b p i j', i=3, j=3)

        inv_cov_matrix = torch.matmul(torch.matmul(rotation, diag), rotation.transpose(-1, -2))
        inv_cov_matrix = inv_cov_matrix.view(*inv_cov_matrix.shape[:-2], 1, 9)

        c00, c01, c02, _, c11, c12, _, _, c22 = inv_cov_matrix.unbind(-1)
        dist = (x * (c00 * x + c01 * y + c02 * z) +\
                y * (c01 * x + c11 * y + c12 * z) +\
                z * (c02 * x + c12 * y + c22 * z))
        
        return dist.unsqueeze(-1)  # Add a dimension to match the shape of the input points


    def _compute_bbx_loss(self, trans: Dict[str, torch.Tensor], pts: torch.Tensor, pc: torch.Tensor, gt: torch.Tensor, max_levels: int) -> torch.Tensor:
        """Compute bounding box loss.
        Args:
            :param trans (torch.Tensor): Transformation matrix.
            :param pts (torch.Tensor): Points.
            :param gt (torch.Tensor): Ground truth.
            :param max_levels (int): Maximum levels.
        Returns:
            :return: Bounding box loss.
        """

        oo = 1e5
        total_loss = 0.0 
        inside_surf = torch.ones(size=(pc.shape[:2]), device=pc.device).unsqueeze(-1)
        gt = torch.cat([gt, inside_surf], dim=1)

        inside = gt[:, None, :, :]

        pts = torch.cat([pts, pc], dim=1)   

        for indx_ in range(max_levels):
            indx_trans = trans[f'level_{indx_}']
            # Add your bounding box loss computation here
            indx_translations, indx_rotations, indx_scales = torch.split(indx_trans, [3, 9, 1], dim=-1)

            # Compute the Mahalanobis distance
            distances = self._compute_mahalanobis_distance(pts=pts, rotations=indx_rotations, translations=indx_translations, scales=indx_scales)
            distances = inside * distances + (1 - inside) * oo
            min_dist_t2i = torch.min(distances, dim=2)[0] 
            min_dist_i2t = gt * torch.min(distances, dim=1)[0]
            min_dist_i2t = torch.sum(min_dist_i2t, dim=1, keepdim=True) / torch.sum(gt, dim=1, keepdim=True)
            total_loss += torch.mean(min_dist_t2i) + torch.mean(min_dist_i2t)

        return total_loss


    def _compute_center_loss(self, offset: Dict[str, torch.Tensor], max_levels: int) -> torch.Tensor:
        """Compute center loss."""
        total_loss = 0.0 
        for indx_ in range(max_levels):
            indx_offset = offset[f'level_{indx_}']
            total_loss += torch.mean(indx_offset ** 2)
        return total_loss
    

    def _compute_balance_loss(self, part_indicator: Dict[str, torch.Tensor], labels: torch.Tensor, max_levels: int, n_query_pts: int) -> torch.Tensor:
        total_loss = 0.0
        labels = rearrange(labels, 'b n p -> b p n')        
        k = 20
        for indx_ in range(max_levels):
            indx_part_indicator = part_indicator[f'level_{indx_}'][:, :n_query_pts, :]
            indx_part_indicator = rearrange(indx_part_indicator, 'b n p -> b p n')           
            indicas = indx_part_indicator * labels
            vals, _ = torch.topk(indicas, k=k, dim=-1)
            total_loss += torch.mean((1. - vals) ** 2)
        return total_loss
    

    def _compute_equillibrium_loss(self, scores: torch.Tensor, nquery_pts: int) -> torch.Tensor:
        """Compute equilibrium loss."""
        total_loss = 0.0
        B = scores.shape[0]
        for indx_ in range(self.Lmax):
            if indx_ == 0: pp_indx_ = 0
            else: pp_indx_ = self.n_parts[indx_-1]
            cp_indx_ = pp_indx_ + self.n_parts[indx_]
            # Compute the equilibrium loss for each level
            indx_scores = scores[:, :nquery_pts, pp_indx_:cp_indx_]
            wk = torch.sum(indx_scores, dim=-2, keepdim=True).reshape(B, -1)
            total_loss += torch.var(input=wk, dim=-1, keepdim=True).mean()
        return total_loss


    def _compute_guidance_loss(self, token_occ: torch.Tensor, cvx_pred: torch.Tensor, max_levels: int, n_query_pts: int) -> torch.Tensor:
        total_loss = 0.0 

        for indx_ in range(max_levels):
            if indx_ == 0: pp_indx_ = 0
            else: pp_indx_ = self.n_parts[indx_-1]
            cp_indx_ = pp_indx_ + self.n_parts[indx_]
            indx_token = token_occ[:, :n_query_pts, pp_indx_:cp_indx_]
            indx_cvx = cvx_pred[:, :n_query_pts, pp_indx_:cp_indx_]
            
            total_loss += torch.mean((indx_token - indx_cvx) ** 2)
        return total_loss


    def _compute_sparsity_loss(self, convex_exists: torch.Tensor, max_levels: int) -> torch.Tensor:
        """Compute sparsity loss."""
        total_loss = 0.0 
        for indx_ in range(max_levels):
            indx_convex_exists = convex_exists[f'level_{indx_}']
            total_loss += indx_convex_exists.abs().mean()
        # loss/=len(indx_convex_exists)
        return total_loss


    def _compute_attention_loss(self, attn_occs: Dict[str, torch.Tensor], cvx_occ: Dict[str, torch.Tensor], max_levels: int) -> torch.Tensor:
        """Compute attention loss."""
        total_loss = 0.0 
        for indx_ in range(max_levels):
            indx_attn_occ = attn_occs[f'level_{indx_}']
            indx_cvx_occ = cvx_occ[f'level_{indx_}']
            total_loss += torch.mean((indx_attn_occ - indx_cvx_occ.detach()) ** 2)
        return total_loss


    def _compute_containment_loss(self, parent_child_indicator, convex_part_indicator, max_levels, n_query_pts) -> torch.Tensor:
        
        total_loss = 0.0 

        for i in range(1, max_levels):
            # parent_normals, parent_offsets, relations = convex_half_spaces[f'level_{i-1}'], convex_offsets[f'level_{i-1}'], relations_dict[f'level_{i}']
            level_parent_child_indicator = parent_child_indicator[f'level_{i}']
            level_convex_part_indicator = convex_part_indicator[f'level_{i}']            
            total_loss += torch.mean((1. - level_parent_child_indicator) * level_convex_part_indicator)

        return total_loss

    def _compute_parent_child_loss(self, part_indicator: Dict[str, torch.Tensor], convex_indicator: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute parent-child loss."""
        total_loss = 0.0 
        for i in range(1, len(part_indicator)):
            level_part_indicator = part_indicator[f'level_{i}']
            level_convex_indicator = convex_indicator[:,:,i:i+1]
            level_part_occ = torch.max(level_part_indicator, dim=-1, keepdim=True)[0]
            total_loss += torch.mean((level_part_occ - level_convex_indicator.detach()) ** 2)
        return total_loss


    def forward(self, data: Dict[str, torch.Tensor], occ_gt: torch.Tensor, query_points: torch.Tensor, pc: torch.Tensor, n_query_pts: int, iter_num: int) -> Dict[str, torch.Tensor]:
        """Compute the loss.
        Args:
            :param data (Dict[str, torch.Tensor]): Data.
            :param occ_gt (torch.Tensor): Ground truth.
            :param query_points (torch.Tensor): Query points.
            :param n_query_pts (int): Number of query points.
        Returns:
            :return: Loss.
        """
        total_loss = 0.0 
            
        sample_loss = self._compute_sample_loss(
            gt=occ_gt, 
            pred=data['all_convex_indicator'], 
            Lmax=data['num_active_blocks']
        )

        overlap_loss = self._compute_overlap_loss(
            sum_indica=data['all_convex_sum_indicator'], 
            trans_sum_indica=data['all_convex_trans_sum'], 
            max_levels=data['num_active_blocks'], 
        )

        bbx_loss = self._compute_bbx_loss(
            trans=data['all_convex_trans'], 
            pts=query_points, 
            pc=pc,
            gt=occ_gt, 
            max_levels=data['num_active_blocks'],
        )

        center_loss = self._compute_center_loss(
            offset=data['all_convex_offsets'], 
            max_levels=data['num_active_blocks']
        )

        balance_loss = self._compute_balance_loss(
            part_indicator=data['all_convex_part_indicator'], 
            labels=occ_gt, 
            max_levels=data['num_active_blocks'], 
            n_query_pts=n_query_pts
        )

        containment_loss = self._compute_containment_loss(
            parent_child_indicator=data['all_parent_child_indicator'],
            convex_part_indicator=data['all_convex_part_indicator'],
            max_levels=data['num_active_blocks'],
            n_query_pts=n_query_pts
        )

        relation_loss = torch.tensor(0., dtype=torch.float32, device=occ_gt.device)
        if data['num_active_blocks'] > 1: 
            relation_loss += self._compute_relation_loss(
                relations=data['all_relations'], 
                max_levels=data['num_active_blocks']
            )

        total_loss +=   self._w_sample * sample_loss +\
                        self._w_overlap * overlap_loss +\
                        self._w_bbx_loss * bbx_loss +\
                        self._w_center * center_loss +\
                        self._w_balance * balance_loss +\
                        self._w_equilibrium * relation_loss +\
                        self._w_containment * containment_loss 
                        # self._w_containment * (containment_loss + sdf_containment_loss)
        
        loss = dict() 
        loss['total_loss'] = total_loss
        loss['sample_loss'] = sample_loss
        loss['overlap_loss'] = overlap_loss
        loss['bbx_loss'] = bbx_loss
        loss['center_loss'] = center_loss
        loss['balance_loss'] = balance_loss
        loss['containment_loss'] = containment_loss
        loss['relation_loss'] = relation_loss

        return loss