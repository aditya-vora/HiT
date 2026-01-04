import torch
import torch.nn as nn 
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from einops import rearrange, repeat

from config import *
from src.models.pointnet import PointNet
from src.models.conv_occnet import LocalPoolPointnet
from src.models.modules import TransformerBlock
from src.models.cvx_decoder import PartParameterization
from utils.model import normalize_3d_coordinate

@dataclass
class ModelArgs:
    pf_dim: int = 3
    ef_dim: int = 256
    zf_dim: int = 512

    use_xyz: bool = True
    use_bn: bool = False
    plane_type: str = 'grid'
    grid_resolution: int = 32

    tf_dim: int = 4
    gf_dim: int = 256

    n_parts: List[int] = field(default_factory=lambda: [8, 16, 36])
    n_levels: int = 3
    n_planes: int = 32 
    mask_mode: str = "mask"
    planef_dim: int = 8


class HierarchicalPartsTransformer(nn.Module):
    """Convex Decoder with embeddings
    Args:
        n_convexs: Number of convex shapes to map the planes to
        zf_dim: Dimension of the embedding
        n_head: Number of attention heads
    Returns:
        ConvexDecoder module
    """
    def __init__(self, n_convexs: int=3, n_head: int=8, zf_dim: int=4) -> None:
        super(HierarchicalPartsTransformer, self).__init__()
        
        self.convex_layer_embeddings = nn.Embedding(n_convexs, zf_dim)
        self.attn = TransformerBlock(d_model=zf_dim, nhead=n_head, dim_feedforward=4*zf_dim, dropout=0.0)
    
    def forward(self, per_point_planes: torch.Tensor = None, scores: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        per_point_planes: multiple planes per point, [B, N, 4D]
        scores: probability of belonging to a particular convex
        returns: convexs
        """
        B, N, _ = per_point_planes.shape

        convex_query_latents = self.convex_layer_embeddings.weight
        tokens = rearrange(convex_query_latents, 'n d -> () n d')
        tokens = repeat(tokens, '() n d -> b n d', b=B)

        # create a mask to restrict the local region in convex mapping.
        if scores is None: 
            mask = None 
        else:
            indexs = torch.argmax(scores, dim=-1)
            mask = F.one_hot(indexs, scores.shape[-1]).float()
            mask = scores + (mask - scores).detach()

            mask = rearrange(mask, 'b n c -> b c n')
        
        convexs, scores = self.attn(tokens, per_point_planes, mask=mask)

        return (convexs, scores)


class HiTDecoder(nn.Module):
    def __init__(
            self,
            pf_dim: int=3,
            zf_dim: int=32,
            tf_dim: int=4,
            gf_dim: int=256,
            planef_dim: int=4,
            n_parts: List[int]=[3,9,27],
            n_planes: int=128,
            mask_mode: str="normal",    
        ) -> None:
        super(HiTDecoder, self).__init__()
        """HiT Decoder Defination
        
        Args:
            pf_dim (int): point feature dimension
            tf_dim (int): transform feature dimension
            zf_dim (int): latent feature dimension
            gf_dim (int): global feature dimension
            planef_dim (int): plane feature dimension
            n_parts (List[int]): number of parts at each level
            n_planes (int): number of planes
            mask_mode (str): mask mode
        Returns:
            None
        """

        self.planef_dim, self.n_planes, self.pf_dim, self.zf_dim, self.n_parts = planef_dim, n_planes, pf_dim, zf_dim, n_parts

        self.token_attn_blocks = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])
        self.multiconvex_decoder = nn.ModuleList([])
        
        self.part_blocks = nn.ModuleList(
            [
                HierarchicalPartsTransformer(
                    n_convexs=self.n_parts[i], 
                    zf_dim=zf_dim, 
                    n_head=1
                ) for i in range(len(self.n_parts))
            ]
        )
                        
        self.multiconvex_decoder = PartParameterization(
            zf_dim=zf_dim,
            pf_dim=pf_dim,
            n_planes=n_planes,
            planef_dim=planef_dim,
        )

        self.num_active_blocks = len(self.n_parts)
        self.mask_mode = mask_mode 

    def increase_active_blocks(self):
        if self.num_active_blocks < len(self.n_parts):
            self.num_active_blocks += 1

    def _get_attention_matrix(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
                
        # query_pos_enc = self._sample_grid_feature(qpts, feats)
        feats = feats.view(B, self.zf_dim, -1) 
        tokens = rearrange(feats, 'b f n -> b n f')
        
        attn_matrixs = {}
        for i in range(self.num_active_blocks):
            convex_codebook = self.convex_codebooks[i]
            if i == 0: convex_latents = tokens    
            convex_latents, attn = convex_codebook(convex_latents, scores=None)            
            attn_matrixs[f'level_{i}'] = attn

        return attn_matrixs
    

    def _get_part_indicator(self, occ: torch.Tensor=None, scores: torch.Tensor=None) -> torch.Tensor:
        """returns the branched attention scores
        Args:
            occ (torch.Tensor): occupancy
            scores (torch.Tensor): attention scores
        Returns:
            torch.Tensor: branched_occ 
        """

        b, n = occ.shape[:2]
        nparts = scores.shape[-1]
        partocc = torch.zeros(
            size=(b, n, nparts), 
            dtype=torch.float32, 
            device=occ.device
        )
        
        part_indexs = torch.argmax(scores, dim=-1, keepdim=True)
        partocc.scatter_(2, part_indexs, occ)

        return partocc


    def _sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone()) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)

        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1).squeeze(-1)
        return c.transpose(1, 2)


    def _map_attention_to_part_indicator(self, indicator: torch.Tensor, part_attention: torch.Tensor) -> None:
        mparts = part_attention.shape[-1]
        max_indexs = torch.argmax(part_attention, dim=-1)
        pre_one_hot_indexs = F.one_hot(max_indexs, num_classes=mparts).float()
        indicator = repeat(indicator, 'b n c -> b n (repeat c)', repeat=mparts)
        part_indicator = part_attention + ((pre_one_hot_indexs * indicator) - part_attention).detach()
        return part_indicator


    def _compute_mahalanobis_distance(self, pts: torch.Tensor, rotations: torch.Tensor, translations: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Computes the Mahalanobis distance for a set of points.

        Args:
            pts (torch.Tensor): Points to compute the distance for.
            rotations (torch.Tensor): Rotation matrices.
            translations (torch.Tensor): Translation vectors.
            scales (torch.Tensor): Scale factors.

        Returns:
            torch.Tensor: The computed Mahalanobis distances.
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
        
        return dist


    def _get_smooth_mask(self, qpts: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        """returns the smooth mask for
        Args:
            qpts (torch.Tensor): query points
            poses (torch.Tensor): convex planes
        Returns:

            torch.Tensor: smooth mask
        """
        # Add your bounding box loss computation here
        trans, rots, scales = torch.split(poses, [3, 9, 1], dim=-1)

        # Compute the Mahalanobis distance
        distances = self._compute_mahalanobis_distance(pts=qpts, rotations=rots, translations=trans, scales=scales)
        distances = distances.permute(0, 2, 1)
        return distances


    def forward(self, qpts: torch.Tensor, pc: torch.Tensor, feats: torch.Tensor, enc_type="volume") -> Dict[str, torch.Tensor]:
        """forward pass of HiT

        Args:
            qpts (torch.Tensor): query points
            pc (torch.Tensor): input point cloud
            feats (torch.Tensor): predicting multiple pla   nes per point
        Returns:
            Dict[str, torch.Tensor]: occupancy and convex occupancy, 
        """
        B = feats.shape[0]
        n_query_pts = qpts.shape[1]

        # define the containers 
        all_convex_indicator, all_convex_sum_indicator = [], []
        all_convex_trans_sum, all_convex_half_planes, all_convex_offsets, all_convex_trans, all_convex_part_indicator, all_convex_part_indicator_hat = {}, {}, {}, {}, {}, {}
        all_convex_part_exists = {}
        
        if enc_type == "volume":
            # query_pos_enc = self._sample_grid_feature(qpts, feats)
            feats = feats.view(B, self.zf_dim, -1) 
            tokens = rearrange(feats, 'b f n -> b n f')
        elif enc_type == "pointnet":
            # Distance-based Nadaraya-Watson estimator interpolation of qpts with pc
            dists = torch.cdist(qpts, pc, p=2) 
            weights = F.softmax(-1 * dists, dim=-1)  
            tokens = torch.einsum('bqn,bnc->bqc', weights, feats)
        
        all_relations, all_part_sdf_logits, all_parent_child_indicator = {}, {}, {}
        all_blend_params = {}   
        for i in range(self.num_active_blocks):
            convex_codebook = self.convex_codebooks[i]

            if i == 0:
                convex_latents = tokens
            
            convex_latents, relations = convex_codebook(convex_latents, scores=None)

            if i > 0:
                relations = rearrange(relations, 'b h c p -> (b h) c p')
                all_relations[f'level_{i}'] = relations

                # straight through estimator
                max_indexs = torch.argmax(relations, dim=-1)
                pre_one_hot_indexs = F.one_hot(max_indexs, num_classes=self.mparts[i-1]).float()
                pre_one_hot_indexs = relations + (pre_one_hot_indexs - relations).detach()

                parent_indicator = all_convex_part_indicator_hat[f'level_{i-1}'].permute(0, 2, 1)
                parent_child_indicator = pre_one_hot_indexs @ parent_indicator
                parent_child_indicator = parent_child_indicator.permute(0, 2, 1)

                all_parent_child_indicator[f'level_{i}'] = parent_child_indicator

            convex_indicator, (convex_trans, part_sdf_logits, convex_indicator_sum_full, convex_normals, convex_offset, convex_part_indicator, blend_params) = self.multiconvex_decoder(qpts, convex_latents)
            convex_part_indicator = convex_part_indicator.permute(0, 2, 1, 3).squeeze(-1)
            convex_indicator_sum, convex_sum_trans = torch.split(convex_indicator_sum_full, [n_query_pts, self.mparts[i]], dim=1)

            if i > 0 and self.mask_mode == "mask": 
                convex_part_indicator_hat = convex_part_indicator * parent_child_indicator 
                convex_part_indicator_hat = torch.clamp(convex_part_indicator_hat, min=1e-6, max=1.0)
            elif i == 0: 
                convex_part_indicator_hat = convex_part_indicator
                        
            all_convex_indicator.append(convex_indicator)
            all_convex_sum_indicator.append(convex_indicator_sum)
            all_convex_part_indicator[f'level_{i}'] = convex_part_indicator  
            all_convex_part_indicator_hat[f'level_{i}'] = convex_part_indicator_hat
            
            all_convex_trans_sum[f'level_{i}'] = convex_sum_trans
            all_convex_offsets[f'level_{i}'] = convex_offset
            all_convex_trans[f'level_{i}'] = convex_trans
            all_convex_half_planes[f'level_{i}'] = convex_normals

            all_part_sdf_logits[f'level_{i}'] = part_sdf_logits
            all_blend_params[f'level_{i}'] = blend_params

        all_convex_indicator = torch.cat(all_convex_indicator, dim=-1)
        all_convex_sum_indicator = torch.cat(all_convex_sum_indicator, dim=-1)

        data = dict() 

        data['num_active_blocks'] = self.num_active_blocks  
  
        data['all_convex_indicator'] = all_convex_indicator
        data['all_convex_part_indicator'] = all_convex_part_indicator
        data['all_convex_sum_indicator'] = all_convex_sum_indicator
        data['all_convex_trans_sum'] = all_convex_trans_sum
        data['all_convex_offsets'] = all_convex_offsets
        data['all_convex_half_planes'] = all_convex_half_planes
        data['all_convex_trans'] = all_convex_trans
        data['all_convex_part_exists'] = all_convex_part_exists
        data['all_convex_part_indicator_hat'] = all_convex_part_indicator_hat

        data['all_relations'] = all_relations
        data['all_parent_child_indicator'] = all_parent_child_indicator
        data['all_blend_params'] = all_blend_params 
        
        return data


class PT_HiTModel_PointNet(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super(PT_HiTModel_PointNet, self).__init__()
        """HiT Model Defination
        Args:
            config (Config): Configuration dataclass containing model parameters.
        Returns:
            None
        """

        self.config = config

        self.encoder = PointNet(
            pf_dim=config.pf_dim,
            zf_dim=config.zf_dim,
            ef_dim=config.ef_dim,
            use_xyz=True,
            use_bn=False
        )

        self.decoder = HiTDecoder(config)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.hidden_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x
    

class PT_HiTModel_ConvOccNet(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super(PT_HiTModel_ConvOccNet, self).__init__()
        """HiT Model Defination
        Args:
            config (Config): Configuration dataclass containing model parameters.
        Returns:
            None
        """
        unet3d_kwargs = {} 
        unet3d_kwargs['num_levels'] = 3 
        unet3d_kwargs['f_maps'] = 32
        unet3d_kwargs['in_channels'] = config.zf_dim
        unet3d_kwargs['out_channels'] = config.zf_dim

        self.encoder = LocalPoolPointnet(
            zf_dim=config.zf_dim,
            inf_dim=config.pf_dim,
            ef_dim=config.ef_dim,
            use_xyz=config.use_xyz,
            use_bn=config.use_bn,
            plane_type=config.plane_type,
            grid_resolution=config.grid_resolution,
            unet3d_kwargs=unet3d_kwargs,
            unet3d=True
        )

        self.decoder = HiTDecoder(
            pf_dim=config.pf_dim,
            tf_dim=config.tf_dim,
            zf_dim=config.zf_dim,
            gf_dim=config.gf_dim,
            planef_dim=config.planef_dim,
            n_parts=config.n_parts,
            n_planes=config.n_planes,
            mask_mode=config.mask_mode
        )

        self.enc_type = "volume"

    def forward(self, pc: torch.Tensor, qpts: torch.Tensor, file_ids: torch.Tensor=None) -> Dict[str, torch.Tensor]:    
        feats = self.encoder(pc)
        data = self.decoder(qpts, pc, feats, enc_type=self.enc_type)
        return data


def HiTModelPointNet(**kwargs) -> nn.Module:
    """function to create a HiTModelPointNet instance."""
    model = PT_HiTModel_PointNet(ModelArgs(**kwargs))
    return model


def HiTModelTable(**kwargs) -> nn.Module:
    """function to create a HiTModelTable instance."""
    # Placeholder for actual implementation
    model = nn.Module()  # Replace with actual model
    return model


def HiTModelConvOccNet(**kwargs) -> nn.Module:
    """function to create a HiTModelConvOccNet instance."""
    model = PT_HiTModel_ConvOccNet(ModelArgs(**kwargs))
    return model


HiT_models = {
    'hit-pointnet': HiTModelPointNet, 
    'hit-table': HiTModelTable , 
    'hit-volume': HiTModelConvOccNet,
}