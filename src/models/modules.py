import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List

# import src.libs.pointnet2.pointnet2_utils as pointnet2_utils

# UNet3D utils.
def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index


def normalize_coordinate(p, padding=0.1, plane='xz'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


def normalize_3d_coordinate(p, padding=0.0):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    # print(torch.min(p), torch.max(p))
    
    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    # p_nor = p_nor + 0.5 # range (0, 1)
    p_nor = (p_nor + 1.0) / 2.0

    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in: int=32, size_out: int=64, size_h: int=None) -> None:
        super(ResnetBlockFC, self).__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.LeakyReLU(negative_slope=0.02)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


# pointnet encoder utilities
class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super(_BNBase, self).__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size, name=""):
        super(BatchNorm2d, self).__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class _ConvBase(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        dilation,
        activation,
        bn,
        init,
        conv=None,
        norm_layer=None,
        bias=True,
        preact=False,
        name="",
    ):
        super(_ConvBase, self).__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = norm_layer(out_size)
            else:
                bn_unit = norm_layer(in_size)

        if preact:
            if bn:
                self.add_module(name + "normlayer", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

        self.add_module(name + "conv", conv_unit)

        if not preact:
            if bn:
                self.add_module(name + "normlayer", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)


class Conv2d(_ConvBase):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm2d,
    ):
        # type: (Conv2d, int, int, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Any, bool, Any, bool, bool, AnyStr, _BNBase) -> None
        super(Conv2d, self).__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            norm_layer=norm_layer,
            bias=bias,
            preact=preact,
            name=name,
        )


class SharedMLP(nn.Sequential):
    def __init__(
        self,
        args,
        bn=False,
        activation=nn.ReLU(inplace=True),
        preact=False,
        first=False,
        name="",
    ):
        super(SharedMLP, self).__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + "layer{}".format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0))
                    else None,
                    preact=preact,
                ),
            )


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz, features=None):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )
