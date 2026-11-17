"""Differentiable, center-anchored directional profiles for images or feature maps.

This is a feature operator, not a target detector. It does not enforce monotonicity,
isotropy, or a foreground decision. Distances are in pixels of the *input tensor*.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RadialProfileConfig:
    """Geometry and readout settings; defaults are a starting point, not tuned."""

    num_directions: int = 8
    num_samples: int = 8  # positive radii; the center is sampled separately
    min_radius: float = 1.0
    max_radius: float = 8.0
    spacing: str = 'linear'  # linear | log; explicit radii override these fields
    radii: tuple[float, ...] | None = None
    components: tuple[str, ...] = ('intensity', 'contrast', 'first', 'second')
    fusion: str = 'learned'  # none | concat (lossless channel stack) | mean | learned
    hidden_channels: int = 8
    out_channels: int = 16
    radial_kernel_size: int = 3
    angular_kernel_size: int = 3
    opposite_pairs: bool = True  # needs an even direction count
    direction_pool: str = 'mean'  # mean (cyclic shift invariant) | flatten
    center_intensity: bool = True  # learned readout uses I(ray)-I(query)
    validity_as_feature: bool = False  # masks gate computation, not semantic input
    fusion_bias: bool = False  # zero profiles must not synthesize an edge response
    learnable_radii: bool = False  # ordered internal radii; outer radius fixed
    learnable_rotation: bool = False  # bounded common rotation; pairs stay opposite
    padding_mode: str = 'border'  # zeros | border | reflection
    chunk_size: int = 512  # query points per sampling/readout chunk


@dataclass
class RadialProfiles:
    """Raw profiles, retaining direction and distance rather than pooling them.

    values: B,N,C,T,K,R; valid: B,N,T,K,R. Invalid entries are zeroed.
    T follows config.components. ``points`` is B,N,2 in (x,y) pixel coordinates.
    All derivatives are signed and scaled by actual radial spacing. ``first``
    is outward *decrease*, -dI/dr, on each interval [r_(j-1),r_j]. ``second`` is
    d2I/dr2 at interior samples r_1..r_(R-1); its final slot is zero and invalid.
    A dense query has N=H*W in row-major order; image_shape records (H,W).
    """

    values: torch.Tensor
    valid: torch.Tensor
    points: torch.Tensor
    radii: torch.Tensor
    angles: torch.Tensor
    center_values: torch.Tensor  # B,N,C, used by optional centered learned readout
    image_shape: tuple[int, int]
    dense: bool


class DirectionalRadialOperator(nn.Module):
    """Sample rays, optionally differentiate, then optionally learn their readout.

    forward(image, points=None):
      image is floating B,C,H,W; optional points is B,N,2 in input-pixel (x,y).
      learned: B,out_channels,H,W or B,N,out_channels.
      mean:    B,C*T,H,W or B,N,C*T, a validity-weighted geometry-only baseline.
      none:    B,C,T,K,R,H,W or B,N,C,T,K,R, preserving both axes.
      concat:  B,C*T*K*R,H,W or B,N,C*T*K*R; lossless channel stacking.

    sample_profiles(...) returns RadialProfiles in query layout for diagnostics
    or custom fusion. Sampling is bilinear; gradients reach the input, floating
    query coordinates, and enabled geometry parameters. Learned fusion uses a
    shared radial Conv1d, learned radial pooling, circular angular convolution,
    optional opposite-pair differences, then a channel projection. No symmetry
    threshold or target-specific decision is built in. Geometry-only modes have
    no readout parameters. The default learned readout removes center brightness,
    excludes validity bits from its feature input, and has no biases. For local
    ROIs use forward_region(full_image, top, left, height, width) to read real
    surrounding pixels instead of turning a crop edge into missing observations.
    Use .to(device/dtype) as for other nn.Modules.
    """

    _COMPONENTS = {'intensity', 'contrast', 'first', 'second'}

    def __init__(self, in_channels: int, config: RadialProfileConfig | None = None):
        super().__init__()
        self.config = cfg = config or RadialProfileConfig()
        if in_channels < 1 or cfg.num_directions < 2 or cfg.chunk_size < 1:
            raise ValueError('Positive in_channels/chunk_size and >=2 directions required')
        if not cfg.components or len(set(cfg.components)) != len(cfg.components) or not set(cfg.components) <= self._COMPONENTS:
            raise ValueError('components must be unique entries from intensity/contrast/first/second')
        if cfg.fusion not in {'none', 'concat', 'mean', 'learned'} or cfg.direction_pool not in {'mean', 'flatten'}:
            raise ValueError('Unsupported fusion or direction_pool')
        if cfg.padding_mode not in {'zeros', 'border', 'reflection'}:
            raise ValueError('Unsupported padding_mode')
        if cfg.opposite_pairs and cfg.num_directions % 2:
            raise ValueError('opposite_pairs requires an even num_directions')
        if cfg.radii is not None:
            radii = torch.tensor(cfg.radii, dtype=torch.float32)
        else:
            if cfg.num_samples < 1 or not 0 < cfg.min_radius <= cfg.max_radius:
                raise ValueError('Need num_samples>=1 and 0<min_radius<=max_radius')
            if cfg.spacing == 'linear':
                radii = torch.linspace(cfg.min_radius, cfg.max_radius, cfg.num_samples)
            elif cfg.spacing == 'log':
                radii = torch.linspace(math.log(cfg.min_radius), math.log(cfg.max_radius), cfg.num_samples).exp()
            else:
                raise ValueError('spacing must be linear or log')
        if radii.ndim != 1 or not radii.numel() or not torch.isfinite(radii).all():
            raise ValueError('radii must be a finite, nonempty vector')
        gaps = torch.diff(torch.cat((radii.new_zeros(1), radii)))
        if not (gaps > 0).all():
            raise ValueError('radii must be positive and strictly increasing')
        if 'second' in cfg.components and radii.numel() < 2:
            raise ValueError('second derivative requires at least two positive radii')
        if cfg.learnable_radii and radii.numel() < 2:
            raise ValueError('learnable_radii needs at least one internal radius')
        self.in_channels = in_channels
        self.num_radii = radii.numel()
        self.max_radius = float(radii[-1])
        self.profile_channels = in_channels * len(cfg.components)
        if cfg.fusion == 'none':
            self.output_channels = None
        elif cfg.fusion == 'concat':
            self.output_channels = self.profile_channels*cfg.num_directions*self.num_radii
        else:
            self.output_channels = self.profile_channels if cfg.fusion == 'mean' else cfg.out_channels
        self.register_buffer('initial_radii', radii)
        self.register_buffer('base_angles', torch.arange(cfg.num_directions) * (2*math.pi/cfg.num_directions))
        if cfg.learnable_radii:
            # Positive intervals prevent sample crossing; normalization fixes support size.
            self.raw_gaps = nn.Parameter(gaps + torch.log(-torch.expm1(-gaps)))
        else:
            self.register_parameter('raw_gaps', None)
        if cfg.learnable_rotation:
            self.rotation = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter('rotation', None)
        if cfg.fusion == 'learned':
            if cfg.hidden_channels < 1 or cfg.out_channels < 1:
                raise ValueError('Positive learned channel counts required')
            if cfg.radial_kernel_size < 1 or cfg.radial_kernel_size % 2 == 0:
                raise ValueError('radial_kernel_size must be positive and odd')
            if cfg.angular_kernel_size < 1 or cfg.angular_kernel_size % 2 == 0 or cfg.angular_kernel_size//2 > cfg.num_directions:
                raise ValueError('angular_kernel_size must be positive, odd, and <=2*K+1')
            h = cfg.hidden_channels
            inputs = self.profile_channels + (len(cfg.components) if cfg.validity_as_feature else 0)
            self.radial = nn.Sequential(
                nn.Conv1d(inputs, h, cfg.radial_kernel_size, padding=cfg.radial_kernel_size//2, bias=cfg.fusion_bias),
                nn.GELU(), nn.Conv1d(h, h, cfg.radial_kernel_size, padding=cfg.radial_kernel_size//2, bias=cfg.fusion_bias), nn.GELU())
            self.radius_logits = nn.Parameter(torch.zeros(self.num_radii))
            self.angular = nn.Conv1d(h, h, cfg.angular_kernel_size, bias=cfg.fusion_bias)
            self.pair_fusion = nn.Conv1d(2*h, h, 1, bias=cfg.fusion_bias) if cfg.opposite_pairs else nn.Identity()
            self.projection = nn.Linear(h if cfg.direction_pool == 'mean' else h*cfg.num_directions, cfg.out_channels, bias=cfg.fusion_bias)

    def sampling_geometry(self):
        """Return current positive radii and angles; this does not detach gradients."""
        if self.raw_gaps is None:
            radii = self.initial_radii
        else:
            gaps = F.softplus(self.raw_gaps).clamp_min(torch.finfo(self.raw_gaps.dtype).eps)
            radii = self.initial_radii[-1] * gaps.cumsum(0) / gaps.sum()
        angles = self.base_angles
        if self.rotation is not None:
            angles = angles + (math.pi/self.config.num_directions)*self.rotation.tanh()
        return radii, angles

    def _queries(self, image, points):
        if image.ndim != 4 or image.shape[1] != self.in_channels or not image.is_floating_point():
            raise ValueError('image must be floating B,in_channels,H,W')
        b, _, height, width = image.shape
        if min(b, height, width) < 1:
            raise ValueError('Empty input dimensions are unsupported')
        dense = points is None
        if dense:
            yy, xx = torch.meshgrid(torch.arange(height, device=image.device, dtype=image.dtype),
                                    torch.arange(width, device=image.device, dtype=image.dtype), indexing='ij')
            points = torch.stack((xx, yy), -1).reshape(1, -1, 2).expand(b, -1, -1)
        else:
            if points.ndim != 3 or points.shape[0] != b or points.shape[-1] != 2 or not points.shape[1]:
                raise ValueError('points must be nonempty B,N,2 in (x,y) coordinates')
            if not points.is_floating_point():
                raise ValueError('points must be floating; integer conversion would sever position gradients')
            points = points.to(device=image.device, dtype=image.dtype)
        return points, dense

    def _sample(self, image, points, dense):
        b, channels, height, width = image.shape
        n = points.shape[1]
        radii, angles = self.sampling_geometry()
        radii, angles = radii.to(image), angles.to(image)
        vectors = torch.stack((angles.cos(), angles.sin()), -1)
        offsets = vectors[:, None, :] * radii[None, :, None]
        coords = torch.cat((points[:, :, None, :],
                            (points[:, :, None, None, :]+offsets[None, None]).flatten(2, 3)), 2)
        # Trig roundoff must not discard an entire axial ray on an image border.
        tolerance = 4*max(1e-7, torch.finfo(image.dtype).eps)*self.max_radius
        inside = ((coords[..., 0] >= -tolerance) & (coords[..., 0] <= width-1+tolerance)
                  & (coords[..., 1] >= -tolerance) & (coords[..., 1] <= height-1+tolerance))
        clamped = torch.stack((coords[..., 0].clamp(0, width-1), coords[..., 1].clamp(0, height-1)), -1)
        coords = torch.where(inside[..., None], clamped, coords)
        # align_corners=False maps pixel centers correctly, including one-pixel axes.
        grid = (coords+.5)*image.new_tensor((2/width, 2/height))-1
        sampled = F.grid_sample(image, grid.reshape(b, -1, 1, 2), mode='bilinear',
                                padding_mode=self.config.padding_mode, align_corners=False)
        sampled = sampled.reshape(b, channels, n, -1).permute(0, 2, 1, 3)
        center = sampled[..., :1, None]
        rings = sampled[..., 1:].reshape(b, n, channels, self.config.num_directions, self.num_radii)
        anchor = center.expand(-1, -1, -1, self.config.num_directions, -1)
        full = torch.cat((anchor, rings), -1)
        valid_center = inside[..., :1, None]
        valid_rings = inside[..., 1:].reshape(b, n, self.config.num_directions, self.num_radii)
        valid_full = torch.cat((valid_center.expand(-1, -1, self.config.num_directions, -1), valid_rings), -1)
        anchored_valid = valid_rings & valid_center
        spacing = torch.diff(torch.cat((radii.new_zeros(1), radii)))
        slope = torch.diff(full, dim=-1)/spacing
        first_valid = valid_full[..., 1:] & valid_full[..., :-1] & valid_center
        parts, masks = [], []
        for component in self.config.components:
            if component == 'intensity':
                value, valid = rings, anchored_valid
            elif component == 'contrast':
                value, valid = anchor-rings, anchored_valid
            elif component == 'first':
                value, valid = -slope, first_valid
            else:
                curvature = 2*torch.diff(slope, dim=-1)/(spacing[:-1]+spacing[1:])
                value = F.pad(curvature, (0, 1))
                valid = F.pad(first_valid[..., :-1] & first_valid[..., 1:], (0, 1), value=False)
            parts.append(value.masked_fill(~valid[:, :, None], 0))
            masks.append(valid)
        return RadialProfiles(torch.stack(parts, 3), torch.stack(masks, 2), points, radii, angles,
                              sampled[..., 0], (height, width), dense)

    def sample_profiles(self, image, points=None):
        """Return raw query-layout profiles; raw output necessarily costs O(B*N*C*T*K*R)."""
        points, dense = self._queries(image, points)
        chunks = [self._sample(image, p, dense) for p in points.split(self.config.chunk_size, 1)]
        first = chunks[0]
        return RadialProfiles(torch.cat([p.values for p in chunks], 1), torch.cat([p.valid for p in chunks], 1),
                              points, first.radii, first.angles, torch.cat([p.center_values for p in chunks], 1), first.image_shape, dense)

    def _readout(self, profiles):
        cfg = self.config
        values, valid = profiles.values, profiles.valid
        b, n, c, t, k, r = values.shape
        if cfg.fusion == 'none':
            return values
        if cfg.fusion == 'concat':
            return values.flatten(2)
        if cfg.fusion == 'mean':
            return (values.sum((-2, -1))/valid.sum((-2, -1)).clamp_min(1)[:, :, None]).flatten(2)
        if cfg.center_intensity and 'intensity' in cfg.components:
            index = cfg.components.index('intensity')
            values = values.clone()
            values[:, :, :, index] = ((values[:, :, :, index]-profiles.center_values[..., None, None])
                                      *valid[:, :, None, index])
        # Shared weights read every direction's ordered radial sequence separately.
        data = values.permute(0, 1, 4, 2, 3, 5).reshape(b*n*k, c*t, r)
        mask = valid.permute(0, 1, 3, 2, 4).reshape(b*n*k, t, r)
        radial = self.radial(torch.cat((data, mask.to(data)), 1) if cfg.validity_as_feature else data)
        valid_radius = mask.any(1)
        weight = self.radius_logits.softmax(0)[None]*valid_radius
        radial = (radial*weight[:, None]).sum(-1)/weight.sum(-1, keepdim=True).clamp_min(1e-8)
        radial = radial.reshape(b*n, k, -1).transpose(1, 2)
        angle_pad = cfg.angular_kernel_size//2
        mixed = F.gelu(self.angular(F.pad(radial, (angle_pad, angle_pad), mode='circular')))
        if cfg.opposite_pairs:
            opposite = (radial-torch.roll(radial, k//2, -1)).abs()
            mixed = F.gelu(self.pair_fusion(torch.cat((mixed, opposite), 1)))
        valid_direction = valid_radius.any(-1).reshape(b*n, k)
        mixed = mixed*valid_direction[:, None]
        if cfg.direction_pool == 'mean':
            mixed = mixed.sum(-1)/valid_direction.sum(-1, keepdim=True).clamp_min(1)
        else:
            mixed = mixed.flatten(1)
        result = self.projection(mixed).reshape(b, n, -1)
        return result*valid.any((-3, -2, -1))[:, :, None]

    def _spatial_layout(self, result, height, width):
        b = result.shape[0]
        if self.config.fusion == 'none':
            return result.permute(0, 2, 3, 4, 5, 1).reshape(b, self.in_channels, len(self.config.components),
                                                         self.config.num_directions, self.num_radii, height, width)
        return result.transpose(1, 2).reshape(b, self.output_channels, height, width)

    def forward(self, image, points=None):
        queries, dense = self._queries(image, points)
        parts = [self._readout(self._sample(image, p, dense)) for p in queries.split(self.config.chunk_size, 1)]
        result = torch.cat(parts, 1)
        return self._spatial_layout(result, *image.shape[-2:]) if dense else result

    def forward_region(self, image, top: int, left: int, height: int, width: int):
        """Return an ROI feature map while sampling rays from its real surroundings.

        The region must lie inside image. For an interior ROI, context of at least
        max_radius pixels prevents crop-edge artifacts. Real image boundaries still
        have missing information and are handled by validity masks. For learned
        floating ROI coordinates, use forward(image, points) instead.
        """
        if any(not isinstance(v, int) for v in (top, left, height, width)):
            raise ValueError('Region dimensions must be integers; use points for floating queries')
        if min(top, left) < 0 or min(height, width) < 1 or top+height > image.shape[-2] or left+width > image.shape[-1]:
            raise ValueError('Region must lie inside image')
        self._queries(image, image.new_zeros((image.shape[0], 1, 2)))
        yy, xx = torch.meshgrid(torch.arange(top, top+height, device=image.device, dtype=image.dtype),
                                torch.arange(left, left+width, device=image.device, dtype=image.dtype), indexing='ij')
        points = torch.stack((xx, yy), -1).reshape(1, -1, 2).expand(image.shape[0], -1, -1)
        return self._spatial_layout(self.forward(image, points), height, width)
