import torch
import torch.nn as nn
from typing import Sequence, Tuple, Union

# Optimize for Tensor Cores on CUDA devices
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import (
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
)
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep

from mmengine.runner import load_checkpoint

# # Modern checkpoint loading approach - try multiple sources
# try:
#     from mmcv.runner import load_checkpoint
# except ImportError:
#     try:
#         from mmengine.runner import load_checkpoint
#     except ImportError:
#         # Fallback to torch.load if mmcv/mmengine not available
#         def load_checkpoint(model, filename, strict=False, revise_keys=None, **kwargs):
#             checkpoint = torch.load(filename, map_location='cpu')
#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             else:
#                 state_dict = checkpoint
            
#             # Handle revise_keys if provided
#             if revise_keys:
#                 for old_key, new_key in revise_keys:
#                     if old_key in state_dict:
#                         state_dict[new_key] = state_dict.pop(old_key)
            
#             model.load_state_dict(state_dict, strict=strict)
#             print(f"Loaded checkpoint from {filename}")
#             return checkpoint


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        pretrained: Union[str, None],
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        # revise_keys=[],
        revise_keys=[("^model\\.encoder\\.", "^vit\\.")],
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.pretrained = pretrained
        self.num_layers = num_layers
        assert self.num_layers // 4
        self.stage_layers = self.num_layers // 4
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(
            img_d // p_d for img_d, p_d in zip(img_size, self.patch_size)
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )
        self.proj_axes = (0, spatial_dims + 1) + tuple(
            d + 1 for d in range(spatial_dims)
        )
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

        self.init_weights(revise_keys=revise_keys)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def init_weights(self, pretrained=None, revise_keys=[]):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
            revise_keys (list, optional): List of (old_key, new_key) tuples for key mapping.
                Defaults to [].
        """

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            print(f"Loading checkpoints from {self.pretrained}")
            try:
                load_checkpoint(
                    self,
                    filename=self.pretrained,
                    strict=False,
                    revise_keys=revise_keys,
                )
                print(f"Successfully loaded checkpoint from {self.pretrained}")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint from {self.pretrained}: {e}")
                print("Continuing with random initialization...")
        elif self.pretrained is None:
            print("No pretrained weights provided, using random initialization")
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[self.stage_layers * 1]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[self.stage_layers * 2]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[self.stage_layers * 3]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)

    def monitor_parameters(self, log_every_n_steps=100):
        """Monitor model parameters and gradients for debugging purposes.
        
        Args:
            log_every_n_steps (int): How often to log parameter statistics.
        """
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        
        self._step_count += 1
        
        if self._step_count % log_every_n_steps == 0:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            print(f"Step {self._step_count}: Model Statistics")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
            # Check for NaN or Inf values in parameters
            has_nan = any(torch.isnan(p).any() for p in self.parameters())
            has_inf = any(torch.isinf(p).any() for p in self.parameters())
            
            if has_nan:
                print("  WARNING: NaN values detected in parameters!")
            if has_inf:
                print("  WARNING: Inf values detected in parameters!")
            
            # Check gradients if they exist
            if any(p.grad is not None for p in self.parameters()):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
                print(f"  Gradient norm: {grad_norm:.6f}")
            
            print("-" * 50)


if __name__ == "__main__":
    import torch

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test input
    x = torch.randn(1, 1, 96, 96, 96).to(device)
    print(f"Input shape: {x.shape}")
    print(f"Input mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")

    # Create model
    model = UNETR(
        pretrained=None,
        in_channels=1,
        out_channels=14,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=1024,
        mlp_dim=4096,
        num_layers=24,
        num_heads=16,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        spatial_dims=3,
        revise_keys=[],
    ).to(device)

    # Monitor model parameters
    model.monitor_parameters(log_every_n_steps=1)

    # Test forward pass
    try:
        with torch.no_grad():
            y = model(x)
        print(f"Output shape: {y.shape}")
        print(f"Output mean: {y.mean().item():.6f}, std: {y.std().item():.6f}")
        
        # Check for NaN or Inf in output
        if torch.isnan(y).any():
            print("WARNING: NaN values in output!")
        if torch.isinf(y).any():
            print("WARNING: Inf values in output!")
            
        print("Model test completed successfully!")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
