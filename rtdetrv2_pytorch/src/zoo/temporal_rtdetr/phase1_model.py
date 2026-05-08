import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, List, Tuple, Optional
from ..rtdetr.rtdetrv2_decoder import RTDETRTransformerv2, TransformerDecoder

class TemporalFusionBlock(nn.Module):
    """
    Fusion block for combining non-key frame features (S) with cached key frame features (CCFF)
    """
    def __init__(self, s_channels: int, hidden_dim: int):
        super().__init__()
        self.s_channels = s_channels
        self.hidden_dim = hidden_dim
        
        # Project S features to hidden_dim
        self.s_proj = nn.Sequential(
            nn.Conv2d(s_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer (combines projected S with CCFF)
        self.fusion = nn.Sequential(
            # kernel_size=3, padding=2, dilation=2
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # 1x1 conv remains the same
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

        # Initialize the final Conv2d and BatchNorm to output exactly 0.0
        nn.init.constant_(self.fusion[3].weight, 0.0)
        nn.init.constant_(self.fusion[4].weight, 0.0)
        nn.init.constant_(self.fusion[4].bias, 0.0)
        
    def forward(self, s_feat: torch.Tensor, ccff_feat: torch.Tensor) -> torch.Tensor:
        """
        Fuse S features with cached CCFF features
        
        Args:
            s_feat: Non-key frame feature [B, s_channels, H, W]
            ccff_feat: Cached key frame feature [B, hidden_dim, H, W]
        
        Returns:
            fused_feat: Fused feature [B, hidden_dim, H, W]
        """
        # Project S to hidden_dim
        s_proj = self.s_proj(s_feat)  # [B, hidden_dim, H, W]
        
        # Concatenate and fuse
        concat = torch.cat([s_proj, ccff_feat], dim=1)  # [B, hidden_dim*2, H, W]
        fused = self.fusion(concat)  # [B, hidden_dim, H, W]
        
        # Residual connection
        fused = fused + ccff_feat
        
        return fused

class LightweightDecoder(RTDETRTransformerv2):
    """
    - REQUIRES query_emb and pos_emb from key frame
    - No denoising
    - No aux_loss
    """
    def __init__(
        self,
        full_decoder: RTDETRTransformerv2,
        num_layers: int = 1,
        decouple_prediction_heads: bool = False,
    ):
        nn.Module.__init__(self)
        
        # self.hidden_dim = full_decoder.hidden_dim       
        self.num_levels = full_decoder.num_levels
        # self.num_classes = full_decoder.num_classes
        # self.num_queries = full_decoder.num_queries

        # Copy decoder layers
        self.num_decoder_layers = min(num_layers, full_decoder.decoder.num_layers)
        self.decoder = TransformerDecoder(
            full_decoder.hidden_dim, 
            copy.deepcopy(full_decoder.decoder.layers[-1]), 
            self.num_decoder_layers
        )

        # Share by direct memory reference.
        self.input_proj = full_decoder.input_proj
        self._set_prediction_modules(
            full_decoder=full_decoder,
            decouple_prediction_heads=decouple_prediction_heads,
        )
        
        self.eval_spatial_size = full_decoder.eval_spatial_size

    def _set_prediction_modules(self, full_decoder: RTDETRTransformerv2, decouple_prediction_heads: bool):
        if decouple_prediction_heads:
            self.query_pos_head = copy.deepcopy(full_decoder.query_pos_head)
            self.dec_score_head = nn.ModuleList(
                [copy.deepcopy(head) for head in list(full_decoder.dec_score_head[-self.num_decoder_layers:])]
            )
            self.dec_bbox_head = nn.ModuleList(
                [copy.deepcopy(head) for head in list(full_decoder.dec_bbox_head[-self.num_decoder_layers:])]
            )
        else:
            self.query_pos_head = full_decoder.query_pos_head
            # Slice the ModuleList to grab only the LAST `num_layers` heads
            self.dec_score_head = nn.ModuleList(
                list(full_decoder.dec_score_head[-self.num_decoder_layers:])
            )
            self.dec_bbox_head = nn.ModuleList(
                list(full_decoder.dec_bbox_head[-self.num_decoder_layers:])
            )
        self.decoupled_prediction_heads = decouple_prediction_heads

    def decouple_prediction_modules(self):
        if self.decoupled_prediction_heads:
            return
        self.query_pos_head = copy.deepcopy(self.query_pos_head)
        self.dec_score_head = nn.ModuleList([copy.deepcopy(head) for head in self.dec_score_head])
        self.dec_bbox_head = nn.ModuleList([copy.deepcopy(head) for head in self.dec_bbox_head])
        self.decoupled_prediction_heads = True
    
    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def forward(self, feats, cached_content, cached_points_unact):
        """
        Forward pass using cached query embeddings from key frame
        
        Args:
            feats: List of multi-scale features [feat1, feat2, feat3]
            cached_content: Cached content from key frame [B, hidden_dim, H, W]
            cached_points_unact: Cached reference points from key frame [B, num_queries, 4] (REQUIRED)
        
        Returns:
            outputs: Dict with 'pred_logits' and 'pred_boxes' only
        """
        # Get input proj
        memory, spatial_shapes = self._get_encoder_input(feats)
        
        out_bboxes, out_logits = self.decoder(
            cached_content,
            cached_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=None,
        )
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        # pred_logits = output['pred_logits']
        # pred_boxes = output['pred_boxes']
        
        return out


class AdaptivePropagationGate(nn.Module):
    """Lightweight APG gate for key/non-key routing."""

    def __init__(self, in_channels: int = 512, hidden_channels: int = 64, pool_size: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.pool_size = int(pool_size)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.fc = nn.Linear(hidden_channels * self.pool_size * self.pool_size, 1)

    def forward(self, prev_key_s5: torch.Tensor, current_s5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prev_key_s5: Previous key S5 feature [B, C, H, W]
            current_s5: Current frame S5 feature [B, C, H, W]

        Returns:
            logits: APG logits [B]
            probs: APG probabilities [B]
        """
        # Residual feature: [B, C, H, W]
        residual = current_s5 - prev_key_s5
        # APG hidden map: [B, hidden_channels, H, W]
        hidden = self.conv(residual)
        # Pooled APG descriptor: [B, hidden_channels, pool_size, pool_size]
        pooled = self.pool(hidden)
        # Flattened APG descriptor: [B, hidden_channels * pool_size * pool_size]
        flattened = pooled.flatten(1)
        logits = self.fc(flattened).squeeze(-1)
        probs = torch.sigmoid(logits)
        return logits, probs


class TemporalRTDETR(nn.Module):
    """
    Temporal RT-DETR for Phase 1 training
    - Key frame: Backbone + Encoder + Decoder
    - Non-key frame: Backbone + Fusion + Lightweight Decoder
    """
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        use_lightweight_decoder: bool = True,
        reuse_position: int = 0,
        enable_apg: bool = False,
        apg_in_channels: int = 512,
        apg_hidden_channels: int = 64,
        apg_pool_size: int = 4,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.use_lightweight_decoder = use_lightweight_decoder
        self.reuse_position = int(reuse_position)
        self.enable_apg = bool(enable_apg)
        self.decoder_num_layers = getattr(getattr(decoder, 'decoder', None), 'num_layers', None)
        if self.reuse_position < 0:
            raise ValueError(f"reuse_position must be >= 0, but got {self.reuse_position}")
        if self.decoder_num_layers is not None and self.reuse_position > self.decoder_num_layers:
            raise ValueError(
                f"reuse_position must be in [0, {self.decoder_num_layers}] for this decoder, "
                f"but got {self.reuse_position}"
            )
        
        # Cached features from key frame
        self.cached_ccff = None
        self.cached_content = None
        self.cached_points_unact = None
        self.cached_key_s5 = None

        device = next(decoder.parameters()).device

        self.fusion_blocks = nn.ModuleList([
            TemporalFusionBlock(s_channels=128, hidden_dim=hidden_dim).to(device),  # S3 + CCFF1
            TemporalFusionBlock(s_channels=256, hidden_dim=hidden_dim).to(device),  # S4 + CCFF2
            TemporalFusionBlock(s_channels=512, hidden_dim=hidden_dim).to(device),  # S5 + CCFF3
        ])

        # Create lightweight decoder if needed
        if use_lightweight_decoder:
            self.lightweight_decoder = LightweightDecoder(
                full_decoder=decoder,
                num_layers=1
            )
        else:
            self.lightweight_decoder = None

        self.apg = None
        if self.enable_apg:
            self.apg = AdaptivePropagationGate(
                in_channels=apg_in_channels,
                hidden_channels=apg_hidden_channels,
                pool_size=apg_pool_size,
            )
        
        print(f"  Success!")
        print(f"  - Use lightweight decoder: {use_lightweight_decoder}")
        print(f"  - Reuse position: {self.reuse_position}")
    
    def forward_key_frame(self, img: torch.Tensor, targets: Optional[List[Dict]] = None) -> Tuple:
        """
        Forward key frame through full pipeline and cache features
        
        Args:
            img: Key frame image [B, C, H, W]
            targets: Ground truth annotations
        
        Returns:
            outputs: Detection outputs
            ccff_features: Cached multi-scale features
            query_embeddings: Cached query embeddings (optional)
        """
        backbone_features = self.backbone(img)
        c3, c4, c5 = backbone_features[-3:]
        encoder_output = self.encoder([c3, c4, c5])
        self.cached_ccff = [feat.detach() for feat in encoder_output]
        self.cached_key_s5 = c5.detach()
        outputs, cached_query_states = self.decoder(encoder_output, return_query=True, targets=targets)

        if self.reuse_position >= len(cached_query_states):
            raise ValueError(
                f"reuse_position={self.reuse_position} is out of range for available positions "
                f"[0, {len(cached_query_states) - 1}]"
            )

        cached_content, cached_points_unact = cached_query_states[self.reuse_position]
        self.cached_content = cached_content[:, :self.num_queries, :].detach()
        self.cached_points_unact = cached_points_unact[:, :self.num_queries, :].detach()
        
        return outputs
    
    def forward_non_key_frame(self, img: torch.Tensor, targets: Optional[List[Dict]] = None, return_fused: bool = False,
                              cached_ccff: Optional[List[torch.Tensor]] = None,
                              cached_content: Optional[torch.Tensor] = None,
                              cached_points_unact: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward non-key frame through lightweight pipeline with fusion
        
        Args:
            img: Non-key frame image [B, C, H, W]
            targets: Ground truth annotations
            return_fused: Whether to return fused features
            cached_ccff: Optional cached multi-scale features (for ONNX export)
            cached_content: Optional cached query content (for ONNX export)
            cached_points_unact: Optional cached reference points (for ONNX export)
        
        Returns:
            outputs: Detection outputs
        """
        # Override internal cache if provided (for deployment/ONNX)
        if cached_ccff is not None:
            self.cached_ccff = cached_ccff
        if cached_content is not None:
            self.cached_content = cached_content
        if cached_points_unact is not None:
            self.cached_points_unact = cached_points_unact

        if self.cached_ccff is None:
            raise RuntimeError("Key frame must be processed first to cache CCFF features")
        if self.cached_content is None or self.cached_points_unact is None:
            raise RuntimeError("Key frame must cache decoder queries before non-key inference")
        
        # Extract multi-scale features from backbone
        backbone_features = self.backbone(img)
        
        # Get last 3 scales: S3, S4, S5
        s3, s4, s5 = backbone_features[-3:]
        s_features = [s3, s4, s5]
        
        # Fuse each scale with cached CCFF
        fused_features = []
        for _, (s_feat, ccff_feat, fusion_block) in enumerate(zip(s_features, self.cached_ccff, self.fusion_blocks)):
            fused = fusion_block(s_feat, ccff_feat)
            fused_features.append(fused)
        
        # Prepare decoder input (fused multi-scale features)
        decoder_input = fused_features
        
        # Use lightweight or full decoder
        if self.use_lightweight_decoder and self.lightweight_decoder is not None:
            # Use single-layer decoder (trainable)
            # Call with only positional argument (memory)
            outputs = self.lightweight_decoder(decoder_input, self.cached_content, self.cached_points_unact)
        else:
            # Use full decoder
            outputs = self.decoder(decoder_input, targets=targets)

        if return_fused:
            return outputs, fused_features
        return outputs

    def extract_s5(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract raw S5 feature from backbone.

        Args:
            img: Input image [B, C, H, W]
        Returns:
            s5: Backbone S5 feature [B, C5, H5, W5]
        """
        backbone_features = self.backbone(img)
        return backbone_features[-1]

    def forward_apg(self, prev_key_s5: torch.Tensor, current_s5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.apg is None:
            raise RuntimeError("APG is disabled. Set enable_apg=True in model config to use forward_apg.")
        return self.apg(prev_key_s5, current_s5)

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

    def forward(self, key_frame: torch.Tensor, non_key_frame: torch.Tensor, 
                key_targets: Optional[List[Dict]] = None, 
                non_key_targets: Optional[List[Dict]] = None) -> Tuple[Dict, Dict]:
        """
        Forward both key and non-key frames
        
        Args:
            key_frame: Key frame image
            non_key_frame: Non-key frame image
            key_targets: Key frame targets
            non_key_targets: Non-key frame targets
        
        Returns:
            key_outputs, non_key_outputs
        """
        # Process key frame
        key_outputs = self.forward_key_frame(key_frame, key_targets)
        
        # Process non-key frame
        non_key_outputs = self.forward_non_key_frame(non_key_frame, non_key_targets)
        
        return key_outputs, non_key_outputs

    def decouple_non_key_prediction_heads(self):
        if not self.use_lightweight_decoder or self.lightweight_decoder is None:
            raise RuntimeError("Lightweight decoder is required to decouple non-key prediction modules")
        self.lightweight_decoder.decouple_prediction_modules()


def build_temporal_rtdetr(cfg):
    """Build Temporal RT-DETR model from config"""
    # Import backbone, encoder, decoder builders
    from ..rtdetr import build_backbone, build_encoder, build_decoder
    
    # Build components
    backbone = build_backbone(cfg)
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)
    
    # Build temporal model
    model = TemporalRTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        num_classes=cfg.num_classes,
        hidden_dim=cfg.hidden_dim,
        num_queries=cfg.num_queries,
        use_lightweight_decoder=cfg.get('use_lightweight_decoder', True),
        reuse_position=cfg.get('reuse_position', 0),
        enable_apg=cfg.get('enable_apg', False),
        apg_in_channels=cfg.get('apg_in_channels', 512),
        apg_hidden_channels=cfg.get('apg_hidden_channels', 64),
        apg_pool_size=cfg.get('apg_pool_size', 4),
    )
    
    return model
