"""
PatchCore-lite anomaly detector.

Builds a patch-level feature memory bank from normal (good) images using a
frozen pretrained ResNet-18 backbone. At inference the anomaly score is the
maximum k-NN distance across all image patches to the memory bank.

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly Detection"
           (CVPR 2022) — this is a simplified CPU-friendly variant.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm


class PatchCore:
    """
    Args:
        k:               Number of nearest neighbours used for distance scoring.
        subsample_ratio: Fraction of patches to keep in the memory bank after
                         random coreset subsampling (trades recall for speed).
    """

    def __init__(self, k=3, subsample_ratio=0.1):
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False

        # Hooks capture intermediate feature maps without modifying the backbone.
        self._features = {}
        backbone.layer1.register_forward_hook(self._make_hook("layer1"))
        backbone.layer2.register_forward_hook(self._make_hook("layer2"))
        backbone.layer3.register_forward_hook(self._make_hook("layer3"))
        self.backbone = backbone

        self.k = k
        self.subsample_ratio = subsample_ratio
        self.memory_bank = None  # numpy (N, C) after fit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_hook(self, name):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def _get_patch_features(self, images):
        """
        Forward images through the backbone and return concatenated multi-scale
        patch features of shape (B, 448, H, W).

        layer1 → (B,  64, H,   W  ) downsampled to layer2 spatial size
        layer2 → (B, 128, H,   W  )
        layer3 → (B, 256, H/2, W/2) upsampled   to layer2 spatial size
        concat → (B, 448, H,   W  )

        Layer1 is included at layer2's spatial resolution so that fine-grained
        low-level features (edges, textures from small contaminants) enrich the
        patch descriptors without multiplying the patch count.
        """
        self._features.clear()
        with torch.no_grad():
            self.backbone(images)

        f1 = self._features["layer1"]  # (B,  64, H1, W1)
        f2 = self._features["layer2"]  # (B, 128, H2, W2)  — reference spatial size
        f3 = self._features["layer3"]  # (B, 256, H3, W3)

        f1_dn = F.adaptive_avg_pool2d(f1, f2.shape[2:])
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([f1_dn, f2, f3_up], dim=1)  # (B, 448, H2, W2)

    def _knn_distances(self, query):
        """
        Mean distance to k nearest neighbours in the memory bank.

        query: numpy (N_q, C)
        returns: numpy (N_q,)
        """
        q = torch.from_numpy(query)
        m = torch.from_numpy(self.memory_bank)

        # Euclidean distances via squared dot-product expansion
        q_sq = (q ** 2).sum(dim=1, keepdim=True)          # (N_q, 1)
        m_sq = (m ** 2).sum(dim=1, keepdim=True)          # (N_m, 1)
        dist_sq = (q_sq + m_sq.T - 2.0 * (q @ m.T)).clamp(min=0.0)
        dists = dist_sq.sqrt()                             # (N_q, N_m)

        topk = dists.topk(self.k, dim=1, largest=False).values  # (N_q, k)
        return topk.mean(dim=1).numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, good_loader):
        """Build the memory bank from a DataLoader of normal (good) images."""
        all_patches = []
        for images, _ in tqdm(good_loader, desc="Building memory bank"):
            feats = self._get_patch_features(images)  # (B, 384, H, W)
            B, C, H, W = feats.shape
            patches = feats.permute(0, 2, 3, 1).reshape(-1, C)
            all_patches.append(patches.numpy())

        all_patches = np.concatenate(all_patches, axis=0).astype(np.float32)

        n_keep = max(1, int(len(all_patches) * self.subsample_ratio))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_patches), size=n_keep, replace=False)
        self.memory_bank = all_patches[idx]
        print(f"Memory bank: {self.memory_bank.shape[0]:,} patches "
              f"(subsampled from {len(all_patches):,})")

    def score(self, image_tensor):
        """
        Return the anomaly score for a single image tensor (C, H, W).
        Higher score means more anomalous.
        """
        feats = self._get_patch_features(image_tensor.unsqueeze(0))
        _, C, H, W = feats.shape
        patches = feats.permute(0, 2, 3, 1).reshape(-1, C).numpy()
        patch_scores = self._knn_distances(patches)
        return float(patch_scores.max())

    def save(self, path):
        np.save(path, self.memory_bank)
        print(f"Memory bank saved → {path}")

    def load(self, path):
        self.memory_bank = np.load(path)
        print(f"Memory bank loaded: {self.memory_bank.shape[0]:,} patches")
