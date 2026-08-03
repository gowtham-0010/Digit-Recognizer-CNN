"""Small shared helper functions for the app."""

import torch
import torch.nn.functional as F


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert raw model logits into a softmax probability distribution."""
    return F.softmax(logits, dim=1)


def top_k_predictions(probabilities: torch.Tensor, k: int = 3):
    """
    Return the top-k (label, confidence) pairs from a (1, num_classes)
    probability tensor, sorted by descending confidence.
    """
    probs = probabilities.squeeze(0)
    top_probs, top_idxs = torch.topk(probs, k)
    return [(int(idx), float(prob)) for idx, prob in zip(top_idxs, top_probs)]
