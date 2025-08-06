import torch 
import flashinfer
from dataclasses import dataclass

@dataclass 
class SamplingConfig:
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    repetition_window: int
    cfg_scale: float

def greedy_sampling(logits):
    """
    Greedy sampling from logits.
    Returns the index of the maximum logit.
    """
    samples = torch.argmax(logits, dim=-1)
    return samples

def top_k_sampling(logits, top_k, temperature):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    samples = flashinfer.sampling.top_k_sampling_from_probs(
        probs=probs, 
        top_k=top_k, 
        deterministic=True,
    )

    return samples

def top_p_sampling(logits, top_p, temperature):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    samples = flashinfer.sampling.top_p_sampling_from_probs(
        probs=probs, 
        top_p=top_p, 
        deterministic=True,
    )

    return samples

def top_k_top_p_sampling(logits, top_k, top_p, temperature, filter_apply_order="top_k_first"):
    logits = logits / temperature

    samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits=logits, 
        top_k=top_k, 
        top_p=top_p,
        filter_apply_order=filter_apply_order, 
        deterministic=True,
    )

    return samples

def min_p_sampling(logits, min_p, temperature):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    samples = flashinfer.sampling.min_p_sampling_from_probs(
        probs=probs, 
        min_p=min_p, 
        deterministic=True,
    )

    return samples

def apply_repetition_penalty(
    logits: torch.Tensor, 
    repetition_cache: torch.Tensor, 
    penalty: float
) -> torch.Tensor:
    """
    Apply repetition penalty to logits based on the repetition cache.

    Args: 
        logits: Tensor of shape (n_codebooks, vocab_size) containing the logits.
        repetition_cache: Tensor of shape (window_size, n_codebooks, vocab_size) indicating which tokens have appeared.
        penalty: Float value to apply as a penalty for repeated tokens.

    Returns:
        Logits tensor of shape (n_codebooks, vocab_size) with the repetition penalty applied.
    """
    # OR operation over window_size dimension. 
    appearance_mask = repetition_cache.any(dim=0)
    
    logits = torch.where((logits > 0) & appearance_mask, logits / penalty, logits)
    logits = torch.where((logits <= 0) & appearance_mask, logits * penalty, logits)
    
    return logits

def update_repetition_penalty_cache(
    repetition_cache: torch.Tensor, 
    output_ids: torch.Tensor, 
    window_size: int
) -> None:
    """
    Update the repetition penalty cache with the newly generated tokens.
    
    Args:
        repetition_cache: Tensor of shape (window_size, n_codebooks, vocab_size).
        output_ids: Tensor of shape (n_codebooks) containing the newly generated tokens.
        window_size: Size of the repetition penalty window.
    
    Returns:
        Updated repetition cache tensor.
    """
    n_codebooks = repetition_cache.shape[1]
    vocab_size = repetition_cache.shape[2]
    # Shift the cache to make room for new tokens
    if window_size > 1:
        # shift the cache to the left and add the new token
        repetition_cache[:-1] = repetition_cache[1:]
        repetition_cache[-1].zero_()
        repetition_cache[-1, torch.arange(n_codebooks), output_ids] = True
    
    else:
        # global cache, just set the new token
        repetition_cache[0, torch.arange(n_codebooks), output_ids] = True
    
    return
