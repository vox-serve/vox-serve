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

def apply_repetition_penalty(logits, repetition_cache, penalty):
    """
    Apply repetition penalty to logits based on the repetition cache.
    The logits and repetition_cache have the same shape (batch_size, vocab_size).
    Conceptually, we apply the following algorithm
    
    for i in range(batch_size):
        for j in range(vocab_size):
            if repetition_cache[i][j]:
                if logits[i][j] > 0: logits[i][j] = logits[i][j] / penalty 
                else: logits[i][j] = logits[i][j] * penalty

    Args:
        logits: Tensor of logits from the model. 
        repetition_cache: Boolean tensor indicating which tokens have appeared in the sequence.
        penalty: Float
    """
    positive_mask = (logits > 0) & repetition_cache
    negative_mask = (logits <= 0) & repetition_cache
    
    logits = torch.where(positive_mask, logits / penalty, logits)
    logits = torch.where(negative_mask, logits * penalty, logits)
    
    return logits