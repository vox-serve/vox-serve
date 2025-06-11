import torch 
import flashinfer
from dataclasses import dataclass

@dataclass 
class SamplingConfig:
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float

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