=============
Core Concepts
=============

Overview
--------

VoxServe is a **streaming-centric serving system** designed to efficiently deploy modern **Speech Language Models (SpeechLMs)**. Unlike traditional LLM serving systems, VoxServe explicitly targets the requirements of **real-time speech generation**, where latency, sustained streaming, and heterogeneous multi-stage pipelines are first-class concerns.

The key contribution of VoxServe is a **unified execution abstraction** that decouples *model architecture* from *system-level optimizations*. This allows a single serving framework to support a wide variety of SpeechLM designs while applying consistent batching, scheduling, and GPU optimizations.

In this page, we will briefly explain the core concepts of VoxServe. For more details, see our `paper <https://arxiv.org/abs/9999.9999>`_.

Motivation
----------

Modern speech language models are increasingly used in real products such as voice assistants and speech generation systems. These models are much harder to serve efficiently than text-only language models. A single request typically involves several stages, including a large language model, optional speech encoders, and an audio detokenizer that turns tokens into sound. Each stage has different performance characteristics and resource needs.

Existing serving systems do not handle this complexity well. Most current deployments rely on custom, model-specific inference pipelines. These pipelines are fragmented, difficult to extend to new models, and poorly optimized for running many streaming requests at the same time. As a result, systems struggle to deliver low delay before the first audio is heard and to keep audio playback smooth once streaming starts.

Streaming speech also introduces different performance goals than text generation. What matters most to users is how quickly the first playable audio arrives and whether audio continues without interruption. Optimizing only for token latency, as is common in text systems, is not enough. The paper is motivated by the need for a single serving system that can support many different speech model architectures while being explicitly designed around these streaming requirements.

Innovation
----------

The key idea behind VoxServe is to separate model-specific details from system-level performance optimization. We introduce a **unified model execution interface** that all supported speech language models must implement. This interface breaks inference into common stages such as preprocessing, language model execution, sampling, and audio detokenization, even though the internal implementations of these stages may differ across models.

Because the serving system interacts with all models through the same interface, VoxServe can apply optimizations uniformly. These include batching requests across models, running detokenization in fixed-size chunks suitable for streaming, managing per-request caches, and using GPU execution techniques like CUDA graphs to reduce overhead.

On top of this abstraction, VoxServe introduces a scheduling strategy designed specifically for streaming speech. Requests are treated differently before the first audio chunk is produced than after streaming has started. New requests are prioritized to minimize time to first audio, while ongoing streams are scheduled based on how close they are to missing their playback deadlines. This takes advantage of the fact that once streaming is stable, some short delays are harmless as long as audio arrives in time.

The system also uses an asynchronous pipeline that overlaps GPU computation with CPU-side work such as sampling and bookkeeping. This reduces idle time and improves overall hardware utilization without complicating the model implementations.
