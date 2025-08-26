#!/usr/bin/env python3
"""
Audiobook Generator for vox-serve

Generates audiobooks from large text files using the vox-serve API with parallel processing.
Intelligently chunks text at sentence boundaries and processes multiple chunks concurrently.
"""

import argparse
import asyncio
import io
import re
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import aiohttp
import numpy as np


class TextChunker:
    """Splits large text into smaller chunks at sentence boundaries."""

    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size

    def chunk_text(self, text: str) -> List[Tuple[int, str]]:
        """Split text into numbered chunks."""
        text = text.strip()
        if not text:
            return []

        sentences = self._split_into_sentences(text)
        chunks = []

        for chunk_id, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > self.max_chunk_size:
                sub_chunks = self._split_long_sentence(sentence)
                for sub_chunk in sub_chunks:
                    chunks.append((len(chunks), sub_chunk))
            else:
                chunks.append((chunk_id, sentence))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using punctuation."""
        sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text)

        result = []
        for sentence in sentences:
            sentence = re.sub(r'[\r\n\t"\'“”‘’()_*]', ' ', sentence)
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)

        return result

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split overly long sentences at commas or spaces."""
        if len(sentence) <= self.max_chunk_size:
            return [sentence]

        # Try splitting at commas first
        comma_parts = sentence.split(',')
        if len(comma_parts) > 1:
            chunks = []
            current = ""
            for part in comma_parts:
                part = part.strip()
                if len(current) + len(part) + 2 <= self.max_chunk_size:
                    if current:
                        current += ", " + part
                    else:
                        current = part
                else:
                    if current:
                        chunks.append(current)
                    current = part
            if current:
                chunks.append(current)
            return chunks

        # Fall back to splitting at spaces
        words = sentence.split()
        chunks = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= self.max_chunk_size:
                if current:
                    current += " " + word
                else:
                    current = word
            else:
                if current:
                    chunks.append(current)
                current = word
        if current:
            chunks.append(current)

        return chunks


class ParallelTTSClient:
    """Handles parallel TTS API requests with concurrency control."""

    def __init__(self, base_url: str = "http://localhost:8000", max_concurrency: int = 10):
        self.base_url = base_url
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def generate_audio_chunks(self, chunks: List[Tuple[int, str]]) -> Dict[int, bytes]:
        """Generate audio for all chunks in parallel."""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._generate_single_chunk(session, chunk_id, text)
                for chunk_id, text in chunks
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            audio_chunks = {}
            for i, result in enumerate(results):
                chunk_id, _ = chunks[i]
                if isinstance(result, Exception):
                    print(f"Error generating chunk {chunk_id}: {result}")
                    continue
                audio_chunks[chunk_id] = result

            return audio_chunks

    async def _generate_single_chunk(self, session: aiohttp.ClientSession, chunk_id: int, text: str) -> bytes:
        """Generate audio for a single text chunk."""
        async with self.semaphore:
            print(f"Generating chunk {chunk_id}: {text[:50]}...")

            data = aiohttp.FormData()
            data.add_field('text', text)
            data.add_field('streaming', 'false')

            async with session.post(f"{self.base_url}/generate", data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                audio_data = await response.read()
                print(f"Completed chunk {chunk_id} ({len(audio_data)} bytes)")
                return audio_data


class AudioConcatenator:
    """Concatenates WAV audio files with volume normalization."""

    @staticmethod
    def concatenate_wav_files(audio_chunks: Dict[int, bytes], normalize_volume: bool = True) -> bytes:
        """Concatenate audio chunks in order with optional volume normalization."""
        if not audio_chunks:
            raise ValueError("No audio chunks to concatenate")

        sorted_chunks = [audio_chunks[i] for i in sorted(audio_chunks.keys())]

        audio_arrays = []
        sample_rate = None
        channels = None
        sample_width = None

        # Read all chunks into numpy arrays
        for i, chunk_data in enumerate(sorted_chunks):
            wav_buffer = io.BytesIO(chunk_data)

            try:
                with wave.open(wav_buffer, 'rb') as wav_file:
                    if sample_rate is None:
                        sample_rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                    elif (wav_file.getframerate() != sample_rate or
                        wav_file.getnchannels() != channels or
                        wav_file.getsampwidth() != sample_width):
                        print(f"Warning: Audio parameters mismatch in chunk {i}")

                    frames = wav_file.readframes(wav_file.getnframes())

                    # Convert to numpy array based on sample width
                    if sample_width == 1:
                        audio_array = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                    elif sample_width == 2:
                        audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sample_width == 4:
                        audio_array = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width}")

                    # Reshape for multi-channel audio
                    if channels > 1:
                        audio_array = audio_array.reshape(-1, channels)

                    audio_arrays.append(audio_array)

            except Exception as e:
                print(f"Error reading chunk {i}: {e}")
                continue

        if not audio_arrays:
            raise ValueError("No valid audio chunks found")

        # Normalize volume if requested
        if normalize_volume:
            audio_arrays = AudioConcatenator._normalize_volumes(audio_arrays)

        # Concatenate all audio arrays
        combined_audio = np.concatenate(audio_arrays, axis=0)

        # Convert back to original format
        if sample_width == 1:
            combined_frames = ((combined_audio + 1.0) * 128.0).clip(0, 255).astype(np.uint8).tobytes()
        elif sample_width == 2:
            combined_frames = (combined_audio * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
        elif sample_width == 4:
            combined_frames = (combined_audio * 2147483647.0).clip(-2147483648, 2147483647).astype(np.int32).tobytes()

        # Create output WAV file
        output_buffer = io.BytesIO()
        with wave.open(output_buffer, 'wb') as output_wav:
            output_wav.setnchannels(channels)
            output_wav.setsampwidth(sample_width)
            output_wav.setframerate(sample_rate)
            output_wav.writeframes(combined_frames)

        return output_buffer.getvalue()

    @staticmethod
    def _normalize_volumes(audio_arrays: List[np.ndarray], target_rms: float = 0.1) -> List[np.ndarray]:
        """Normalize volume levels across all audio chunks to target RMS level."""
        normalized_arrays = []

        for audio_array in audio_arrays:
            # Calculate RMS (root mean square) for volume measurement
            rms = np.sqrt(np.mean(audio_array ** 2))

            if rms > 1e-6:  # Avoid division by zero for silent audio
                # Calculate scaling factor to reach target RMS
                scale_factor = target_rms / rms

                # Apply scaling with clipping to prevent distortion
                normalized_audio = audio_array * scale_factor
                normalized_audio = np.clip(normalized_audio, -1.0, 1.0)

            else:
                normalized_audio = audio_array

            normalized_arrays.append(normalized_audio)

        return normalized_arrays

    @staticmethod
    def save_audio(audio_data: bytes, output_path: str):
        """Save audio data to file."""
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        print(f"Audio saved to {output_path}")


async def main():
    """Main function that orchestrates the audiobook generation process."""
    parser = argparse.ArgumentParser(description="Generate audiobook from text file using vox-serve API")
    parser.add_argument("input_file", help="Input text file path")
    parser.add_argument("-o", "--output", default="audiobook.wav", help="Output audio file path")
    parser.add_argument("--url", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--concurrency", type=int, default=100, help="Maximum concurrent requests")
    parser.add_argument("--chunk-size", type=int, default=100, help="Maximum characters per chunk")
    parser.add_argument("--no-normalize", action="store_true", help="Disable volume normalization")

    args = parser.parse_args()

    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)

    print(f"Reading text from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    if not text.strip():
        print("Error: Input file is empty")
        sys.exit(1)

    print(f"Text length: {len(text)} characters")

    # Chunk the text
    chunker = TextChunker(max_chunk_size=args.chunk_size)
    chunks = chunker.chunk_text(text)

    print(f"Split into {len(chunks)} chunks (max {args.chunk_size} chars each)")

    if not chunks:
        print("Error: No chunks generated from input text")
        sys.exit(1)

    # Generate audio chunks in parallel
    client = ParallelTTSClient(base_url=args.url, max_concurrency=args.concurrency)

    print(f"Starting audio generation with {args.concurrency} concurrent requests...")
    start_time = time.time()

    try:
        audio_chunks = await client.generate_audio_chunks(chunks)
    except Exception as e:
        print(f"Error during audio generation: {e}")
        sys.exit(1)

    generation_time = time.time() - start_time

    if not audio_chunks:
        print("Error: No audio chunks were generated successfully")
        sys.exit(1)

    print(f"Generated {len(audio_chunks)}/{len(chunks)} chunks in {generation_time:.2f} seconds")

    # Report missing chunks
    if len(audio_chunks) < len(chunks):
        missing_chunks = set(range(len(chunks))) - set(audio_chunks.keys())
        print(f"Warning: Missing chunks: {sorted(missing_chunks)}")

    # Concatenate audio chunks
    normalize_volume = not args.no_normalize
    if normalize_volume:
        print("Concatenating audio chunks with volume normalization...")
    else:
        print("Concatenating audio chunks without volume normalization...")

    try:
        concatenated_audio = AudioConcatenator.concatenate_wav_files(audio_chunks, normalize_volume)
        AudioConcatenator.save_audio(concatenated_audio, args.output)
    except Exception as e:
        print(f"Error during audio concatenation: {e}")
        sys.exit(1)

    total_time = time.time() - start_time
    print(f"Audiobook generation completed in {total_time:.2f} seconds")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
