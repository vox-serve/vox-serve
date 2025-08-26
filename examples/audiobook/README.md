# Audiobook Example

(This client code is vibe-coded with Claude Code)

This example demonstrates an example of generating audiobooks from large text files using the API server with parallel processing.

## Usage

```bash
# Basic usage with default settings
python audiobook.py sample_text.txt

# Specify output file and server URL  
python audiobook.py sample_text.txt -o my_audiobook.wav --url http://localhost:8000
```

## Options

- `input_file`: Path to the text file to convert (required)
- `-o, --output`: Output audio file path (default: `audiobook.wav`)
- `--url`: API server URL (default: `http://localhost:8000`)
- `--concurrency`: Maximum concurrent requests (default: 100)
- `--chunk-size`: Maximum characters per chunk (default: 100)
- `--no-normalize`: Disable volume normalization (enabled by default)

It contains sample text from a [public domain book](https://www.gutenberg.org/files/11/11-0.txt).