# vox-serve

A PyTorch-based text-to-speech inference server using the Orpheus model with distributed architecture.

## Usage

### Start the Server
```bash
cd vox-serve
python api_server.py
```

The API server automatically starts the scheduler process, so you only need to run one command.

### Generate Speech
```bash
# Generate audio from text (returns audio file directly)
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "tara"}' \
  -o output.wav
```

### Available Voices
`zoe`, `zac`, `jess`, `leo`, `mia`, `julia`, `leah`

### API Endpoints
- `POST /generate` - Generate speech from text and return audio file directly
- `GET /health` - Health check