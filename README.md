# MediaGen

Fully-local pipeline to generate short-form **Hinglish reels** with a
consistent on-screen character, captions, and one-command YouTube Shorts
upload. No paid APIs.

> **MVP scope:** one genre (`motivation`), one character (`Arjun`), 30-second
> 9:16 vertical reel, optional YouTube auto-upload. Instagram is generated
> file + caption; you upload manually for now (their API has account-ban risk
> for unofficial automation).

## Pipeline

```
topic -> Ollama LLM -> JSON script (hook + 4-6 lines + visuals + hashtags)
                    -> XTTS-v2 voiceover per line (Hinglish, voice-cloned)
                    -> SDXL still per line (character LoRA optional)
                    -> LTX-Video / CogVideoX-2b image->video per line
                    -> faster-whisper word-level captions
                    -> ffmpeg/moviepy compose + BGM + burn subs
                    -> YouTube Data API v3 upload
```

## Hardware target

- NVIDIA GPU, ~8 GB VRAM. CPU offload + VAE tiling are on by default.
- If GPU is tighter, set `video.pipeline: image_kenburns` in
  `config/config.yaml` — that skips video gen and uses a pan/zoom on the
  still frame. Still looks decent; ships fastest.

## One-time setup

```bash
# 1. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Local LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b    # or llama3.1:8b

# 3. Env
cp .env.example .env      # then edit if needed

# 4. Character assets
# Drop a 6-10s clean voice sample at:
#   characters/motivation/voice_ref.wav
# (Optional) Train/download a character LoRA and wire it up in
# config/genres/motivation.yaml.

# 5. YouTube OAuth (only if you want auto-upload)
# - Cloud Console -> enable "YouTube Data API v3"
# - OAuth client (Desktop) -> download client_secret.json
# - Place at credentials/client_secret.json
# First run will open a browser to authorize.
```

## Generate a reel

```bash
# Fast path (no GPU video gen) — verifies the pipeline end-to-end:
python -m src.main \
    --genre motivation \
    --topic "Failure is data, not identity" \
    --video-backend image_kenburns \
    --no-upload

# Full path with LTX-Video:
python -m src.main --genre motivation --topic "Aaj ka effort kal ka result"
```

Final mp4 lands in `output/`. Intermediates (script.json, per-line wav/png/mp4,
SRT) stay in `workspace/<reel-id>/` for debugging.

## Adding genres

1. Copy `config/genres/motivation.yaml` to e.g. `config/genres/health.yaml`,
   change character + prompt style.
2. Copy `prompts/script_motivation.txt` to `prompts/script_health.txt`.
3. Drop `voice_ref.wav` under `characters/health/`.
4. Run `python -m src.main --genre health --topic "..."`.

## Notes & honest limits

- LTX-Video / CogVideoX-2b produce 3-5 second clips. The composer loops/trims
  per line to match voiceover length — for longer lines you'll see a subtle
  re-loop. Keep lines short.
- Without a character LoRA, faces drift between shots even with the same
  prompt. Train one when you've validated the pipeline.
- Instagram: a true Business/Creator account + Facebook Graph API is the only
  safe auto-upload path. Hooks for that can be added in
  `src/instagram_upload.py` later.
- All AI-generated content should be labeled as such on YouTube ("Altered
  content" toggle in YT Studio) per their disclosure policy.

## Layout

```
config/         genre + global YAML
prompts/        LLM prompt templates per genre
characters/     voice refs + optional LoRAs
src/            pipeline modules
output/         final reels (gitignored)
workspace/      intermediates (gitignored)
credentials/    YT OAuth (gitignored)
```
