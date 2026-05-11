# Motivation character — Arjun

Drop two files here to make this genre runnable end-to-end:

1. `voice_ref.wav` — a 6-10 second clean recording of the voice you want
   Arjun to speak in. Mono or stereo, 16-22kHz, no music behind it. This is
   the voice XTTS-v2 will clone for every reel.

2. *(optional)* `arjun.safetensors` — a character LoRA trained on 15-30
   reference photos of "Arjun" to lock his face/clothing across shots. If you
   add one, update `config/genres/motivation.yaml`:

       character:
         lora_path: characters/motivation/arjun.safetensors
         lora_weight: 0.8

Without a LoRA you still get consistent style via the per-shot
`character.visual_prompt`, but face identity will drift between shots.
