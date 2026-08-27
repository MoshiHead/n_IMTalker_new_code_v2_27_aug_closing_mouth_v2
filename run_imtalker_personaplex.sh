#!/usr/bin/env bash
set -euo pipefail
ROOT="${SPEECH2AVATAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IM="$ROOT/IMTalker"
BNB="$ROOT/checkpoints/personaplex_bnb4"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PORT="${PORT:-8998}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
VOICE_PROMPT="${VOICE_PROMPT:-Robert_5.pt}"
DEFAULT_PROMPT_FILE="$IM/prompts/Robert_8998_default.txt"
TEXT_PROMPT_FILE="${TEXT_PROMPT_FILE:-$DEFAULT_PROMPT_FILE}"
PROMPT_CACHE="${PROMPT_CACHE:-0}"
CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1
if [[ -n "${TEXT_PROMPT:-}" ]]; then
  TEXT_PROMPT_VALUE="$TEXT_PROMPT"
elif [[ -f "$TEXT_PROMPT_FILE" ]]; then
  TEXT_PROMPT_VALUE="$(<"$TEXT_PROMPT_FILE")"
else
  echo "Missing text prompt file: $TEXT_PROMPT_FILE" >&2
  exit 1
fi
case "$PROMPT_CACHE" in
  0|false|no|off) PROMPT_CACHE=0 ;;
  1|true|yes|on) PROMPT_CACHE=1 ;;
  *) echo "PROMPT_CACHE must be 0 or 1." >&2; exit 2 ;;
esac
required=(
 "$VENV_DIR/bin/python"
 "$IM/imtalker_personaplex_try_vad2_8998.py" "$IM/seedvc_runtime.py" "$IM/liveTry.py" "$IM/liveTry_cached.py" "$IM/ws_av_binary_codec.py" "$IM/mouth_closing.py"
 "$IM/experiments/original_pod_8998/FM.py" "$IM/experiments/original_pod_8998/FMT.py"
 "$IM/static/index_v3_binary_fullscreen_robot_try_vad2.html" "$IM/static/assets/robert_idle_10s.mp4" "$IM/static/assets/audio-processor-aj-nodrop.js"
 "$IM/static/assets/decoderWorker.min.js" "$IM/static/assets/decoderWorker.min.wasm" "$IM/static/assets/encoderWorker.min-DpsJ02BN.js"
 "$IM/assets/3robert.jpeg" "$IM/checkpoints/renderer.ckpt" "$IM/checkpoints/wav2vec2-base-960h/config.json"
 "$ROOT/checkpoints/fullgen_static_2s_6400_resume/last.ckpt" "$ROOT/checkpoints/personaplex_unitalk_strict2s_2gpu_15k/last.pt"
 "$ROOT/checkpoints/lora/3robert_audio3_ditto_static_motion.pt"
 "$ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt"
 "$BNB/model_bnb_4bit.pt" "$BNB/tokenizer-e351c8d8-checkpoint125.safetensors" "$BNB/tokenizer_spm_32k_3.model" "$BNB/voices/$VOICE_PROMPT"
)
for path in "${required[@]}"; do [[ -e "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }; done
declare -A hashes=(
 ["$IM/imtalker_personaplex_try_vad2_8998.py"]="fe8521b4aef39c570abcfe207e3ac60b7c4bdff28b9172e14b0fa6c9033585ac"
 ["$IM/static/index_v3_binary_fullscreen_robot_try_vad2.html"]="a185e65b9c78d60271351c96d90baf7f555f8a2dbeb13596bc77970088fed63d"
 ["$IM/static/assets/robert_idle_10s.mp4"]="6bdfb847fb3dd2a76d42278a138e26e2729bf5ed938f6733a3b428768a9e7916"
 ["$IM/experiments/original_pod_8998/FM.py"]="8620d6cad2b945276a792a1d63159369654cbb83f9114ab5788f93a3d8daf5d9"
 ["$IM/experiments/original_pod_8998/FMT.py"]="286eb512e710926b0a88d1bc47f14aef5cfc3ef6fc0987fc3cf0d9e7bd004c5d"
 ["$IM/liveTry.py"]="81fa259f751aed7e423d6b8404a8bd744a3a0d701c5cfc525feeced695c8835d"
 ["$IM/liveTry_cached.py"]="0204f5a1e00cada1358a04563309495fb840bdcf4acddd4a3f03910f8c5fce24"
 ["$IM/seedvc_runtime.py"]="fe46773af65e010e3d6f41732f0fa1c3e3cf6a8221d9c68718e15561062337f7"
 ["$IM/ws_av_binary_codec.py"]="c090b6a5a076743055f1dd34301662405a28d5cb1636556e9de4c895ddffe4d3"
 ["$IM/mouth_closing.py"]="3a81f39b4b5d86c5000671b77e0721f0179d887e8929a888da43fe8cb629c945"
 ["$BNB/voices/Robert_5.pt"]="a9684503d2a9d37f527341c9a0385b9ed0943eac955b40159bc34f4796563c3d"
 ["$ROOT/checkpoints/fullgen_static_2s_6400_resume/last.ckpt"]="000d595124516f6437e218213a31c2ede2350ebfda7bb121a957ef5d52b0e88e"
 ["$ROOT/checkpoints/personaplex_unitalk_strict2s_2gpu_15k/last.pt"]="c9c86d108f81fbdef57e1548ca403b78a68acc32c5a37dab12265d72654f55b9"
 ["$ROOT/checkpoints/lora/3robert_audio3_ditto_static_motion.pt"]="e29a41ff004b228d7efee15cad0f32f4d4bc5466563709e2ba78b158d4e340bb"
 ["$IM/checkpoints/renderer.ckpt"]="ca1686c1157b8ef5de43eabdeb846db4612694f5f74012be38742b0871808755"
 ["$ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt"]="20a6d6eb58608d6d202bac46958e595e243635fdeeb8f04eb1afbe2ac7f2f16d"
)
for path in "${!hashes[@]}"; do actual="$(sha256sum "$path" | awk '{print $1}')"; [[ "$actual" == "${hashes[$path]}" ]] || { echo "Checksum mismatch: $path" >&2; exit 1; }; done
source "$VENV_DIR/bin/activate"
python - <<'PY'
import torch, torchaudio, bitsandbytes, aiohttp, av, sphn
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.__version__.startswith("2.8.0+cu128"), torch.__version__
print("CUDA:", torch.cuda.get_device_name(0)); print("Torch:", torch.__version__)
PY
python -m py_compile "$IM/imtalker_personaplex_try_vad2_8998.py" "$IM/seedvc_runtime.py" "$IM/liveTry.py" "$IM/liveTry_cached.py" "$IM/experiments/original_pod_8998/FM.py" "$IM/experiments/original_pod_8998/FMT.py" "$IM/mouth_closing.py"
echo "Preflight OK: try_vad2, $VOICE_PROMPT, prompt cache=$PROMPT_CACHE, 2.0s/25-step chunks, 50 frames, CFG 1.24, NFE 3, renderer sub-batch 6, FP32, Opus."
[[ "$CHECK_ONLY" -eq 1 ]] && exit 0
if ss -ltnp | grep -q ":${PORT}\\b"; then echo "Port $PORT is occupied:" >&2; ss -ltnp | grep ":${PORT}\\b" >&2; exit 1; fi
mkdir -p "$IM/logs" "$IM/closing_animations" "$ROOT/pacing_compare/integrated"
cd "$IM"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$IM:$BNB/moshi:$BNB:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export IMTALKER_CACHED_ENGINE="$PROMPT_CACHE" IMTALKER_PROMPT_STATE_CACHE="$PROMPT_CACHE" IMTALKER_TRANSITION_BLEND_FRAMES=0
exec python -u "$IM/imtalker_personaplex_try_vad2_8998.py" \
 --host 0.0.0.0 --port "$PORT" --html_path "$IM/static/index_v3_binary_fullscreen_robot_try_vad2.html" \
 --generator_path "$ROOT/checkpoints/fullgen_static_2s_6400_resume/last.ckpt" --renderer_path "$IM/checkpoints/renderer.ckpt" \
 --adapter_path "$ROOT/checkpoints/personaplex_unitalk_strict2s_2gpu_15k/last.pt" \
 --adapter_type unitalk_last_layer --adapter_num_layers 12 --adapter_dropout 0.0 --adapter_window_mode lookahead --adapter_future_steps 0 \
 --ref_path "$IM/assets/3robert.jpeg" --wav2vec_model_path "$IM/checkpoints/wav2vec2-base-960h" \
 --moshi_root "$BNB" --mimi_hf_repo nvidia/personaplex-7b-v1 --moshi_weight "$BNB/model_bnb_4bit.pt" \
 --mimi_weight "$BNB/tokenizer-e351c8d8-checkpoint125.safetensors" --tokenizer "$BNB/tokenizer_spm_32k_3.model" \
 --text_prompt "$TEXT_PROMPT_VALUE" \
 --quantize_4bit --voice_prompt "$VOICE_PROMPT" --voice_prompt_dir "$BNB/voices" \
 --enable_moshi_reply --direct_reply_hidden --reply_hidden_steps_per_chunk 25 \
 --audio_chunk_sec 2.0 --wav2vec_sec 2.0 --fm_chunk_frames 50 --helium_deque_size 25 \
 --prebuffer_chunks 1 --render_sub_batch 6 --renderer_precision fp32 --frame_q_backpressure 32 --buffer_ms 160 --skip_fm_audio_encoder \
 --assistant_speech_rms_threshold 0.006 --assistant_speech_hold_chunks 1 --motion_ref_blend 0.0 --motion_prior_noise_blend 0.0 \
 --a_cfg_scale 1.24 --nfe 3 --seed 42 --noise_seed 42 --shared_noise --fp32 --tf32 \
 --silence_helium_path "$ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt" \
 --jpeg_quality 90 --device cuda --reply_audio_gain 1.0 --output_audio_codec opus \
 --blink_motion_path "$ROOT/checkpoints/lora/3robert_audio3_ditto_static_motion.pt" --enable_eye_blink_composite
