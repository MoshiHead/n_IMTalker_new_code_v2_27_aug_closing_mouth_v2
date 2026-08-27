#!/usr/bin/env bash
set -euo pipefail

ROOT="${SPEECH2AVATAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IMTALKER_DIR="$ROOT/IMTalker"
CHECKPOINT_DIR="$ROOT/checkpoints"
PERSONAPLEX_DIR="$CHECKPOINT_DIR/personaplex_bnb4"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
usage() {
  cat <<'EOF'
Usage: ./prepare_imtalker_personaplex.sh --hf-token TOKEN

  --hf-token TOKEN  Use TOKEN to download the required Hugging Face assets.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token)
      [[ $# -ge 2 && -n "$2" ]] || {
        echo "--hf-token requires a token value." >&2
        usage >&2
        exit 2
      }
      HF_TOKEN="$2"
      export HF_TOKEN
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

[[ -f "$IMTALKER_DIR/requirement.txt" ]] || {
  echo "Run this script from a complete speech2avatar clone." >&2
  exit 1
}

if [[ "$(id -u)" -eq 0 ]]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.11 python3.11-venv \
    ffmpeg git git-lfs htop tmux curl ca-certificates build-essential
  git lfs install
else
  echo "Not root: skipping apt packages. Python 3.11, ffmpeg, git-lfs, and build tools must already exist."
fi

command -v "$PYTHON_BIN" >/dev/null || {
  echo "Missing $PYTHON_BIN." >&2
  exit 1
}

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==80.9.0"
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$IMTALKER_DIR/requirement.txt"
python -m pip install \
  "huggingface_hub[cli]==0.36.2" \
  hf_transfer tensorboard \
  "sphn==0.2.1" einops sentencepiece \
  "aiohttp==3.14.3" "av==17.1.0" "aiortc==1.15.0" \
  "bitsandbytes==0.50.0"

if [[ -z "${HF_TOKEN:-}" ]] && ! hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face access is required for the gated PersonaPlex assets." >&2
  echo "Run again with: ./prepare_imtalker_personaplex.sh --hf-token TOKEN" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p \
  "$IMTALKER_DIR/checkpoints/wav2vec2-base-960h" \
  "$CHECKPOINT_DIR/fullgen_static_2s_6400_resume" \
  "$CHECKPOINT_DIR/personaplex_unitalk_strict2s_2gpu_15k" \
  "$CHECKPOINT_DIR/lora" \
  "$CHECKPOINT_DIR/personaplex_lookahead_rms_adapter/stats" \
  "$PERSONAPLEX_DIR"

echo "[1/7] IMTalker renderer and Wav2Vec files"
for file in \
  renderer.ckpt \
  wav2vec2-base-960h/config.json \
  wav2vec2-base-960h/pytorch_model.bin \
  wav2vec2-base-960h/preprocessor_config.json \
  wav2vec2-base-960h/feature_extractor_config.json; do
  hf download cbsjtu01/IMTalker "$file" --local-dir "$IMTALKER_DIR/checkpoints"
done

echo "[2/7] Two-second IMTalker generator and adapter"
hf download niloy629/hdtf_preprocess \
  live_winner/fullgen_static_2s_6400_resume/last.ckpt \
  live_winner/adapters/personaplex_unitalk_strict2s_2gpu_15k_last.pt \
  --repo-type dataset --local-dir "$CHECKPOINT_DIR"
ln -sfn \
  "$CHECKPOINT_DIR/live_winner/fullgen_static_2s_6400_resume/last.ckpt" \
  "$CHECKPOINT_DIR/fullgen_static_2s_6400_resume/last.ckpt"
ln -sfn \
  "$CHECKPOINT_DIR/live_winner/adapters/personaplex_unitalk_strict2s_2gpu_15k_last.pt" \
  "$CHECKPOINT_DIR/personaplex_unitalk_strict2s_2gpu_15k/last.pt"

echo "[3/7] Blink motion and silence Helium seed"
hf download niloy629/hdtf_preprocess \
  lora/3robert_audio3_ditto_static_motion.pt \
  personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt \
  --repo-type dataset --local-dir "$CHECKPOINT_DIR"

echo "[4/7] PersonaPlex bnb4 package and weights"
hf download brianmatzelle/personaplex-7b-v1-bnb-4bit \
  --local-dir "$PERSONAPLEX_DIR"

echo "[5/7] PersonaPlex Mimi and tokenizer"
hf download nvidia/personaplex-7b-v1 \
  tokenizer-e351c8d8-checkpoint125.safetensors \
  tokenizer_spm_32k_3.model \
  --local-dir "$PERSONAPLEX_DIR"

echo "[6/7] PersonaPlex voices and bundled Robert_5 voice"
hf download nvidia/personaplex-7b-v1 voices.tgz --local-dir "$PERSONAPLEX_DIR"
tar --no-same-owner -xzf "$PERSONAPLEX_DIR/voices.tgz" -C "$PERSONAPLEX_DIR"
install -m 0644 "$ROOT/bundled_assets/Robert_5.pt" "$PERSONAPLEX_DIR/voices/Robert_5.pt"

echo "[7/7] Install bundled Moshi and verify deployment"
[[ -f "$PERSONAPLEX_DIR/moshi/pyproject.toml" ]] || {
  echo "PersonaPlex download is missing bundled moshi source." >&2
  exit 1
}
python -m pip install -e "$PERSONAPLEX_DIR/moshi" --no-deps

SPEECH2AVATAR_ROOT="$ROOT" VENV_DIR="$VENV_DIR" \
  "$ROOT/run_imtalker_personaplex.sh" --check-only

echo
echo "Preparation complete. Start the server with:"
echo "  cd $ROOT && bash run_imtalker_personaplex.sh"
