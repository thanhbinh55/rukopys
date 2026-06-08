#!/bin/bash
# scripts/install_env.sh — Cài đặt toàn bộ environment một lần
# Chạy: bash scripts/install_env.sh
# Hardware: L40 48GB, CUDA 12.4+, Ubuntu 24

set -e  # Dừng nếu có lỗi

echo "========================================"
echo "RUKOPYS HTR — Environment Setup"
echo "Hardware: L40 48GB / 8vCPU / 64GB RAM"
echo "========================================"

python -m pip install -U pip wheel setuptools packaging ninja

# 1. PyTorch với CUDA 12.4
echo "[1/8] Installing PyTorch (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. HuggingFace ecosystem
echo "[2/8] Installing HuggingFace ecosystem..."
pip install \
    "transformers==4.57.1" \
    "peft==0.17.1" \
    "accelerate>=1.0.0,<2.0.0" \
    "datasets>=2.20.0" \
    "huggingface_hub>=0.24.0" \
    "tokenizers>=0.20.0"

# 3. Flash Attention 2 (bắt buộc cho L40 Ada Lovelace)
echo "[3/8] Installing Flash Attention 2 (optional; SDPA fallback is supported)..."
MAX_JOBS="${MAX_JOBS:-4}" pip install flash-attn --no-build-isolation || {
    echo "[warn] Flash Attention install failed — continuing with USE_FLASH_ATTN=0 / SDPA fallback"
}

# 4. Qwen VL utilities
echo "[4/8] Installing Qwen-VL utilities..."
pip install qwen-vl-utils
pip install timm  # vision backbone

# 5. YOLO
echo "[5/8] Installing Ultralytics (YOLOv8)..."
pip install ultralytics

# 6. OCR fallback
echo "[6/8] Installing PaddleOCR (optional fallback)..."
pip install paddlepaddle-gpu paddleocr || {
    echo "[warn] PaddleOCR GPU install failed — trying CPU version"
    pip install paddlepaddle paddleocr || echo "[warn] PaddleOCR not available"
}

# 7. Synthetic data generation
echo "[7/8] Installing TRDG (synthetic data)..."
pip install trdg || echo "[warn] TRDG not available"

# 8. Data & utilities
echo "[8/8] Installing data utilities..."
pip install \
    "pandas>=2.0.0" \
    "numpy>=1.24.0" \
    "pillow>=10.0.0" \
    "scikit-learn>=1.3.0" \
    "rapidfuzz>=3.0.0" \
    "pyyaml>=6.0" \
    tensorboard

echo ""
echo "========================================"
echo "Installation complete!"
echo "Run: python scripts/check_env.py"
echo "========================================"
