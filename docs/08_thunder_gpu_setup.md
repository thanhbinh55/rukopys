# Thunder GPU Setup

Hướng dẫn này giả định dùng 1x L40 48GB, Ubuntu 24, disk tối thiểu 200GB.

## 1. Tạo instance

Chọn image Ubuntu có NVIDIA driver/CUDA sẵn nếu Thunder cung cấp. Disk nên để 200GB trở lên vì model Qwen3-VL, YOLO artifact, crop cache và checkpoint sẽ tăng nhanh.

Sau khi SSH vào máy:

```bash
sudo apt update
sudo apt install -y git git-lfs tmux htop nvtop python3-venv python3-pip unzip
nvidia-smi
```

## 2. Đưa code lên máy GPU

Nếu repo đã có Git remote:

```bash
git clone <your-repo-url> rukopys
cd rukopys
```

Nếu đang copy từ máy local:

```bash
rsync -av --exclude .venv --exclude local_working --exclude htr_artifacts \
  /path/to/rukopys/ ubuntu@<server-ip>:/home/ubuntu/rukopys/
ssh ubuntu@<server-ip>
cd /home/ubuntu/rukopys
```

## 3. Cài môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
bash scripts/install_env.sh
```

Qwen3-VL cần `transformers>=4.57.0`. Nếu gặp lỗi import `Qwen3VLForConditionalGeneration`, chạy:

```bash
pip install -U "transformers>=4.57.0" qwen-vl-utils
```

## 4. Cấu hình Kaggle token

Tạo token ở Kaggle Account settings, upload lên server tại `~/.kaggle/kaggle.json`:

```bash
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

## 5. Tải dataset trên disk GPU

Source code không chứa dataset. Script sẽ tải dataset từ Kaggle trên chính disk của instance:

```text
https://www.kaggle.com/datasets/bnthanh/rukopys-dataset
```

Dataset này đã có `train`, `test`, `silver`, và `sample_submission.csv`.

```bash
cd /home/ubuntu/rukopys
source .venv/bin/activate
bash setup_kaggle_dataset.sh
source .env.kaggle
python scripts/show_run_config.py
python scripts/check_env.py
```

Sau khi tải, data nằm tại:

```text
data/kaggle_input/rukopys-dataset/rukopys_raw/
```

Kiểm tra nhanh cấu hình trước khi chạy phase tốn GPU:

```bash
python scripts/show_run_config.py
python scripts/check_env.py
```

Muốn baseline không dùng silver dù thư mục silver có tồn tại:

```bash
USE_SILVER=0 python scripts/02_train_recognizer.py
```

Muốn dùng silver nhưng giới hạn số sample:

```bash
MAX_SILVER=8000 python scripts/02_train_recognizer.py
```

`kaggle.json` nếu đặt tạm ở root repo sẽ được script copy vào `~/.kaggle/kaggle.json`. File này đã nằm trong `.gitignore`; không commit token Kaggle.

Nếu cần test bằng dataset local có sẵn, dùng `USE_LOCAL_RAW=1 bash setup_kaggle_dataset.sh`, nhưng workflow chuẩn trên server là tải từ Kaggle.

## 6. Model cache nằm trên instance

Qwen3-VL, YOLO base weights và các model Hugging Face sẽ được tải ở nơi bạn chạy script. Nếu chạy trên Thunder thì model nằm trên disk Thunder, không tải về máy local.

Sau khi `source .env.kaggle`, cache Hugging Face được đặt tại:

```bash
$PROJECT_ROOT/local_working/hf_cache
```

## 7. Chạy các phase

Khuyến nghị chạy trong `tmux` để tránh mất SSH:

```bash
tmux new -s rukopys
cd /home/ubuntu/rukopys
source .venv/bin/activate
source .env.kaggle
```

Chạy lần lượt:

```bash
python scripts/00_setup_inspect.py
python scripts/00b_generate_synthetic.py
python scripts/00c_prepare_hkr.py
python scripts/01_train_detector.py
python scripts/02_train_recognizer.py
python scripts/03_validate.py
python scripts/04_inference.py
```

Nếu chưa có silver data, Phase 1 và Phase 2 sẽ bỏ qua phần silver. `02b_pseudo_label.py` cũng có thể bỏ qua cho đến khi có silver.

## 8. Artifact quan trọng

Các file cần giữ lại:

```text
local_working/htr_artifacts/config.json
local_working/htr_artifacts/det_yolo/best.pt
local_working/htr_artifacts/qwen3_lora/best_checkpoint/
local_working/htr_artifacts/best_config.json
submission.csv
```

`submission.csv` là file dùng để submit Kaggle.
