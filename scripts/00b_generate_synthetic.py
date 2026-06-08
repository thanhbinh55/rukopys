#!/usr/bin/env python3
"""
scripts/00b_generate_synthetic.py — Phase 0B: Generate synthetic crops với TRDG.
Output: htr_artifacts/rec_data/synthetic/ + synth_manifest.csv

Hardware: L40 48GB / 8vCPU / 64GB RAM
Thời gian: ~30-60 phút cho 100k samples
"""
import os, json, random, time
from pathlib import Path
import pandas as pd

ROOT     = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
ART      = Path(os.getenv('HTR_ART_DIR', ROOT / 'local_working' / 'htr_artifacts'))
OUT_DIR  = ART / 'rec_data' / 'synthetic'
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = ART / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

N_GEN    = int(os.getenv('N_SYNTH', '100000'))
FONT_DIR = Path(os.getenv('FONT_DIR', './fonts/cyrillic'))
BATCH_SAVE = 1000  # Lưu manifest mỗi N samples (checkpoint)


def load_texts(art: Path) -> list:
    """Load text strings từ gold data + word list."""
    texts = []

    # 1. Từ gold annotations (highest quality)
    for jsonl_path in [art / 'train_split.jsonl']:
        if not jsonl_path.exists():
            continue
        print(f'[synth] Reading texts from {jsonl_path.name}...')
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for reg in (r.get('regions') or r.get('annotations') or []):
                    t = reg.get('text', '').strip()
                    rtype = reg.get('type', 'handwritten')
                    # Lấy text từ tất cả scorable types
                    if (3 <= len(t) <= 120
                            and reg.get('language', 'uk') == 'uk'
                            and rtype not in ('image', 'graph')):
                        texts.append(t)

    print(f'[synth] Gold texts: {len(texts):,}')

    # 2. Ukrainian word lists
    for wl_path in [
        Path('./data/ukr_words.txt'),
        Path('/kaggle/input/ukr-words/words.txt'),
        Path('./data/ukr_corpus.txt'),
    ]:
        if wl_path.exists():
            words = [w.strip() for w in open(wl_path, 'r', encoding='utf-8')
                     if w.strip() and len(w.strip()) >= 2]
            # Tạo câu ngẫu nhiên từ word list
            generated = []
            for _ in range(min(60000, N_GEN // 2)):
                n_words = random.randint(3, 12)
                sent = ' '.join(random.choices(words, k=n_words))
                # Thêm punctuation ngẫu nhiên
                if random.random() < 0.3:
                    sent += random.choice(['.', ',', '!', '?', ':'])
                generated.append(sent)
            texts.extend(generated)
            print(f'[synth] Word list loaded: {len(words):,} words → {len(generated):,} sentences')
            break

    # 3. Dictation-style sentences (Ukrainian grammar patterns)
    dictation_templates = [
        "Україна є суверенною та незалежною державою",
        "Київ є столицею України та найбільшим містом країни",
        "Мова українського народу є однією з найдавніших слов'янських мов",
        "Освіта відіграє важливу роль у розвитку суспільства",
        "Природа України дуже різноманітна і багата",
    ]
    for tmpl in dictation_templates:
        words = tmpl.split()
        for _ in range(200):
            n = random.randint(len(words)//2, len(words))
            subset = random.sample(words, n)
            texts.append(' '.join(subset))

    random.shuffle(texts)
    # Deduplicate
    texts = list(dict.fromkeys(texts))
    print(f'[synth] Total unique texts: {len(texts):,}')
    return texts


def get_fonts(font_dir: Path) -> list:
    """Tìm Cyrillic fonts."""
    font_paths = []
    if font_dir.exists():
        for ext in ['*.ttf', '*.otf', '*.TTF', '*.OTF']:
            font_paths.extend([str(p) for p in font_dir.glob(ext)])

    # Tìm system fonts có Cyrillic support
    system_font_dirs = [
        Path('/usr/share/fonts'),
        Path('/usr/local/share/fonts'),
        Path(os.path.expanduser('~/.fonts')),
        Path('/kaggle/input/cyrillic-fonts'),
    ]
    for sfd in system_font_dirs:
        if sfd.exists():
            for ext in ['*.ttf', '*.otf']:
                font_paths.extend([
                    str(p) for p in sfd.rglob(ext)
                    if any(kw in p.stem.lower() for kw in
                           ['dejavu', 'liberation', 'ubuntu', 'noto', 'roboto',
                            'open', 'freemono', 'freeserif', 'freesans',
                            'cyrillic', 'slavic', 'ukr'])
                ])

    print(f'[synth] Fonts found: {len(font_paths)}')
    if not font_paths:
        print('[warn] No Cyrillic fonts found. Using TRDG default (may not support Ukrainian).')
        print('[warn] Download fonts: https://fonts.google.com/?subset=cyrillic')
    return font_paths


def generate(texts: list, n: int, font_paths: list) -> list:
    """Generate synthetic crops với TRDG."""
    try:
        from trdg.generators import GeneratorFromStrings
    except ImportError:
        print('[synth] TRDG not installed. Run: pip install trdg')
        return []

    gen_texts = (texts * ((n // max(len(texts), 1)) + 1))[:n]
    random.shuffle(gen_texts)

    # Config tối ưu cho Ukrainian
    generator = GeneratorFromStrings(
        strings=gen_texts,
        count=n,
        fonts=font_paths if font_paths else [],
        language='uk',
        size=64,            # height cố định — resize nếu cần
        skewing_angle=3,
        random_skew=True,
        blur=1,
        random_blur=True,
        background_type=0,  # 0=Gaussian noise (giống scan quality)
        distorsion_type=0,
        is_handwritten=False,
        width=-1,           # auto width
        alignment=0,
        text_color='#2a2a2a',
        orientation=0,
        space_width=1.0,
        fit=True,
    )

    manifest = []
    progress_path = LOGS_DIR / 'synth_progress.json'
    t0 = time.time()

    for i, (img, label) in enumerate(generator):
        if i >= n:
            break

        img_name = f'synth_{i:08d}.jpg'
        img_path = OUT_DIR / img_name

        # Resize nếu cần
        if img.height != 64:
            from PIL import Image
            ratio = 64 / max(1, img.height)
            new_w = min(1920, max(32, int(img.width * ratio)))
            img = img.resize((new_w, 64))

        img.save(str(img_path), 'JPEG', quality=88)
        manifest.append({
            'image_path': str(img_path),
            'text': label,
            'weight': 0.3,
            'source': 'synthetic_trdg',
            'type': 'handwritten',
        })

        # Checkpoint mỗi BATCH_SAVE ảnh
        if (i + 1) % BATCH_SAVE == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta_min = (n - i - 1) / max(speed, 1e-6) / 60
            json.dump({'done': i+1, 'total': n, 'speed': round(speed, 1)}, open(progress_path,'w'))
            print(f'[synth] {i+1:,}/{n:,} ({speed:.0f} img/s, ETA {eta_min:.1f} min)')

    return manifest


def main():
    print('PHASE 0B — SYNTHETIC DATA GENERATION (TRDG)')
    print('=' * 50)
    print(f'Target: {N_GEN:,} images → {OUT_DIR}')

    # Check resume
    existing = list(OUT_DIR.glob('synth_*.jpg'))
    if existing:
        print(f'[synth] Found {len(existing):,} existing images')
        manifest_path = ART / 'rec_data' / 'synth_manifest.csv'
        if manifest_path.exists():
            existing_df = pd.read_csv(manifest_path)
            if len(existing_df) >= N_GEN:
                print(f'[synth] Already have {len(existing_df):,} >= {N_GEN:,} samples. Skipping.')
                print('[done] Phase 0B skipped (already complete)')
                return
            n_remaining = N_GEN - len(existing_df)
            print(f'[synth] Generating {n_remaining:,} more samples...')
        else:
            n_remaining = N_GEN
    else:
        n_remaining = N_GEN

    texts = load_texts(ART)
    if not texts:
        print('[warn] No texts found. Run Phase 0A first: python scripts/00_setup_inspect.py')
        print('[synth] Generating with placeholder Ukrainian texts...')
        texts = [
            "Доброго ранку", "Привіт друже", "Україна понад усе",
            "Слава Україні", "Героям слава", "Навчання — це важливо",
            "Сьогодні гарна погода", "Задача вирішена правильно",
        ] * 1000

    fonts = get_fonts(FONT_DIR)
    manifest = generate(texts, n_remaining, fonts)

    if manifest:
        # Merge з existing manifest якếu có
        manifest_path = ART / 'rec_data' / 'synth_manifest.csv'
        new_df = pd.DataFrame(manifest)
        if manifest_path.exists():
            old_df = pd.read_csv(manifest_path)
            df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            df = new_df

        df.to_csv(manifest_path, index=False)
        print(f'\n[DONE] Phase 0B complete!')
        print(f'       {len(df):,} synthetic crops → {OUT_DIR}')
        print(f'       Manifest → {manifest_path}')
    else:
        print('[warn] No synthetic data generated. Install TRDG: pip install trdg')
        # Tạo empty manifest để pipeline không bị lỗi
        pd.DataFrame(columns=['image_path','text','weight','source','type']).to_csv(
            ART / 'rec_data' / 'synth_manifest.csv', index=False)


if __name__ == '__main__':
    main()
