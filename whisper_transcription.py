# ===============================================================
#  Whisper ➜ training_data.json / .jsonl +  markup.json / .jsonl
# ---------------------------------------------------------------
#  依赖：
#     pip install --upgrade openai-whisper tqdm numpy
#  Windows 需自装 ffmpeg 并加入 PATH
# ===============================================================

import os, re, json, html, warnings, subprocess, tempfile, wave, struct
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import numpy as np
import whisper

warnings.filterwarnings("ignore")

# ---------- 可选：词频、token 计数（仅调试用，不写入文件） ----------
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    count_tokens = lambda s: len(enc.encode(s))
except ModuleNotFoundError:
    count_tokens = lambda s: 0

# ---------- 配置路径 ----------
BASE_DIR   = Path(__file__).parent
INPUT_DIR  = BASE_DIR / "data_audio"
OUT_DIR    = BASE_DIR / "data_result"
OUT_DIR.mkdir(exist_ok=True)

OUT_TRAIN_JSON   = OUT_DIR / "training_data.json"
OUT_TRAIN_JSONL  = OUT_DIR / "training_data.jsonl"
OUT_MARKUP_JSON  = OUT_DIR / "markup.json"
OUT_MARKUP_JSONL = OUT_DIR / "markup.jsonl"

SUPPORTED   = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma", ".webm")
FFMPEG_OPTS = ["-ar", "16000", "-ac", "1"]
WHISPER_MOD = "base"
USE_FP16    = False   # 若 GPU 且显存足够可改 True
PAUSE_TH_MS = 350     # 静音阈值：连续低能量 > 350ms 视为停顿

# ---------- Whisper 模型 ----------
print("⏳ Loading Whisper model:", WHISPER_MOD)
model = whisper.load_model(WHISPER_MOD)

# ---------- 工具函数 ----------
def read_wave_mono(p: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(p), "rb") as wf:
        sr = wf.getframerate()
        n  = wf.getnframes()
        sig = wf.readframes(n)
        fmt = "<" + "h"*n*wf.getnchannels()
        data = np.frombuffer(sig, np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() == 2:
            data = data[::2]
        return data, sr

def calc_pauses(data: np.ndarray, sr: int) -> tuple[int, float]:
    frame = int(sr * 0.02)  # 20 ms
    if len(data) < frame: return 0, 0.0
    energy = (data[:len(data)//frame*frame]
              .reshape(-1, frame) ** 2).mean(axis=1)
    thr = energy.mean() * 0.5
    pauses, cur = [], 0
    for e in energy:
        if e < thr:
            cur += 20
        else:
            if cur >= PAUSE_TH_MS: pauses.append(cur/1000)
            cur = 0
    if cur >= PAUSE_TH_MS:
        pauses.append(cur/1000)
    return len(pauses), round(sum(pauses), 2)

def convert_to_wav(src: Path) -> Path:
    if src.suffix.lower() == ".wav": return src
    dst = Path(tempfile.mktemp(suffix=".wav"))
    cmd = ["ffmpeg", "-loglevel", "quiet", "-y", "-i", str(src),
           *FFMPEG_OPTS, str(dst)]
    subprocess.run(cmd, check=True)
    return dst

def transcribe_audio(src: Path) -> Dict[str, Any]:
    wav = convert_to_wav(src)
    result = model.transcribe(str(wav), fp16=USE_FP16)
    text = result["text"].strip()
    data, sr = read_wave_mono(wav)
    pause_cnt, pause_tot = calc_pauses(data, sr)
    if wav != src: wav.unlink(missing_ok=True)   # 删除临时 wav
    return {
        "plain": text,
        "pause_count": pause_cnt,
        "pause_total": pause_tot,
        "errors": []        # 如需后期写错误可扩展
    }

# ---------- 主流程 ----------
def main():
    files = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in SUPPORTED]
    if not files:
        print("⚠️  data_audio 目录为空")
        return

    train_dict: Dict[str, Any] = {}
    markup_list: List[Dict[str, str]] = []

    for fp in tqdm(files, desc="Transcribing"):
        rec = transcribe_audio(fp)
        # --- 此处曾写 tokens，现完全移除 ---
        train_dict[fp.name] = rec
        markup_list.append({
            "file": fp.name,
            "html": rec["plain"].replace("  ", " <mark>[pause]</mark> ")
        })

    # ---------- 写 JSON ----------
    OUT_TRAIN_JSON.write_text(json.dumps(train_dict, ensure_ascii=False, indent=2), "utf-8")
    OUT_MARKUP_JSON.write_text(json.dumps(markup_list, ensure_ascii=False, indent=2), "utf-8")

    # ---------- 写 JSONL ----------
    OUT_TRAIN_JSONL.unlink(missing_ok=True)
    with OUT_TRAIN_JSONL.open("w", encoding="utf-8") as fjl:
        for rec in train_dict.values():
            fjl.write(json.dumps({
                "input": {
                    "plain": rec["plain"],
                    "pause_count": rec["pause_count"],
                    "pause_total": rec["pause_total"],
                    "errors": rec["errors"]
                },
                "output": {}
            }, ensure_ascii=False) + "\n")

    OUT_MARKUP_JSONL.unlink(missing_ok=True)
    with OUT_MARKUP_JSONL.open("w", encoding="utf-8") as mljl:
        for row in markup_list:
            mljl.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("✅ All done! 生成文件：",
          OUT_TRAIN_JSON.name, OUT_TRAIN_JSONL.name,
          OUT_MARKUP_JSON.name, OUT_MARKUP_JSONL.name)

if __name__ == "__main__":
    main()
