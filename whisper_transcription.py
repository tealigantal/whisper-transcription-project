# ===============================================================
#  Whisper ➜ training_data.json  +  markup.json
#  功能：
#    • 语音转文字 (Whisper)
#    • 填充词 <filler> 标注
#    • 停顿/静音 ⏸ 标注（仅写入 markup）
#    • 生成训练用纯文本（含停顿 *次数* & *总时长*，不含逐点时间）
# ---------------------------------------------------------------
#  依赖：
#     pip install -U openai-whisper  # Whisper 官方包（含 torch）
#     pip install -U tiktoken        # (可选) token 统计更准
#     pip install -U wordfreq        # (可选) 生僻词辨识
#  Windows 需自装 ffmpeg 并加入 PATH
# ===============================================================

import os, subprocess, whisper, json, re, tempfile, sys, warnings, html
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------- token 计数 ----------
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(enc.encode(text))
except ModuleNotFoundError:
    def count_tokens(text: str) -> int:
        return len(text.split())

# ---------- wordfreq (optional) ----------
try:
    from wordfreq import zipf_frequency
    WORD_FREQ_READY = True
    def is_common_word(w: str) -> bool:
        return zipf_frequency(w.lower(), "en") > 1.5
except ModuleNotFoundError:
    WORD_FREQ_READY = False
    def is_common_word(w: str) -> bool:
        return True

# ---------- 路径 ----------
INPUT_DIR  = r"C:\\Users\\24179\\Desktop\\whisper-transcription-project\\data_audio"
OUT_TRAIN  = r"C:\\Users\\24179\\Desktop\\whisper-transcription-project\\data_result\\training_data.json"
OUT_MARKUP = r"C:\\Users\\24179\\Desktop\\whisper-transcription-project\\data_result\\markup.json"
os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)

# ---------- 其他参数 ----------
SUPPORTED    = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma", ".webm")
FFMPEG_OPTS  = ["-ar", "16000", "-ac", "1"]
WHISPER_MOD  = "base"   # tiny / base / small / medium / large
WHISPER_DEV  = "cpu"    # 改 "cuda" 如有显卡
CONF_THR     = 0.7      # 置信度阈值
PAUSE_THR    = 0.4      # 秒，视为停顿
BAR_LEN      = 28
FILLERS      = {"um", "uh", "like", "really", "actually"}

# ---------- 工具 ----------

def ensure_ffmpeg():
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode:
        sys.exit("❌ 未检测到 ffmpeg，请安装并配置 PATH")


def to_wav(path: str) -> str:
    """保证音频为 16‑kHz / 单声道 WAV。若不是则转码到临时文件。"""
    if path.lower().endswith(".wav"):
        meta=subprocess.run(["ffprobe","-v","error","-select_streams","a:0","-show_entries","stream=sample_rate,channels","-of","default=noprint_wrappers=1:nokey=1",path],text=True,capture_output=True).stdout.strip().splitlines()
        if meta and meta[0]=="16000" and meta[1]=="1":
            return path
    fd,tmp=tempfile.mkstemp(suffix=".wav");os.close(fd)
    if subprocess.run(["ffmpeg","-y","-i",path,*FFMPEG_OPTS,tmp],capture_output=True).returncode:
        os.remove(tmp);raise RuntimeError("ffmpeg 转码失败:"+path)
    return tmp


def whisper_transcribe(model, wav):
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        return model.transcribe(wav, verbose=False, word_timestamps=True)

# ---------- 停顿 / 标注 ----------

def split_words(segs):
    return [w for seg in segs for w in seg.get("words", [])]


def detect_pauses(words, thr=PAUSE_THR):
    pauses, prev_end = [], None
    for w in words:
        if prev_end is not None:
            gap = w["start"] - prev_end
            if gap >= thr:
                pauses.append(round(gap, 2))  # 仅记录时长
        prev_end = w["end"]
    return pauses


def build_markup(words, pauses):
    """生成纯文本 & 含 HTML 标注字符串。停顿在 markup 中插 ⏸。"""
    plain, html_parts, errs = [], [], []
    pauses_iter = iter(pauses)
    next_pause_idx, next_pause = 0, next(pauses_iter, None)

    for w in words:
        # 插入该词前的停顿标记
        if next_pause is not None and w["start"]-0.001 > 0 and next_pause_idx < len(pauses):
            html_parts.append(f'<span class="pause" data-dur="{next_pause}">⏸ {next_pause}s</span>')
            next_pause_idx += 1
            next_pause = next(pauses_iter, None)

        tok = w["word"]
        plain.append(tok)
        low = w.get("confidence", 1) < CONF_THR
        uncommon = not is_common_word(tok) if WORD_FREQ_READY else False
        cls = None
        if tok.lower() in FILLERS:
            cls = "filler"
        elif (low or uncommon) and not tok.istitle():
            cls = "err"; errs.append({"w": tok.strip(), "conf": round(w.get("confidence",1),3)})
        html_parts.append(f'<span class="{cls}">{html.escape(tok)}</span>' if cls else html.escape(tok))

    return " ".join(plain).strip(), "".join(html_parts).strip(), errs


def make_compact(text, max_chars=400):
    return " ".join(t for t in re.sub(r"\s+", " ", text).split() if t.lower() not in FILLERS)[:max_chars]

# ---------- 主入口 ----------

def main():
    ensure_ffmpeg()
    print(f"Whisper Splitter • {datetime.now():%F %T}")
    model = whisper.load_model(WHISPER_MOD, device=WHISPER_DEV)

    audios = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(SUPPORTED)]
    if not audios:
        print("⚠️ 输入目录为空"); return

    train, markup, tot_tokens = {}, {}, 0

    for idx, file in enumerate(audios, 1):
        bar = "█"*int(BAR_LEN*idx/len(audios)) + "░"*(BAR_LEN-int(BAR_LEN*idx/len(audios)))
        print(f"\r[{bar}] {idx}/{len(audios)} {file}", end="", flush=True)

        wav = to_wav(os.path.join(INPUT_DIR, file))
        res = whisper_transcribe(model, wav)
        words = split_words(res["segments"])
        pauses = detect_pauses(words)
        plain, html_out, errs = build_markup(words, pauses)

        tok = count_tokens(plain); tot_tokens += tok
        key = f"{os.path.splitext(file)[0]}_T{tok}{os.path.splitext(file)[1]}"

        train[key] = {
            "plain": plain,
            "compact": make_compact(plain),
            "tokens": tok,
            "errors": errs,
            "pause_count": len(pauses),
            "pause_total": round(sum(pauses), 2)
        }
        markup[key] = html_out

    print("\n✅ 完成，总 token:", tot_tokens)
    with open(OUT_TRAIN, "w", encoding="utf-8") as fp: json.dump(train, fp, ensure_ascii=False, indent=2)
    with open(OUT_MARKUP, "w", encoding="utf-8") as fp: json.dump(markup, fp, ensure_ascii=False, indent=2)
    print("📄 保存:", OUT_TRAIN, "|", OUT_MARKUP)

if __name__ == "__main__":
    main()
