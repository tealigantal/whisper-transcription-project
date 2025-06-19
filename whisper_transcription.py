# ===============================================================
#  Whisper ➜ training_data.json  +  markup.json
#  ---------------------------------------------------------------
#  依赖：
#     pip install -U openai-whisper           # 必需
#     pip install -U tiktoken                 # (可选, 准确 token)
#     pip install -U wordfreq                 # (可选, 词典判错)
#  ffmpeg 需本机安装并加入 PATH
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
    print("⚠️  未装 tiktoken → 用空格计 token；pip install tiktoken 可更精确")
    def count_tokens(text: str) -> int:
        return len(text.split())

# ---------- wordfreq (optional) ----------
try:
    from wordfreq import zipf_frequency
    WORD_FREQ_READY = True
    def is_common_word(w: str) -> bool:
        return zipf_frequency(w.lower(), "en") > 1.5
except ModuleNotFoundError:
    print("⚠️  未装 wordfreq → 仅按置信度标错；pip install wordfreq 可更准")
    WORD_FREQ_READY = False
    def is_common_word(w: str) -> bool:
        return True  # 全当常见词

# ---------- 路径 ----------
INPUT_DIR  = r"C:\Users\24179\Desktop\whisper-transcription-project\data_audio"
OUT_TRAIN  = r"C:\Users\24179\Desktop\whisper-transcription-project\data_result\training_data.json"
OUT_MARKUP = r"C:\Users\24179\Desktop\whisper-transcription-project\data_result\markup.json"
os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)

# ---------- 其他参数 ----------
SUPPORTED    = (".mp3",".wav",".m4a",".flac",".aac",".ogg",".wma",".webm")
FFMPEG_OPTS  = ["-ar","16000","-ac","1"]
WHISPER_MOD  = "base"
WHISPER_DEV  = "cpu"          # "cuda" 若显卡
CONF_THR     = 0.7
BAR_LEN      = 28
FILLERS      = {"um","uh","like","really","actually"}

# ---------- 工具函数 ----------
def ensure_ffmpeg():
    if subprocess.run(["ffmpeg","-version"],capture_output=True).returncode:
        sys.exit("❌ ffmpeg 未安装，请配置 PATH")

def to_wav(path: str) -> str:
    if path.lower().endswith(".wav"):
        pr = subprocess.run(
            ["ffprobe","-v","error","-select_streams","a:0",
             "-show_entries","stream=sample_rate,channels",
             "-of","default=noprint_wrappers=1:nokey=1",path],
            text=True, capture_output=True
        ).stdout.strip().splitlines()
        if pr and pr[0]=="16000" and pr[1]=="1":
            return path
    fd,tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    if subprocess.run(["ffmpeg","-y","-i",path,*FFMPEG_OPTS,tmp],
                      capture_output=True).returncode:
        os.remove(tmp); raise RuntimeError("ffmpeg 转码失败: "+path)
    return tmp

def whisper_transcribe(model, wav):
    with open(os.devnull,"w") as devnull,\
         redirect_stdout(devnull), redirect_stderr(devnull):
        return model.transcribe(wav, verbose=False, word_timestamps=True)

def split_words(segments):
    return [w for seg in segments for w in seg.get("words", [])]

def build_markup(words):
    plain, html_parts, errs = [], [], []
    for w in words:
        tok = w["word"]
        plain.append(tok)
        low_conf = w.get("confidence",1) < CONF_THR
        uncommon = not is_common_word(tok) if WORD_FREQ_READY else False
        if (low_conf or uncommon) and not tok.istitle():
            html_parts.append(f'<span class="err">{html.escape(tok)}</span>')
            errs.append({
                "w": tok.strip(),
                "conf": round(w.get("confidence",1),3),
                "start": round(w["start"],2),
                "end": round(w["end"],2)
            })
        else:
            html_parts.append(html.escape(tok))
    return "".join(plain).strip(), "".join(html_parts).strip(), errs

def make_compact(text: str, max_chars=400):
    cleaned = " ".join(
        t for t in re.sub(r'\s+',' ',text).split()
        if t.lower() not in FILLERS
    )
    return cleaned[:max_chars]

# ---------- 主流程 ----------
def main():
    ensure_ffmpeg()
    print(f"Whisper Splitter • {datetime.now():%F %T}")
    print("📥 加载模型…")
    model = whisper.load_model(WHISPER_MOD, device=WHISPER_DEV)
    print("✅ Whisper Ready\n")

    audio_files=[f for f in os.listdir(INPUT_DIR) if f.lower().endswith(SUPPORTED)]
    if not audio_files:
        print("⚠️  输入目录为空"); return

    train_json, markup_json = {}, {}
    total_tok, temps = 0, []

    for idx,fname in enumerate(audio_files, 1):
        pct = idx/len(audio_files)
        bar = "█"*int(BAR_LEN*pct)+"░"*(BAR_LEN-int(BAR_LEN*pct))
        print(f"\r[{bar}] {idx}/{len(audio_files)} {fname}", end="", flush=True)

        wav = to_wav(os.path.join(INPUT_DIR,fname))
        if wav not in fname: temps.append(wav)

        res   = whisper_transcribe(model, wav)
        words = split_words(res["segments"])
        plain, markup, errs = build_markup(words)
        tokens   = count_tokens(plain)
        total_tok += tokens
        compact  = make_compact(plain)

        base,ext = os.path.splitext(fname)
        key      = f"{base}_T{tokens}{ext}"

        train_json[key] = {
            "plain": plain,
            "compact": compact,
            "tokens": tokens,
            "errors": errs
        }
        markup_json[key] = markup

    print("\n✅ 全部处理完成")
    print("🔢 总 token:", total_tok)

    with open(OUT_TRAIN,"w",encoding="utf-8") as fp:
        json.dump(train_json, fp, ensure_ascii=False, indent=2)
    with open(OUT_MARKUP,"w",encoding="utf-8") as fp:
        json.dump(markup_json, fp, ensure_ascii=False, indent=2)

    print("📄 训练数据:", OUT_TRAIN)
    print("📄 前端标注:", OUT_MARKUP)

    for t in temps: os.remove(t)

if __name__ == "__main__":
    main()
