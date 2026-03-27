import argparse
import os
import re
import warnings

import soundfile as sf

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gpt-sovits")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
REF_AUDIO_PATH = os.path.join(INPUT_DIR, "ref.mp3")
REF_TEXT_PATH = os.path.join(INPUT_DIR, "ref_text.txt")
OUTPUT_PATH = os.path.join(INPUT_DIR, "output.wav")

TTS_CONFIG = {
    "custom": {
        "device": "cpu",
        "is_half": False,
        "version": "v4",
        "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "vits_weights_path": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    }
}


def detect_lang(text: str) -> str:
    if re.search(r"[\u3040-\u30ff]", text):
        return "all_ja"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "all_zh"
    return "en"


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def synthesize(text: str) -> None:
    text = text.strip()
    if not text:
        raise ValueError("输入文本为空。")

    if not os.path.exists(REF_AUDIO_PATH):
        raise FileNotFoundError(f"缺少参考音频: {REF_AUDIO_PATH}")
    if not os.path.exists(REF_TEXT_PATH):
        raise FileNotFoundError(f"缺少参考文本: {REF_TEXT_PATH}")

    ref_text = read_text(REF_TEXT_PATH)
    if not ref_text:
        raise ValueError(f"参考文本为空: {REF_TEXT_PATH}")

    os.makedirs(INPUT_DIR, exist_ok=True)
    tts = TTS(TTS_CONFIG)
    result = list(
        tts.run(
            {
                "text": text,
                "text_lang": detect_lang(text),
                "ref_audio_path": REF_AUDIO_PATH,
                "prompt_text": ref_text,
                "prompt_lang": detect_lang(ref_text),
                "text_split_method": "cut5",
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "batch_size": 1,
                "batch_threshold": 0.75,
                "split_bucket": True,
                "speed_factor": 1.0,
                "fragment_interval": 0.3,
                "seed": 42,
                "parallel_infer": True,
                "repetition_penalty": 1.35,
                "sample_steps": 32,
                "super_sampling": False,
                "streaming_mode": False,
            }
        )
    )
    if not result:
        raise RuntimeError("TTS 推理无返回音频。")
    sr, audio = result[-1]
    sf.write(OUTPUT_PATH, audio, sr)
    print(f"输出文件: {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="固定使用 input/ref.wav + input/ref_text.txt 的极简 TTS 脚本"
    )
    parser.add_argument("text", nargs="?", help="要生成的文本（可不传，运行后交互输入）")
    args = parser.parse_args()

    text = args.text if args.text else input("请输入文本（中文/English/日本語）：").strip()
    synthesize(text)


if __name__ == "__main__":
    main()
