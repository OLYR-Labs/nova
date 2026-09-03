from __future__ import annotations

import time

import torch
from faster_qwen3_tts import FasterQwen3TTS


MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def main():
    print()
    print("=" * 60)
    print("        NOVA FASTER QWEN3-TTS BENCHMARK")
    print("=" * 60)

    print()
    print("[BENCHMARK] PyTorch:", torch.__version__)
    print("[BENCHMARK] CUDA:", torch.version.cuda)
    print("[BENCHMARK] GPU:", torch.cuda.get_device_name(0))

    print()
    print("[BENCHMARK] Loading FasterQwen3TTS...")

    load_start = time.perf_counter()

    model = FasterQwen3TTS.from_pretrained(
        MODEL_NAME,
        device="cuda:0",
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    load_time = time.perf_counter() - load_start

    print(
        f"[BENCHMARK] Model load time: "
        f"{load_time:.3f}s"
    )

    if torch.cuda.is_available():

        print(
            f"[BENCHMARK] VRAM allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        print(
            f"[BENCHMARK] VRAM reserved: "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )

    tests = [
        "Hello B.",
        "Hello B. I am NOVA.",
        (
            "Hello B. I am NOVA. "
            "All core systems are online. "
            "I am ready to assist you."
        ),
        (
            "I have inspected the project and "
            "identified several areas that require "
            "attention before we continue."
        ),
    ]

    print()
    print("-" * 60)
    print("WARM GENERATION TESTS")
    print("-" * 60)

    for index, text in enumerate(tests, start=1):

        print()
        print(f"[TEST {index}]")
        print(f"Text: {text}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        result = model.generate_custom_voice(
            text=text,
            language="English",
            speaker="ryan",
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        generation_time = (
            time.perf_counter() - start
        )

        # FasterQwen3TTS may return either:
        #   wavs, sample_rate
        # or another compatible structure.
        if isinstance(result, tuple):
            audio = result[0]
            sample_rate = result[1]
        else:
            audio = result
            sample_rate = 24000

        if isinstance(audio, (list, tuple)):
            audio = audio[0]

        try:
            audio_samples = len(audio)
            audio_duration = (
                audio_samples / sample_rate
            )
        except Exception:
            audio_duration = 0

        rtf = (
            generation_time / audio_duration
            if audio_duration > 0
            else 0
        )

        print(
            f"Generation time : "
            f"{generation_time:.3f}s"
        )

        print(
            f"Audio duration   : "
            f"{audio_duration:.3f}s"
        )

        print(
            f"Real-time factor : "
            f"{rtf:.3f}x"
        )

        if rtf < 1:
            print(
                "Performance       : "
                "FASTER THAN REAL-TIME"
            )
        else:
            print(
                "Performance       : "
                "SLOWER THAN REAL-TIME"
            )

    print()
    print("=" * 60)
    print("FASTER QWEN3-TTS BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()