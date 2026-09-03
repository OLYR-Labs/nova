from __future__ import annotations

import time

import torch

from voice import NOVAVoice


def main():

    print()
    print("=" * 60)
    print("             NOVA TTS LATENCY BENCHMARK")
    print("=" * 60)

    voice = NOVAVoice(
        speaker="ryan",
        device="cuda:0",
        audio_device=3,
    )

    print()
    print("[BENCHMARK] Loading model...")

    load_start = time.perf_counter()

    voice.load()

    load_time = time.perf_counter() - load_start

    print(
        f"[BENCHMARK] Model load time: "
        f"{load_time:.3f}s"
    )

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        print(
            f"[BENCHMARK] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

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

        audio, sample_rate = voice.generate_audio(
            text
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        generation_time = (
            time.perf_counter() - start
        )

        audio_samples = len(audio)

        audio_duration = (
            audio_samples / sample_rate
        )

        real_time_factor = (
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
            f"{real_time_factor:.3f}x"
        )

        if real_time_factor < 1:

            print(
                "Performance       : "
                "FASTER THAN REAL-TIME"
            )

        elif real_time_factor == 1:

            print(
                "Performance       : "
                "REAL-TIME"
            )

        else:

            print(
                "Performance       : "
                "SLOWER THAN REAL-TIME"
            )

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()