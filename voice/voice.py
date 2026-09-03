from __future__ import annotations

import re
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import torch

from faster_qwen3_tts import FasterQwen3TTS


class NOVAVoice:
    """
    NOVA local Qwen3-TTS voice engine.

    Design:
        Text
          ↓
        Sentence/chunk splitter
          ↓
        Qwen3-TTS
          ↓
        CUDA
          ↓
        Audio RAM
          ↓
        sounddevice

    The unstable community streaming API is intentionally NOT used.

    Instead, NOVA uses pipelined chunk generation:

        chunk 1 → generate → play
                       ↓
                  generate chunk 2
                       ↓
                  play chunk 2
                       ↓
                  ...

    This dramatically improves perceived response latency because
    NOVA starts speaking before the complete response has been
    synthesized.
    """

    def __init__(
        self,
        state=None,
        model_name: str = (
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        ),
        speaker: str = "ryan",
        device: str = "cuda",
        audio_device: Optional[int] = 3,
    ):
        self.state = state

        self.model_name = model_name
        self.speaker = speaker.lower()
        self.device = device
        self.audio_device = audio_device

        self.model: Optional[FasterQwen3TTS] = None

        self.loaded = False

        self._load_lock = threading.Lock()
        self._speak_lock = threading.Lock()

        self._speaking = False
        self._stop_event = threading.Event()

    # =========================================================
    # GPU STATUS
    # =========================================================

    @staticmethod
    def gpu_status():

        cuda_available = torch.cuda.is_available()

        if not cuda_available:
            return {
                "cuda": False,
                "device": "cpu",
                "gpu": None,
                "cuda_version": torch.version.cuda,
            }

        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "Unknown NVIDIA GPU"

        return {
            "cuda": True,
            "device": "cuda:0",
            "gpu": gpu_name,
            "cuda_version": torch.version.cuda,
        }

    # =========================================================
    # LOAD
    # =========================================================

    def load(self):

        if self.loaded:
            return

        with self._load_lock:

            if self.loaded:
                return

            print()
            print("[NOVA VOICE] Loading faster-qwen3-tts...")
            print(f"[NOVA VOICE] Model   : {self.model_name}")
            print(f"[NOVA VOICE] Speaker : {self.speaker}")
            print(f"[NOVA VOICE] Device  : {self.device}")

            gpu = self.gpu_status()

            if gpu["cuda"]:

                print("[NOVA VOICE] CUDA enabled.")
                print(f"[NOVA VOICE] GPU     : {gpu['gpu']}")
                print(f"[NOVA VOICE] CUDA    : {gpu['cuda_version']}")

            else:

                print(
                    "[NOVA VOICE] WARNING: CUDA unavailable."
                )

            self.model = FasterQwen3TTS.from_pretrained(
                self.model_name,
                device=self.device,
            )

            # -------------------------------------------------
            # WARMUP
            # -------------------------------------------------

            print("[NOVA VOICE] Warming up model...")

            try:

                self.model.warmup()

            except Exception as error:

                print(
                    f"[NOVA VOICE] Warmup warning: {error}"
                )

            self.loaded = True

            print(
                "[NOVA VOICE] faster-qwen3-tts ready."
            )

    # =========================================================
    # SPEAKERS
    # =========================================================

    def get_supported_speakers(self):

        self.load()

        try:

            return self.model.get_supported_speakers()

        except AttributeError:

            return []

    # =========================================================
    # TEXT CHUNKING
    # =========================================================

    @staticmethod
    def split_text(
        text: str,
        max_chars: int = 180,
    ) -> list[str]:
        """
        Split a response into natural speech chunks.

        Priority:

            1. Sentence boundaries
            2. Commas / semicolons
            3. Word boundaries

        Short responses remain a single chunk.
        """

        text = str(text or "").strip()

        if not text:
            return []

        # -----------------------------------------------------
        # Normalize whitespace.
        # -----------------------------------------------------

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if len(text) <= max_chars:

            return [text]

        # -----------------------------------------------------
        # First split into sentences.
        # -----------------------------------------------------

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        chunks: list[str] = []

        current = ""

        for sentence in sentences:

            # -------------------------------------------------
            # Normal sentence fits.
            # -------------------------------------------------

            if len(sentence) <= max_chars:

                if not current:

                    current = sentence

                elif len(current) + 1 + len(sentence) <= max_chars:

                    current = (
                        current
                        + " "
                        + sentence
                    )

                else:

                    chunks.append(current)

                    current = sentence

                continue

            # -------------------------------------------------
            # Long sentence.
            # Split using punctuation/word boundaries.
            # -------------------------------------------------

            pieces = re.split(
                r"(?<=[,;:])\s+",
                sentence,
            )

            for piece in pieces:

                piece = piece.strip()

                if not piece:
                    continue

                if not current:

                    current = piece

                elif len(current) + 1 + len(piece) <= max_chars:

                    current = (
                        current
                        + " "
                        + piece
                    )

                else:

                    chunks.append(current)

                    current = piece

        if current:

            chunks.append(current)

        # -----------------------------------------------------
        # Safety split anything still too long.
        # -----------------------------------------------------

        final_chunks: list[str] = []

        for chunk in chunks:

            if len(chunk) <= max_chars:

                final_chunks.append(chunk)

                continue

            words = chunk.split()

            current = ""

            for word in words:

                if not current:

                    current = word

                elif len(current) + 1 + len(word) <= max_chars:

                    current += " " + word

                else:

                    final_chunks.append(current)

                    current = word

            if current:

                final_chunks.append(current)

        return final_chunks

    # =========================================================
    # GENERATE AUDIO
    # =========================================================

    def generate_audio(
        self,
        text: str,
    ):

        text = str(text or "").strip()

        if not text:

            raise ValueError(
                "Cannot generate speech from empty text."
            )

        self.load()

        print(
            f"[NOVA TTS] Generating: "
            f'"{text}"'
        )

        start = time.perf_counter()

        with torch.inference_mode():

            result = self.model.generate_custom_voice(
                text=text,
                speaker=self.speaker,
                language="English",
            )

        generation_time = (
            time.perf_counter()
            - start
        )

        # -----------------------------------------------------
        # Normalize result.
        # -----------------------------------------------------

        if isinstance(result, tuple):

            wavs, sample_rate = result

        else:

            wavs = result
            sample_rate = 24000

        if wavs is None:

            raise RuntimeError(
                "Faster Qwen3-TTS returned no audio."
            )

        if isinstance(wavs, (list, tuple)):

            if len(wavs) == 0:

                raise RuntimeError(
                    "Faster Qwen3-TTS returned empty audio."
                )

            audio = wavs[0]

        else:

            audio = wavs

        # -----------------------------------------------------
        # Torch → NumPy
        # -----------------------------------------------------

        if torch.is_tensor(audio):

            audio = (
                audio
                .detach()
                .float()
                .cpu()
                .numpy()
            )

        else:

            audio = np.asarray(
                audio,
                dtype=np.float32,
            )

        audio = np.squeeze(audio)

        if audio.ndim != 1:

            audio = audio.reshape(-1)

        if audio.size == 0:

            raise RuntimeError(
                "Generated audio is empty."
            )

        # -----------------------------------------------------
        # Validate.
        # -----------------------------------------------------

        if not np.isfinite(audio).all():

            raise RuntimeError(
                "Generated audio contains NaN or infinity."
            )

        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        if peak > 1.0:

            audio = audio / peak

        duration = (
            len(audio)
            / int(sample_rate)
        )

        print(
            f"[NOVA TTS] Ready | "
            f"generation={generation_time:.2f}s | "
            f"audio={duration:.2f}s"
        )

        return audio, int(sample_rate)

    # =========================================================
    # PLAY AUDIO
    # =========================================================

    def play_audio(
        self,
        audio,
        sample_rate: int,
    ):

        if audio is None:

            raise ValueError(
                "Audio data is None."
            )

        if sample_rate <= 0:

            raise ValueError(
                f"Invalid sample rate: {sample_rate}"
            )

        if self._stop_event.is_set():

            return False

        if self.state is not None:

            try:

                self.state.set_nova_speaking(True)

            except Exception:

                pass

        self._speaking = True

        try:

            sd.play(
                audio,
                sample_rate,
                device=self.audio_device,
                blocking=False,
            )

            # -------------------------------------------------
            # Wait manually so stop() can interrupt playback.
            # -------------------------------------------------

            duration = (
                len(audio)
                / sample_rate
            )

            start = time.perf_counter()

            while (
                time.perf_counter() - start
                < duration
            ):

                if self._stop_event.is_set():

                    sd.stop()

                    return False

                time.sleep(0.01)

            sd.stop()

            return True

        finally:

            self._speaking = False

            if self.state is not None:

                try:

                    self.state.set_nova_speaking(False)

                except Exception:

                    pass

    # =========================================================
    # SPEAK SINGLE CHUNK
    # =========================================================

    def speak_chunk(
        self,
        text: str,
    ):

        if not text:

            return False

        audio, sample_rate = (
            self.generate_audio(text)
        )

        return self.play_audio(
            audio,
            sample_rate,
        )

    # =========================================================
    # PIPELINED CHUNKED SPEECH
    # =========================================================

    def speak_chunked(
        self,
        text: str,
        max_chars: int = 180,
    ):
        """
        Generate and play speech in a pipeline.

        Example:

            chunk 1 generation
                  ↓
            chunk 1 playback
                  ↓
            chunk 2 generation happens during playback
                  ↓
            chunk 2 playback
                  ↓
            ...

        The first chunk begins playback as soon as it is ready.
        """

        text = str(text or "").strip()

        if not text:

            return False

        self.load()

        chunks = self.split_text(
            text,
            max_chars=max_chars,
        )

        if not chunks:

            return False

        print()
        print(
            f"[NOVA TTS] Response split into "
            f"{len(chunks)} chunk(s)."
        )

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            print(
                f"[NOVA TTS] Chunk "
                f"{index}/{len(chunks)}: "
                f"{chunk}"
            )

        self._stop_event.clear()

        with self._speak_lock:

            self._speaking = True

            try:

                # -------------------------------------------------
                # IMPORTANT:
                #
                # We generate the NEXT chunk in a worker while
                # the CURRENT chunk is being played.
                #
                # Only one TTS generation happens at a time.
                # -------------------------------------------------

                next_audio = None
                next_sample_rate = None

                generation_thread = None
                generation_error = None

                def generate(
                    chunk_text,
                ):

                    nonlocal next_audio
                    nonlocal next_sample_rate
                    nonlocal generation_error

                    try:

                        (
                            next_audio,
                            next_sample_rate,
                        ) = self.generate_audio(
                            chunk_text
                        )

                    except Exception as error:

                        generation_error = error

                # -------------------------------------------------
                # Generate first chunk.
                # -------------------------------------------------

                print()
                print(
                    "[NOVA TTS] Preparing first chunk..."
                )

                generation_thread = threading.Thread(
                    target=generate,
                    args=(chunks[0],),
                    daemon=True,
                    name="NOVA-TTS-Generator",
                )

                generation_thread.start()

                # Wait for first chunk.
                generation_thread.join()

                if generation_error:

                    raise generation_error

                # -------------------------------------------------
                # Play each chunk while preparing the next one.
                # -------------------------------------------------

                for index in range(
                    len(chunks)
                ):

                    if self._stop_event.is_set():

                        sd.stop()

                        return False

                    # -------------------------------------------------
                    # Start generating next chunk BEFORE playback.
                    #
                    # This means generation can overlap with audio.
                    # -------------------------------------------------

                    if index + 1 < len(chunks):

                        next_generation_audio = [
                            None
                        ]

                        next_generation_rate = [
                            None
                        ]

                        next_generation_error = [
                            None
                        ]

                        def generate_next():

                            try:

                                (
                                    next_generation_audio[0],
                                    next_generation_rate[0],
                                ) = self.generate_audio(
                                    chunks[index + 1]
                                )

                            except Exception as error:

                                next_generation_error[0] = error

                        next_thread = threading.Thread(
                            target=generate_next,
                            daemon=True,
                            name="NOVA-TTS-Next",
                        )

                        next_thread.start()

                    else:

                        next_thread = None
                        next_generation_audio = [None]
                        next_generation_rate = [None]
                        next_generation_error = [None]

                    # -------------------------------------------------
                    # Current audio is ready.
                    # -------------------------------------------------

                    print(
                        f"[NOVA TTS] Playing "
                        f"chunk {index + 1}/"
                        f"{len(chunks)}..."
                    )

                    if next_audio is None:

                        raise RuntimeError(
                            "No audio generated for chunk."
                        )

                    playback_start = (
                        time.perf_counter()
                    )

                    success = self.play_audio(
                        next_audio,
                        next_sample_rate,
                    )

                    playback_time = (
                        time.perf_counter()
                        - playback_start
                    )

                    print(
                        f"[NOVA TTS] Playback "
                        f"{playback_time:.2f}s"
                    )

                    if not success:

                        return False

                    # -------------------------------------------------
                    # Wait for next generation.
                    # -------------------------------------------------

                    if next_thread is not None:

                        next_thread.join()

                        if next_generation_error[0]:

                            raise next_generation_error[0]

                        next_audio = (
                            next_generation_audio[0]
                        )

                        next_sample_rate = (
                            next_generation_rate[0]
                        )

                return True

            except Exception as error:

                print()
                print(
                    "[NOVA VOICE ERROR]"
                )
                print(error)

                if self.state is not None:

                    try:

                        self.state.observe(
                            f"Voice error: {error}"
                        )

                    except Exception:

                        pass

                return False

            finally:

                sd.stop()

                self._speaking = False

                if self.state is not None:

                    try:

                        self.state.set_nova_speaking(
                            False
                        )

                    except Exception:

                        pass

    # =========================================================
    # SPEAK
    # =========================================================

    def speak(
        self,
        text: str,
        blocking: bool = True,
    ):

        text = str(text or "").strip()

        if not text:

            return None

        if not blocking:

            thread = threading.Thread(
                target=self.speak_chunked,
                args=(text,),
                daemon=True,
                name="NOVA-Voice",
            )

            thread.start()

            return thread

        return self.speak_chunked(
            text
        )

    # =========================================================
    # STOP
    # =========================================================

    def stop(self):

        self._stop_event.set()

        sd.stop()

        self._speaking = False

        if self.state is not None:

            try:

                self.state.set_nova_speaking(False)

            except Exception:

                pass

    # =========================================================
    # STATUS
    # =========================================================

    def is_loaded(self) -> bool:

        return self.loaded

    def is_speaking(self) -> bool:

        return self._speaking

    # =========================================================
    # OUTPUT DEVICES
    # =========================================================

    @staticmethod
    def list_output_devices():

        print()
        print(
            "[NOVA VOICE] OUTPUT DEVICES"
        )

        devices = sd.query_devices()

        for index, device in enumerate(devices):

            if device["max_output_channels"] > 0:

                print(
                    f"{index}: {device['name']}"
                )


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("             NOVA FAST VOICE")
    print("=" * 60)

    gpu = NOVAVoice.gpu_status()

    print()
    print("GPU STATUS:")
    print(gpu)

    voice = NOVAVoice(
        speaker="ryan",
        device="cuda",
        audio_device=3,
    )

    voice.load()

    voice.speak(
        "Hello B. I am NOVA. "
        "This is a chunked voice test. "
        "The first part should begin playing before the entire response has finished generating. "
        "This allows NOVA to feel much more responsive.",
        blocking=True,
    )

    print()
    print("=" * 60)
    print("NOVA FAST VOICE TEST COMPLETE")
    print("=" * 60)