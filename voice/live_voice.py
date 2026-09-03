from __future__ import annotations

import contextlib
import io
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from faster_whisper import WhisperModel


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# NOVA MODULES
# ============================================================

from providers.ollama_provider import OllamaProvider
from providers.groq_provider import GroqProvider

from core.router import Router
from core.orchestrator import Orchestrator

from voice.voice import NOVAVoice


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# MICROPHONE
# ------------------------------------------------------------

MIC_DEVICE = 9

MIC_CHANNELS = 2

MIC_SAMPLE_RATE = 48000

WHISPER_SAMPLE_RATE = 16000


# ------------------------------------------------------------
# SPEAKER
# ------------------------------------------------------------

AUDIO_DEVICE = 3


# ------------------------------------------------------------
# WHISPER
# ------------------------------------------------------------

WHISPER_MODEL = "small"

WHISPER_LANGUAGE = "en"


# ------------------------------------------------------------
# AUDIO FRAME
# ------------------------------------------------------------

FRAME_MS = 30

FRAME_SAMPLES = int(
    MIC_SAMPLE_RATE
    * FRAME_MS
    / 1000
)


# ------------------------------------------------------------
# SPEECH DETECTION
# ------------------------------------------------------------

START_FRAMES = 2

SILENCE_DURATION = 0.65

MAX_SPEECH_DURATION = 10.0

PRE_ROLL_DURATION = 0.30

MIN_SPEECH_RMS = 0.0005


# ------------------------------------------------------------
# NOVA BRAIN
# ------------------------------------------------------------

LOCAL_MODEL = "gpt-oss:20b"

GROQ_MODEL = "openai/gpt-oss-20b"


# ------------------------------------------------------------
# NOVA VOICE
# ------------------------------------------------------------

VOICE_SPEAKER = "ryan"

VOICE_DEVICE = "cuda"

VOICE_MAX_CHARS = 180


# ============================================================
# OUTPUT SUPPRESSION
# ============================================================

@contextlib.contextmanager
def suppress_internal_output():

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with contextlib.redirect_stdout(stdout_buffer):

        with contextlib.redirect_stderr(stderr_buffer):

            yield


# ============================================================
# RMS
# ============================================================

def calculate_rms(audio) -> float:

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:

        return 0.0

    return float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )


# ============================================================
# PEAK
# ============================================================

def calculate_peak(audio) -> float:

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:

        return 0.0

    return float(
        np.max(
            np.abs(audio)
        )
    )


# ============================================================
# SELECT STRONGEST CHANNEL
# ============================================================

def select_strongest_channel(
    audio,
    verbose=False,
):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.ndim == 1:

        return audio

    if audio.ndim != 2:

        raise ValueError(
            f"Unexpected microphone shape: "
            f"{audio.shape}"
        )

    if audio.shape[1] == 1:

        return audio[:, 0]

    rms_values = [
        calculate_rms(
            audio[:, channel]
        )
        for channel in range(
            audio.shape[1]
        )
    ]

    strongest_channel = int(
        np.argmax(
            rms_values
        )
    )

    if verbose:

        print(
            f"[MIC] Channel RMS: "
            f"{', '.join(f'{v:.6f}' for v in rms_values)}"
        )

        print(
            f"[MIC] Selected channel: "
            f"{strongest_channel}"
        )

    return audio[
        :,
        strongest_channel
    ]


# ============================================================
# RESAMPLE
# ============================================================

def resample_audio(
    audio,
    source_rate,
    target_rate,
):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)

    if audio.size == 0:

        return audio

    if source_rate == target_rate:

        return audio

    if (
        source_rate == 48000
        and target_rate == 16000
    ):

        resampled = resample_poly(
            audio,
            up=1,
            down=3,
        )

    else:

        from math import gcd

        divisor = gcd(
            source_rate,
            target_rate,
        )

        up = target_rate // divisor

        down = source_rate // divisor

        resampled = resample_poly(
            audio,
            up=up,
            down=down,
        )

    return np.asarray(
        resampled,
        dtype=np.float32,
    )


# ============================================================
# MICROPHONE INFO
# ============================================================

def get_microphone_info():

    devices = sd.query_devices()

    if (
        MIC_DEVICE < 0
        or MIC_DEVICE >= len(devices)
    ):

        raise RuntimeError(
            f"Invalid microphone device "
            f"index: {MIC_DEVICE}"
        )

    device = devices[MIC_DEVICE]

    print()

    print(
        "[MIC] Selected device:"
    )

    print(
        f"      Index : {MIC_DEVICE}"
    )

    print(
        f"      Name  : {device['name']}"
    )

    print(
        f"      Input : "
        f"{device['max_input_channels']} channels"
    )

    print(
        f"      Rate  : "
        f"{device['default_samplerate']} Hz"
    )

    print(
        f"      Target: "
        f"{WHISPER_SAMPLE_RATE} Hz for Whisper"
    )

    sd.check_input_settings(
        device=MIC_DEVICE,
        channels=MIC_CHANNELS,
        samplerate=MIC_SAMPLE_RATE,
        dtype="float32",
    )

    return device


# ============================================================
# MICROPHONE CALIBRATION
# ============================================================

def calibrate_microphone():

    print(
        "[MIC] Calibrating microphone..."
    )

    samples = []

    with sd.InputStream(
        device=MIC_DEVICE,
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
    ) as stream:

        calibration_frames = int(
            0.6
            / (FRAME_MS / 1000)
        )

        for _ in range(
            calibration_frames
        ):

            data, overflowed = (
                stream.read(
                    FRAME_SAMPLES
                )
            )

            if overflowed:

                print(
                    "[MIC] Warning: "
                    "input overflow during calibration."
                )

            samples.append(
                data.copy()
            )

    if not samples:

        return MIN_SPEECH_RMS

    calibration_audio = np.concatenate(
        samples,
        axis=0,
    )

    strongest = select_strongest_channel(
        calibration_audio,
        verbose=True,
    )

    noise_rms = calculate_rms(
        strongest
    )

    threshold = max(
        MIN_SPEECH_RMS,
        noise_rms * 4.0,
    )

    print(
        f"[MIC] Noise RMS: "
        f"{noise_rms:.8f}"
    )

    print(
        f"[MIC] Speech threshold: "
        f"{threshold:.8f}"
    )

    return threshold


# ============================================================
# LISTEN FOR ONE UTTERANCE
# ============================================================

def listen_for_speech(
    speech_threshold,
):

    print()
    print(
        "🎤 Listening..."
    )

    pre_roll_frames = max(
        1,
        int(
            PRE_ROLL_DURATION
            / (FRAME_MS / 1000)
        ),
    )

    silence_frames_required = max(
        1,
        int(
            SILENCE_DURATION
            / (FRAME_MS / 1000)
        ),
    )

    max_frames = max(
        1,
        int(
            MAX_SPEECH_DURATION
            / (FRAME_MS / 1000)
        ),
    )

    pre_roll = deque(
        maxlen=pre_roll_frames
    )

    captured = []

    speech_started = False

    speech_frame_count = 0

    silence_count = 0

    # --------------------------------------------------------
    # Keep channel selection stable during one utterance.
    # --------------------------------------------------------

    selected_channel = None

    with sd.InputStream(
        device=MIC_DEVICE,
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
    ) as stream:

        while True:

            data, overflowed = (
                stream.read(
                    FRAME_SAMPLES
                )
            )

            if overflowed:

                print(
                    "[MIC] Warning: "
                    "input overflow."
                )

            # ------------------------------------------------
            # Determine strongest channel only when waiting
            # for speech.
            # ------------------------------------------------

            if selected_channel is None:

                if data.ndim == 2 and data.shape[1] > 1:

                    channel_rms = [
                        calculate_rms(
                            data[:, channel]
                        )
                        for channel in range(
                            data.shape[1]
                        )
                    ]

                    selected_channel = int(
                        np.argmax(
                            channel_rms
                        )
                    )

                else:

                    selected_channel = 0

            frame = (
                data[:, selected_channel]
                if data.ndim == 2
                else data
            )

            rms = calculate_rms(
                frame
            )

            is_speech = (
                rms >= speech_threshold
            )

            # ------------------------------------------------
            # WAITING FOR SPEECH
            # ------------------------------------------------

            if not speech_started:

                pre_roll.append(
                    frame.copy()
                )

                if is_speech:

                    speech_frame_count += 1

                else:

                    speech_frame_count = 0

                if (
                    speech_frame_count
                    >= START_FRAMES
                ):

                    speech_started = True

                    print(
                        "🎤 Speech detected..."
                    )

                    print(
                        f"[MIC] Using channel "
                        f"{selected_channel}"
                    )

                    captured.extend(
                        list(pre_roll)
                    )

                    silence_count = 0

                    continue

            # ------------------------------------------------
            # RECORDING
            # ------------------------------------------------

            else:

                captured.append(
                    frame.copy()
                )

                if is_speech:

                    silence_count = 0

                else:

                    silence_count += 1

                if (
                    silence_count
                    >= silence_frames_required
                ):

                    break

                if (
                    len(captured)
                    >= max_frames
                ):

                    print(
                        "[MIC] Maximum "
                        "speech duration reached."
                    )

                    break

    if not captured:

        return None

    audio = np.concatenate(
        captured
    ).astype(
        np.float32
    )

    if audio.size == 0:

        return None

    # --------------------------------------------------------
    # Trim exact silence.
    # --------------------------------------------------------

    nonzero = np.flatnonzero(
        np.abs(audio) > 1e-8
    )

    if nonzero.size:

        audio = audio[
            nonzero[0]:
            nonzero[-1] + 1
        ]

    else:

        return None

    duration = (
        len(audio)
        / MIC_SAMPLE_RATE
    )

    print(
        f"[MIC] Captured "
        f"{duration:.2f}s"
    )

    print(
        f"[MIC] Audio RMS: "
        f"{calculate_rms(audio):.6f}"
    )

    print(
        f"[MIC] Audio peak: "
        f"{calculate_peak(audio):.6f}"
    )

    return audio


# ============================================================
# WHISPER
# ============================================================

def transcribe(
    whisper,
    audio,
):

    if audio is None:

        return ""

    print(
        "[STT] Transcribing..."
    )

    start = time.perf_counter()

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:

        return ""

    audio = resample_audio(
        audio,
        MIC_SAMPLE_RATE,
        WHISPER_SAMPLE_RATE,
    )

    print(
        f"[STT] Samples: "
        f"{len(audio)}"
    )

    segments, info = whisper.transcribe(

        audio,

        language=WHISPER_LANGUAGE,

        beam_size=5,

        best_of=1,

        temperature=0.0,

        condition_on_previous_text=False,

        vad_filter=False,

        no_speech_threshold=0.8,

        compression_ratio_threshold=None,

        log_prob_threshold=None,
    )

    text_parts = []

    for segment in segments:

        if segment.text:

            text_parts.append(
                segment.text.strip()
            )

    text = " ".join(
        text_parts
    ).strip()

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"[STT] {elapsed:.3f}s"
    )

    if text:

        print()
        print(
            f"You: {text}"
        )

    else:

        print(
            "[STT] No speech recognized."
        )

    return text


# ============================================================
# NOVA BRAIN
# ============================================================

def think(
    nova,
    text,
):

    with suppress_internal_output():

        response = nova.run(
            text,
            mode="auto",
        )

    if response is None:

        return ""

    return str(
        response
    ).strip()


# ============================================================
# NOVA VOICE
# ============================================================

def speak(
    voice,
    response,
):

    if not response:

        return

    print()
    print(
        "[TTS] Starting chunked speech..."
    )

    start = time.perf_counter()

    success = voice.speak_chunked(
        response,
        max_chars=VOICE_MAX_CHARS,
    )

    total_time = (
        time.perf_counter()
        - start
    )

    if success:

        print(
            f"[TTS] Complete in "
            f"{total_time:.3f}s"
        )

    else:

        print(
            "[TTS] Speech interrupted "
            "or failed."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 60
    )

    print(
        "             NOVA LIVE VOICE"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # MICROPHONE
    # ========================================================

    get_microphone_info()

    speech_threshold = (
        calibrate_microphone()
    )

    # ========================================================
    # WHISPER
    # ========================================================

    print()

    print(
        "Loading speech recognition..."
    )

    start = time.perf_counter()

    whisper = WhisperModel(
        WHISPER_MODEL,
        device="cuda",
        compute_type="float16",
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Speech recognition ready "
        f"({elapsed:.2f}s)."
    )

    print(
        "Whisper device: CUDA"
    )

    print(
        f"Whisper model: "
        f"{WHISPER_MODEL}"
    )

    print(
        f"Whisper input: "
        f"{WHISPER_SAMPLE_RATE} Hz"
    )

    # ========================================================
    # NOVA BRAIN
    # ========================================================

    print()

    print(
        "Loading NOVA brain..."
    )

    local = OllamaProvider(
        model=LOCAL_MODEL
    )

    groq = GroqProvider(
        model=GROQ_MODEL
    )

    router = Router(
        local_provider=local,
        groq_provider=groq,
    )

    nova = Orchestrator(
        router
    )

    print(
        "NOVA brain ready."
    )

    # ========================================================
    # NOVA VOICE
    # ========================================================

    print()

    print(
        "Loading NOVA voice..."
    )

    voice = NOVAVoice(
        speaker=VOICE_SPEAKER,
        device=VOICE_DEVICE,
        audio_device=AUDIO_DEVICE,
    )

    with suppress_internal_output():

        voice.load()

    print(
        "NOVA voice ready."
    )

    # ========================================================
    # READY
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "NOVA IS READY"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "NOVA: Hello B. I am ready to talk."
    )

    # ========================================================
    # LIVE LOOP
    # ========================================================

    while True:

        try:

            # ------------------------------------------------
            # LISTEN
            # ------------------------------------------------

            audio = listen_for_speech(
                speech_threshold
            )

            if audio is None:

                continue

            # ------------------------------------------------
            # STT
            # ------------------------------------------------

            text = transcribe(
                whisper,
                audio,
            )

            if not text:

                continue

            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            normalized = (
                text.lower()
                .strip()
                .rstrip(".!?")
            )

            if normalized in {

                "exit",
                "quit",
                "goodbye",
                "stop",
                "shutdown",

            }:

                print()

                print(
                    "NOVA: Goodbye B."
                )

                break

            # ------------------------------------------------
            # BRAIN
            # ------------------------------------------------

            print()

            print(
                "[NOVA] Thinking..."
            )

            brain_start = (
                time.perf_counter()
            )

            response = think(
                nova,
                text,
            )

            brain_time = (
                time.perf_counter()
                - brain_start
            )

            print(
                f"[BRAIN] "
                f"{brain_time:.3f}s"
            )

            if not response:

                print(
                    "NOVA: I couldn't generate "
                    "a response."
                )

                continue

            # ------------------------------------------------
            # TEXT
            # ------------------------------------------------

            print()

            print(
                f"NOVA: {response}"
            )

            # ------------------------------------------------
            # CHUNKED TTS
            # ------------------------------------------------

            speak(
                voice,
                response,
            )

            print()

        except KeyboardInterrupt:

            print()

            voice.stop()

            print(
                "NOVA stopped."
            )

            break

        except Exception as error:

            print()

            print(
                "[NOVA ERROR]"
            )

            print(
                repr(error)
            )

            print()

            print(
                "NOVA is still running."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()