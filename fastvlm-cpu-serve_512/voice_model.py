from faster_whisper import WhisperModel
import pyttsx3

class VoiceTranslator:
    def __init__(self):
        print("Loading ASR model...")
        self.asr = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8"
        )

        print("Loading TTS...")
        self.tts = pyttsx3.init()

    def run(
        self,
        input_audio: str,
        output_audio: str = "english_output.wav"
    ):
        # Speech -> English Text
        segments, _ = self.asr.transcribe(
            input_audio,
            task="translate"
        )

        english_text = " ".join(
            seg.text for seg in segments
        )

        print("\nTranscript:")
        print(english_text)

        # English Text -> Speech
        self.tts.save_to_file(
            english_text,
            output_audio
        )
        self.tts.runAndWait()

        return {
            "text": english_text,
            "audio_file": output_audio
        }


# -------------------------
# Usage
# -------------------------

pipeline = VoiceTranslator()

result = pipeline.run(
    "input.wav",
    "output.wav"
)

print(result)