import edge_tts
import asyncio
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TextFrame, AudioRawFrame
from loguru import logger


class ExpressiveTTSProcessor(FrameProcessor):
    """
    Prosody-aware Edge-TTS generator that dynamically adjusts pitch, rate, and volume
    based on the user's acoustic state.
    """
    
    # Emotional prosody tuning matrix for Edge-TTS pitch/rate attributes
    PROSODY_MAP = {
        "excited_animated": {"rate": "+12%", "pitch": "+5Hz"},
        "low_energy_whisper": {"rate": "-10%", "pitch": "-3Hz"},
        "questioning_curious": {"rate": "+0%", "pitch": "+3Hz"},
        "stressed_strained": {"rate": "-5%", "pitch": "-4Hz"},
        "flat_monotone": {"rate": "+5%", "pitch": "+0Hz"},
        "neutral_calm": {"rate": "+0%", "pitch": "+0Hz"}
    }

    def __init__(self, voice: str = "en-US-EmmaNeural"):
        super().__init__()
        self.voice = voice
        self.current_affect = "neutral_calm"

    def set_affect_state(self, affect: str):
        """Allows real-time updating of the acoustic target profile."""
        if affect in self.PROSODY_MAP:
            self.current_affect = affect

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            settings = self.PROSODY_MAP.get(self.current_affect, self.PROSODY_MAP["neutral_calm"])
            logger.info(f"🔊 Synthesizing TTS with affect [{self.current_affect}]: {settings}")

            communicate = edge_tts.Communicate(
                text=frame.text,
                voice=self.voice,
                rate=settings["rate"],
                pitch=settings["pitch"]
            )

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    await self.push_frame(
                        AudioRawFrame(
                            audio=chunk["data"],
                            sample_rate=24000,
                            num_channels=1
                        )
                    )
        else:
            await self.push_frame(frame, direction)