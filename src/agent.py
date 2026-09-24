import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import edge_tts

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TextFrame, AudioRawFrame, UserAudioRawFrame, TranscriptionFrame
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.websocket.server import WebsocketServerTransport, WebsocketServerParams

# Direct local imports inside src/
from audio_prosody import EnhancedProsodyAnalyzer
from memory import MemoryManager

# Resolve project root boundaries cleanly for environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("❌ GROQ_API_KEY is missing in your .env file!")
    sys.exit(1)


class ExpressiveTTSProcessor(FrameProcessor):
    PROSODY_MAP = {
        "excited_animated": {"rate": "+10%", "pitch": "+4Hz"},
        "low_energy_whisper": {"rate": "-8%", "pitch": "-3Hz"},
        "questioning_curious": {"rate": "+0%", "pitch": "+3Hz"},
        "stressed_strained": {"rate": "-5%", "pitch": "-4Hz"},
        "flat_monotone": {"rate": "+4%", "pitch": "+0Hz"},
        "neutral_calm": {"rate": "+0%", "pitch": "+0Hz"}
    }

    def __init__(self, voice: str = "en-US-EmmaNeural"):
        super().__init__()
        self.voice = voice
        self.current_affect = "neutral_calm"

    def set_affect_state(self, affect: str):
        if affect in self.PROSODY_MAP:
            self.current_affect = affect

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            settings = self.PROSODY_MAP.get(self.current_affect, self.PROSODY_MAP["neutral_calm"])
            logger.info(f"🔊 Synthesizing Edge-TTS [{self.current_affect}]: Rate={settings['rate']}, Pitch={settings['pitch']}")

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


class RealtimeProsodyProcessor(FrameProcessor):
    def __init__(self, context: OpenAILLMContext, tts_engine: ExpressiveTTSProcessor):
        super().__init__()
        self.analyzer = EnhancedProsodyAnalyzer(sample_rate=16000)
        self.context = context
        self.tts_engine = tts_engine
        self.audio_buffer = bytearray()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserAudioRawFrame):
            self.audio_buffer.extend(frame.audio)

            if len(self.audio_buffer) >= 32000:
                profile = self.analyzer.extract_prosody_profile(bytes(self.audio_buffer))
                self.audio_buffer.clear()

                affect = profile["detected_affect"]
                if affect != "neutral_calm":
                    self.tts_engine.set_affect_state(affect)
                    hint = (
                        f"[System Cue: User vocal affect is {affect} "
                        f"(Pitch Contour: {profile['pitch_contour']}, Energy: {profile['rms_energy']})]"
                    )
                    logger.info(f"🎙️ Acoustic Hint Injected: {hint}")

        await self.push_frame(frame, direction)


class MemoryRetrievalProcessor(FrameProcessor):
    def __init__(self, memory_manager: MemoryManager, context: OpenAILLMContext, user_id: str = "user_01"):
        super().__init__()
        self.memory = memory_manager
        self.context = context
        self.user_id = user_id

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text.strip()
            if user_text:
                memories = self.memory.retrieve_memories(
                    user_id=self.user_id,
                    query=user_text,
                    limit=2,
                    score_threshold=0.52
                )
                if memories:
                    memory_hint = f"[System Note: Relevant User Context -> {'; '.join(memories)}]"
                    logger.info(f"🧠 Injected Long-Term Memory: {memory_hint}")
                    self.context.messages.append({"role": "system", "content": memory_hint})


async def main():
    logger.info("⚡ Initializing Vox-Companion Full Multimodal Pipeline...")

    memory_manager = MemoryManager()
    memory_manager.store_memory("user_01", "User prefers vegetarian recipes and simple dishes.")
    memory_manager.store_memory("user_01", "User is studying Electronics and Communication Engineering.")

    transport = WebsocketServerTransport(
        host="0.0.0.0",
        port=8765,
        params=WebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False
        )
    )

    stt = GroqSTTService(
        api_key=GROQ_API_KEY,
        model="whisper-large-v3-turbo"
    )

    llm = GroqLLMService(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant"
    )

    tts = ExpressiveTTSProcessor(voice="en-US-EmmaNeural")

    system_prompt = (
        "You are Vox, a warm, supportive, and emotionally intelligent voice companion. "
        "You respond naturally in concise, conversational sentences as if on a phone call. "
        "Pay attention to system notes regarding the user's acoustic tone and retrieved facts. "
        "Never use bullet points, bolding, markdown formatting, or emojis in your responses."
    )

    context = OpenAILLMContext(messages=[{"role": "system", "content": system_prompt}])
    context_aggregator = llm.create_context_aggregator(context)
    prosody_processor = RealtimeProsodyProcessor(context, tts)
    memory_processor = MemoryRetrievalProcessor(memory_manager, context, user_id="user_01")

    pipeline = Pipeline(
        [
            transport.input(),
            prosody_processor,
            stt,
            memory_processor,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            vad_analyzer=SileroVADAnalyzer()
        )
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("🔌 Client disconnected.")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    logger.info("🎙️ Vox-Companion Engine Online on ws://localhost:8765")
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Vox-Companion agent...")