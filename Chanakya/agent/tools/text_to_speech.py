import asyncio
import os
import re
from datetime import datetime

import edge_tts
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = os.environ.get("MODEL")


class TextToSpeech:
    VOICES = {
        "female": "en-US-JennyNeural",
        "male": "en-US-GuyNeural",
        "british": "en-GB-SoniaNeural",
        "indian": "en-IN-NeerjaNeural",
    }

    def __init__(self, voice: str = "en-US-JennyNeural", rate: str = "+10%"):
        self._voice = voice
        self._rate = rate
        self._filter_chain = None

    def _generate_filename(self, text: str) -> str:
        """Generate a filename based on text content with timestamp."""
        clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", text[:20])
        clean_text = re.sub(r"\s+", "-", clean_text.strip()).lower()

        if not clean_text:
            clean_text = "article"

        if len(clean_text) > 20:
            clean_text = clean_text[:20].rstrip("-")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{clean_text}_{timestamp}"

    def _build_filter_chain(self) -> RunnableSequence:

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
        )

        prompt = PromptTemplate(
            template="""You are a text cleaner for a text-to-speech pipeline.
            Your job:
               - Extract ONLY the main article or blog content from the text below
               - Remove: ads, navigation links, cookie notices, related article titles,
                 author bios, social share text, footer content, subscription prompts
               - Do NOT summarize
               - Do NOT rephrase any sentence
               - Do NOT add anything new
               - Return ONLY the clean article text, nothing else

            Input text:
                 {raw_text}""",
            template_format="f-string",
            input_variables=["raw_text"],
        )

        return prompt | llm

    def filter_text(self, raw_text: str) -> dict:
        try:
            if self._filter_chain is None:
                self._filter_chain = self._build_filter_chain()

            result = self._filter_chain.invoke({"raw_text": raw_text})
            clean_text = result.content.strip()

            if not clean_text:
                return {"success": False, "error": "LLM returned empty text"}

            return {
                "success": True,
                "text": clean_text,
                "word_count": len(clean_text.split()),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_voice(self, voice: str):
        self._voice = voice

    def set_rate(self, rate: str):
        self._rate = rate

    async def _generate_audio(self, text: str, output_path: str):
        communicate = edge_tts.Communicate(text, self._voice, rate=self._rate)
        folder = os.path.dirname(output_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        await communicate.save(output_path)

    def text_to_speech(
        self,
        text: str,
        output_path: str = "output.mp3",
    ) -> dict:
        """
        Converts text to speech using Microsoft Edge TTS.
        Free, no API key needed, high quality neural voices.

        Available voices:
          en-US-JennyNeural (default), en-US-GuyNeural,
          en-GB-SoniaNeural, en-IN-NeerjaNeural
        """
        try:
            word_count = len(text.split())

            if word_count > 5000:
                print(f"  Warning: {word_count} words — audio may take a moment")

            # Generate descriptive filename, add voice/ prefix and .mp3 extension
            filename = self._generate_filename(text)
            full_output_path = f"voice/{filename}.mp3"

            asyncio.run(self._generate_audio(text, full_output_path))

            if not os.path.exists(full_output_path):
                return {"success": False, "error": "Audio file was not created"}

            file_size_kb = os.path.getsize(full_output_path) // 1024

            return {
                "success": True,
                "audio_file": full_output_path,
                "voice": self._voice,
                "word_count": word_count,
                "file_size_kb": file_size_kb,
            }

        except Exception as error:
            return {"success": False, "error": str(error)}


@tool
def text_to_speech_tool(text: str, output_path: str = "output.mp3") -> dict:
    """
    Converts text to speech using Microsoft Edge TTS with automatic text filtering.
    Free, no API key needed, high quality neural voices.

    This tool automatically filters the input text to remove ads, navigation,
    and other non-content elements before converting to speech.

    Available voices:
      en-US-JennyNeural (default), en-US-GuyNeural,
      en-GB-SoniaNeural, en-IN-NeerjaNeural

    Args:
        text: The text to convert to speech (will be filtered automatically)
        output_path: Path where the audio file should be saved

    Returns:
        Dict containing success status, audio file path, voice used, word count, and file size
    """
    
  
    tts = TextToSpeech()

    # First filter the text
    filter_result = tts.filter_text(text)
    if not filter_result["success"]:
        return {
            "success": False,
            "error": f"Text filtering failed: {filter_result['error']}",
        }

    # Then convert filtered text to speech

    return tts.text_to_speech(filter_result["text"], output_path)
