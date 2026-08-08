# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import base64
import os
import time
from typing import List, Optional
from urllib.parse import urlparse


import openai
import requests
from pydub.utils import mediainfo

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.agents import ChatAgent
from camel.models import BaseModelBackend

import logging
logger = logging.getLogger(__name__)


class AudioAnalysisToolkit(BaseToolkit):
    r"""A class representing a toolkit for audio operations.

    This class provides methods for processing and understanding audio data.
    """

    def __init__(self, cache_dir: Optional[str] = None, audio_reasoning_model: Optional[BaseModelBackend] = None):
        self.cache_dir = 'tmp/'
        if cache_dir:
            self.cache_dir = cache_dir

        self.client = openai.OpenAI()
        self.audio_reasoning_model = audio_reasoning_model
        self._bench_timings: dict = {}
        
    def get_audio_duration(self, file_path):
        info = mediainfo(file_path)
        duration = float(info['duration'])
        return duration


    def ask_question_about_audio(self, audio_path: str, question: str) -> str:
        r"""Ask any question about the audio and get the answer using
            multimodal model.

        Args:
            audio_path (str): The path to the audio file.
            question (str): The question to ask about the audio.

        Returns:
            str: The answer to the question.
        """

        logger.debug(
            f"Calling ask_question_about_audio method for audio file \
            `{audio_path}` and question `{question}`."
        )

        parsed_url = urlparse(audio_path)
        is_url = all([parsed_url.scheme, parsed_url.netloc])
        encoded_string = None

        if is_url:
            res = requests.get(audio_path)
            res.raise_for_status()
            audio_data = res.content
            encoded_string = base64.b64encode(audio_data).decode('utf-8')
        else:
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            audio_file.close()
            encoded_string = base64.b64encode(audio_data).decode('utf-8')

        file_suffix = os.path.splitext(audio_path)[1]
        file_format = file_suffix[1:]

        if self.audio_reasoning_model:
            text_prompt = f"Transcribe all the content in the speech into text."
            _t0 = time.perf_counter()
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=open(audio_path, "rb")
            )
            self._bench_timings['transcription_elapsed'] = time.perf_counter() - _t0

            transcript = transcription.text

            reasoning_prompt = f"""
            <speech_transcription_result>{transcript}</speech_transcription_result>

            Please answer the following question based on the speech transcription result above:
            <question>{question}</question>
            """
            
            audio_reasoning_agent = ChatAgent(
                "You are a helpful assistant that can answer questions about the given speech transcription.",
                model=self.audio_reasoning_model
                )
            
            reasoning_result = audio_reasoning_agent.step(reasoning_prompt)
            response: str = str(reasoning_result.msg.content)
            duration = self.get_audio_duration(audio_path)
            response += f"\n\nAudio duration: {duration} seconds"

            logger.debug(f"Response: {response}")
            return response


        else:
            text_prompt = f"""Answer the following question based on the given \
            audio information:\n\n{question}"""

            _t0 = time.perf_counter()
            completion = self.client.chat.completions.create(
                # model="gpt-4o-audio-preview",
                model = "gpt-4o-mini-audio-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specializing in \
                        audio analysis.",
                    },
                    {  # type: ignore[list-item, misc]
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded_string,
                                    "format": file_format,
                                },
                            },
                        ],
                    },
                ],
            )  # type: ignore[misc]
            self._bench_timings['transcription_elapsed'] = time.perf_counter() - _t0

            # get the duration of the audio
            duration = self.get_audio_duration(audio_path)

            response: str = str(completion.choices[0].message.content)
            response += f"\n\nAudio duration: {duration} seconds"

            logger.debug(f"Response: {response}")
            return response
        

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions
            in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing the
                functions in the toolkit.
        """
        return [FunctionTool(self.ask_question_about_audio)]
