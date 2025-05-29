
from typing import List
import json

import nemo.collections.asr as nemo_asr
import ollama
from common_ml.model import VideoModel
from common_ml.tag_formatting import VideoTag
from loguru import logger

from config import config


class EuroSTT(VideoModel):
    def __init__(self):
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name="stt_multilingual_fastconformer_hybrid_large_pc_blend_eu")
        self.frame_size = 80
        self.translator = ollama.Client(config["llama"])
        self.llama_model = "llama3.3:70b"
        self.prompt = "Translate the following French text to English, please maintain the timestamps from the input."

    def tag(self, fpath: str) -> List[VideoTag]:
        hypothesis = self.model.transcribe(
            [fpath], return_hypotheses=True, channel_selector='average')[0]
        toks = self.model.tokenizer.ids_to_tokens(
            [tok.item() for tok in hypothesis.y_sequence])
        word_level_timestamps = self._get_word_level_timestamps(
            hypothesis.timestamp, toks)
        word_level_timestamps = [
            (ts * self.frame_size).item() for ts in word_level_timestamps]
        timesteps_w_words = list(
            zip(hypothesis.text.split(), word_level_timestamps))
        # convert tuples to lists
        timesteps_w_words = [[word, ts] for word, ts in timesteps_w_words]

        prompt = f"{self.prompt}\n" + str(timesteps_w_words) + \
            "\nOutput your response in the following format: {\"translation\": [[word1, timestamp1], [word2, timestamp2], ...]}. Do not output anything else."

        raw_response = self.translator.generate(
            model=self.llama_model,
            stream=False,
            prompt=prompt,
            options={'seed': 1, "temperature": 0.0})["response"]

        try:
            response = raw_response[raw_response.index(
                "{"):raw_response.index("}") + 1]
            response = json.loads(response)
            timesteps_w_words = response['translation']
        except Exception as e:
            logger.debug(f"Raw response:\n {raw_response}")
            logger.error(f"Error parsing translation response: {e}")
            return []

        if not timesteps_w_words:
            logger.debug("No words found in transcription.")
            return []

        tags = []
        for word, ts in timesteps_w_words:
            ts = int(ts)
            tags.append(VideoTag(
                start_time=ts,
                end_time=ts,
                text=word,
            ))

        return tags

    def _get_word_level_timestamps(self, timestamps: list, tokens: list) -> list:
        word_timestamps = []
        for ts, tok in zip(timestamps, tokens):
            if tok.startswith('▁'):
                word_timestamps.append(ts)
        return word_timestamps
