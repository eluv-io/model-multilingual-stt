
from typing import List
import json

import nemo.collections.asr as nemo_asr\
from common_ml.model import VideoModel
from common_ml.tag_formatting import VideoTag
from loguru import logger

from config import config


class EuroSTT(VideoModel):
    def __init__(self):
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name="stt_multilingual_fastconformer_hybrid_large_pc_blend_eu")
        self.frame_size = 80

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
