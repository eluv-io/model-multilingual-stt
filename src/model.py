from typing import List
from dataclasses import dataclass

import nemo.collections.asr as nemo_asr
import torch
from loguru import logger

from common_ml.tagging.models.tag_types import Tag

@dataclass(frozen=True)
class OutputTag:
    start_time: int
    end_time: int
    tag: str

class EuroSTT:
    def __init__(self):
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name="stt_multilingual_fastconformer_hybrid_large_pc_blend_eu")
        self.frame_size = 80

    def tag(self, audio_tensor: torch.Tensor) -> List[OutputTag]:
        hypothesis = self.model.transcribe(
            audio=audio_tensor, return_hypotheses=True)[0]
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
            tags.append(OutputTag(
                start_time=ts,
                end_time=ts,
                tag=word,
            ))

        return tags

    def _get_word_level_timestamps(self, timestamps: list, tokens: list) -> list:
        word_timestamps = []
        for ts, tok in zip(timestamps, tokens):
            if tok.startswith('▁'):
                word_timestamps.append(ts)
        return word_timestamps
