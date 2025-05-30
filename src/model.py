
from typing import List
import json

import nemo.collections.asr as nemo_asr
import ollama
from common_ml.model import VideoModel
from common_ml.tag_formatting import VideoTag
from common_ml.utils.metrics import timeit
from loguru import logger
from deepmultilingualpunctuation import PunctuationModel
import spacy

from config import config


class EuroSTT(VideoModel):
    def __init__(self, cfg: dict):
        self.config = cfg
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name="stt_multilingual_fastconformer_hybrid_large_pc_blend_eu")
        self.frame_size = 80
        self.translator = ollama.Client(cfg["llama_host"])
        self.llama_model = cfg["llama_model"]
        self.prompt = cfg["prompt"]
        self.punctuation_model = PunctuationModel()
        self.capitalization_model = spacy.load("en_core_web_sm")

    def get_config(self) -> dict:
        return self.config

    def set_config(self, cfg: dict) -> None:
        self.config = cfg

    def tag(self, fpath: str) -> List[VideoTag]:
        # --- 1.  ASR ------------------------------------------------------------
        hyp = self.model.transcribe([fpath], return_hypotheses=True,
                                    channel_selector='average')[0]

        # word‑level end‑timestamps (same length as hyp.text.split())
        ends = self._get_word_level_timestamps(hyp.timestamp,
                                               self.model.tokenizer.ids_to_tokens(
                                                   [t.item() for t in hyp.y_sequence]))
        ends = [float(ts * self.frame_size) for ts in ends]          # seconds

        if len(ends) == 0:
            logger.warning(f"No words detected in {fpath}.")
            return []

        # --- 2.  MT -------------------------------------------------------------
        prompt = (
            f"{self.prompt}\n{hyp.text}\n"
            'Output ONLY json: {"translation": "<Translation>"}'
        )

        with timeit("translating"):
            raw = self.translator.generate(model=self.llama_model,
                                           stream=False,
                                           prompt=prompt,
                                           options={'seed': 1, 'temperature': 0})["response"]

        try:
            trans = json.loads(
                raw[raw.index("{"):raw.index("}") + 1])["translation"]
        except Exception as e:
            logger.error(f"Bad translation payload: {e}\n{raw}")
            return []

        # reference window
        src_first, src_last = ends[0], ends[-1]
        tgt_words = trans.split()
        n = len(tgt_words)
        step = (src_last - src_first) / n if n else 0

        # --- 3.  align translation to timeline -------------------------------
        tags = []
        for i, word in enumerate(tgt_words):
            st = src_first + i * step
            tags.append(VideoTag(start_time=st, end_time=st, text=word))

        return self.prettify_tags(tags)

    def prettify_tags(self, asr_tags: List[VideoTag]) -> List[VideoTag]:
        if len(asr_tags) == 0:
            return []
        max_gap = config["postprocessing"]["sentence_gap"]
        full_transcript = [asr_tags[0].text]
        last_start = asr_tags[0].start_time
        for tag in asr_tags[1:]:
            if tag.start_time - last_start > max_gap:
                full_transcript.append(tag.text)
            else:
                full_transcript[-1] += ' ' + tag.text
            last_start = tag.start_time
        corrected_transcript = [self.correct_text(t) for t in full_transcript]
        corrected_transcript = ' '.join(corrected_transcript)
        for tag, word in zip(asr_tags, corrected_transcript.split()):
            tag.text = word

        return asr_tags

    def correct_text(self, text: str) -> str:
        if text == "":
            return text
        res = self.capitalize_proper_nouns(text)
        res = self.punctuation_model.restore_punctuation(res)
        if not res.endswith("."):
            res += "."
        sentence_delimiters = ['.', '?', '!']
        # iterate through first character of each sentence and capitalize it
        capitalized = []
        for i, c in enumerate(res):
            if i == 0 or i > 1 and res[i-2] in sentence_delimiters:
                capitalized.append(c.upper())
            else:
                capitalized.append(c)
        return ''.join(capitalized)

    def capitalize_proper_nouns(self, sentence: str) -> str:
        model = self.capitalization_model
        proper_acronyms = ["us", "uk", "usa", "dc", "nyc", "la", "sf",
                           "nba", "nfl", "mlb", "ncaa", "nasa", "fbi", "cia", "nypd", "lapd"]
        doc = model(sentence)
        capitalized_sentence = ""
        for token in doc:
            if token.pos_ == "PROPN" and token.text in proper_acronyms:
                capitalized_sentence += token.text.upper()
            elif token.pos_ == "PROPN":
                capitalized_sentence += token.text.capitalize()
            elif token.text == "i" or token.text.startswith("i'"):
                capitalized_sentence += token.text.capitalize()
            else:
                capitalized_sentence += token.text
            capitalized_sentence += token.whitespace_

        return capitalized_sentence

    def _get_word_level_timestamps(self, timestamps: list, tokens: list) -> list:
        word_timestamps = []
        for ts, tok in zip(timestamps, tokens):
            if tok.startswith('▁'):
                word_timestamps.append(ts)
        return word_timestamps
