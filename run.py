
import argparse
from typing import List, Callable
import os
import sys
import json
from dacite import from_dict
import setproctitle
from dataclasses import dataclass

from common_ml.utils import nested_update
from common_ml.model import run_live_mode

from src.tagger import SpeechTagger, RuntimeConfig
from config import config

def make_tag_fn(cfg: RuntimeConfig, tags_out: str) -> Callable:
    """
    Create a function that processes audio files using SpeechTagger
    
    Args:
        cfg: Runtime configuration
        
    Returns:
        Function that takes list of audio file paths
    """
    print(cfg)
    tagger = SpeechTagger(cfg, tags_out)
    
    def tag_fn(audio_paths: List[str]) -> None:
        for fname in audio_paths:
            tagger.tag(fname)
    
    return tag_fn

if __name__ == '__main__':
    setproctitle.setproctitle("model-asr")
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_paths', nargs='*', type=str, default=[])
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--live', action='store_true', help='Run in live mode (read files from stdin)')
    args = parser.parse_args()
    
    if args.config is None:
        cfg = config["runtime"]["default"]
    else:
        cfg = json.loads(args.config)
        cfg = nested_update(config["runtime"]["default"], cfg)

    runtime_config = from_dict(data=cfg, data_class=RuntimeConfig)

    tags_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    if not os.path.exists(tags_out):
        os.makedirs(tags_out)
    
    tag_fn = make_tag_fn(runtime_config, tags_out)

    if args.live:
        print('Running in live mode', file=sys.stderr)
        run_live_mode(tag_fn)
    else:
        tag_fn(args.audio_paths)