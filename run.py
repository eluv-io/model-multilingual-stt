
import argparse
import os
import sys
import json
from dacite import from_dict
import setproctitle

from common_ml.tagging.run_helpers import catch_errors, get_params, start_loop_from_producer

from src.producer import *

if __name__ == '__main__':
    catch_errors()
    setproctitle.setproctitle("model-asr")
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=False)
    args, _ = parser.parse_known_args()
    
    params = get_params()
    params = from_dict(data=params, data_class=RuntimeConfig)

    producer = ASRProducer(cfg=params)

    start_loop_from_producer(producer, output_path=args.output_path)