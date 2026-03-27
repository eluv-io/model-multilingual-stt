
from dacite import from_dict
import setproctitle

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

from src.producer import *

if __name__ == '__main__':
    setproctitle.setproctitle("model-asr")
    catch_errors()
    
    params = get_params()
    params = from_dict(data=params, data_class=RuntimeConfig)

    producer = ASRProducer(cfg=params)

    run_default(producer)