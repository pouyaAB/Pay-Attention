# from nf_model import ModelController
# from nf_model_prednet import ModelController
# from nf_model_generate_att_cycle_top_path_e2e import ModelController
from nf_model_generate_att_cycle_top_path_e2e import ModelController
import argparse
from local_config import config
import signal
import sys

locals().update(config)

parser = argparse.ArgumentParser(description='RNN MDN trainer')
parser.add_argument('--load', default=True, type=bool,
                    help='whether to load the models')
args = parser.parse_args()

def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    MC = ModelController(image_size=image_size, latent_size=latent_size, batch_size=batch_size,
                         sequence_size=sequence_size, num_channels=num_channels, load_models=args.load)

    MC.train()
