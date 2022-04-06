import os
import random
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.strings import colorize
import parlai.utils.logging as logging


# MODEL_FILE = 'zoo:bot_adversarial_dialogue/multi_turn/model'
# DATAPATH = ''
# NUM_EXAMPLES = 1

# os.system("parlai display_model --task bot_adversarial_dialogue --modelfile " + MODEL_FILE + " --datatype valid --datapath " + DATAPATH + " --num-examples " + str(NUM_EXAMPLES))

def run_classifier(opt):
    random.seed(42)

    # Create model and assign it to the specified task
    agent = create_agent(opt)
    print("AGENT: ", agent)
#     world = create_task(opt, agent)
#     agent.opt.log()

    # Show some example dialogs.
#     turn = 0
#     with world:
#         for _k in range(int(opt['num_examples'])):
#             world.parley()
#             if opt['verbose'] or opt.get('display_add_fields', ''):
#                 print(world.display() + "\n~~")
# #             else:
# #                 simple_display(opt, world, turn)
#             turn += 1
#             if world.get_acts()[0]['episode_done']:
#                 turn = 0
#             if world.epoch_done():
#                 logging.info("epoch done")
#                 turn = 0
#                 break

@register_script("classifier", aliases=["classifier"])
class Classifier(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(True, True)
        parser.add_argument('-n', '-ne', '--num-examples', default=1)
        parser.set_defaults(datatype='valid')
        return parser

    def run(self):
        run_classifier(self.opt)
        
if __name__ == '__main__':
    Classifier.main()