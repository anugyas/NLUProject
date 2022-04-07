import os
import random
random.seed(42)
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.strings import colorize
import parlai.utils.logging as logging
from parlai.core.torch_agent import Batch

def run_classifier(agent, data_from_targetLM="NONE"):
    # TODO: figure out how to create Batch object using parsed data from Target LM
    harmfulness_score = agent.score(input_vec)
    print("Harmfulness Score: ", 111111)
    return harmfulness_score

@register_script("classifier", aliases=["classifier"])
class Classifier(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(True, True)
        parser.add_argument('-n', '-ne', '--num-examples', default=1)
        print("Parser: ", parser)
#         parser.set_defaults(interactive_mode=True)
#         parser.set_defaults(print_scores=True)
        parser.set_defaults(batch_size=1)
        parser.set_defaults(task='bot_adversarial_dialogue')
        parser.set_defaults(model_file='zoo:bot_adversarial_dialogue/multi_turn/model')
        parser.set_defaults(datatype='valid')
        return parser

    def run(self):
        return create_agent(self.opt)
        
        

classifier_agent = Classifier.main()

# for i in range(10):
#     run_classifier(classifier_agent, data_from_targetLM="AAAAAAAAAAAAA"+str(i))