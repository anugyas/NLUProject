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
from parlai.utils.safety import OffensiveLanguageClassifier

def create_classifier():

    clf_file = 'zoo:bot_adversarial_dialogue/multi_turn/model'
    offensiveness_classifier = OffensiveLanguageClassifier(custom_model_file = clf_file)
    
    return clf_file, offensiveness_classifier
# TODO: to be integrated w/ target lm
# prob_offensiveness = offensiveness_classifier.contains_offensive_language(text_from_target_lm)
