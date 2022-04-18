import trl
import datasets
import torch
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()
from datasets import load_dataset, Dataset
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt
import yaml
from parlai.core.agents import create_agent_from_model_file
from parlai.core.teachers import register_teacher, DialogTeacher
from parlai.scripts.eval_model import EvalModel
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier


config = {
    "lm_name": "gpt2-large",
    "ref_lm_name": "gpt2-large",
    "tk_name": "gpt2",
    "steps": 25600,
    "batch_size": 64,
    "forward_batch_size": 16,
    "ppo_epochs": 4,   
    "txt_in_len": 5,
    "txt_out_len": 150,
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1,
    "response_save_file": f'./data/response/few_shot_sample.responses.all.jsonl',
}


@register_teacher("test_cases")
class MyTeacher(DialogTeacher):
  def __init__(self, opt, shared=None):
    opt['datafile'] = f'./data/query/test_cases.txt'
    super().__init__(opt, shared)
  
  def setup_data(self, datafile):
    print(f" ~~ Loading from {datafile} ~~ ")
    with open(self.opt['datafile']) as f:
      lines = [line.strip() for line in f]

    # Get first dialogue utterances written by humans
    for text in lines:
      yield (text, '__notok__'), True


class RLAgent():
    def __init__(self, config,device,classifer=None):
        self.config=config
        self.device= device
        self.model = GPT2HeadWithValueModel.from_pretrained(config['lm_name'])
        self.model_ref = GPT2HeadWithValueModel.from_pretrained(config['ref_lm_name'])
        self.tokenizer = GPT2Tokenizer.from_pretrained(config['tk_name'])
        if classifier:
            self.classifier= classifier
        else:
            self.classifier= None
        
        self.wandb= None

    def train(self, data):
        
        
        data['tokens']=  data['prompt'].progress_apply(lambda x: self.tokenizer.encode(x, return_tensors="pt").to(self.device)[0,:])
        data['query'] = data['tokens'].progress_apply(lambda x: gpt2_tokenizer.decode(x))
        
        fbs= self.config["forward_batch_size"]
        clf_file, clf = create_classifier()

        for epoch in tqdm(range(int(np.ceil(config["steps"]/config['batch_size'])))):
            if device=='cuda':
                torch.cuda.empty_cache()
            logs = dict()
            game_data = dict()
            timing = dict()
            t0 = time.time()
            
            #### get a batch from the dataset
            data_batch = data.sample(config['batch_size'])
            game_data['query'] = data_batch['query'].tolist()
            query_tensors = torch.stack(data_batch['tokens'].tolist())
            
            #### generate questions(test_cases) from gpt2(red_lm)
            t = time.time()
            # total_length = config['txt_in_len']+config['txt_out_len']
            response_tensors = []
            for i in range(int(config['batch_size']/fbs)):
                response  = respond_to_batch(self.model, query_tensors[i*fbs:(i+1)*fbs],
                                            txt_len=config['txt_out_len'])
                response_tensors.append(response)
            response_tensors = torch.cat(response_tensors)
            game_data['response'] = [self.tokenizer.decode(response_tensors[i, :]) for i in range(config['batch_size'])]
            # import pdb; pdb.set_trace()
            timing['time/get_response'] = time.time()-t

            #### tokenize text for sentiment analysis
            t = time.time()
            texts = [q + r for q,r in zip(game_data['query'], game_data['response'])]
            classifier_inputs, attention_masks = build_bert_batch_from_txt(texts, sentiment_tokenizer, device)    
            timing['time/build_input_sentiment'] = time.time()-t

            #### get classifier score
            response_save_file = config['response_save_file']
            EvalModel.main(task='rl_test_cases', model_file="zoo:blender/blender_400Mdistill/model", skip_generation=False, batchsize=64, world_logs=response_save_file)
            with open(response_save_file) as f:
                responses = [json.loads(line) for line in f]

            humans = [response['dialog'][0][0]['text'] for response in responses]
            bots = [response['dialog'][0][1]['text'] for response in responses]
            texts = ['\n'.join([turn['text'] for turn in response['dialog'][0]]) for response in responses]
            t = time.time()
            preds, confs = zip(*[clf.contains_offensive_language(text) for text in tqdm(texts)])
            probs = np.array([conf if pred else (1 - conf) for pred, conf in zip(preds, confs)])
            rewards = probs
            # rewards = []
            # for i in range(int(config['batch_size']/fbs)):
            #     res = classifier_model.forward(classifier_inputs[i*fbs:(i+1)*fbs],
            #                                 attention_masks[i*fbs:(i+1)*fbs])[0][:, 1].detach()
            #     rewards.append(res)
            # rewards = torch.cat(rewards)
            timing['time/get_sentiment_preds'] = time.time()-t
            
            #### Run PPO training 
            t = time.time()
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            timing['time/optimization'] = time.time()-t
            
            #### Log everything
            timing['time/epoch'] = time.time()-t0
            table_rows = [list(r) for r in zip(game_data['query'], game_data['response'], rewards.cpu().tolist())]
            
            if self.wandb:
                logs.update({'game_log': self.wandb.Table(
                columns=['query', 'response', 'reward'],
                rows=table_rows)})
                logs.update(timing)
                logs.update(stats)
                logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
                logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
                logs['env/reward_dist'] = rewards.cpu().numpy()
                self.wandb.log(logs)




