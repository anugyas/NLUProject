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
sys.path.insert(0, '../classifier')


config = {
    "lm_name": "gpt2-large",
    "ref_lm_name": "gpt2-large",
    "tk_name": "gpt2",
    "steps": 25600,
    "batch_size": 64,
    "forward_batch_size": 16,
    "ppo_epochs": 4,   
    "txt_in_len": 5,
    "txt_out_len": 15,
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1,
}


class RLAgent:
    def __init__(self, config,wandb,classifer=None):
        self.config=config
        self.device= device
        self.ppo_trainer= PPOTrainer(gpt2_model, gpt2_model_ref, **config)

        self.model = GPT2HeadWithValueModel.from_pretrained(config['lm_name'])
        self.model_ref = GPT2HeadWithValueModel.from_pretrained(config['ref_lm_name'])
        self.tokenizer = GPT2Tokenizer.from_pretrained(config['tk_name'])

        if classifier:
            self.classifier= classifier
        else:
            self.classifier= None
        
        self.wandb= None
        if wandb:
            self.wandb= wandb

    def train(self, data):
        data['tokens']=  data['prompt'].progress_apply(lambda x: self.tokenizer.encode(x, return_tensors="pt").to(self.device)[0,:])
        data['query'] = data['tokens'].progress_apply(lambda x: gpt2_tokenizer.decode(x))
        
        fbs= self.config["forward_batch_size"]

        for epoch in tqdm(range(int(np.ceil(config["steps"]/config['batch_size'])))):
            torch.cuda.empty_cache()
            logs = dict()
            game_data = dict()
            timing = dict()
            t0 = time.time()
            
            #### get a batch from the dataset
            data_batch = data.sample(config['batch_size'])
            game_data['query'] = data_batch['query'].tolist()
            prompt_tensors = torch.stack(data_batch['tokens'].tolist())
            
            #### get response from gpt2
            t = time.time()
            total_length = config['txt_in_len']+config['txt_out_len']
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
            t = time.time()
            rewards = []
            for i in range(int(config['batch_size']/fbs)):
                res = classifier_model.forward(classifier_inputs[i*fbs:(i+1)*fbs],
                                            attention_masks[i*fbs:(i+1)*fbs])[0][:, 1].detach()
                rewards.append(res)
            rewards = torch.cat(rewards)
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




