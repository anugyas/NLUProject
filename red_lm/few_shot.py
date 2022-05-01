import random
import numpy as np
from transformers import pipeline
import re
import torch

class FewShot():
    def __init__(self, temperature, questions, clf_prob_scores, initial_prompt, total_num_questions,max_length, num_sequences, filename, num_fs_prompts):
        self.temperature = temperature
        self.total_num_questions = len(questions)
        self.zero_shot_gen_data = questions
        self.offensiveness_scores = np.asarray(clf_prob_scores)
        self.total_num_questions = total_num_questions
        self.num_questions = 0
        self.initial_prompt = initial_prompt
        self.questions = []
        self.max_length = max_length
        self.num_sequences = num_sequences
        self.filename = filename
        self.num_fs_prompts = num_fs_prompts
        open(filename, 'w').close()
    
    def softmax_scores(self):
        softmax = torch.nn.Softmax(dim=0)
        scores = torch.tensor(self.offensiveness_scores / self.temperature).float()
        return softmax(scores).detach().cpu().numpy()

    def sample(self):
        prob_scores = self.softmax_scores()
        samples = np.random.choice(self.zero_shot_gen_data, size=self.num_fs_prompts,
                                   replace=False, p=prob_scores)
        return samples.tolist()
    
    def get_prompt(self):
        questions = self.sample()
        prompt = self.initial_prompt
        for i, question in enumerate(questions):
            prompt += ' ' + question
            prompt += '\n{}.'.format(i + 2)
        return prompt

    def process_questions(self, prompt, sequences):
        pattern = re.compile(r'^[1-9]\..+?\?')
        questions = []
        for sequence in sequences:
            generated_text = sequence['generated_text']
            generated_text = generated_text.split(prompt)[1]
            texts = generated_text.split('\n')[1:]
            # import pdb; pdb.set_trace()
            for text in texts:
                if pattern.fullmatch(text):
                    question = re.sub(r'^[1-9]\.\s', '', text)
                    questions.append(question)
        return questions

    def generate_test_cases(self, model_name=None):
        if model_name:
            generator = pipeline('text-generation', model=model_name,
                                max_length=self.max_length,
                                num_return_sequences=self.num_sequences)
        else:
            generator = pipeline('text-generation',max_length=self.max_length,
                                num_return_sequences=self.num_sequences)

        # import pdb; pdb.set_trace()
        while self.num_questions < self.total_num_questions:
            prompt = self.get_prompt()
            sequences = generator(prompt)
            questions = self.process_questions(prompt, sequences)
            self.questions += questions
            self.save_to_file(questions)
            self.num_questions += len(questions)

        return self.questions
    
    def save_to_file(self, questions):
        with open(self.filename, 'a') as output:
            for question in questions:
                output.write(question + '\n')
        return

    def save_model(self, path):
        if self.generator:
            self.generator.save_pretrained(path)