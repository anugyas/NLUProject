import random
import numpy as np
from transformers import pipeline


class FewShot():
    def __init__(self, temperature, questions, clf_prob_scores, initial_prompt,
                 num_questions, max_length, num_sequences, filename, k):
        self.temperature = temperature
        self.total_num_questions = len(questions)
        self.zero_shot_gen_data = questions
        self.offensiveness_scores = np.asarray(clf_prob_scores)
        self.adjusted_scores = self.offensiveness_scores / self.temperature
        self.num_questions = num_questions
        self.initial_prompt = initial_prompt
        self.questions = []
        self.max_length = max_length
        self.num_sequences = num_sequences
        self.filename = filename
        self.k = k
        open(filename, 'w').close()
    
    def sample(self):
        samples = np.random.choice(self.zero_shot_gen_data, size=self.k,
                                   replace=False, p=self.adjusted_scores)
        return samples.tolist()
    
    def get_prompt(self):
        questions = self.sample()
        prompt = self.initial_prompt
        for i, question in enumerate(questions):
            prompt += ' ' + question
            prompt += '\n{}.'.format(i + 2)
        prompt += '\n'
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
            generator = pipeline('text-generation', max_length=self.max_length,
                                num_return_sequences=self.num_sequences)

        while self.num_questions < self.total_num_questions:
            prompt = self.get_prompt()
            sequences = generator(prompt)
            questions = process_questions(prompt, sequences)
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