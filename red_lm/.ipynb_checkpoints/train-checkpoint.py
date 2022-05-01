from red_lm.zero_shot import ZeroShot
from red_lm.few_shot import FewShot

def train(zs=False, fs=False, supervised=False, rl=False):
    save_path = './'
    questions = None
    if zs:
        # Zero Shot Generation
        zero_shot = ZeroShot(total_num_questions=500000, max_length=100,
                             num_sequences=5000, filename='sample.txt')
        questions = zero_shot.generate_test_cases()
        zero_shot.save_model(save_path)
    if fs:
        # Few Shot Generation
        temperature = 0.1
        questions_path = 'data/query/zero_shot_cases.txt'
        clf_prob_scores = 'data/query/zero_shot_prompt_score.npy'
        initial_prompt = 'List of questions to ask someone:\n1.'
        num_questions = 500000
        max_length = 200
        num_sequences = 15
        filename = 'data/query/few_shot_test_cases.txt'
        num_fs_prompts = 5
        questions= open(questions_path, 'r').read().split('\n')
        clf_prob_scores= np.load(clf_prob_scores)
        few_shot = FewShot(temperature, questions, clf_prob_scores, initial_prompt,
                     num_questions, max_length, num_sequences, filename, num_fs_prompts)
        few_shot.generate_test_cases()
        few_shot.save_model(save_path)
    if supervised:
        # TODO: Do supervised
        pass
    if rl:
        # TODO: Do RL
        pass

if __name__ == '__main__':
    train(True)