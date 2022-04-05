from zero_shot import ZeroShot

def train(zs=False, fs=False, supervised=False, rl=False):
    questions = None
    if zs:
        zero_shot = ZeroShot(total_num_questions=500000, max_length=100,
                             num_sequences=5000, filename='sample.txt')
        questions = zero_shot.generate_test_cases()
    if fs:
        # TODO: Do Few Shot Learning
        pass
    if supervised:
        # TODO: Do supervised
        pass
    if rl:
        # TODO: Do RL
        pass

if __name__ == '__main__':
    train(True)