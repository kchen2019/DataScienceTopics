text = "Just for the record darling, not all positive change feels positive in the beginning. Forgiveness is not an occasional act, it is a constant attitude. Nothing in the world is more dangerous than sincere ignorance and conscientious stupidity. We must accept finite disappointment, but never lose infinite hope. Sometimes, you face difficulties not because you are doing something wrong, but because you are doing something right. Strength does not come from winning. Your struggles develop your strengths. When you go through hardships and decide not to surrender, that is strength. Whenever you find yourself doubting how far you can go, just remember how far you have come. At any given moment you have the power to say: This is not how the story is going to end."

class MarkovChain:

    def __init__(self):
        self.memory = {}

    def _learn_key(self, key, value):
        if key not in self.memory:
            self.memory[key] = []

        self.memory[key].append(value)

    def learn(self, text):
        tokens = text.split(" ")
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens) - 1)]
        for bigram in bigrams:
            self._learn_key(bigram[0], bigram[1])


m = MarkovChain()

m.learn(text)

wordmap = m.memory

from collections import Counter

wordprob={}
for key in wordmap.keys():
    c = Counter(wordmap[key])
    s = sum(c.values())
    wordprob[key]={k: round(v/s,2) for k, v in c.items()}

import random
def random_sample(prob_dict):
    cuts = [0]
    for i in range(len(prob_dict)):
        key = list(prob_dict.keys())[i]
        prob = prob_dict[key]
        cuts_add = prob + cuts[i]
        cuts.append(cuts_add)
    cuts.append(1)
    sample = random.uniform(0,1)
    idx = cuts.index([x for x in cuts if x >= sample][0])
    return(list(prob_dict.keys())[idx-1])

def create_sentence(start_word, max_length):
    sentence = [start_word]
    while len(sentence) <= max_length:
        next_word = random_sample(wordprob[start_word])
        sentence.append(next_word)
        start_word = next_word
    return(' '.join(sentence))