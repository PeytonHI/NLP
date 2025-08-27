from collections import Counter

def load_conllu(file):
    data = []
    with open(file) as fin:
        current = []
        for line in fin:
            arr = line.strip('\n').split('\t')
            if arr[0] == "":  # blank line
                if current != []:
                    data.append(current)
                    current = []
                continue
            
            d = {
                "id": arr[0],
                "form": arr[1],
                "lemma": arr[2],
                "upos": arr[3],
                "xpos": arr[4],
                "feats": arr[5],
                "head": arr[6],
                "deprel": arr[7],
                "deps": arr[8],
                "misc": arr[9]
            }
            current.append(d)
        
        if current != []:
            data.append(current)
    return data


class HMMClassifier():
    def __init__(self):
        self.tags = Counter()  # Count occurrences of each tag
        self.words = Counter()  # Count occurrences of each word
        self.transitions = Counter()  # Count tag bigrams (t_(i-1), t_i)
        self.emissions = Counter()  # Count (word, tag) pairs
        self.initial_tags = Counter()  # Count initial tags
        self.trained = False
    
    
    def train(self, data):
        for sentence in data:
            for i, token in enumerate(sentence):
                tag = token['upos']
                word = token['form']

                # Count tags and words
                self.tags[tag] += 1
                self.words[word] += 1
                self.emissions[(word, tag)] += 1

                if i == 0:  # First word in the sentence
                    self.initial_tags[tag] += 1
                else:  # Transition probabilities
                    prev_tag = sentence[i - 1]['upos']
                    self.transitions[(tag, prev_tag)] += 1


        self.trained = True

    def transition(self, t_i, t_iminus1):
        return self.transitions[(t_iminus1, t_i)] / self.tags[t_iminus1] if self.tags[t_iminus1] > 0 else 0


    def emission(self, w_i, t_i):
        
        return self.emissions[(w_i, t_i)] / self.tags[t_i] if self.tags[t_i] > 0 else 0

    def initial(self, t):
        total_sentences = sum(self.initial_tags.values())
        return self.initial_tags[t] / total_sentences if total_sentences > 0 else 0

    
    def predict(self, sentence):
        if not self.trained:
            raise Exception("must train first!")

        viterbi = [{}]
        backpointers = [{}]

        # Initialization step
        for t in self.tags:
            viterbi[0][t] = self.initial(t) * self.emission(sentence[0], t)
            backpointers[0][t] = None

        # Recursive step
        for i in range(1, len(sentence)):
            viterbi.append({})
            backpointers.append({})
            for t in self.tags:
                max_prob, best_prev = max(
                    (viterbi[i - 1][prev_t] * self.transition(t, prev_t) * self.emission(sentence[i], t), prev_t)
                    for prev_t in self.tags
                )
                viterbi[i][t] = max_prob
                backpointers[i][t] = best_prev

        # Termination step
        best_final_tag = max(viterbi[-1], key=viterbi[-1].get)
        best_parse = []
        for i in range(len(sentence) - 1, -1, -1):
            best_parse.insert(0, best_final_tag)
            best_final_tag = backpointers[i][best_final_tag]

        return viterbi, backpointers, best_parse
    
    
def main():
    data = load_conllu(r"C:\Users\peyto\Desktop\school24\497\hw4\example.conllu")
    print("Data:", data)

    model = HMMClassifier()
    model.train(data)

    print("Transition Probabilities")
    for t_iminus1 in sorted(model.tags):
        for t_i in sorted(model.tags):
            print(f"p({t_i} | {t_iminus1}) = {model.transition(t_i, t_iminus1)}")
    
    print("Emission Probabilities")
    for t in sorted(model.tags):
        for w in sorted(model.words):
            print(f"p({w} | {t}) = {model.emission(w, t)}")

    viterbi, backpointers, best_parse = model.predict("the old man".split())
    print(best_parse)


if __name__ == "__main__":
    main()