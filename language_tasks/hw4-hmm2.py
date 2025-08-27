from collections import Counter

# Author: Peyton
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
        self.tags = Counter()
        self.words = Counter()
        self.tags = Counter()
        self.initial_tags = Counter()
        self.transitions = Counter()
        self.emissions = Counter ()
        # TODO: implement
        self.trained = False
    

    def train(self, data):
        # TODO: implement
        for sentence in data:
            for i, token in enumerate(sentence):
                # print(i, token)
                tag = token['upos'] 
                word = token['form']
                self.tags[tag] += 1
                self.words[word] += 1
                self.emissions[(word, tag)] +=1

                if i == 0:
                    self.initial_tags[tag] += 1
                else:
                    prev_tag = sentence[i - 1]['upos'] 
                    self.transitions[(tag, prev_tag)] += 1

        self.trained = True
    
    # transition formula: probability of transitioning from previous tag to current tag
    def transition(self, t_i, t_iminus1):
        return self.transitions[(t_iminus1, t_i)] / self.tags[t_iminus1] if self.tags[t_iminus1] > 0 else 0

        # TODO: implement
        pass

    # emission formula: probability of current tag emitting the word
    def emission(self, w_i, t_i):
        return self.emissions[(t_i, w_i)] / self.tags[t_i] if self.tags[t_i] > 0 else 0
        # TODO: implement
        pass

    # probability of starting sentence with a tag
    def initial(self, t):
        total_sents = sum(self.initial_tags.values())
        return self.initial_tags[t] / total_sents if total_sents > 0 else 0
        # TODO: implement
        pass
    
    # best parse path: most probable sequence of tags 
    def predict(self, sentence):
        if not self.trained:
            raise Exception("must train first!")

        viterbi = [{}]
        backpointers = [{}]

        # initial probabilities: probability of starting with tag for each tag for given sentence
        for t in self.tags:
            viterbi[0][t] = self.initial(t) * self.emission(sentence[0], t)
            backpointers[0][t] = None

        # 2nd index to end, iterates through sentences and creates new viterbis and bps 
        for i in range(1, len(sentence)):
            viterbi.append({})
            backpointers.append({})
            for t in self.tags: # loop through each possible tag current word
                # highest prob of reaching current tag t based on prev tags: prob prev tag * prob transition from prev tag to current tag * prob current word emitting current tag
                max_prob, best_prev = max(
                    (viterbi[i - 1][prev_t] * self.transition(t, prev_t) * self.emission(sentence[i], t), prev_t)
                    for prev_t in self.tags
                )
                viterbi[i][t] = max_prob # store max prob for current tag at position i
                backpointers[i][t] = best_prev # store best prev tag that leads to max prob

        # final step
        best_final_tag = max(viterbi[-1], key=viterbi[-1].get) # highest prob last position: best tag for last word
        best_parse = []
        # last word to first, insert best tag at beginning of best parse, update best tag to prev tag from bp. 
        for i in range(len(sentence) - 1, -1, -1):
            best_parse.insert(0, best_final_tag)
            best_final_tag = backpointers[i][best_final_tag]

        return viterbi, backpointers, best_parse
        
        
def main():
    data = load_conllu(r"C:\Users\peyto\Desktop\school24\497\hw4\example.conllu")
    # print(data)

    model = HMMClassifier()
    model.train(data)

    print("Transition Probabilities")
    for t_iminus1 in sorted(model.tags):
        print(sorted(model.tags))
        print(t_iminus1)
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