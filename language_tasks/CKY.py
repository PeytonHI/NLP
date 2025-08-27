# Author: jaden, Peyton

# 
def cky_parser(sentence, rules):
    sen_len = len(sentence)
    table = [[set() for _ in range(sen_len)] for _ in range(sen_len)]

    # fill terminals
    for j in range(sen_len):
        word = sentence[j]
        if word in rules:  # check if word exists in rules
            for rule in rules[word]:
                table[j][j].add(rule)

        # fill non-terminals
        for i in range(j - 1, -1, -1): # from end, down to 0
            for k in range(i, j): # from end to 0
                for left_rule in table[i][k]: 
                    for right_rule in table[k + 1][j]:
                        if (left_rule, right_rule) in rules:
                            # for rule in rules[(left_rule, right_rule)]:
                                table[i][j].add(rules[left_rule, right_rule]) 

    return table

# grammar rules
rules = {
    # Non-terminal rules
    ('S',): {('VP', 'NP'), ('NP', 'VP')},

    ('VP',): {
        ('he', 'N'),  # VP -> he N
        ('he', 'NP'),  # VP -> he NP
    },

    ('NP',): {
        ('PRON',),  # NP -> PRON
        ('Det',),   # NP -> Det
        ('ADJ', 'N')  # NP -> ADJ N
    },

    # Terminal rules (words)
    'he': {('AUX',)},  # Terminal: 'he' -> AUX
    'wahine': {('NOUN',)},  # Terminal: 'wahine' -> NOUN
    'ʻoe': {('PRON',)},  # Terminal: 'ʻoe' -> PRON
    'oluolu': {('ADJ',)},  # ADJ -> oluolu
    'maikai': {('ADJ',)},  # ADJ -> maikai
    'ono': {('ADJ',)},  # ADJ -> ono
    'hou': {('ADJ',)},  # ADJ -> hou
    'nui': {('ADJ',)},  # ADJ -> nui
    'haole': {('N',)},  # N -> haole
    'alii': {('N',)},  # N -> alii
    'haumana': {('N',)},  # N -> haumana
    'Hawaii': {('N',)},  # N -> Hawaii
    'noho': {('N',)},  # N -> noho
    'pepa': {('N',)},  # N -> pepa
    'kanaka': {('N',)},  # N -> kanaka
    'aina': {('N',)},  # N -> aina
    'olelo': {('N',)},  # N -> olelo
    'puke': {('N',)},  # N -> puke
    'hula': {('N',)},  # N -> hula
    'ai': {('N',)},  # N -> ai
    'kuleana': {('N',)},  # N -> kuleana
    'kela': {('Det',)},  # Det -> kela
    'kena': {('Det',)},  # Det -> kena
    'keia': {('Det',)},  # Det -> keia
    'au': {('PRON',)},  # PRON -> au
    'oe': {('PRON',)},  # PRON -> oe
}

# S -> VP NP = NP VP
# VP -> he N = are a N
# VP -> he N = am a N
# VP -> he N = is a N
# VP -> he NP = are a NP
# VP -> he NP = is a NP
# VP -> he NP = is NP
# NP -> PRON = PRON
# NP -> Det = Det
# NP -> ADJ N = ADJ N
# ADJ -> oluolu = cool/comfortable
# ADJ -> maikai = good
# ADJ -> ono = delicious
# ADJ -> hou = new
# ADJ -> nui = big
# N -> haole = haole
# N -> alii = chief
# N -> wahine = woman
# N -> haumana = student
# N -> Hawaii = Hawaiian
# N -> noho = chair
# N -> pepa = paper
# N -> kanaka = person
# N -> aina = land
# N -> olelo = speech/language
# N -> puke = book
# N -> hula = hula
# N -> ai = food
# N -> kuleana = kuleana
# Det -> kela = this
# Det -> kela = that
# Det -> kena = that (near)
# Det -> keia = this
# PRON -> au = I
# PRON -> oe = you

# parse sents from file
with open(r'C:\Users\peyto\Desktop\school24\497\hw4\elbert.conllu', 'r', encoding='UTF-8') as f:
    sentence = []
    for line in f:
        if line.startswith('#') or line.strip() == '': # skip empty/comment lines
            continue
        sects = line.strip().split('\t')
        if len(sects) > 1: # get word
            word = sects[1]
            sentence.append(word)

    # parse sentence
    table = cky_parser(sentence, rules)

    # print table
    for i in range(len(sentence)):
        for j in range(len(sentence)):
            if table[i][j]:
                print(f"parse_table[{i}][{j}]: {table[i][j]}")
