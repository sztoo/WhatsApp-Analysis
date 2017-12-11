import os
import re
import time
import enchant # spellchecking library 
from nltk.corpus import stopwords 
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist

d = enchant.Dict("en_US") # english 

def read_data():
    file_path = ''.join([os.path.dirname(os.path.abspath(__file__)), '/data/_chat.txt'])
    
    try: 
        f = open(file_path, 'r')
        raw_data = list(f)
        data_size = len(raw_data)
        print("data size: {} phrases".format(data_size))
        data = []
        
        for line in raw_data: 
            data.extend(re.findall("(?:.*?\:){3}(?:\s)(.*?.*)", line))
    finally: 
        f.close()
    return data
    
def tokenize(data): 
    user_dict = dict()
    tokens = dict()
    to_skip = ["Messages to this chat and calls are now secured with end-to-end encryption.", "‎Missed Voice Call", "Missed Video Call"]
    names = []
    try:
        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r"\w+")
        for line in data[:]:
            if any(skip in line for skip in to_skip):
                continue
            name = re.findall("(.*?)(?:\:)", line.lower())[0]
            names.append(name)
            sentence = re.findall("(?:.*?\:\s)(.*?.*)", line.lower())[0]
            words = [w for w in list(tokenizer.tokenize(sentence)) if w not in stop_words] # check for stop words
            words = [w for w in words if d.check(w)==True] # further filtering with enchant library
            user_dict.setdefault(name, []).append(sentence)
            tokens.setdefault(name, []).extend(words)
    except IndexError: 
        print("error: IndexError")
    return(user_dict, names, tokens)

def sentiment_analysis(names, tokens):
    num = 3 # top 50 common words used
    user1_top = FreqDist(tokens[names[0]]).most_common(num)
    user2_top = FreqDist(tokens[names[1]]).most_common(num)
    users_top = [user1_top, user2_top]
    user1, user2 = [], []
    users = [user1, user2]
    checks = ['a','n','v'] # adjective, noun, verb
    for idx, user in enumerate(users_top):
        obj_list, pos_list, neg_list = [], [], []
        occurence = [] # total occurences of words
        for tup in user:
            obj, pos, neg = [], [], []
            occurence.append(tup[1])
            for check in checks:
                word = tup[0]
                score = list(swn.senti_synsets(word, check))
                if len(score) != 0: # if word is found 
                    obj.append(score[0].obj_score()) 
                    pos.append(score[0].pos_score()) 
                    neg.append(score[0].neg_score()) 
                else: 
                    occurence[:len(occurence)-1]
            pos_list.append((sum(pos)/len(checks))*tup[1]) # weighted value = mean * freq 
            neg_list.append((sum(neg)/len(checks))*tup[1])
            obj_list.append((sum(obj)/len(checks))*tup[1])
        weighted_pos = sum(pos_list)/sum(occurence)
        weighted_neg = sum(neg_list)/sum(occurence)
        weighted_obj = sum(obj_list)/sum(occurence)
        users[idx].extend([weighted_obj, weighted_pos, weighted_neg])
    return(num, users_top, users)

def output(user_dict, names, tokens, num, users_top, users):
    idx = 0
    for user in user_dict:
        print("="*15)
        print("\n{}: \n{} phrases\n{} tokens\n\ntop {} words:".format(user, len(user_dict[user]), len(tokens[user]), num))
        print(users_top[idx])
        print("obj_score: {}\npos_score: {}\nneg_score: {}\n".format(users[idx][0], users[idx][1], users[idx][2]))
        idx += 1

def main(): 
    data = read_data()
    user_dict, names, tokens = tokenize(data)
    num, users_top, users = sentiment_analysis(names, tokens)
    output(user_dict, names, tokens, num, users_top, users)

if __name__ == "__main__":
    main()