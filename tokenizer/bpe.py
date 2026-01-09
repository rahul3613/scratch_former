from collections import Counter
import json
import regex as re
import time
from tqdm import tqdm


PATTERN = re.compile(
    r"\n+"
    r"| 's|'t|'re|'ve|'m|'ll|'d"
    r"| ?(?:\p{L}\p{M}*)+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{M}\p{N}]+"
    r"|[ \t]+(?!\S)"
    r"|[ \t]+"
)

    
vocab_dict = {}
    
 
print("Loading Data:")
with open("corpus.txt", "r") as f:
    text = f.read()

text = text.lower()
parts = PATTERN.findall(text)

for part in parts:
    try:
        vocab_dict[part]["freq"] += 1
    except KeyError:
        vocab_dict[part] = {"freq":1, "token_ids": list(part.encode('utf-8'))}
            
        
print("Total Words", len(vocab_dict))



def update_list(token_ids, x, y, curr_vocab_length):
    new_list = []
    i = 0
    
    while i < len(token_ids):
        if token_ids[i:i+2] == [x,y]:
            new_list.append(curr_vocab_length)
            i += 2
        else:
            new_list.append(token_ids[i])
            i += 1
            
    return new_list



vocab_length = 25000
curr_vocab_length = 255
merges = []
o_time = time.time()
avg_time = 0

freq_counter = Counter()
pair_words_dict = {}

for word, value in tqdm(vocab_dict.items()):
    token_ids = value["token_ids"]
    list_len = len(token_ids)
    
    if list_len > 1:
        for i in range(list_len-1):
            token_key = (token_ids[i], token_ids[i+1])
            freq_counter[token_key] += value["freq"]
            
            v = pair_words_dict.get(token_key)
            if v is None:
                pair_words_dict[token_key] = set([word])
            else:
                v.add(word)
                

while curr_vocab_length < vocab_length:
                                
    (x,y), freq = freq_counter.most_common(1)[0]

    curr_vocab_length += 1

    n_time = time.time()
    avg_time = 0.9 * avg_time + 0.1 * (n_time - o_time)
    print("x:", x, "  y:", y, "  z:", curr_vocab_length, "  freq:", freq, "  time:", round(avg_time, 2), "  eta:", round(avg_time * (vocab_length - curr_vocab_length), 0))
    o_time = n_time
    

    words = pair_words_dict[(x, y)]
    for word in words:
        voc_val = vocab_dict[word]
        voc_tok_ids = voc_val["token_ids"]
        
        new_token_ids = update_list(voc_tok_ids, x, y, curr_vocab_length)
        voc_val["token_ids"] = new_token_ids
        
        if len(new_token_ids) > 1:
            for i in range(0, len(new_token_ids)):
                if new_token_ids[i] == curr_vocab_length:
                    if i > 0:
                        w = new_token_ids[i-1]
                        freq_counter[(w, x)] -= voc_val["freq"]
                        
                        token_key = (w, curr_vocab_length)
                        freq_counter[token_key] += voc_val["freq"]
                        
                        v = pair_words_dict.get(token_key)
                        if v is None:
                            pair_words_dict[token_key] = set([word])
                        else:
                            v.add(word)
                        
                        
                    if i < len(new_token_ids) - 1:
                        z = new_token_ids[i+1]
                        freq_counter[(y, z)] -= voc_val["freq"]
                                           
                        if curr_vocab_length != z:     
                            token_key = (curr_vocab_length, z)
                            freq_counter[token_key] += voc_val["freq"]
                            
                            v = pair_words_dict.get(token_key)
                            if v is None:
                                pair_words_dict[token_key] = set([word])
                            else:
                                v.add(word)
        

    del freq_counter[(x, y)]
    del pair_words_dict[(x, y)]
    

    merges.append({"x": x, "y": y, "z": curr_vocab_length})

with open("merges.json", "w") as f:
    json.dump(merges, f)
