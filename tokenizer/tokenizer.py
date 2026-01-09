import json

with open("tokenizer/merges.json", "r") as f:
    merges = json.load(f)


def encode(text, merges=merges):
    token_ids = list(text.lower().encode("utf-8"))
    
    for merge in merges:
        x, y, z = merge["x"], merge["y"], merge["z"]
        
        new_token_ids = []
        i = 0
        while i < len(token_ids):
            if token_ids[i:i+2] == [x,y]:
                new_token_ids.append(z)
                i += 2
            else:
                new_token_ids.append(token_ids[i])
                i += 1
        
        token_ids = new_token_ids
    return token_ids
    
    
def decode(token_ids, merges=merges):
    for merge in reversed(merges):
        x, y, z = merge["x"], merge["y"], merge["z"]
        
        new_token_ids = []
        for token in token_ids:
            if token == z:
                new_token_ids.extend([x, y])
            else:
                new_token_ids.append(token)
        
        token_ids = new_token_ids
    return bytes(token_ids).decode()
    