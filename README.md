# WhatsApp-Analysis

### Goal
This project is to study the sentiment analysis between two users on WhatsApp based on their vocabularies to better understand their conversational experience.

### Install Requirements 
``` 
$ pip install -r requirements.txt
$ python -m nltk.downloader all
```

### Run 
```
$ python main.py
```

Currently, one would need to export the chats `_chat.txt` from WhatsApp manually. Place it under `/data/_chat.txt`.

### Results 
```
data size: 27525 phrases
===============

user 1:
12574 phrases
29428 tokens

top 3 words:
[('like', 581), ('go', 476), ('want', 427)]
obj_score: 0.7641509433962265
pos_score: 0.08333333333333333
neg_score: 0.05660377358490566

===============

user 2:
14784 phrases
22825 tokens

top 3 words:
[('like', 552), ('want', 454), ('go', 373)]
obj_score: 0.7461324631375391
pos_score: 0.08333333333333333
neg_score: 0.06079284505680444
```
