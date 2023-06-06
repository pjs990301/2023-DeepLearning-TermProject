from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import streamlit as st
import itertools
import time

st.title('2학2석')

#############################################    -> 실험결과 FIX
random_seed = 4
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
##################################

model_name = 'kykim/electra-kor-base'
task = "COPA"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
).cuda()

model.load_state_dict(torch.load('kykim-electra-kor-base-COPA-9.pt'))

eval_data = pd.read_csv("./COPA/SKT_COPA_Dev.tsv", delimiter="\t")
eval_text, eval_question, eval_1, eval_2, eval_labels = (
    eval_data["sentence"].values,
    eval_data["question"].values,
    eval_data["1"].values,
    eval_data["2"].values,
    eval_data["Answer"].values - 1,
)

dataset = [
    {"data": t + args.sep_token + q + args.sep_token + f + args.sep_token + s, "label": l}
    for t, q, f, s, l in zip(eval_text, eval_question, eval_1, eval_2, eval_labels)
]

eval_loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
    shuffle=True
)


def load_data():

    random_number = random.randint(1, 500)
    subset_loder = itertools.islice(eval_loader, random_number, random_number + 1)
    # subset_loder = itertools.islice(eval_loader, 0, 1)

    for idx, eval in enumerate(subset_loder):
        eval_text, eval_label = eval["data"], eval["label"].cuda()
        sentences = eval_text[0].split("[SEP]")

        eval_tokens = tokenizer(
                    eval_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                )

    return eval_text, eval_label, sentences, eval_tokens


def predict(eval_text, eval_label, sentences, eval_tokens) :

    with torch.no_grad():
        model.eval()

        input_ids = eval_tokens.input_ids.cuda()
        attention_mask = eval_tokens.attention_mask.cuda()
        eval_out = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=eval_label
                )

        eval_classification_results = eval_out.logits.argmax(-1)

    return eval_classification_results


eval_text, eval_label, sentences, eval_tokens = load_data()

st.title('2학2석')
# st.write(sentences)
# st.write(eval_classification_results)

if sentences[1] == "원인" :
    st.header("주어진 문장의 원인을 찾으세요!")
    st.subheader(sentences[0])

elif sentences[1] == "결과" :
    st.header("주어진 문장의 결과 찾으세요!")
    st.subheader(sentences[0])

col1, col2 = st.columns(2)
with col1:
    st.header("문장 1")
    st.write(sentences[2])

with col2:
    st.header("문장 2")
    st.write(sentences[3])

seletion = st.selectbox("정답을 선택해주세요 ❗",(sentences[2], sentences[3]))

result = None
seletion_n = None

if seletion == sentences[2] :
    seletion_n = 0
else :
    seletion_n = 1


if st.button("선택완료") :
    # st.write(seletion)
    result = predict(eval_text, eval_label, sentences, eval_tokens)
    # st.write(result)

    if seletion_n == result :
        st.success("Correct", icon ="🎉")
        st.balloons()
    else :
        st.error("Not Correct", icon="😥")

    if seletion_n == 0:
        # Code for selection 0
        pass
    else:
        # Code for selection 1
        pass

       



            
            