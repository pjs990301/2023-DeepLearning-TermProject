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
import wandb

#############################################    -> 실험결과 FIX
random_seed = 4
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
##################################

Model_list = [
    "lassl/bert-ko-small",
    "lassl/roberta-ko-small",
    "tunib/electra-ko-small",
    "tunib/electra-ko-en-small",
    "monologg/koelectra-small-v3-discriminator",
    "beomi/KcELECTRA-base-v2022",
    "jason9693/soongsil-bert-small",
    "monologg/kocharelectra-small-discriminator",
    "monologg/kocharelectra-base-discriminator",
    "beomi/kcbert-base",
    "beomi/KcELECTRA-base",
    "tunib/electra-ko-base",
    "tunib/electra-ko-en-base",
    "kykim/bert-kor-base",
    "kykim/funnel-kor-base",
    "kykim/electra-kor-base",
    "lassl/bert-ko-base",
    "monologg/kobigbird-bert-base",
    "hyunwoongko/brainsbert-base",
    "jinmang2/kpfbert",
    "beomi/kcbert-large",
    "klue/roberta-large",
]

for model_name in Model_list :
    # model_name = "kykim/electra-kor-base"
    print(model_name)
    task = "COPA"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    ).cuda()

    parser = ArgumentParser()
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
    args, _ = parser.parse_known_args()

    wandb.init(project="DL_COPA", name=f"{task}-{model_name}-{random_seed}", entity="deeplearningcopa")

    train_data = pd.read_csv("./COPA/SKT_COPA_Train.tsv", delimiter="\t")
    train_text, train_question, train_1, train_2, train_labels = (
        train_data["sentence"].values,
        train_data["question"].values,
        train_data["1"].values,
        train_data["2"].values,
        train_data["Answer"].values -1,
    )

    dataset = [
        {"data": t + args.sep_token + q + args.sep_token + f + args.sep_token + s, "label": l}
        for t, q, f, s, l in zip(train_text, train_question, train_1, train_2, train_labels)
    ]
    train_loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    eval_data = pd.read_csv("./COPA/SKT_COPA_Dev.tsv", delimiter="\t")
    eval_text, eval_question, eval_1, eval_2, eval_labels = (
        eval_data["sentence"].values,
        eval_data["question"].values,
        eval_data["1"].values,
        eval_data["2"].values,
        eval_data["Answer"].values -1,
    )

    dataset = [
        {"data": t + args.sep_token + q + args.sep_token + f + args.sep_token + s, "label": l}
        for t, q, f, s, l in zip(eval_text, eval_question, eval_1, eval_2, eval_labels)
    ]
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    optimizer = AdamW(params=model.parameters(), lr=1e-5, weight_decay=3e-7)


    for epoch in range(args.epoch):
        model.train()
        train_acc = 0
        for train in tqdm(train_loader):
            optimizer.zero_grad()
            text, label = train["data"], train["label"].cuda()
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )

            input_ids = tokens.input_ids.cuda()
            attention_mask = tokens.attention_mask.cuda()
            output = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label,
            )

            loss = output.loss
            loss.backward()
            optimizer.step()
            classification_results = output.logits.argmax(-1)

            acc = 0
            for res, lab in zip(classification_results, label):
                if res == lab:
                    acc += 1
            train_acc += acc

        wandb.log({"loss": loss})
        wandb.log({"acc": train_acc / len(train_data)})
        print({"loss": loss})
        print({"acc": train_acc / len(train_data)})

        with torch.no_grad():
            model.eval()
            eval_acc = 0
            for eval in tqdm(eval_loader):
                eval_text, eval_label = eval["data"], eval["label"].cuda()
                eval_tokens = tokenizer(
                    eval_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                )
                input_ids = eval_tokens.input_ids.cuda()
                attention_mask = eval_tokens.attention_mask.cuda()

                eval_out = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=eval_label
                )

                eval_classification_results = eval_out.logits.argmax(-1)
                eval_loss = eval_out.loss

                acc = 0
                for res, lab in zip(eval_classification_results, eval_label):
                    if res == lab:
                        acc += 1
                eval_acc += acc

            wandb.log({"eval_loss": eval_loss})
            wandb.log({"eval_acc": eval_acc / len(eval_data)})
            wandb.log({"epoch": epoch + 1})
            print({"eval_loss": eval_loss})
            print({"eval_acc": eval_acc / len(eval_data)})
            print({"epoch": epoch + 1})

            # 모델 저장하기 편하도록 torch.save 디렉토리 변경
            if not os.path.exists(f"model_save/{model_name.replace('/', '-')}"):
                os.makedirs(f"model_save/{model_name.replace('/', '-')}")

            torch.save(model.state_dict(),f"model_save/{model_name.replace('/', '-')}/{model_name.replace('/', '-')}-{task}-{epoch}.pt")

    wandb.finish()