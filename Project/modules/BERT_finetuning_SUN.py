''"""
Created by: Claire Z. Sun
Date: 2021.01.17
''"""

### References:
# Documentation:    https://huggingface.co/transformers/model_doc/bert.html
# BERTweet:         https://github.com/VinAIResearch/BERTweet
# Tutorial:         https://colab.research.google.com/drive/1PHv-IRLPCtv7oTcIGbsgZHqrB5LPvB7S#scrollTo=9aHyGuTFgyPO
# Dataloader:       https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f
# Class imbalance:  https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification
# Stopping early:   https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Create Dataset classes for training set and raw tweets

class TwitterTrainDataset(torch.utils.data.Dataset):
    # Labelled training data: label(int) = {0,1,2} --> sentiment-label-lookup = {'negative':0, 'neutral': 1, 'positive':2}
    def __init__(self, text, label, tokenizer, max_len):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        label = self.label[item]
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False,truncation=True, padding='max_length', return_attention_mask=True,return_tensors='pt')
        #encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False,padding='max_length', return_attention_mask=True, return_tensors='pt')
        return {'text': text,
                'label': torch.tensor(label, dtype=torch.long),
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()}


class TwitterRawDataset(torch.utils.data.Dataset):
    # Unlabeled raw tweets
    def __init__(self, text, tokenizer, max_len):
        self.text = text
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, truncation=True, padding='max_length', return_attention_mask=True,return_tensors='pt')

        return {'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()}



def create_dataloader(text, label, training: bool, tokenizer, max_len, batch_size):
    if training:
        dataset = TwitterTrainDataset(text, label, tokenizer,max_len)
    else:
        dataset = TwitterRawDataset(text, tokenizer, max_len)

    return torch.utils.data.DataLoader(dataset, batch_size, num_workers=0)




# Fine-tuning BERT models

class SentimentClassifier_BERT(torch.nn.Module):
    # Use a dropout layer for some regularization and a fully-connected layer for output.
    # Note returning raw output of the last layer for the cross-entropy loss function in PyTorch to work

    def __init__(self, bert, n_classes):
        super(SentimentClassifier_BERT, self).__init__()
        self.bert = bert
        # self.drop = torch.nn.Dropout(p=0.2)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        # output = self.drop(pooled_output)
        # return self.out(output)
        return self.out(pooled_output)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        label = d["label"].to(device)

        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, label)

        correct_predictions += torch.sum(preds == label)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            label = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, label)

            correct_predictions += torch.sum(preds == label)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)



def predict_train(model, data_loader_train, device):
    model = model.eval()

    tweets = []
    predictions = []
    prediction_probs = []
    labels = []

    with torch.no_grad():
        for d in data_loader_train:
            texts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            label = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            probs = torch.nn.functional.softmax(outputs, dim=1)

            tweets.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            labels.extend(label)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    labels = torch.stack(labels).cpu()
    return tweets, predictions, prediction_probs, labels



def infer_raw(model, data_loader_raw, device):
    model = model.eval().to(device)
    predictions = []
    # prediction_probs = []

    with torch.no_grad():
        for d in data_loader_raw:
            texts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            # probs = torch.nn.functional.softmax(outputs, dim=1)

            predictions.extend(preds)
            # prediction_probs.extend(probs)

    predictions = torch.stack(predictions).cpu()
    # prediction_probs = torch.stack(prediction_probs).cpu()

    return predictions.detach().numpy().astype('int32')



def show_prediction_performance(y_label, y_pred):
    sentiments = ['negative', 'neutral', 'positive']
    cm = confusion_matrix(y_label, y_pred)
    # df_cm = pd.DataFrame(cm, index=None, columns=None)
    hmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(sentiments, rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(sentiments, rotation=0, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.show()
    print(classification_report(y_label, y_pred, target_names=sentiments))


def bert_base(df_text,pretrained_bert,tokenizer_bert,finetuned_save,device,batch_size):
    model = SentimentClassifier_BERT(pretrained_bert,3)
    model.load_state_dict(torch.load(finetuned_save))
    data_loader_raw = create_dataloader(df_text, None, False, tokenizer_bert, 100, batch_size)
    pred_labels = infer_raw(model, data_loader_raw, device)
    return pred_labels

