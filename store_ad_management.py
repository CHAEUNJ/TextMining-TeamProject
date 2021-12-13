import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp
from torch.utils.data import Dataset

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm, tqdm_notebook
from tqdm import tqdm
from tqdm import trange


class BERTDataset(Dataset):
    # sentences 의 index와 label index 매개변수 제거
    def __init__(self, dataset, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        # sentences 리스트와 label 리스트를 만드는 코드 변경
        self.sentences = [transform([i]) for i in dataset['contents']]
        self.labels = list(dataset['label'])

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,  # Dimensionality of the encoder layers and the pooler layer
                 num_classes=3,  # 분류할 class label의 개수
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

"""
class StoreAdManagement:
    def __init__(self, post_list):
        self.post_list = post_list

        self.model = torch.load('./model3.pt', map_location=torch.device('cpu'))
        self.device = torch.device('cpu')

        bertmodel, vocab = get_pytorch_kobert_model()

        tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

        self.max_len = 128 # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
        self.batch_size = 16
        self.warmup_ratio = 0.1
        self.num_epochs = 5
        self.max_grad_norm = 1
        self.log_interval = 200
        self.learning_rate = 5e-5

    def calc_accuracy(self, X, Y):
        max_vals, max_indices = torch.max(X, 1)
        train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
        return train_acc

    def checkAd(self, lst):
      _data = self.formatter(lst)
      _dataset = BERTDataset(_data, self.tok, self.max_len, True, False)
      _dataloader = torch.utils.data.DataLoader(_dataset, batch_size=self.batch_size, num_workers=5, shuffle=False)
      pred = np.mean(self.predict(_dataloader))
      if pred < 1 :
          return True
      else:
          return False

    def formatter(self, sentence):
      data_format = pd.DataFrame()
      data_format['contents'] = sentence
      data_format['label'] = pd.Series([2] * len(sentence))
      return data_format

    def predict(self, _dataloader):
        # test_acc = 0.0
        lst_results = []
        self.model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(_dataloader):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            valid_length= valid_length
            label = label.long().to(self.device)
            out = self.model(token_ids, valid_length, segment_ids)
            # test_acc += calc_accuracy(out, label)
            for i in out:
                logits=i
                logits = logits.detach().cpu().numpy()
                lst_results.append(np.argmax(logits))
        # print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
        return lst_results

    def start_check_ad(self):
        store_ad_tmp = []

        for data in self.post_list:
            # title
            title = data[1]
            # text
            text = data[3]
            # prediction
            pred = self.checkAd(text)

            store_ad_tmp.append([title, pred])

        return store_ad_tmp
"""

def get_blog_post_list():
    post_list_tmp = []

    input_location = "일산"
    input_food = "피자"
    file_path = "./blog_data/" + input_location + "/" + input_food

    for idx in range(978):
        file_name = "blog_data_" + str(idx) + ".txt"

        data_idx = 0

        post_idx = ""
        post_store_map = ""
        post_title = ""
        post_content = []
        post_date = ""

        for line in open(file_path + "/" + file_name, 'r', encoding='utf-8'):
            if line[0] == "$":
                data_idx += 1

            elif line[0] == ";":
                line = line.lstrip(';').rstrip("\n")
                if data_idx == 0:
                    post_idx = line
                elif data_idx == 1:
                    post_title = line
                elif data_idx == 2:
                    post_store_map = line
                elif data_idx == 3:
                    post_content.append(line)
                elif data_idx == 4:
                    post_date = line

        post_list_tmp.append([post_idx, post_title, post_store_map, post_content, post_date])

    return post_list_tmp


print("Program start")

post_list = get_blog_post_list()

bertmodel, vocab = get_pytorch_kobert_model()
print("1")
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
print("2")
max_len = 128  # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
batch_size = 16
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

device = torch.device('cpu')
print("3")
#bertmodel.load_state_dict('./model3.pt')
model = torch.load('./model3.pt', map_location=torch.device('cpu'))
model.to(device)
print("4")

def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


def checkAd(lst):
    print(6)
    _data = formatter(lst)
    print(7)
    _dataset = BERTDataset(_data, tok, max_len, True, False)
    print(8)
    _dataloader = torch.utils.data.DataLoader(_dataset, batch_size=batch_size, shuffle=False)
    print(9)
    pred = np.mean(predict(_dataloader))
    if pred < 1:
        return True
    else:
        return False


def formatter(sentence):
    data_format = pd.DataFrame()
    data_format['contents'] = sentence
    data_format['label'] = pd.Series([2] * len(sentence))
    return data_format


def predict(_dataloader):
    # test_acc = 0.0
    lst_results = []
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(_dataloader):
        print(10)
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        # test_acc += calc_accuracy(out, label)
        for i in out:
            print(11)
            logits = i
            logits = logits.detach().cpu().numpy()
            lst_results.append(np.argmax(logits))
    # print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    return lst_results


def start_check_ad():
    store_ad_tmp = []

    for data in post_list:
        # title
        title = data[1]
        # text
        text = data[3]
        # prediction
        print(5)
        pred = checkAd(text)

        store_ad_tmp.append([title, pred])

    return store_ad_tmp


print(start_check_ad())
