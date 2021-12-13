# 참고 사이트 : https://nicola-ml.tistory.com/63?category=816369, rhinoMorph - MachineLearning

"""
def read_data(filename):
    with open(filename, 'r', encoding="cp949") as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data


data = read_data(r'ratings_morphed.txt')
data_text = [line[1] for line in data]
data_senti = [line[2] for line in data]

print(data_text[0])
print(data_senti[0])

from sklearn.model_selection import train_test_split

train_data_text, test_data_text, train_data_senti, test_data_senti \
    = train_test_split(
        data_text,
        data_senti,
        stratify=data_senti,
        test_size=0.3,
        random_state=156
)

from collections import Counter

train_data_senti_freq = Counter(train_data_senti)
print('train_data_senti_freq:', train_data_senti_freq)
test_data_senti_freq = Counter(test_data_senti)
print('test_data_senti_freq:', test_data_senti_freq)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5).fit(train_data_text)
X_train = vect.transform(train_data_text)

feature_names = vect.get_feature_names()
print("특성 개수:", len(feature_names))
print("처음 20개 특성:\m", feature_names[:20])

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
y_train = pd.Series(train_data_senti)
scores = cross_val_score(LogisticRegression(solver="liblinear"), X_train, y_train, cv=3)
print('교차 검증 점수:', scores)
print('교차 검증 점수 평균:', scores.mean())

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 3, 5]}
grid = GridSearchCV(LogisticRegression(solver="liblinear"), param_grid, cv=3)
grid.fit(X_train, y_train)
print("최고 교차 검증 점수:", round(grid.best_score_, 3))
print("최적의 매개변수:", grid.best_params_)

X_test = vect.transform(test_data_text)
y_test = pd.Series(test_data_senti)
print("테스트 데이터 점수:", grid.score(X_test, y_test))

import rhinoMorph

rn = rhinoMorph.startRhino()
print('rn\n', rn)
new_input = 'ㅅㅂ 바텀 ㅈㄴ 못하네!!!!'

# 입력 데이터 형태소 분석하기
inputdata = []
morphed_input = rhinoMorph.onlyMorph_list(rn, new_input, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'])
morphed_input = ' '.join(morphed_input)                     # 한 개의 문자열로 만들기
inputdata.append(morphed_input)                               # 분석 결과를 리스트로 만들기
X_input = vect.transform(inputdata)
print(float(grid.predict(X_input)))
result = grid.predict(X_input) # 0은 부정,1은 긍정
print("결과:", result)
"""
"""

# 음식점 이름 추출 Ko ner 코드
import ssl
import urllib.request

from tqdm import tqdm
import re

import json
import requests
from bs4 import BeautifulSoup

from konlpy.tag import Okt
#from konlpy.tag import Komoran
from konlpy.tag import Hannanum


def remove_tag(content):
   cleanr =re.compile('<.*?>')
   cleantext = re.sub(cleanr, '', content)
   cleanr =re.compile('&.*?;')
   cleantext2 = re.sub(cleanr, '', cleantext)
   return cleantext2


def edit_sent(sentence):
    #sent = "부천역 맛집 추천::남도뽀글이_소곱창 가성비 끝판왕"
    # "미간 찌푸려지게 맛있는 부천 곱창 맛집♥ “구들짱 황소곱창”"
    # "[부천 맛집]곱창 가성비 맛집_남도뽀글이"
    # "[강남 초밥맛집 청춘스시]:인생초밥집을찾았다ㅠㅠㅋ존맛x100"
    # "초밥/강남 스시,맛집] 은행골 초밥:: 살살 녹는 강남 초밥맛집"

    edited_sentence = ""
    list_special_char = ['&', '♪', '★', '「', '」', '♡', '♥', "'", '"', "“", "”", "[", "]", ":", ",", "-", "/", "_", "(", ")", "#"]

    for word in sentence:
        if word in list_special_char:
            #edited_sentence += " " + word + " "
            edited_sentence += " P "
        else:
            edited_sentence += word

    edited_sentence = edited_sentence.replace("맛집", " 맛집")

    return edited_sentence.split()
    #return ha.pos(edited_sentence)
    #return okt.pos(edited_sentence, stem=False)


def get_map_title(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"}
    res = requests.get(url, headers=headers)
    res.raise_for_status()  # 문제시 프로그램 종료
    soup = BeautifulSoup(res.text, "lxml")

    src_url = "https://blog.naver.com/" + soup.iframe["src"]

    response = requests.get(src_url)

    bs = BeautifulSoup(response.text, 'html.parser')

    tag = bs.find('strong', attrs={'class': 'se-map-title'})

    if tag:
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', str(tag.contents[0]))

        return cleantext
    else:
        return None


ssl._create_default_https_context = ssl._create_unverified_context

client_id = "OuGFA4zkGrGJ0mavPWL5"        # 오픈 API Token ID
client_secret = "bI05UsiyAI"              # 오픈 API Token Pass

location = "일산"
food = "국밥"

encText = urllib.parse.quote(location + food + "맛집")  # 검색을 원하는 문자열
displayN = '100'                          # 검색 결과 출력 건수 지정
startN = '1'                              # 검색 시작 위치로 최대 1000까지 가능
sortN = 'sim'                            # 정렬 옵션: sim (유사도순), date (날짜순)

url = "https://openapi.naver.com/v1/search/blog?query=" \
      + encText + "&display=" + displayN + "&start=" + startN + "&sort=" + sortN

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id", client_id)
request.add_header("X-Naver-Client-Secret", client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()

okt = Okt()
#ko = Komoran()
ha = Hannanum()

if rescode == 200:
    response_body = response.read()
    #print(response_body.decode('utf-8'))
    data_json = json.loads(response_body.decode('utf-8'))
    data_list = [(data["title"], data["description"], data["link"]) for data in data_json.get("items")]

    f = open("blog_list.txt", 'w', encoding='utf-8')

    for data in tqdm(data_list):
        if food in data[0] and location in data[0]:
        #if "모음" not in data[0] and "리스트" not in data[0] and "목록" not in data[0] and "소개" not in data[0] and "공유" not in data[0]:
            #print("Okt - ", okt.pos(remove_tag(data[0])))
            #print("Komoran - ", ko.pos(remove_tag(data[0])))
            #print("Hannanum - ", ha.pos(remove_tag(data[0])))
            #print("Noun - ", ha.nouns(remove_tag(data[0])), "\n")
            f.write(";" + remove_tag(data[0]) + "\n")

            map_title = ""
            tmp = get_map_title(data[2])
            if tmp:
                map_title = tmp

            f.write("$" + data[2] + "\t" + map_title + "\n")
            f.write("$" + remove_tag(data[1]) + "\n")

            for word in edit_sent(remove_tag(data[0])):
                #f.write(word[0] + "\t" + word[1] + "\n")
                f.write(word + "\n")

    f.close()

else:
    print("Error Code:" + rescode)
"""


############################# NER 음식점 이름 추출 with new train dataset #########################
"""
import pandas as pd
import eli5
from sklearn_crfsuite import CRF
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report


def edit_sent(sentence):
    #sent = "부천역 맛집 추천::남도뽀글이_소곱창 가성비 끝판왕"
    # "미간 찌푸려지게 맛있는 부천 곱창 맛집♥ “구들짱 황소곱창”"
    # "[부천 맛집]곱창 가성비 맛집_남도뽀글이"
    # "[강남 초밥맛집 청춘스시]:인생초밥집을찾았다ㅠㅠㅋ존맛x100"
    # "초밥/강남 스시,맛집] 은행골 초밥:: 살살 녹는 강남 초밥맛집"

    edited_sentence = ""
    list_special_char = ['&', '♪', '★', '「', '」', '♡', '♥', "'", '"', "“", "”", "[", "]", ":", ",", "-", "/", "_", "(", ")", "#"]

    for word in sentence:
        if word in list_special_char:
            edited_sentence += " " + word + " "
        else:
            edited_sentence += word

    edited_sentence = edited_sentence.replace("맛집", " 맛집")

    return han.pos(edited_sentence)
    #return okt.pos(edited_sentence, stem=False)


data_sents = []
data = []
sent_no = 0

for line in open("blog_list_total.txt"):
    if line[0] == ';':
        data_sents.append(line.split('\n')[0])
        sent_no += 1
    elif line[0] != '$':
        row = line.split()
        if row:
            row.insert(0, str(sent_no))
        if len(row):
            data.append(row)
            print(row)

data = pd.DataFrame(data)
data.columns = ['Sentence #', 'Word', 'POS', 'Tag']


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(data)
sent = getter.get_next()
sentences = getter.sentences

print(len(sentences))


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word': word,
        'word[-2:]': word[-2:],
        'word[-1:]': word[-1:],
        'word.isdigit()': word.isdigit(),
        #'postag': postag
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word': word1,
            '-1:word[-1:]': word1[-1:]
            #'-1:postag': postag1
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word': word1,
            '+1:word[-1:]': word1[-1:]
            #'+1:postag': postag1
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

print(len(X))
print(len(y))
print(len(sentences))


# Graident Descent 대신 Limited-memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS) 사용
crf = CRF(algorithm='lbfgs',
          c1=0.1, # overfitting을 방지하기 위한 L1 regularization(정규화) 가중치
          c2=0.1, # overfitting을 방지하기 위한 L2 regularization 가중치
          max_iterations=100,
          all_possible_transitions=False)

pred = cross_val_predict(estimator=crf, X=X, y=y, cv=2)
report = flat_classification_report(y_pred=pred, y_true=y)
print(report)

crf.fit(X[:330], y[:330])
#y_test_pred = crf.predict(X[330:])
#report = flat_classification_report(y_pred=y_test_pred, y_true=y[330:])
#print(report)

#eli5.show_weights(crf, top=30)

from konlpy.tag import Hannanum

han = Hannanum()

sent = "선릉역_맛집_족보있는 국밥 ( 얼큰 돼지국밥 )"
    #"선릉역맛집 농민백암순대 본점에서 웨팅후 맛본 순대국밥"
    #"잠실 목살 맛집_방이돈가"
    #"[잠실역/송파나루역] - 송리단길 맛집 :: 또봄 (바질페스토&토마토파스타 / 돼지목살스테이크&트러플대파귀리리조트)"
    #"[일산 회 맛집] 웨스턴돔 횟집 '어촌' 오늘은 방어회와 전복치!!"

sent_pos = edit_sent(sent)
sent_x = sent2features(sent_pos)

y_test_pred = crf.predict([sent_x])

print(sent_pos)
print(y_test_pred[0])

print(sent)
for i in range(len(y_test_pred[0])):
    if y_test_pred[0][i] == 'ST':
        print(sent_pos[i][0])
"""

# kobert 활용한 ko ner crf 코드
"""
import sys

sys.path.append("C:\\Users\\82109\\PycharmProjects\\homework\\pytorchBertCrfNer")

print(sys.path)

from pytorchBertCrfNer.kobert_ner_crf import KoBERT_NER_CRF

ko = KoBERT_NER_CRF()
result = ko.ko_ner_crf("강남역에 새로 오픈한 은행골에서 밥을 먹고 일산으로 갔다가 다시 강남역으로 돌아오니 5시였다.")
#"어제는 서울에서 놀다가 다시 서울에서 밥먹고 또 서울에서 집으로 왔다.")
#"진짜 제가 정말 좋아하는 고기집이 있어요!\n바로 김포공항에서 송정역 근방에 위치해 있는 서울 송정역 맛집 우리도한번잘구워보세 인데요~~\n
# 진짜 간판부터 뭔가 정겹지 않으세요?ㅠㅠ 여기 가게 분위기도 좋아해서 더 자주 갑니다~~\n데이트장소로도 최고봉인 이 곳!!")
#"[일상이야기]서울 송정역 맛집을 찾으신다면 꼭 방문해야 할 우리도한번잘구워보세")

print(result)
"""

# 카카오 API를 활용한 음식점 이름 list 불러오기
"""
import requests

searching = '부천역 육회'
url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query={}&category_group_code=FD6&size={}&page={}'
headers = {
    "Authorization": "KakaoAK f3900d00b90c2ae4cc01fe341dd2a59f"
}

total_places = []

total_count = requests.get(url.format(searching, 1, 1), headers=headers).json()['meta']['pageable_count']

for i in range(1, total_count+1):
    res = requests.get(url.format(searching, 15, i), headers=headers)
    if res.status_code == 200:
        for data in res.json()['documents']:
            total_places.append(str(i)+data['place_name'])
    else:
        break

print((total_places, len(total_places)))
print((set(total_places), len(set(total_places))))

print(total_count)
res = requests.get(url.format(searching, 15, 3), headers=headers).json()['documents']
print([data['place_name'] for data in res])
res = requests.get(url.format(searching, 15, 4), headers=headers).json()['documents']
print([data['place_name'] for data in res])
res = requests.get(url.format(searching, 15, 5), headers=headers).json()['documents']
print([data['place_name'] for data in res])
"""

"""
# 참고 사이트: https://soyoung-new-challenge.tistory.com/46, Feature 기반 감정분석

import re
from konlpy.tag import Kkma
from konlpy.tag import Okt

kk = Kkma()
okt = Okt()

taste_good_feature = {'간':['맞','적절','딱','환상','담백'],
                      '음식':['깔끔'],
                      '맛':['있','좋','나다','최고']}

taste_bad_feature = {'간':['세','쎄','강하다','별로'],
                     '음식':['별로','쏘다쏘다','최악'],
                     '맛':['별로','최악']}

taste_good_emotion = ['고소', '부드럽', '신선', '촉촉', '싱싱', '정갈', '최고']
taste_bad_emotion = ['싱겁', '느끼', '짜다', '느끼다', '짜다', '딱딱하다', '차갑다']


def sent_proess(review):
    sentence = re.sub('([a-zA-Z])', '', review)
    sentence = re.sub('[ㄱ-ㅎㅏ-ㅣ]+', '', sentence)
    sentence = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sentence)

    return sentence#kk.morphs(sentence)


def check_sent_emotion(sentence):
    list_good_emotion_words = []
    list_bad_emotion_words = []

    for word in sentence:
        for emotion_word in taste_good_emotion:
            if emotion_word in word:
                list_good_emotion_words.append(word)
        for emotion_word in taste_bad_emotion:
            if emotion_word in word:
                list_bad_emotion_words.append((word))

    return list_good_emotion_words, list_bad_emotion_words


def check_sent_feature(sentence, keyword):
    list_good_feature_words = []
    list_bad_feature_words = []

    review = " ".join(sentence)

    a = re.findall(keyword + '+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+', review)
    b = re.findall(keyword + '+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+', review)
    c = re.findall('[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+' + keyword + '[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+', review)
    #print(a, b, c)

    for word in a:
        for feature_word in taste_good_feature[keyword]:
            if feature_word in word:
                list_good_feature_words.append(word)
        for feature_word in taste_bad_feature[keyword]:
            if feature_word in word:
                list_bad_feature_words.append(word)

    for word in b:
        for feature_word in taste_good_feature[keyword]:
            if feature_word in word:
                list_good_feature_words.append(word)
        for feature_word in taste_bad_feature[keyword]:
            if feature_word in word:
                list_bad_feature_words.append(word)

    for word in c:
        for feature_word in taste_good_feature[keyword]:
            if feature_word in word:
                list_good_feature_words.append(word)
        for feature_word in taste_bad_feature[keyword]:
            if feature_word in word:
                list_bad_feature_words.append(word)

    return list_good_feature_words, list_bad_feature_words


review = "맛도 좋고 간도 너무 딱 맞아서 좋았습니다. 음식도 모두 정갈하긴 했지만 싱거워서 아쉬웠다. 짜기도 했고 딱딱했지만 최고였다."

d = okt.morphs(review, stem=True)

for keyword in taste_good_feature.keys():
    print(check_sent_feature(d, keyword))

print(check_sent_emotion(d))
"""
"""
import eli5

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.svm import LinearSVC

corpus = []
y_data = []

for data in open("1.txt", 'r', encoding='utf-8'):
    if data[0] == ';':
        data = data.replace(";", "")
        data = data.replace("\n", "")
        corpus.append(data)
        y_data.append(1)

for data in open("2.txt", 'r', encoding='utf-8'):
    if data[0] == ';':
        data = data.replace(";", "")
        data = data.replace("\n", "")
        corpus.append(data)
        y_data.append(0)

for data in open("3.txt", 'r', encoding='utf-8'):
    if data[0] == ';':
        data = data.replace(";", "")
        data = data.replace("\n", "")
        corpus.append(data)
        y_data.append(0)


#morphed_corpus = okt.morphs(" ".join(corpus))

vect = CountVectorizer()
vect.fit(corpus)
print(vect.vocabulary_)

x_data = vect.transform(corpus)
x_data = TfidfTransformer().fit_transform(x_data)

print(x_data)

svm = LinearSVC()
svm.fit(x_data, y_data)

test_data = vect.transform(["강남 맛집 굴토리 굴국밥", "[일산 대화 맛집] 가야밀면 돼지국밥 / tvN 아르곤 출연진 봤다."])
test_data = TfidfTransformer().fit_transform(test_data)

y_pred = svm.predict(test_data)
print(y_pred)

eli5.show_weights(svm)
"""
########################################## TOTAL #############################################

import sys

sys.path.append("C:\\Users\\82109\\PycharmProjects\\homework\\pytorchBertCrfNer")

import os

import threading

from blog_post_scrap import BlogPostScrap
from pytorchBertCrfNer.kobert_ner_crf import KoBERT_NER_CRF
from store_score_management import StoreScoreManagement
from store_keyword_management import StoreKeywordManagement

from wordcloud import WordCloud
from tqdm import tqdm
from collections import Counter
from difflib import SequenceMatcher

global input_food
global input_location

global file_list_post

global post_list
global store_list
global store_name
global store_ad
global store_rank
global store_keyword

def get_blog_post_list(path):
    post_list_tmp = []

    for idx in range(len(file_list_post)):
        file_name = "blog_data_" + str(idx) + ".txt"

        data_idx = 0

        post_idx = ""
        post_store_map = ""
        post_title = ""
        post_content = []
        post_date = ""

        for line in open(path + "/" + file_name, 'r', encoding='utf-8'):
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


def get_store_list_from_post(path):
    store_list_tmp = []

    store_file_name = path + "/store_list.txt"
    s_f = open(store_file_name, 'w', encoding='utf-8')

    for post_data in post_list:
        if post_data[2] != "":
            store_list_tmp.append(post_data[2])

    for store_data in list(set(store_list_tmp)):
        s_f.write(store_data + "\n")

    return list(set(store_list_tmp))


def get_store_list_from_file(path):
    store_list_tmp = []

    store_file_name = path + "/store_list.txt"
    for line in open(store_file_name, 'r', encoding='utf-8'):
        store_tmp = line.split()
        if store_tmp[len(store_tmp)-1].endswith("점"):
            store_tmp[len(store_tmp) - 1] = ""
        if " ".join(store_tmp).rstrip(" ") != "":
            store_list_tmp.append(" ".join(store_tmp).rstrip(" "))

    return list(set(store_list_tmp))


def get_store_name_from_post(path):
    ko = KoBERT_NER_CRF()

    store_name_tmp = []
    store_name_f = open(path + "/store_name.txt", "w", encoding="utf-8")

    for idx in range(len(post_list)):
        result = Counter()

        map_match_flag = 0

        if post_list[idx][2] != "":
            for store_data in store_list:
                if store_data.replace(" ", "") in post_list[idx][2].replace(" ", ""):
                    print("Map Match -", store_data)
                    post_list[idx][2] = store_data
                    map_match_flag = 1
                    break

        if map_match_flag == 0:
            post_list[idx][2] = ""

        if post_list[idx][2] == "":
            result = Counter(ko.ko_ner_crf(post_list[idx][1]))
            for data in post_list[idx][3]:
                result += Counter(ko.ko_ner_crf(data))

            for dict_data in dict(result).keys():
                for store_data in store_list:
                    if SequenceMatcher(None, dict_data, store_data).ratio() >= 0.90:
                        print("NER -", store_data)
                        print(result, "\n")
                        post_list[idx][2] = store_data
                        break
                if post_list[idx][2] != "":
                    break

        if post_list[idx][2] == "":
            print("\n", post_list[idx][0], post_list[idx][1])
            print(result, "\n")
        #else:
        #    print("\n", idx, post_list[idx][2], "\n")

        store_name_tmp.append(post_list[idx][2])
        store_name_f.write(post_list[idx][2] + "\n")

    return store_name_tmp


def get_store_name_from_file(path):
    store_name_tmp = []

    for store_data in open(path + "/store_name.txt", "r", encoding="utf-8"):
        store_name_tmp.append(store_data.rstrip("\n"))

    return store_name_tmp


def get_store_ad_from_file(path):
    store_ad_tmp = []

    for store_data in open(path + "/store_ad.txt", "r", encoding="utf-8"):
        if "TRUE" in store_data:
            store_ad_tmp.append("TRUE")
        else:
            store_ad_tmp.append("FALSE")

    return store_ad_tmp


def get_store_score_from_file(path):
    store_rank_tmp = []

    for line in open(path + "/store_rank.txt", "r", encoding="utf-8"):
        data = line.split("\t")
        store_rank_tmp.append(data)

    return store_rank_tmp


def get_store_score_from_post(path):
    storeScoreM = StoreScoreManagement(post_list, store_name, store_ad)
    store_score_result = storeScoreM.start_score()

    # 0: 음식점명, 1: 포스팅수, 2: 서비스, 3: 분위기, 4: 가격, 5: 재방문, 6: 맛
    store_score_result = sorted(store_score_result, key=lambda x: (x[2] * 0.1) + (x[4] * 0.1) + (x[6] * 0.1) + (x[8] * 0.1) + (x[10] * 0.6), reverse=True)

    store_rank_f = open(path + "/store_rank.txt", "w", encoding="utf-8")

    for data in store_score_result:
        for i in range(12):
            store_rank_f.write(str(data[i]) + "\t")
        store_rank_f.write("\n")

    return store_score_result


def get_store_keyword_from_file(path):
    store_keyword_tmp = []

    for data in open(path + "/store_keyword.txt", "r", encoding="utf-8"):
        data = data.rstrip("\n")
        data_list = data.split("\t")
        store_keyword_tmp.append(data_list)

    return store_keyword_tmp


def get_store_keyword_from_post(path):
    storeKeywordM = StoreKeywordManagement(post_list, store_name, store_ad, store_rank)
    store_keyword_tmp = storeKeywordM.get_keyword(input_location, input_food)

    store_keyword_f = open(path + "/store_keyword.txt", "w", encoding="utf-8")

    for data in store_keyword_tmp:
        for word in data:
            store_keyword_f.write(word + "\t")
        store_keyword_f.write("\n")

    return store_keyword_tmp


def code_main(location, food):
    global input_food
    global input_location

    global file_list_post

    global post_list
    global store_list
    global store_name
    global store_ad
    global store_rank
    global store_keyword

    #input_location = input("지역: ")
    #input_food = input("음식: ")
    input_location = location
    input_food = food

    file_path = "./blog_data/" + input_location + "/" + input_food

    ##########Web Scraping##############
    if os.path.isdir(file_path):
        print(file_path + "\t\t\t\t\talready exits!")
    else:
        scrap = BlogPostScrap()
        scrap.start_scraping(location=input_location, food=input_food)

    file_list = os.listdir(file_path)
    file_list_post = [file for file in file_list if file.startswith("blog_data")]

    post_list = get_blog_post_list(file_path)

    print("Log - Data #:", str(len(post_list)), "\n")

    #########Get Store List#############
    updated_flag = 0

    if os.path.isfile(file_path + "/store_list.txt"):
        print(file_path + "/store_list.txt\t\talready exits!")
        store_list = get_store_list_from_file(file_path)
        updated_flag = 1
    else:
        print("Please update store_list.txt!")
        store_list = get_store_list_from_post(file_path)

    print("Log - Store List #:", str(len(store_list)), "\n")

    ##########Get Store Name#############
    if updated_flag == 1:
        if os.path.isfile(file_path + "/store_name.txt"):
            print(file_path + "/store_name.txt\t\talready exits!")
            store_name = get_store_name_from_file(file_path)
        else:
            store_name = get_store_name_from_post(file_path)

        print("Log - Store Name #:", str(len(store_name) - store_name.count("")), "\n")

    ###########Get Store AD##############
        if os.path.isfile(file_path + "/store_ad.txt"):
            print(file_path + "/store_ad.txt\t\talready exits!")
            store_ad = get_store_ad_from_file(file_path)
        else:
            print("Please update store_ad.txt!")

        print("Log - Store Ad #:", str(store_ad.count("FALSE")), "\n")

    ############Get Store Score###########
        if os.path.isfile(file_path + "/store_rank.txt"):
            print(file_path + "/store_rank.txt\t\talready exits!")
            store_rank = get_store_score_from_file(file_path)
        else:
            store_rank = get_store_score_from_post(file_path)

        print("Log -Store Rank OK\n")

    ###########Get Store Keyword############
        if os.path.isfile(file_path + "/store_keyword.txt"):
            print(file_path + "/store_keyword.txt\t\talready exits!")
            store_keyword = get_store_keyword_from_file(file_path)
        else:
            store_keyword = get_store_keyword_from_post(file_path)

        print("Log -Store Keyword OK\n")

    ################Output##################
        if not os.path.isdir(file_path + "/word_cloud"):
            os.makedirs(file_path + "/word_cloud")

        for i in range(10):
            print(i+1, "위 음식점:", store_rank[i][0])
            print(" * 포스팅 수:", store_rank[i][1])
            print("\n * 서비스 점수:", store_rank[i][3])
            print(" * 분위기 점수:", store_rank[i][5])
            print(" * 가격 점수:", store_rank[i][7])
            print(" * 재방문 점수:", store_rank[i][9])
            print(" * 맛 점수:", store_rank[i][11])
            print(" * 키워드:", store_keyword[i][:5], "\n\n")

            if not os.path.isfile(file_path + "/word_cloud/#" + store_rank[i][0] + " word_cloud.jpg"):
                wc = WordCloud(font_path="BMJUA_ttf.ttf", background_color="white", max_font_size=60)
                cloud = wc.generate_from_frequencies(dict(Counter(store_keyword[i])))

                cloud.to_file(file_path + "/word_cloud/#" + store_rank[i][0] + " word_cloud.jpg")


import functools
from PIL import Image

from PyQt5 import QtCore, QtGui, QtWidgets
import test_rc
import threading


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        MainWindow.setWindowTitle("TextMining TeamProject")
        MainWindow.resize(360, 640)
        MainWindow.setStyleSheet("font: 9pt \"배달의민족 주아\";""color: rgb(229, 229, 229);")


        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 361, 81))
        self.frame.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)

        self.widget = QtWidgets.QWidget(self.frame)
        self.widget.setGeometry(QtCore.QRect(10, 0, 341, 80))
        self.widget.setStyleSheet("border-image: url(:/newPrefix/gui_image/search_img.png);")

        self.widget_2 = QtWidgets.QWidget(self.widget)
        self.widget_2.setGeometry(QtCore.QRect(30, 23, 31, 31))
        self.widget_2.setStyleSheet("border-image: url(:/newPrefix/gui_image/search_icon.png);")

        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setGeometry(QtCore.QRect(70, 18, 231, 41))
        self.lineEdit.setStyleSheet("border-image: url(:/newPrefix/gui_image/search_window.png);\n"
                                    "color: rgb(39, 174, 96);\n""font: 18pt;")
        self.lineEdit.returnPressed.connect(self.enterInputFunction)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 95, 56, 12))
        self.label.setStyleSheet("color: rgb(75, 75, 75);")
        self.label.setText("맛집 랭킹")
        self.label.hide()

        self.pbar = QtWidgets.QProgressBar(self.centralwidget)
        self.pbar.setGeometry(QtCore.QRect(70, 250, 250, 30))

        self.lbl_pbar = QtWidgets.QLabel(self.centralwidget)
        self.lbl_pbar.setGeometry(QtCore.QRect(80, 280, 200, 30))
        self.lbl_pbar.setStyleSheet("font: 13pt;\n""color: rgb(68, 68, 68);")
        self.lbl_pbar.setText("맛집 랭킹을 가져오고 있어요!")
        self.lbl_pbar.hide()

        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(0, 120, 361, 471))
        self.scrollArea.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.scrollArea.setWidgetResizable(True)

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 346, 469))

        self.restaurant_list_layout = QtWidgets.QVBoxLayout()

        self.widget_list = []
        self.label_list = []
        self.score_list = []

        for idx in range(10):
            self.widget_list.append(QtWidgets.QWidget(self.scrollAreaWidgetContents))
            self.widget_list[idx].setGeometry(QtCore.QRect(10, 10 + (130 * idx), 321, 111))
            self.widget_list[idx].setFixedHeight(111)
            self.widget_list[idx].setFixedWidth(321)
            self.widget_list[idx].setStyleSheet("border-image: url(:/newPrefix/gui_image/list_bg.png);")

            self.lbl_ranking = QtWidgets.QLabel(self.widget_list[idx])
            self.lbl_ranking.setGeometry(QtCore.QRect(15, 40, 31, 31))
            self.lbl_ranking.setStyleSheet("font: 20pt;\n""color: rgb(169, 169, 169);")
            self.lbl_ranking.setText(str(idx + 1))

            self.lbl_taste = QtWidgets.QLabel(self.widget_list[idx])
            self.lbl_taste.setGeometry(QtCore.QRect(50, 70, 21, 16))
            self.lbl_taste.setStyleSheet("font: 10pt;\n""color: rgb(68, 68, 68);")
            self.lbl_taste.setText("맛")

            self.lbl_service = QtWidgets.QLabel(self.widget_list[idx])
            self.lbl_service.setGeometry(QtCore.QRect(130, 70, 41, 16))
            self.lbl_service.setStyleSheet("font: 10pt;\n""color: rgb(68, 68, 68);")
            self.lbl_service.setText("서비스")

            self.lbl_vibes = QtWidgets.QLabel(self.widget_list[idx])
            self.lbl_vibes.setGeometry(QtCore.QRect(230, 70, 41, 16))
            self.lbl_vibes.setStyleSheet("font: 10pt;\n""color: rgb(68, 68, 68);")
            self.lbl_vibes.setText("분위기")

            self.lbl_price = QtWidgets.QLabel(self.widget_list[idx])
            self.lbl_price.setGeometry(QtCore.QRect(50, 90, 31, 16))
            self.lbl_price.setStyleSheet("font: 10pt;\n""color: rgb(68, 68, 68);")
            self.lbl_price.setText("가격")

            self.lbl_visit_again = QtWidgets.QLabel(self.widget_list[idx])
            self.lbl_visit_again.setGeometry(QtCore.QRect(130, 90, 41, 16))
            self.lbl_visit_again.setStyleSheet("font: 10pt;\n""color: rgb(68, 68, 68);")
            self.lbl_visit_again.setText("재방문")

            self.lbl_star_1 = QtWidgets.QLabel(self.widget_list[idx])
            self.lbl_star_1.setGeometry(QtCore.QRect(65, 70, 16, 16))
            self.lbl_star_1.setText(
                "<html><head/><body><p><img src=\":/newPrefix/gui_image/star_icon.png\"/></p></body></html>")

            self.lbl_star_2 = QtWidgets.QLabel(self.widget_list[idx])
            self.lbl_star_2.setGeometry(QtCore.QRect(165, 70, 16, 16))
            self.lbl_star_2.setText(
                "<html><head/><body><p><img src=\":/newPrefix/gui_image/star_icon.png\"/></p></body></html>")

            self.lbl_star_3 = QtWidgets.QLabel(self.widget_list[idx])
            self.lbl_star_3.setGeometry(QtCore.QRect(265, 70, 16, 16))
            self.lbl_star_3.setText(
                "<html><head/><body><p><img src=\":/newPrefix/gui_image/star_icon.png\"/></p></body></html>")

            self.lbl_star_4 = QtWidgets.QLabel(self.widget_list[idx])
            self.lbl_star_4.setGeometry(QtCore.QRect(75, 90, 16, 16))
            self.lbl_star_4.setText(
                "<html><head/><body><p><img src=\":/newPrefix/gui_image/star_icon.png\"/></p></body></html>")

            self.lbl_star_5 = QtWidgets.QLabel(self.widget_list[idx])
            self.lbl_star_5.setGeometry(QtCore.QRect(165, 90, 16, 16))
            self.lbl_star_5.setText(
                "<html><head/><body><p><img src=\":/newPrefix/gui_image/star_icon.png\"/></p></body></html>")
            # 음식점 이름
            self.label_list.append([QtWidgets.QLabel(self.widget_list[idx]),
                                    QtWidgets.QLabel(self.widget_list[idx]), QtWidgets.QLabel(self.widget_list[idx])])
            self.label_list[idx][0].setGeometry(QtCore.QRect(50, 10, 130, 31))
            self.label_list[idx][0].setStyleSheet("font: 17pt;\n""color: rgb(5, 118, 23);")
            # 블로그 리뷰 수
            self.label_list[idx][1].setGeometry(QtCore.QRect(190, 18, 131, 16))
            self.label_list[idx][1].setStyleSheet("font: 14pt;\n""color: rgb(169, 169, 169);")
            # 키워드
            self.label_list[idx][2].setGeometry(QtCore.QRect(50, 45, 251, 16))
            self.label_list[idx][2].setStyleSheet("font: 11pt;\n""color: rgb(39, 174, 96);\n"
                                                  "border-image: url(:/newPrefix/gui_image/highlight.png);")
            self.label_list[idx][2].setObjectName(str(idx))
            self.label_list[idx][2].mousePressEvent = functools.partial(self.showWordCloud, self.label_list[idx][2])


            # 맛 점수
            self.score_list.append([QtWidgets.QLabel(self.widget_list[idx]), QtWidgets.QLabel(self.widget_list[idx]),
                                    QtWidgets.QLabel(self.widget_list[idx]), QtWidgets.QLabel(self.widget_list[idx]),
                                    QtWidgets.QLabel(self.widget_list[idx])])
            self.score_list[idx][0].setGeometry(QtCore.QRect(80, 70, 21, 16))
            self.score_list[idx][0].setStyleSheet("font: 10pt;\n""color: rgb(68, 68, 68);")
            # 서비스 점수
            self.score_list[idx][1].setGeometry(QtCore.QRect(180, 70, 21, 16))
            self.score_list[idx][1].setStyleSheet("font: 10pt;\n""color: rgb(68, 68, 68);")
            # 분위기 점수
            self.score_list[idx][2].setGeometry(QtCore.QRect(280, 70, 21, 16))
            self.score_list[idx][2].setStyleSheet("font: 10pt;\n""color: rgb(68, 68, 68);")
            # 음식 점수
            self.score_list[idx][3].setGeometry(QtCore.QRect(90, 90, 21, 16))
            self.score_list[idx][3].setStyleSheet("font: 10pt;\n""color: rgb(68, 68, 68);")
            # 재방문 점수
            self.score_list[idx][4].setGeometry(QtCore.QRect(180, 90, 21, 16))
            self.score_list[idx][4].setStyleSheet("font: 10pt;\n""color: rgb(68, 68, 68);")

        for idx in range(10):
            self.restaurant_list_layout.addWidget(self.widget_list[idx])

        self.scrollAreaWidgetContents.setLayout(self.restaurant_list_layout)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 358, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.scrollArea.hide()
        self.pbar.hide()

        self.step = 0

        self.timer = QtCore.QTimer()

    def timerEvent(self):
        if self.step >= 100:
            self.timer.stop()

            self.pbar.hide()
            self.lbl_pbar.hide()

            # label update
            for rank_idx in range(10):
                self.label_list[rank_idx][0].setText(store_rank[rank_idx][0])
                self.label_list[rank_idx][1].setText("블로그 리뷰 " + str(store_rank[rank_idx][1]))
                self.label_list[rank_idx][2].setText((" " + " ".join(store_keyword[rank_idx][:5])).replace(" ", " #"))
                self.label_list[rank_idx][2].setFixedWidth(len(self.label_list[rank_idx][2].text()) * 11)
                self.score_list[rank_idx][0].setText(store_rank[rank_idx][3])
                self.score_list[rank_idx][1].setText(store_rank[rank_idx][5])
                self.score_list[rank_idx][2].setText(store_rank[rank_idx][7])
                self.score_list[rank_idx][3].setText(store_rank[rank_idx][9])
                self.score_list[rank_idx][4].setText(store_rank[rank_idx][11])

            self.label.show()
            self.scrollArea.show()

            self.step = 0
            return

        self.step = self.step + 1
        self.pbar.setValue(self.step)

    def enterInputFunction(self):
        self.scrollArea.hide()

        location, food = self.lineEdit.text().split()
        print(location, food)

        t = threading.Thread(target=code_main, args=(location, food))
        t.start()

        self.timer.setInterval(35)
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start()
        self.pbar.show()
        self.lbl_pbar.show()

    def showWordCloud(self, key_word_idx, event):
        location, food = self.lineEdit.text().split()
        file_path = "./blog_data/" + location + "/" + food + "/word_cloud/#"

        image = Image.open(file_path + store_rank[int(key_word_idx.objectName())][0] + " word_cloud.jpg")
        image.show()


app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)

MainWindow.show()


sys.exit(app.exec_())
