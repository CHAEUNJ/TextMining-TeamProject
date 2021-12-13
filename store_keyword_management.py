import re
from krwordrank.sentence import summarize_with_sentences
from konlpy.tag import Hannanum


class StoreKeywordManagement:
    def __init__(self, post_list, store_name, store_ad, store_rank):
        self.post_list = post_list
        self.store_name_list = store_name
        self.store_ad = store_ad
        self.store_rank = store_rank

        self.hann = Hannanum()

    def get_keyword(self, location, food):
        store_keyword_tmp = []

        stopwords = []
        stopwords_key = [location, "메뉴", "주문"]

        for word in open("stopwords_ko.txt", "r", encoding="utf-8"):
            stopwords.append(word.rstrip("\n"))

        for idx in range(10):
            store_name = self.store_rank[idx][0]
            store_content = []

            for i, name in enumerate(self.store_name_list):
                if name != store_name or self.store_ad[i] == "TRUE":
                    continue
                for line in self.post_list[i][3]:
                    #print(line)

                    line = re.sub('([a-zA-Z])', '', line)
                    line = re.sub('[ㄱ-ㅎㅏ-ㅣ]+', '', line)
                    line = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', line)
                    line = line.replace(name, "")
                    line = line.replace(name.replace(food, ""), "")
                    line = re.sub(r'[0-9]+', '', line)

                    if len(line.replace(" ", "")) == 0:
                        continue

                    line = self.hann.nouns(line)
                    line = [word for word in line if word not in stopwords]
                    line = " ".join(line)

                    if len(line.replace(" ", "")) == 0:
                        continue

                    #print(i, line)
                    store_content.append(line)

            key_words, key_sents = summarize_with_sentences(
                store_content,
                stopwords=stopwords_key,
                num_keywords=100,
                num_keysents=10)

            store_keyword_tmp.append(list(key_words.keys()))

        return store_keyword_tmp
