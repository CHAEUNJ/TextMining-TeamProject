import re
import math

from konlpy.tag import Kkma
from collections import Counter


class StoreScoreManagement:
    service_good_feature = {'서비스': ['좋', '친절', '괜찮', '최고', '빠르', '짱', '훌륭', '추천', '감사', '구수', '최상', '대박',
                                    '훈훈', '특별', '개이득', '최고', '만족', '세련', '최고', '감동'],
                            '사장': ['친절', '스윗', '센스'],
                            '알바': ['친절', '스윗', '센스'],
                            '직원': ['친절', '스윗', '센스'],
                            '일을': ['잘', '빠르게'],
                            '일도': ['잘', '빠르게'],
                            '서빙': ['잘', '빠르게']}
    service_bad_feature = {
        '서비스': ['아쉽', '최악', '나쁘', '느리', '빡치', '비추', '별로', '그냥', '낙제', '쏘다쏘다', '엉망', '실망', '불친절', '문제', '컴플레인',
                '거지', '그닥', '그다지', '구려', '불편', '엉성', '헬', '개판'],
        '알바': ['불친절', '똑바로', '재수'],
        '사장': ['불친절', '똑바로', '재수'],
        '직원': ['불친절', '똑바로', '재수'],
        '일을': ['못', '느리게', '답답'],
        '일도': ['못', '느리게', '천천히'],
        '서빙': ['못', '느리게', '천천히', '답답']}

    taste_good_feature = {'간': ['맞', '적절', '딱', '환상', '담백'],
                          '음식': ['깔끔'],
                          '맛': ['있', '좋', '나다', '최고']}
    taste_bad_feature = {'간': ['세', '쎄', '강하다', '별로'],
                         '음식': ['별로', '쏘다쏘다', '최악'],
                         '맛': ['별로', '최악']}

    taste_good_emotion = ['고소', '부드럽', '신선', '촉촉', '싱싱', '정갈', '최고']
    taste_bad_emotion = ['싱겁', '느끼다하다', '짜다', '느끼다', '짜다', '딱딱하다', '차갑다']

    cost_good_feature = {'가격': ['괜찮', '착하다', '저렴', '적당', '싸다', '좋다', '합리적', '훌륭', '최고', '만족', '마음', '든든', '알맞다',
                                '무난', '괜춘', '최상', '최상', '굿', '엄지', '낮'],
                         '가성비': ['괜찮', '착하다', '저렴', '적당', '싸다', '좋다', '합리적', '훌륭', '최고', '만족', '마음', '든든', '알맞다',
                                 '무난', '괜춘', '최상', '최상', '굿', '엄지'],
                         '양': ['많', '적당', '푸짐하고', '괜찮다', '넉넉', '충분', '든든']
                         }
    cost_bad_feature = {
        '가격': ['비싸', '있다', '나쁘', '사악', '비효율', '높다', '부담', '아쉽', '쏘다쏘다', '별로', '그닥', '그다지', '쎄', 'ㅎㄷㄷ', '높', '거품'],
        '가성비': ['별로'],
        '양': ['적다', '작다', '아쉽', '적다', '다소', '별로'],
        }

    atmosphere_good_feature = {
        '분위기': ['좋', '괜찮', '조용', '깔끔', '적당', '깡패', '굉장', '아담', '완벽', '아기자기', '고급', '최고', '세련', '만족', '아늑', '훌륭', '예쁘',
                '이쁘', '짱',
                '심쿵', '따뜻', '깨끗', '독특', '매력', '모던', '취향저격', '맘', '마음', '클래식', '아름', '인상적', '귀엽', '포근'],
        '인테리어': ['좋', '괜찮', '조용', '깔끔', '적당', '깡패', '굉장', '아담', '완벽', '아기자기', '고급', '최고', '세련', '만족', '아늑', '훌륭', '예쁘',
                 '이쁘', '짱',
                 '심쿵', '따뜻', '깨끗', '독특', '매력', '모던', '취향저격', '맘', '마음', '클래식', '아름', '인상적', '귀엽', '포근']}
    atmosphere_bad_feature = {'분위기': ['나쁘다', '바쁘다', '어수선하다', '이상하다', '촌스럽다', '별로', '부담스럽다', '시끄럽', '복잡'],
                              '인테리어': []}

    visit_good_feature = {'의사': ['있다', '충만', '백프로', '백프롭', '많', '만땅', '마구', '그득', '만점', '넘침'],
                          '다시': ['가다', '오다', '방문', '찾다', '가보다', '한번', '갈다', '찾아가다', '가야지', '갈거다', '방문하다보고',
                                 '생각나다', '방문한다면', '와보고', '재방문', '접하다', '간다면', '갈다때가', '먹다고프다', '방문한다임', '오자고', '가기로',
                                 '갈다생각이다', '가면'],
                          '굳이': []}
    visit_bad_feature = {'의사': ['글쎄'],
                         '굳이': ['다시', '많이', '여기까지', '줄서서', '찾아', '시키다', '가다', '찾다', '여기', '기다리다', '줄을', '사먹'],
                         '다시': []}

    negative_word_emotion = ['안', '않', '못', '없', '아닌', '아니']

    def __init__(self, post_list, store_name, store_ad):
        self.post_list = post_list
        self.store_name_list = store_name
        self.store_ad_list = store_ad

        self.kkma = Kkma()

    def preprocessing(self, review):
        total_review = ''
        # 인풋리뷰
        for idx in range(len(review)):
            r = review[idx]
            sentence = re.sub('([a-zA-Z])', '', r)
            sentence = re.sub('[ㄱ-ㅎㅏ-ㅣ]+', '', sentence)
            sentence = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sentence)

            if len(sentence) == 0:
                continue

            total_review += sentence

        return total_review

    def get_feature_keywords(self, feature_keywords, review):
        feature_temp = []
        for keyword in feature_keywords:
            if re.findall(keyword, review):
                sub_list = ['게', '고', '음', '며', '데', '만', '도', '면']

                for sub in sub_list:
                    if sub + ' ' in review:
                        review = re.sub(sub + ' ', sub + ',', review)

                a = re.findall(keyword + '+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+', review)  # K한 한 한글
                b = re.findall(keyword + '+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+', review)  # K 한 한글
                c = re.findall('[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+' + keyword + '[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+', review)  # 한 K한글 예쁜 분위기가

                for ngram in a:
                    t = ()
                    feature_temp.append(t + (ngram, keyword))
                for ngram in b:
                    t = ()
                    feature_temp.append(t + (ngram, keyword))
                for ngram in c:
                    t = ()
                    feature_temp.append(t + (ngram, keyword))

        return feature_temp

    def get_feature_emotions(self, feature_good_dict, feature_bad_dict, feature_temp):
        good_feature_emotion_list = []
        bad_feature_emotion_list = []

        for ngrams in feature_temp:
            keyword = ngrams[1]
            ngram = ngrams[0]
            is_bad_feature = None

            good_emotion_list = feature_good_dict[keyword]
            bad_emotion_list = feature_bad_dict[keyword]
            for emotion in good_emotion_list:
                if re.findall(emotion, ngram):
                    is_bad_feature = False
            for emotion in bad_emotion_list:
                if re.findall(emotion, ngram):
                    is_bad_feature = True
            for negative in self.negative_word_emotion:
                if re.findall(negative, ngram):
                    if is_bad_feature:
                        is_bad_feature = False
                        break
                    elif not is_bad_feature:
                        is_bad_feature = True
                        break
                    else:
                        is_bad_feature = True
                        break
            if is_bad_feature:
                bad_feature_emotion_list.append(ngram)
            elif not is_bad_feature:
                good_feature_emotion_list.append(ngram)
            else:
                pass
        return good_feature_emotion_list, bad_feature_emotion_list

    def get_taste_emotion(self, taste_good_emotions, taste_bad_emotions):
        bad_taste_emotion_list = []
        good_taste_emotion_list = []
        for ngrams in taste_good_emotions:
            ngram = ngrams[0]
            is_bad_taste = False
            for negative in self.negative_word_emotion:
                if re.findall(negative, ngram):
                    is_bad_taste = True
            if is_bad_taste:
                bad_taste_emotion_list.append(ngram)
            else:
                good_taste_emotion_list.append(ngram)

        for ngrams in taste_bad_emotions:
            ngram = ngrams[0]
            is_bad_taste = True
            for negative in self.negative_word_emotion:
                if re.findall(negative, ngram):
                    is_bad_taste = False
            if is_bad_taste:
                bad_taste_emotion_list.append(ngram)
            else:
                good_taste_emotion_list.append(ngram)

        return good_taste_emotion_list, bad_taste_emotion_list

    def start_score(self):
        store_score_tmp = []
        store_score_result_tmp = []

        c_store_name_list = Counter(self.store_name_list)

        for idx in range(len(self.post_list)):
            if self.store_name_list[idx] == "" or self.store_ad_list[idx] == "TRUE":
                continue

            store_content = self.post_list[idx][3]
            store_content.append(self.post_list[idx][1])

            review = self.preprocessing(store_content)

            service_temp = self.get_feature_keywords(self.service_good_feature.keys(), review)
            good_service, bad_service = self.get_feature_emotions(self.service_good_feature, self.service_bad_feature, service_temp)

            atmosphere_temp = self.get_feature_keywords(self.atmosphere_good_feature.keys(), review)
            good_atmosphere, bad_atmosphere = self.get_feature_emotions(self.atmosphere_good_feature, self.atmosphere_bad_feature, atmosphere_temp)

            cost_temp = self.get_feature_keywords(self.cost_good_feature.keys(), review)
            good_cost, bad_cost = self.get_feature_emotions(self.cost_good_feature, self.cost_bad_feature, cost_temp)

            visit_temp = self.get_feature_keywords(self.visit_good_feature.keys(), review)
            good_visit, bad_visit = self.get_feature_emotions(self.visit_good_feature, self.visit_bad_feature, visit_temp)

            taste_temp = self.get_feature_keywords(self.taste_good_feature.keys(), review)
            good_taste, bad_taste = self.get_feature_emotions(self.taste_good_feature, self.taste_bad_feature, taste_temp)
            taste_good_emotion_temp = self.get_feature_keywords(self.taste_good_emotion, review)
            taste_bad_emotion_temp = self.get_feature_keywords(self.taste_bad_emotion, review)
            good_taste2, bad_taste2 = self.get_taste_emotion(taste_good_emotion_temp, taste_bad_emotion_temp)

            #print("#################", idx, "#################")
            #print(good_service, bad_service)
            #print(good_atmosphere, bad_atmosphere)
            #print(good_cost, bad_cost)
            #print(good_visit, bad_visit)
            #print(good_taste, bad_taste)
            #print(good_taste2, bad_taste2)

            service_result = [good_service, bad_service]
            atmosphere_result = [good_atmosphere, bad_atmosphere]
            cost_result = [good_cost, bad_cost]
            visit_result = [good_visit, bad_visit]
            taste_result = [good_taste + good_taste2, bad_taste + bad_taste2]

            save_flag = 0

            for i, data in enumerate(store_score_tmp):
                if data[0] == self.store_name_list[idx]:
                    store_score_tmp[i][1][0] += service_result[0]
                    store_score_tmp[i][1][1] += service_result[1]

                    store_score_tmp[i][2][0] += atmosphere_result[0]
                    store_score_tmp[i][2][1] += atmosphere_result[1]

                    store_score_tmp[i][3][0] += cost_result[0]
                    store_score_tmp[i][3][1] += cost_result[1]

                    store_score_tmp[i][4][0] += visit_result[0]
                    store_score_tmp[i][4][1] += visit_result[1]

                    store_score_tmp[i][5][0] += taste_result[0]
                    store_score_tmp[i][5][1] += taste_result[1]

                    save_flag = 1
                    break

            if save_flag == 0:
                store_score_tmp.append([self.store_name_list[idx], service_result, atmosphere_result,
                                                         cost_result, visit_result, taste_result])

        for data in store_score_tmp:
            list_tmp = []

            list_tmp.append(data[0])

            c_name = c_store_name_list[data[0]]
            list_tmp.append(c_name)

            for i in range(1, 6):
                if len(data[i][0]) + len(data[i][1]) != 0:
                    list_tmp.append(round((len(data[i][0]) / (len(data[i][0]) + len(data[i][1]))) * 100 * (math.log(c_name, 100) + 1), 1))
                    list_tmp.append(round(len(data[i][0]) / (len(data[i][0]) + len(data[i][1])) * 5, 1))
                else:
                    list_tmp.append(0)
                    list_tmp.append(0)

            store_score_result_tmp.append(list_tmp)

        return store_score_result_tmp
