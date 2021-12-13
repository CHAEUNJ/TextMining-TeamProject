import re
import math

from konlpy.tag import Kkma
from collections import Counter


class StoreScoreManagement:
    # 감성사전을 미리 만들어 놓음
    # target을 key에 저장하고 그 key에 대한 긍정, 부정 평가 단어를 value로 저장함 
    service_good_features = {'서비스': ['좋', '친절', '괜찮', '최고', '빠르', '짱', '훌륭', '추천', '감사', '구수', '최상', '대박',
                                     '훈훈', '특별', '개이득', '최고', '만족', '세련', '최고', '감동'],
                             '사장': ['친절', '스윗', '센스'],
                             '알바': ['친절', '스윗', '센스'],
                             '직원': ['친절', '스윗', '센스'],
                             '일을': ['잘', '빠르게'],
                             '일도': ['잘', '빠르게'],
                             '서빙': ['잘', '빠르게']}
    service_bad_features = {
        '서비스': ['아쉽', '최악', '나쁘', '느리', '빡치', '비추', '별로', '그냥', '낙제', '쏘다쏘다', '엉망', '실망', '불친절', '문제', '컴플레인',
                '거지', '그닥', '그다지', '구려', '불편', '엉성', '헬', '개판'],
        '알바': ['불친절', '똑바로', '재수'],
        '사장': ['불친절', '똑바로', '재수'],
        '직원': ['불친절', '똑바로', '재수'],
        '일을': ['못', '느리게', '답답'],
        '일도': ['못', '느리게', '천천히'],
        '서빙': ['못', '느리게', '천천히', '답답']}

    taste_good_features = {'간': ['맞', '적절', '딱', '환상', '담백'],
                           '음식': ['깔끔'],
                           '맛': ['있', '좋', '나다', '최고']}
    taste_bad_features = {'간': ['세', '쎄', '강하다', '별로'],
                          '음식': ['별로', '쏘다쏘다', '최악'],
                          '맛': ['별로', '최악']}

    taste_good_emotion = ['고소', '부드럽', '신선', '촉촉', '싱싱', '정갈', '최고']
    taste_bad_emotion = ['싱겁', '느끼다하다', '짜다', '느끼다', '짜다', '딱딱하다', '차갑다']

    cost_good_features = {'가격': ['괜찮', '착하다', '저렴', '적당', '싸다', '좋다', '합리적', '훌륭', '최고', '만족', '마음', '든든', '알맞다',
                                 '무난', '괜춘', '최상', '최상', '굿', '엄지', '낮'],
                          '가성비': ['괜찮', '착하다', '저렴', '적당', '싸다', '좋다', '합리적', '훌륭', '최고', '만족', '마음', '든든', '알맞다',
                                  '무난', '괜춘', '최상', '최상', '굿', '엄지'],
                          '양': ['많', '적당', '푸짐하고', '괜찮다', '넉넉', '충분', '든든']
                          }
    cost_bad_features = {
        '가격': ['비싸', '있다', '나쁘', '사악', '비효율', '높다', '부담', '아쉽', '쏘다쏘다', '별로', '그닥', '그다지', '쎄', 'ㅎㄷㄷ', '높', '거품'],
        '가성비': ['별로'],
        '양': ['적다', '작다', '아쉽', '적다', '다소', '별로'],
    }

    atmosphere_good_features = {
        '분위기': ['좋', '괜찮', '조용', '깔끔', '적당', '깡패', '굉장', '아담', '완벽', '아기자기', '고급', '최고', '세련', '만족', '아늑', '훌륭', '예쁘',
                '이쁘', '짱',
                '심쿵', '따뜻', '깨끗', '독특', '매력', '모던', '취향저격', '맘', '마음', '클래식', '아름', '인상적', '귀엽', '포근'],
        '인테리어': ['좋', '괜찮', '조용', '깔끔', '적당', '깡패', '굉장', '아담', '완벽', '아기자기', '고급', '최고', '세련', '만족', '아늑', '훌륭', '예쁘',
                 '이쁘', '짱',
                 '심쿵', '따뜻', '깨끗', '독특', '매력', '모던', '취향저격', '맘', '마음', '클래식', '아름', '인상적', '귀엽', '포근']}
    atmosphere_bad_features = {'분위기': ['나쁘다', '바쁘다', '어수선하다', '이상하다', '촌스럽다', '별로', '부담스럽다', '시끄럽', '복잡'],
                               '인테리어': []}

    visit_good_features = {'의사': ['있다', '충만', '백프로', '백프롭', '많', '만땅', '마구', '그득', '만점', '넘침'],
                           '다시': ['가다', '오다', '방문', '찾다', '가보다', '한번', '갈다', '찾아가다', '가야지', '갈거다', '방문하다보고',
                                  '생각나다', '방문한다면', '와보고', '재방문', '접하다', '간다면', '갈다때가', '먹다고프다', '방문한다임', '오자고', '가기로',
                                  '갈다생각이다', '가면'],
                           '굳이': []}
    visit_bad_features = {'의사': ['글쎄'],
                          '굳이': ['다시', '많이', '여기까지', '줄서서', '찾아', '시키다', '가다', '찾다', '여기', '기다리다', '줄을', '사먹'],
                          '다시': []}

    negative_word_emotion = ['안', '않', '못', '없', '아닌', '아니']

    def __init__(self, post_list, store_name, store_ad):
        self.post_list = post_list
        self.store_name_list = store_name
        self.store_ad_list = store_ad

        self.kkma = Kkma()

    # 전처리
    # 영어, 한글 자음 또는 모음, 특수문자 제거
    def preprocessing(self, review):

        # 전처리 끝난 문장을 저장해놓을 string
        review_post = ''

        # 인풋리뷰
        for idx in range(len(review)):
            r = review[idx]
            sentence = re.sub('([a-zA-Z])', '', r)  # 영어거나
            sentence = re.sub('[ㄱ-ㅎㅏ-ㅣ]+', '', sentence)  # 한글 자음 또는 모음이거나
            sentence = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sentence)  # 특수 문자의 경우 지운다.

            if len(sentence) == 0:
                continue

            # 지운 문장 저장
            review_post += sentence

        return review_post

    # 감성사전의 keys가 포함되어있는 구문을 구하는 함수
    def get_feature_keys(self, feature_keys, review):
        feature_temp = []

        lst_conj = ['게', '고', '음', '며', '데', '만', '도', '면']
        for keys in feature_keys:

            # 감성사전의 keys(target)이 리뷰에 포함되어 있다면, 
            if re.findall(keys, review):

                # 접속사 뒤에 ','를 스페이스로 바꾸고
                for conj in lst_conj:
                    if conj + ' ' in review:
                        review = re.sub(conj + ' ', conj + ',', review)

                # 다음과 같은 세가지 경우의 구문을 걸러냄

                # target(한글) (한글) (한글)
                a = re.findall(keys + '+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+', review)

                # target (한글) (한글)
                b = re.findall(keys + '+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+', review)

                # (한글) target(한글)
                c = re.findall('[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+' + keys + '[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+', review)

                # 찾아낸 구문들을 target과 함께 feature_temp에 저장하고 return 함
                for ngram in a:
                    t = ()
                    feature_temp.append(t + (ngram, keys))
                for ngram in b:
                    t = ()
                    feature_temp.append(t + (ngram, keys))
                for ngram in c:
                    t = ()
                    feature_temp.append(t + (ngram, keys))

        return feature_temp

    # target에 대한 평가 구문들에서 긍정, 부정 평가 표현을 찾아내 구별하여 저장
    def get_feature_emotions(self, feature_good_dict, feature_bad_dict, feature_temp):
        good_feature_emotion_list = []
        bad_feature_emotion_list = []

        for ngrams in feature_temp:
            review = ngrams[0]
            target = ngrams[1]

            is_bad_feature = None

            good_emotion_list = feature_good_dict[target]
            bad_emotion_list = feature_bad_dict[target]

            # 긍정 평가 표현을 찾음
            for emotion in good_emotion_list:
                if re.findall(emotion, review):
                    is_bad_feature = False
                    break

            # 부정 평가 표현을 찾음
            for emotion in bad_emotion_list:
                if re.findall(emotion, review):
                    is_bad_feature = True
                    break

            # 부정 표현(negative_word_emotion 변수 확인)이 있다면 is_bad_feature 값을 반전을 취함
            for negative in self.negative_word_emotion:
                if re.findall(negative, review):
                    if is_bad_feature:
                        is_bad_feature = False
                        break
                    else:
                        is_bad_feature = True
                        break

            # bad라면 bad_feature_emotion_list에
            if is_bad_feature:
                bad_feature_emotion_list.append(review)
            # bad 평가가 아니라면 good_feature_emotion_list에 추가한다.
            elif not is_bad_feature:
                good_feature_emotion_list.append(review)
            else:
                pass

        # 긍정 평가 내용과, 부정 평가 내용을 구분한 리스트를 리턴한다.
        return good_feature_emotion_list, bad_feature_emotion_list

    # 맛에 대한 평가가 아닌, 맛이 어떠한지 찾는 함수
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

    # main 함수
    def start_score(self):
        store_score_tmp = []
        store_score_result_tmp = []

        c_store_name_list = Counter(self.store_name_list)

        for idx in range(len(self.post_list)):

            # 식당 이름이 없거나 광고인 경우 제외한다.
            if self.store_name_list[idx] == "" or self.store_ad_list[idx] == "TRUE":
                continue

            # 포스팅 내용과 제목을 store_content에 저장한 뒤
            # print(self.post_list[idx][3])
            store_content = [self.post_list[idx][3]]
            # print(self.post_list[idx][1])
            store_content.append(self.post_list[idx][1])

            # 전처리를 하고
            review = self.preprocessing(store_content)

            # 서비스, 분위기, 가격, 재방문의사, 맛에 대한 감성분석
            service_temp = self.get_feature_keys(self.service_good_features.keys(), review)
            good_service, bad_service = self.get_feature_emotions(self.service_good_features, self.service_bad_features,
                                                                  service_temp)

            atmosphere_temp = self.get_feature_keys(self.atmosphere_good_features.keys(), review)
            good_atmosphere, bad_atmosphere = self.get_feature_emotions(self.atmosphere_good_features,
                                                                        self.atmosphere_bad_features, atmosphere_temp)

            cost_temp = self.get_feature_keys(self.cost_good_features.keys(), review)
            good_cost, bad_cost = self.get_feature_emotions(self.cost_good_features, self.cost_bad_features, cost_temp)

            visit_temp = self.get_feature_keys(self.visit_good_features.keys(), review)
            good_visit, bad_visit = self.get_feature_emotions(self.visit_good_features, self.visit_bad_features,
                                                              visit_temp)

            taste_temp = self.get_feature_keys(self.taste_good_features.keys(), review)
            good_taste, bad_taste = self.get_feature_emotions(self.taste_good_features, self.taste_bad_features,
                                                              taste_temp)
            taste_good_emotion_temp = self.get_feature_keys(self.taste_good_emotion, review)
            taste_bad_emotion_temp = self.get_feature_keys(self.taste_bad_emotion, review)
            good_taste2, bad_taste2 = self.get_taste_emotion(taste_good_emotion_temp, taste_bad_emotion_temp)

            # print("#################", idx, "#################")
            # print(good_service, bad_service)
            # print(good_atmosphere, bad_atmosphere)
            # print(good_cost, bad_cost)
            # print(good_visit, bad_visit)
            # print(good_taste, bad_taste)
            # print(good_taste2, bad_taste2)

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
                    list_tmp.append(round(
                        (len(data[i][0]) / (len(data[i][0]) + len(data[i][1]))) * 100 * (math.log(c_name, 100) + 1), 1))
                    list_tmp.append(round(len(data[i][0]) / (len(data[i][0]) + len(data[i][1])) * 5, 1))
                else:
                    list_tmp.append(0)
                    list_tmp.append(0)

            store_score_result_tmp.append(list_tmp)

        return store_score_result_tmp
