import os
import sys
import ssl
import re
import urllib.request #파이썬에서 http 프로토콜에 따라 서버 응답 요청을 도와주는 모듈 request는 클라이언트의 요청을 처리
import json
from tqdm import tqdm

#contents 수집을 위한
import requests
from bs4 import BeautifulSoup


class BlogPostScrap:
    # 문서 스크래핑
    def blog_scraping(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"}
        res = requests.get(url, headers=headers)
        res.raise_for_status()  # 문제시 프로그램 종료
        soup = BeautifulSoup(res.text, "lxml")

        # 제목 수집
        global blog_title2
        ttext = soup.select_one('.se-title-text span')

        # 구버전 판별
        if ttext == None:
            ttext = soup.find("span", attrs={"class": "pcol1 itemSubjectBoldfont"})
            # 중간버전 판별
            if ttext == None:
                ttext = soup.find("h3", attrs={"class": "se_textarea"})
        if ttext != None:
            blog_title2 = ttext.get_text().strip()

        # 날짜 수집
        dtext = soup.find("span", attrs={"class": "se_publishDate"})

        # 구버전 판별
        if dtext == None:
            dtext = soup.find("p", {"class": "date fil5 pcol2 _postAddDate"}).get_text()
            dtext = re.sub(r'[^0-9]', '', dtext[:13])
            blog_date = dtext

        elif dtext != None:
            dtext = soup.find("span", attrs={"class": "se_publishDate"}).get_text()
            dtext = re.sub(r'[^0-9]', '', dtext[:13])
            blog_date = dtext

        # 본문 수집
        ptext = soup.find_all("p", attrs={"se-text-paragraph"})[1:]

        # 구버전 판별
        if len(ptext) == 0:
            ptext = soup.select('.post-view .view p')
            ptext += soup.select('.post-view .view > div > span')
            # 중간버전 판별
            if len(ptext) == 0:
                ptext = soup.select('.se_paragraph p')
                # 구버전의 다른 속성 판별
                if len(ptext) == 0:
                    ptext = soup.select('.post-view p')
                    ptext += soup.select('.post-view > span')
                    ptext += soup.select('.post-view > div')

        #     #해쉬태그 수집
        #     hash_tag1 = soup.find_all("span", attrs={"__se-hash-tag"})
        #     #구버전 판별
        #     if(len(hash_tag1)==0):
        #         hash_tag1 = soup.select('.item .ell')
        #         #본문내 해쉬태그판별
        #         if(len(hash_tag1)==0):
        #             for i in ptext:
        #                 hash_tag_test = i.get_text()
        #                 if hash_tag_test.startswith("#"):
        #                     hash_tag1 += i
        #             print(hash_tag1)

        #     hash_tag_result = ""

        #     #해쉬태그 작업
        #     for i in hash_tag1:
        #         hash_tag_result += i.get_text()

        #     hash_tag_result= hash_tag_result.replace('#',',')
        #     hash_tag_result = hash_tag_result[1:]

        # 이모지제거
        only_BMP_pattern = re.compile("["
                                      u"\U0001F600-\U0001F64F"  # emoticons
                                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                      u"\U00002702-\U000027B0"
                                      "]+", flags=re.UNICODE)

        # 이모지 후보1. 장점 : 이모지는 확실히 지워진다. 단점 : 다른 특수기호가 지워질수도
        # re.compile('[^ \.\,\?\!a-zA-Z0-9\u3131-\u3163\uac00-\ud7a3\\n]+')

        # 이모지 후보2 장점 : 이모지가 다 지워지지 않는다. 단점 : 모든 이모지가 제거되진 않는다.
        '''
        re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        '''

        # 본문에서 특수 공백을 제거한다.
        text = ""
        for i in ptext:
            if len(i.get_text()) > 300:
                text = "error"
                continue

            if i.get_text() == "\u200b":
                continue
            else:
                # 앞공백제거
                text += i.get_text().strip()

                # 앞공백을 제외한 글자 길이가 1이상일 때만
                if len(i.get_text().strip()) > 3:
                    # 리스트의 마지막 줄이 아닐때만
                    if i != ptext[-1]:
                        # \n;를 붙여준다.
                        text += "\n;"
                    else:
                        continue
                else:
                    continue
                text = only_BMP_pattern.sub(r'', text)

        #             #본문에서 해쉬태그를 제거한다.
        #             for j in hash_tag1:
        #                 k = j.get_text()
        #                 text = re.sub(k, '', text)

        # 마지막에 ;혼자있으면 제거
        if len(text) != 0:
            if text[-1] == ";":
                text = text[:-2]

        #     return text, blog_title2, blog_date, hash_tag_result
        return text, blog_title2, blog_date

    # 네이버의 url이 그대로 사용할 수 있는게 아닌, iframe을 통한 다른 url이 있었음.
    # iframe 제거 후 blog.naver.com 붙이는 함수
    def delete_iframe(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"}
        res = requests.get(url, headers=headers)
        res.raise_for_status()  # 문제시 프로그램 종료
        soup = BeautifulSoup(res.text, "lxml")

        src_url = "https://blog.naver.com/" + soup.iframe["src"]

        return src_url

    def get_map_title(self, url):
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
            return ""

    def start_scraping(self, location, food):
        # 중복된 검색에 대해서 시간 줄이기
        file_path = "./blog_data/" + location + "/" + food

        ssl._create_default_https_context = ssl._create_unverified_context

        client_id = "OuGFA4zkGrGJ0mavPWL5"        # 오픈 API Token ID
        client_secret = "bI05UsiyAI"              # 오픈 API Token Pass

        #URL 인코딩
        encText = urllib.parse.quote(location + food + "맛집")  # 검색을 원하는 문자열
        # encText = urllib.parse.quote("맛집"+"협찬")  # 검색을 원하는 문자열
        displayN = '100'                          # 검색 결과 출력 건수 지정
        startN = '1'                              # 검색 시작 위치로 최대 1000까지 가능
        sortN = 'sim'                            # 정렬 옵션: sim (유사도순), date (날짜순)

        url = "https://openapi.naver.com/v1/search/blog?query=" \
              + encText + "&display=" + displayN + "&start={}&sort=" + sortN

        total_data = []

        for i in range(0, 10):
            request = urllib.request.Request(url.format(str(100*i + 1)))
            request.add_header("X-Naver-Client-Id", client_id)
            request.add_header("X-Naver-Client-Secret", client_secret)
            response = urllib.request.urlopen(request)

            response_body = response.read()
            data_json = json.loads(response_body.decode('utf-8'))
            data_list = [data["link"] for data in data_json.get("items")]



            total_data += data_list

        #print("총 읽어온 갯수:", len(total_data))
        #print("중복 제거한 갯수:", len(set(total_data)))

        find_url = self.delete_iframe("https://blog.naver.com/missms9307/222342456025")
        blog_content, blog_title, blog_date = self.blog_scraping(find_url)

        #print(blog_title)
        #print(blog_date)
        #print(blog_content)

        # 제목저장
        num = 0

        # 위치, 음식별 폴더를 생성한다.
        os.makedirs(file_path)

        for i in tqdm(total_data):
            # 각 필요요소를 변수에 담는다.

            number = str(num)

            # 블로그 결과만을 취급하기 위함
            if i.startswith("https://blog.naver.com/"):

                # 해당 url을 함수에 담는다.
                find_url = self.delete_iframe(i)
                # 블로그를 스크롤링하여 필요한 정보를 가져온다.
                #         blog_content, blog_title, blog_date, hash_tag_result = blog_scraping(find_url)
                blog_content, blog_title, blog_date = self.blog_scraping(find_url)
                rest_name = self.get_map_title(i)

                # 맛집 모음, 브이로그 일상 포스팅을 제외하기 위함
                if "리스트" in blog_title:
                    continue
                if "브이로그" in blog_title:
                    continue
                if "모음" in blog_title:
                    continue

                if len(blog_content) < 100:
                    continue

                if blog_content == "error":
                    continue

                # 텍스트 파일로 저장한다.
                filename = file_path + "/" + 'blog_data_' + number + '.txt'

                f = open(filename, 'w', encoding="UTF-8")

                f.write(';' + number + '\n')
                f.write('$\n')
                f.write(';' + blog_title + '\n')
                f.write('$\n')
                f.write(';' + rest_name + '\n')
                f.write('$\n')
                f.write(';' + blog_content + '\n')
                #         f.write('$\n')
                #         f.write(';'+hash_tag_result+'\n')
                f.write('$\n')
                f.write(';' + blog_date + '\n')

            else:
                continue

            f.close()
            num += 1
