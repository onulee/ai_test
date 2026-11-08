from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import requests
from bs4 import BeautifulSoup
import time
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# url = "https://namu.wiki/RecentChanges"
# options = Options()
# options.add_experimental_option("excludeSwitches", ["enable-automation"])
# options.add_experimental_option("useAutomationExtension", False)
# options.add_argument("--disable-blink-features=AutomationControlled")
# browser = webdriver.Chrome(options=options)
# browser.maximize_window() # 화면 최대창 확대
# browser.get(url)
# time.sleep(2)
# # 파일저장
# soup = BeautifulSoup(browser.page_source,'lxml')
# with open('lab02/file/namu1.html','w',encoding='utf-8') as f:
#     f.write(soup.prettify())
    
# 3. 파일 BeautifulSoup변환
with open('lab02/file/namu1.html','r',encoding='utf-8') as f:
    soup = BeautifulSoup(f,'lxml')   

page_url_base = "https://namu.wiki"
page_urls = []
    
# items = soup.find_all('div',{'class':'"div.dem\\+8eeF.-PZ4jqRx"'})     
# items = soup.find('div',{'class':'kmK+Qqcj yF3PgY1W'})     
data = soup.find('div',{'class':'qjuFv8xZ'})     
items = data.find('div',{'class':'rcjap5iR'})  #1개   
divs = items.find_all('div',{'class':'dem+8eeF -PZ4jqRx'})     
# print('개수 : ',len(divs)) #100개
# print(divs[0].find('a').get_text().strip()) #100개
print(divs[0].find('a')['href']) #100개
href = divs[0].find('a')['href']
print(page_url_base+href)
data = []
for i,div in enumerate(divs):
    title = div.find('a').get_text().strip()
    # print(div.find('a')['href']) #100개
    href = div.find('a')['href']
    page_url= page_url_base+href
    data.append({
        '제목':title,
        '링크':page_url
    })
    # page_urls.append(page_url_base+href)
    # 중복 url을 제거합니다.
    # page_urls = list(set(page_urls))
    # print(i+1,':',page_url_base+href)
    
df = pd.DataFrame(data)
print(df)

df.to_csv('lab02/file/namu_link.csv')





   
