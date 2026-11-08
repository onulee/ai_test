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
data = []
df = pd.read_csv('lab02/file/namu_link.csv')
# print(df)
# print(df['링크'])
for i,url in enumerate(df['링크']):
    print(url)
    if i==50:
        break
    # url = df['링크'][0]
    options = Options()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--disable-blink-features=AutomationControlled")
    browser = webdriver.Chrome(options=options)
    browser.maximize_window() # 화면 최대창 확대
    browser.get(url)
    time.sleep(2)
    # 파일저장
    soup = BeautifulSoup(browser.page_source,'lxml')
# with open('lab02/file/namu_category1.html','w',encoding='utf-8') as f:
#     f.write(soup.prettify())
    
# 3. 파일 BeautifulSoup변환
# with open('lab02/file/namu_category1.html','r',encoding='utf-8') as f:
#     soup = BeautifulSoup(f,'lxml') 
    try:
        title = soup.find('h1',{'class':'wxxx093Y'}).get_text(strip=True)    
        print(title)

        category = soup.find('ul',{'class':'Tj1tYIlS'}).get_text(strip=True)  
        print(category)  

        content_text = soup.find('div',{'class':'_37JOhaLP x9cmj5+d'}).get_text(strip=True)
        print(content_text)
        data.append({
            'title':title,
            'category':category,
            'content_text':content_text
        })
    except Exception as e:
        print(e)

df = pd.DataFrame(data)
df.to_csv('lab02/file/namu_content.csv')