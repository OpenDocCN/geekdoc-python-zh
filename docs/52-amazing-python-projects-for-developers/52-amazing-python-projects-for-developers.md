

# 52个令人惊叹的Python项目，献给开发者

EDCORNER LEARNING

# 52个令人惊叹的Python项目，献给开发者

Edcorner Learning

# 目录

**简介**

**模块一 项目 1-10**

- 1. LinkedIn邮箱抓取器
- 2. Cricbuzz数据抓取器
- 3. 歌词下载器
- 4. 合并CSV文件
- 5. 合并PDF文件
- 6. 垃圾短信检测器
- 7. 电影信息抓取器
- 8. 电影信息Telegram机器人
- 10. Python音乐播放器

**模块二 项目 11-20**

- 11. 带语音的新闻更新器
- 12. 新闻抓取器
- 13. 噪声消除器
- 14. NSE股票数据
- 15. 猜数字游戏
- 16. 文件分类器
- 17. PageSpeed工具
- 18. 绘图应用
- 19. 密码管理器
- 20. 密码管理器图形界面

**模块三 项目 21-30**

- 21. 可朗读的PDF与图片文本阅读器
- 22. PDF转CSV转换器
- 23. 抄袭检查器
- 24. 带图形界面的番茄钟
- 25. Pyduku
- 26. PYQT5密码生成器
- 27. Python自动绘图
- 28. Pyweather天气应用
- 29. 使用Python生成二维码
- 30. 竞赛条形图动画

**模块四 项目 31-40**

- 31. 随机密码生成器
- 32. 随机维基百科文章
- 33. 从列表中随机选取单词
- 34. 高质量YouTube视频下载器
- 35. 树莓派-Sonoff控制
- 36. 递归密码生成器
- 37. 无需API的Reddit抓取器
- 38. 图片尺寸压缩器
- 39. 石头剪刀布游戏
- 40. 使用笔记本摄像头实现房间安防

**模块五 项目 41-50**

- 41. 从HackerNews网站抓取新闻
- 42. 名言抓取器
- 43. 抓取Medium文章
- 44. 屏幕录制器
- 45. 使用Python发送邮件
- 46. 从CSV文件发送邮件
- 47. 发送短信
- 48. 设置闹钟
- 49. 关机或重启你的设备
- 50. 正弦与余弦

51. 网站屏蔽器

52. 短信自动化

如何下载此项目：

# 简介

Python是一种通用的、解释型的、交互式的、面向对象的、具有动态语义的强大编程语言。它易于学习并精通。Python是少数几种既声称简单又强大的语言之一。Python优雅的语法、动态类型以及其解释型特性，使其成为在众多大型平台上进行脚本编写和健壮应用开发的理想语言。

Python通过模块和包来辅助开发，这促进了程序的模块化和代码重用。Python解释器以及广泛的标准化库，均可在所有关键平台上免费获取源代码或二进制形式，并可自由分发。学习Python不需要任何先决条件。然而，应该对编程语言有基本的理解。

**本书包含52个Python项目，适合所有开发者/学生练习不同的项目和场景。将这些知识应用于专业任务或日常学习项目中。**

**在本书末尾，你可以通过我们的链接下载所有这些项目。**

所有52个项目被划分为不同的模块，每个项目在开发者执行日常任务方面都有其独特之处。每个项目都有其源代码，学习者可以复制并在自己的系统上练习/使用。如果任何项目有特殊要求，书中已作说明。

学习愉快！！

# 模块一 项目 1-10

## 1. LinkedIn邮箱抓取器

### 前提条件：

- 1. 执行 `pip install -r requirements.txt` 以确保你拥有必要的库。
- 2. 确保你已安装 **chromedriver** 并将其添加到系统路径。
- 3. 准备好你想要抓取的LinkedIn帖子的 **URL**（*确保该帖子在评论区有一些邮箱地址*）
- 4. 准备好你的 **LinkedIn** 账户凭据

### 执行应用程序

- 1. 将代码中URL、email和password变量的值替换为你自己的数据
- 2. 如果你的IDE有运行选项，点击 **运行**，或者直接在终端输入 `python main.py`。
- 3. 从帖子中抓取到的姓名和对应的邮箱地址应出现在 **emails.csv** 文件中。

**依赖项：**
**selenium**
**email-validator**

### 源代码：

```python
from selenium import webdriver
from email_validator import validate_email, EmailNotValidError
import csv

def LinkedInEmailScraper(userEmail, userPassword):
    emailList = {}

    browser = webdriver.Chrome()
    # 示例 => 'https://www.linkedin.com/posts/faangpath_hiring-womxn-ghc2020-activity-6721287139721650176-QFCV/'
    url = '[插入LinkedIn帖子的URL]'
    browser.get(url) # 访问目标帖子的页面

    browser.implicitly_wait(5)

    commentDiv = browser.find_element_by_xpath(
        '/html/body/main/section[1]/section[1]/div/div[3]/a[2]'
    ) # 查找评论按钮
    loginLink = commentDiv.get_attribute('href')
    browser.get(loginLink)

    email = browser.find_element_by_xpath('//*[@id="username"]')
    password = browser.find_element_by_xpath('//*[@id="password"]')
    email.send_keys(userEmail) # 在邮箱字段输入邮箱
    password.send_keys(userPassword) # 在密码字段输入密码
    submit = browser.find_element_by_xpath(
        '//*[@id="app__container"]/main/div[3]/form/div[3]/button')
    submit.submit() # 提交表单

    browser.implicitly_wait(5)

    commentSection = browser.find_element_by_css_selector(
        '.comments-comments-list') # 查找评论区

    for _ in range(
        3
    ): # 这也可以设置为任何数字，或者如果你想搜索整个帖子的评论区，可以设置为 "while True"
        try:
            moreCommentsButton = commentSection.find_element_by_class_name(
                'comments-comments-list__show-previous-container'
            ).find_element_by_tag_name('button')
            moreCommentsButton.click()
            browser.implicitly_wait(5)
        except:
            print('评论检查结束')
            break

    browser.implicitly_wait(20)

    comments = commentSection.find_elements_by_tag_name(
        'article') # 查找所有单独的评论

    for comment in comments:
        try:
            commenterName = comment.find_element_by_class_name(
                'hoverable-link-text') # 查找评论者姓名
            commentText = comment.find_element_by_tag_name('p')
            commenterEmail = commentText.find_element_by_tag_name(
                'a').get_attribute('innerHTML') # 查找评论者邮箱
            # 验证邮箱地址
            validEmail = validate_email(commenterEmail)
            commenterEmail = validEmail.email
        except:
            continue

        emailList[commenterName.get_attribute('innerHTML')] = commenterEmail

    browser.quit()
    return emailList

def DictToCSV(input_dict):
    """
    将字典转换为csv
    """
    with open('./LinkedIn Email Scraper/emails.csv', 'w') as f:
        f.write('name,email\n')
        for key in input_dict:
            f.write('%s,%s\n' % (key, input_dict[key]))
        f.close()

if __name__ == '__main__':
    userEmail = '[插入你的LinkedIn账户邮箱地址]'
    userPassword = '[插入你的LinkedIn账户密码]'

    emailList = LinkedInEmailScraper(userEmail, userPassword)
    DictToCSV(emailList)
```

## 2. Cricbuzz数据抓取器

这个Python脚本将抓取cricbuzz.com以获取比赛的实时比分。

### 设置

- 安装依赖项

    `pip install -r requirements.txt`
- 运行文件

    `python live_score.py`

**依赖项：**
beautifulsoup4==4.9.3
bs4==0.0.1
pypiwin32==223
pywin32==228
soupsieve==2.0.1

urllib3==1.26.5
win10toast==0.9

```python
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from win10toast import ToastNotifier
import time

URL = 'http://www.cricbuzz.com/cricket-match/live-scores'

def notify(title, score):
    # Windows桌面通知功能
    toaster = ToastNotifier()
    # toaster.show_toast(score, "Get! Set! GO!", duration=5,icon_path='cricket.ico')
    toaster.show_toast("CRICKET LIVE SCORE",
                       score,
                       duration=30,
                       icon_path='ipl.ico')

while True:
    request = Request(URL, headers={'User-Agent': 'XYZ/3.0'})
    response = urlopen(request, timeout=20).read()
```

## 3. 歌词下载

此脚本可用于下载任意数量歌手的任意数量歌曲的歌词，直至达到API使用限制。

该脚本使用 [Genius API](https://docs.genius.com/)。这是一个专门的音乐平台。

### 设置说明

- 你需要一个API客户端（免费），请按照[此处](https://docs.genius.com/)的步骤操作。
- 使用 `pip install lyricsgenius` 安装专用包。
- 准备就绪，请遵循代码中注释提到的指南。
- 该脚本具有很强的交互性，请确保遵循指南。

### 源代码：

```python
import lyricsgenius as lg

# File for writing the Lyrics
filename = input('Enter a filename: ') or 'Lyrics.txt'
file = open(filename, "w+")

# Acquire a Access Token to connect with Genius API
genius = lg.Genius(
    'Client_Access_Token_Goes_Here',
    # Skip song listing
    skip_non_songs=True,
    # Terms that are redundant song names with same lyrics, e.g. Old Town Raod and Old Town Road Remix
    # have same lyrics
    excluded_terms=["(Remix)", "(Live)"],
    # In order to keep headers like [Chorus], [Bridge] etc.
    remove_section_headers=True)

# List of Artist and Maximum Songs
input_string = input("Enter name of Artists separated by spaces: ")
artists = input_string.split(" ")

def get_lyrics(arr, max_song):
    """
    Returns: Number of songs grabbed by Function
    Saves : Text File with Lyrics
    Parameters :
        arr : Artist
        max_song : Number of maximum songs to be grabbed
    """
    # Write lyrics of k songs by each artist in arr
    c = 0
    # A counter
    for name in arr:
        try:
            songs = (genius.search_artist(name,
                                          max_songs=max_song,
                                          sort='popularity')).songs
            s = [song.lyrics for song in songs]
            # A custom delimiter
            file.write("\n \n <|endoftext|> \n \n".join(s))
            c += 1
            print(f"Songs grabbed:{len(s)}")
        except:
            print(f"some exception at {name}: {c}")

# Function Call
get_lyrics(artists, 3)
```

## 4. 合并CSV文件

借助以下简单的Python脚本，可以合并目录中存在的CSV文件。

### 依赖项

需要Python 3和`pandas`

安装依赖：`pip install -r "requirements.txt"`

或

安装pandas：`pip install pandas`

### 如何使用

### 运行

将所有需要合并的CSV文件放入包含该脚本的目录中。

可以从代码编辑器或IDE运行，或者在命令行中输入 `python merge_csv_files.py`。

最终输出将是同一目录下的 `combined_csv.csv` 文件。

**依赖：**

**pandas==1.1.0**

**源代码：**

```python
import glob
import pandas as pd

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
```

## 5. 合并PDF

一个简单的Python脚本，执行后可合并两个PDF文件。

### 先决条件

运行 - "pip install PyPDF2"

### 如何运行脚本

可以通过运行 "python merge_pdfs.py" 来执行。

**依赖：**
**PyPDF2==1.26.0**

**源代码：**

```python
#!/usr/bin/env python

from PyPDF2 import PdfFileMerger


# By appending in the end
def by_appending():
    merger = PdfFileMerger()
    # Either provide file stream
    f1 = open("samplePdf1.pdf", "rb")
    merger.append(f1)
    # Or direct file path
    merger.append("samplePdf2.pdf")

    merger.write("mergedPdf.pdf")


# By inserting at after an specified page no.
def by_inserting():
    merger = PdfFileMerger()
    merger.append("samplePdf1.pdf")
    merger.merge(0, "samplePdf2.pdf")
    merger.write("mergedPdf1.pdf")

if __name__ == "__main__":
    by_appending()
    by_inserting()
```

## 6. 垃圾短信检测

包/脚本的简短描述

- 使用的库：
    - pandas
    - string
    - re
    - nltk
    - sklearn
    - pickle

- Python代码包含基于Kaggle数据集（数据集链接在代码内）的垃圾短信检测脚本。

### 设置说明

- 下载代码
- 下载数据集
- 运行笔记本中的单元格

### 脚本的详细解释（如需要）

无

### 输出

- Hello, I am James Bond: Not Spam
- Winner! Winner! Winner! Congrats! Call at xyz or email us at to claim your prize! Limited Time Offer!: Spam

### 垃圾短信检测源代码：

```python
# importing required libraries
import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
import re
warnings.filterwarnings("ignore")

# reading the dataset
msg = pd.read_csv(
    "./Message_Spam_Detection/Cleaned_Dataset.csv", encoding='latin-1')

msg.drop(['Unnamed: 0'], axis=1, inplace=True)

# seperating target and features
y = pd.DataFrame(msg.label)
x = msg.drop(['label'], axis=1)

# countvectorization
cv = CountVectorizer(max_features=5000)
temp1 = cv.fit_transform(x['final_text'].values.astype('U')).toarray()
tf = TfidfTransformer()
temp1 = tf.fit_transform(temp1)
temp1 = pd.DataFrame(temp1.toarray(), index=x.index)
x = pd.concat([x, temp1], axis=1, sort=False)

# drop final_text col
x.drop(['final_text'], axis=1, inplace=True)

# converting to int datatype
y = y.astype(int)

# randomforstclassifier model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(x, y)

# User input
text = input("Enter text: ")

# data cleaning/preprocessing - removing punctuation and digits
updated_text = ""
for i in range(len(text)):
    if text[i] not in string.punctuation:
        if text[i].isdigit() == False:
            updated_text = updated_text+text[i]
text = updated_text

# data cleaning/preprocessing - tokenization and convert to lower case
text = re.split("\W+", text.lower())

# data cleaning/preprocessing - stopwords
updated_list = []
stopwords = nltk.corpus.stopwords.words('english')
for i in range(len(text)):
    if text[i] not in stopwords:
        updated_list.append(text[i])
text = updated_list

# data cleaning/preprocessing - lemmentizing
updated_list = []
wordlem = nltk.WordNetLemmatizer()
for i in range(len(text)):
    updated_list.append(wordlem.lemmatize(text[i]))
text = updated_list

# data cleaning/preprocessing - mergining token
text = " ".join(text)

text = cv.transform([text])
text = tf.transform(text)
pred = model.predict(text)
if pred == 0:
    print("Not Spam")
else:
    print("Spam")
```

### 数据清洗源代码：

```python
# importing required libraries
import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
import re
warnings.filterwarnings("ignore")
nltk.download('stopwords')
nltk.download('wordnet')

# reading the dataset
# dataset: https://www.kaggle.com/uciml/sms-spam-collection-dataset
msg = pd.read_csv("./Message_Spam_Detection/dataset.csv", encoding='latin-1')
msg.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
msg.rename(columns={"v1": "label", "v2": "text"}, inplace=True)

# mapping ham=0 and spam=1
for i in msg.index:
    if msg['label'][i] == 'ham':
```

msg['label'][i] = 0
else:
    msg['label'][i] = 1

# 删除重复列
msg = msg.drop_duplicates()

# 数据清洗/预处理 - 移除标点符号和数字
msg['cleaned_text'] = ""
for i in msg.index:
    updated_list = []
    for j in range(len(msg['text'][i])):
        if msg['text'][i][j] not in string.punctuation:
            if msg['text'][i][j].isdigit() == False:
                updated_list.append(msg['text'][i][j])
    updated_string = "".join(updated_list)
    msg['cleaned_text'][i] = updated_string
msg.drop(['text'], axis=1, inplace=True)

# 数据清洗/预处理 - 分词并转换为小写
msg['token'] = ""
for i in msg.index:
    msg['token'][i] = re.split("\W+", msg['cleaned_text'][i].lower())

# 数据清洗/预处理 - 停用词
msg['updated_token'] = ""
stopwords = nltk.corpus.stopwords.words('english')
for i in msg.index:
    updated_list = []
    for j in range(len(msg['token'][i])):
        if msg['token'][i][j] not in stopwords:
            updated_list.append(msg['token'][i][j])
    msg['updated_token'][i] = updated_list
msg.drop(['token'], axis=1, inplace=True)

# 数据清洗/预处理 - 词形还原
msg['lem_text'] = ""
wordlem = nltk.WordNetLemmatizer()
for i in msg.index:
    updated_list = []
    for j in range(len(msg['updated_token'][i])):
        updated_list.append(wordlem.lemmatize(msg['updated_token'][i][j]))
    msg['lem_text'][i] = updated_list
msg.drop(['updated_token'], axis=1, inplace=True)

# 数据清洗/预处理 - 合并词元
msg['final_text'] = ""
for i in msg.index:
    updated_string = " ".join(msg['lem_text'][i])
    msg['final_text'][i] = updated_string
msg.drop(['cleaned_text', 'lem_text'], axis=1, inplace=True)

# 清洗后的数据集
msg.to_csv('Cleaned_Dataset.csv')

## 7. 电影信息爬虫

此脚本通过爬取 IMDB 网站获取电影详情。

### 前提条件

- beautifulsoup4
- requests
- 运行 `pip install -r requirements.txt` 以安装所需的外部模块。

### 如何运行脚本

执行 `python3 movieInfoScraper.py` 并在提示时输入电影名称。

**要求：**
- beautifulsoup4
- requests==2.23.0

**源代码：**

```python
import os
import zipfile
import sys
import argparse

# 添加命令行界面的代码
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--zippedfile", required=True, help="Zipped file")
args = vars(parser.parse_args())

# 捕获用户定义的 zip 文件
zip_file = args['zippedfile']

file_name = zip_file

# 检查输入的 zip 文件是否存在于目录中
if os.path.exists(zip_file) == False:
    sys.exit("No such file present in the directory")

# 解压 zip 文件的函数
def extract(zip_file):
    file_name = zip_file.split(".zip")[0]
    if zip_file.endswith(".zip"):
        # 将使用此路径将解压后的文件保存在当前目录
        current_working_directory = os.getcwd()
        new_directory = current_working_directory + "/" + file_name
        # 解压文件的逻辑
        with zipfile.ZipFile(zip_file, 'r') as zip_object:
            zip_object.extractall(new_directory)
        print("Extracted successfully!!!")
    else:
        print("Not a zip file")

extract(zip_file)
```

## 8. 电影信息 Telegram 机器人

### 描述

一个使用 Python 制作的 Telegram 机器人，它爬取 IMDb 网站并具有以下功能：

1. 回复电影名称，提供电影的类型和评分。
2. 回复类型，提供属于该类型的热门电影和电视剧列表。

### 设置说明

1. 安装所需的包：
    ```
    pip install -r requirements.txt
    ```
2. 在 Telegram 中创建一个机器人：
    1. 前往 @BotFather，点击 /start，然后输入 /newbot 并为其命名。
    2. 选择一个用户名并获取令牌。
3. 将令牌粘贴到 .env 文件中（以 [.env.example](.env.example) 为例）。
4. 运行 Python 脚本以启动机器人。
5. 输入 /start 命令以开始与聊天机器人对话。
6. 输入 /name <movie_name> 以获取电影的类型和评分。机器人会回复大约三个结果。
7. 输入 /genre <genre> 以获取属于该类型的电影和电视剧列表。

**要求：**
- APScheduler==3.6.3
- beautifulsoup4==4.9.3
- certifi==2020.12.5
- python-dateutil==2.8.1
- python-decouple==3.4
- python-telegram-bot==13.1
- pytz==2020.4
- requests==2.25.1
- six==1.15.0
- soupsieve==2.1
- tornado==6.1
- urllib3==1.26.2

**源代码：**

```python
import logging
import requests
import re
import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
import ssl
import itertools
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import decouple

# 启用日志记录
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

TOKEN = decouple.config("API_KEY")

# 定义一些命令处理器。它们通常接受两个参数：update 和 context。错误处理器也会在错误中接收引发的 TelegramError 对象。

def start(update, context):
    """当发出 /start 命令时发送消息。"""
    update.message.reply_text(
        '这个机器人能做什么？\n\n这个机器人从 IMDb 网站提供任何电影的简要信息'
        + '\n发送 /name movie_name 以了解电影的类型和评分。\n发送 /genre genre_name 以'
        + '获取属于该类型的电影列表'
    )

def help(update, context):
    """当发出 /help 命令时发送消息。"""
    update.message.reply_text('帮助！')

def genre(update, context):
    """当发出 /genre 命令时发送电影列表。"""
    url = 'https://www.imdb.com/search/title/'
    genre = str(update.message.text)[7:]
    print(genre)
    r = requests.get(url+'?genres='+genre)
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.find('title')
    if title.string == 'IMDb: Advanced Title Search - IMDb':
        update.message.reply_text("抱歉，没有这种类型。请重试")
    else:
        res = []
        res.append(title.string+'\n')
        tags = soup('a')
        for tag in tags:
            movie = re.search('<a href="/title/.*>(.*?)</a>', str(tag))
            try:
                if "&amp;" in movie.group(1):
                    movie.group(1).replace("&amp;", "&")
                res.append(movie.group(1))
            except:
                pass
        stri = ""
        for i in res:
            stri += i+'\n'
        update.message.reply_text(stri)

def name(update, context):
    """当发出 /name 命令时，在 IMDb 网站发送电影名称的前 3 个搜索结果。"""
    movie = str(update.message.text)[6:]
    print(movie)
    res = get_info(movie)
    stri = ""
    for i in res:
        for a in i:
            stri += a+'\n'
        stri += '\n'
    update.message.reply_text(stri)

def error(update, context):
    """记录由更新引起的错误。"""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def get_info(movie):
    """爬取 IMDb 并获取电影的类型和评分"""
    url = 'https://www.imdb.com/find?q='
    r = requests.get(url+movie+'&ref_=nv_sr_sm')
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.find('title')
    tags = soup('a')
    pre_url = ""
    count = 0
    lis = []
    res = []
    for tag in tags:
        if(count > 2):
            break
        m = re.search('<a href=.*>(.*?)</a>', str(tag))
        try:
            lis = []
            link = re.search('/title/(.*?)/', str(m))
            new_url = 'https://www.imdb.com'+str(link.group(0))
            if new_url != pre_url:
                html = requests.get(new_url)
                soup2 = BeautifulSoup(html.text, "html.parser")
                movietitle = soup2.find('title').string.replace('- IMDb', ' ')
                a = soup2('a')
                span = soup2('director')
                for item in span:
                    print(item)
                genrestring = "类型 : "
                for j in a:
                    genre = re.search(
                        '<a href="/search/title\?genres=.*> (.*?)</a>', str(j))
                    try:
                        genrestring += genre.group(1)+' '
                    except:
                        pass
                atag = soup2('strong')
                for i in atag:
                    rating = re.search('<strong title="(.*?) based', str(i))
                    try:
                        rstring = "IMDb 评分 : "+rating.group(1)
                    except:
                        pass
                details = "更多详情 : "+new_url
                lis.append(movietitle)
                lis.append(genrestring)
                lis.append(rstring)
                lis.append(details)
                pre_url = new_url
                count += 1
```

## 9. 网站快照工具

### 设置

```
pip install selenium
pip install chromedriver-binary==XX.X.XXXX.XX.X
```

- 'XX.X.XXXX.XX.X' 是 Chrome 驱动程序的版本号。
- 'chrome driver' 的版本需要与你的 Google Chrome 浏览器版本相匹配。

*如何查找你的 Google Chrome 版本*

1.  点击屏幕右上角的菜单图标。
2.  点击“帮助”，然后选择“关于 Google Chrome”。
3.  你的 Chrome 浏览器版本号可以在这里找到。

### 执行

```
python snapshot_of_given_website.py <url>
```

运行此脚本后，快照将保存在当前目录中。

**依赖要求：**
**selenium==3.141.0**
**chromedriver-binary==85.0.4183.38.0**

**源代码：**

```python
# -*- coding: utf-8 -*-
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_binary

script_name = sys.argv[0]

options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

try:
    url = sys.argv[1]

    driver.get(url)
    page_width = driver.execute_script('return document.body.scrollWidth')
    page_height = driver.execute_script('return document.body.scrollHeight')
    driver.set_window_size(page_width, page_height)
    driver.save_screenshot('screenshot.png')
    driver.quit()
    print("SUCCESS")

except IndexError:
    print('Usage: %s URL' % script_name)
```

## 10. Python 音乐播放器

```python
import pygame
import tkinter as tkr
from tkinter.filedialog import askdirectory
import os

music_player = tkr.Tk()
music_player.title("My Music Player")
music_player.geometry("450x350")
directory = askdirectory()
os.chdir(directory)
song_list = os.listdir()

play_list = tkr.Listbox(music_player, font="Helvetica 12 bold", bg='yellow', selectmode=tkr.SINGLE)
for item in song_list:
    pos = 0
    play_list.insert(pos, item)
    pos += 1
pygame.init()
pygame.mixer.init()

def play():
    pygame.mixer.music.load(play_list.get(tkr.ACTIVE))
    var.set(play_list.get(tkr.ACTIVE))
    pygame.mixer.music.play()
def stop():
    pygame.mixer.music.stop()
def pause():
    pygame.mixer.music.pause()
def unpause():
    pygame.mixer.music.unpause()
Button1 = tkr.Button(music_player, width=5, height=3, font="Helvetica 12 bold", text="PLAY", command=play,
bg="blue", fg="white")
Button2 = tkr.Button(music_player, width=5, height=3, font="Helvetica 12 bold", text="STOP", command=stop,
bg="red", fg="white")
Button3 = tkr.Button(music_player, width=5, height=3, font="Helvetica 12 bold", text="PAUSE",
command=pause, bg="purple", fg="white")
Button4 = tkr.Button(music_player, width=5, height=3, font="Helvetica 12 bold", text="UNPAUSE",
command=unpause, bg="orange", fg="white")

var = tkr.StringVar()
song_title = tkr.Label(music_player, font="Helvetica 12 bold", textvariable=var)

song_title.pack()
Button1.pack(fill="x")
Button2.pack(fill="x")
Button3.pack(fill="x")
Button4.pack(fill="x")
play_list.pack(fill="both", expand="yes")
music_player.mainloop()
```

# 模块 2 项目 11-20

## 11. 带语音功能的新闻更新器

### 从这里获取 News API 密钥：
[NewsAPI](https://newsapi.org/)

### 添加你的 API 密钥：

```python
newsapi = NewsApiClient(api_key='在此处添加')
```

### 依赖项：

*Newsapi*

```
pip install newsapi
```

*pyttsx3*

```
pip install pyttsx3
```

*pyaudio*

```
pip install pyaudio
```

### 源代码：

```python
from newsapi import NewsApiClient
import pyttsx3
import speech_recognition as sr
from time import sleep

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def new():
    newsapi = NewsApiClient(api_key="")# 添加你的 api 密钥
    data = newsapi.get_top_headlines(q='corona',country='in',
                                    language='en',
                                    page_size=5)

    at = data['articles']

    for x,y in enumerate(at):
        print(f'{x} {y["description"]}')
        speak(f'{x} {y["description"]}')

    speak("that's it for now i'll updating you in some time ")


if __name__ == "__main__":
    while True:
        new()
        sleep(600)
```

## 12. 新闻抓取器

### 金融新闻抓取器
一个使用 Python 中的 beautiful soup 4 制作的抓取器。专为从 moneycontrol.com 提取新闻而定制。如需不同的抓取器，请提交拉取请求。

__开始抓取的主页：https://www.moneycontrol.com/news/technical-call-221.html__
![](images/home.JPG)

__程序通过提取这些按钮中的网站链接来抓取后续页面的新闻__
![](images/nextpage.JPG)

__生成的 JSON 文件包含标题、日期和图片链接，并按页码索引__
![](images/result.JPG)

### 源代码：

```python
import re
import json
import requests
import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict

submission = defaultdict(list)
#main url
src_url = 'https://www.moneycontrol.com/news/technical-call-221.html'

#get next page links and call scrap() on each link
def setup(url):
    nextlinks = []
    src_page = requests.get(url).text
    src = BeautifulSoup(src_page, 'lxml')

    #ignore <a> with void js as href
    anchors = src.find("div", attrs={"class": "pagenation"}).findAll(
        'a', {'href': re.compile('^(?!void).)*$')})
    nextlinks = [i.attrs['href'] for i in anchors]
    for idx, link in enumerate(tqdm(nextlinks)):
        scrap('https://www.moneycontrol.com'+link, idx)

#scraps passed page url
def scrap(url, idx):
    src_page = requests.get(url).text
    src = BeautifulSoup(src_page, 'lxml')

    span = src.find("ul", {"id": "cagetory"}).findAll('span')
    img = src.find("ul", {"id": "cagetory"}).findAll('img')

    #<img> has alt text attr set as heading of news, therefore get img link and heading from same tag
    imgs = [i.attrs['src'] for i in img]
    titles = [i.attrs['alt'] for i in img]
    date = [i.get_text() for i in span]

    #list of dicts as values and indexed by page number
    submission[str(idx)].append({'title': titles})
    submission[str(idx)].append({'date': date})
    submission[str(idx)].append({'img_src': imgs})

    #save data as json named by current date
    def json_dump(data):
        date = datetime.date.today().strftime("%B %d, %Y")
        with open('moneycontrol_'+str(date)+'.json', 'w') as outfile:
            json.dump(submission, outfile)

    setup(src_url)
    json_dump(submission)
```

## 13. 降噪

降噪脚本
实现一个功能，通过减少背景噪音来过滤音频文件，类似于“Audacity”。

### 使用的库

首先导入以下 Python 库

- NumPy
- scipy.io.wavfile
- Matplotlib
- Os

将音频文件和你的代码保存在同一文件夹中
运行 Python 代码

### “降噪脚本”所用方法的详细解释

- 导入所需的库（NumPy、scipy.io.wavfile 和 Matplotlib）
- 使用 scipy.io.wavfile 库读取输入音频文件
- 将音频文件转换为包含给定音频文件所有信息的数组，并初始化帧值。
- 计算带噪音频文件每个窗口的第一次傅里叶变换
- 从输入频谱中减去噪声频谱均值，并进行逆短时傅里叶变换（istft）
- 最终得到一个背景噪音大幅减少的音频文件

**依赖要求：**
**matplotlib==3.2.2**
**numpy==1.18.5**
**scipy==1.5.0**

**源代码：**

```python
# Spectral Subtraction: Method used for noise reduction
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt

file = input("Enter the file path: ")
sr, data = wav.read(file)
fl = 400 #frame_length
frames = [] #empty list
for i in range(0,int(len(data)/(int(fl/2))-1)):
    arr = data[int(i*int(fl/2)):int(i*int(fl/2)+fl)]
    frames.append(arr) #appending each array data into the frames list
```

frames = np.array(frames)  # 将帧列表转换为数组
ham_window = np.hamming(fl)  # 使用np.hamming
windowed_frames = frames * ham_window  # 将帧数组与ham_window相乘
dft = []  # 空列表，用于存储windowed_frames的FFT结果
for i in windowed_frames:
    dft.append(np.fft.fft(i))  # 现在对每个窗口进行傅里叶变换
dft = np.array(dft)  # 将dft转换为数组

dft_mag_spec = np.abs(dft)  # 将dft转换为绝对值
dft_phase_spec = np.angle(dft)  # 计算dft的角度
noise_estimate = np.mean(dft_mag_spec, axis=0)  # 求均值
noise_estimate_mag = np.abs(noise_estimate)  # 取绝对值

estimate_mag = (dft_mag_spec - 2 * noise_estimate_mag)  # 减法方法
estimate_mag[estimate_mag < 0] = 0
estimate = estimate_mag * np.exp(1j * dft_phase_spec)  # 计算最终估计值
ift = []  # 作为输入列表，用于存储estimate的逆傅里叶变换结果
for i in estimate:
    ift.append(np.fft.ifft(i))  # 追加到ift列表中

clean_data = []
clean_data.extend(ift[0][:int(fl / 2)])  # 扩展clean_data，包含ift列表
for i in range(len(ift) - 1):
    clean_data.extend(ift[i][int(fl / 2):] + ift[i + 1][:int(fl / 2)])
clean_data.extend(ift[-1][int(fl / 2):])  # 扩展clean_data，包含ift列表
clean_data = np.array(clean_data)  # 将其转换为数组

# 最后绘制图表，显示噪声差异
fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(1, 1, 1)
ax.plot(np.linspace(0, 64000, 64000), data, label='原始信号', color="orange")
ax.plot(np.linspace(0, 64000, 64000), clean_data, label='滤波后信号', color="purple")
ax.legend(fontsize=12)
ax.set_title('谱减法', fontsize=15)
filename = os.path.basename(file)
cleaned_file = "(Filtered_Audio)" + filename  # 最终滤波后的音频
wav.write(cleaned_file, rate=sr, data=clean_data.astype(np.int16))
plt.savefig(filename + "(Spectral Subtraction graph).jpg")  # 保存的文件名为 audio.wav(Spectral Subtraction graph).jpg

## 14. NSE 股票数据

运行此脚本将允许用户根据自己的选择，浏览从 [NSE 网站](https://www.nseindia.com) 抓取的 NSE 股票数据。

### 设置说明

要运行此脚本，您需要在系统上安装 Python 和 pip。安装完 Python 和 pip 后，从终端运行以下命令，从项目的同一文件夹（目录）安装依赖项。

```
pip install -r requirements.txt
```

由于此脚本使用 selenium，您需要从 [此链接](https://sites.google.com/a/chromium.org/chromedriver/downloads) 安装 chrome webdriver。

满足项目的所有要求后，在项目文件夹中打开终端并运行

```
python stocks.py
```

或

```
python3 stocks.py
```

具体取决于 Python 版本。确保您在安装了所需模块的同一虚拟环境中运行该命令。

**依赖项：**
- **requests**
- **beautifulsoup4**
- **selenium**

### 源代码：

```
import requests
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
```

```
driver_path = input('Enter path for chromedriver: ')
```

```
# 类别及其URL标识符
most_active = {'Most Active equities - Main Board': 'mae_mainboard_tableC', 'Most Active equities - SME': 'mae_sme_tableC', 'Most Active equities - ETFs': 'mae_etf_tableC',
               'Most Active equities - Price Spurts': 'mae_pricespurts_tableC', 'Most Active equities - Volume Spurts': 'mae_volumespurts_tableC'}
top_20 = {'NIFTY 50 Top 20 Gainers': 'topgainer-Table', 'NIFTY 50 Top 20 Losers': 'toplosers-Table'}
```

```
# 根据用户选择生成请求URL的函数
def generate_url():
    category_choice = category.get()
    if (category_choice in most_active):
        page = 'most-active-equities'
    else:
        page = 'top-gainers-loosers'
    url = 'https://www.nseindia.com/market-data/{}'.format(page)
    return url
```

```
# 从生成的URL抓取股票数据的函数
def scraper():
    url = generate_url()
    driver = webdriver.Chrome(driver_path)
    driver.get(url)

    # 等待结果加载
    time.sleep(5)
    html = driver.page_source

    # 开始抓取生成的HTML数据
    soup = BeautifulSoup(html, 'html.parser')

    # 根据选择抓取div
    category_choice = category.get()
    if category_choice in most_active:
        category_div = most_active[category_choice]
    else:
        category_div = top_20[category_choice]

    # 查找要抓取的表格
    results = soup.find("table", {"id": category_div})
    rows = results.findChildren('tr')

    table_data = []
    row_values = []
    # 将股票数据追加到列表中
    for row in rows:
        cells = row.findChildren(['th', 'td'])
        for cell in cells:
            value = cell.text.strip()
            value = " ".join(value.split())
            row_values.append(value)
        table_data.append(row_values)
        row_values = []

    # 格式化存储在列表中的股票数据
    stocks_data = ""
    for stock in table_data:
        single_record = ""
        for cell in stock:
            format_cell = "{:<20}"
            single_record += format_cell.format(cell[:20])
        single_record += "\n"
        stocks_data += single_record

    # 将格式化后的数据添加到tkinter GUI中
    query_label.config(state=tk.NORMAL)
    query_label.delete(1.0, "end")
    query_label.insert(1.0, stocks_data)
    query_label.config(state=tk.DISABLED)
    driver.close()

# 创建tkinter窗口
window = tk.Tk()
window.title('NSE 股票数据')
window.geometry('1200x1000')
window.configure(bg='white')

style = ttk.Style()
style.configure('my.TButton', font=('Helvetica', 16))
style.configure('my.TFrame', background='white')

# 标题标签文本
ttk.Label(window, text="NSE 股票市场数据",
          background='white', foreground="SpringGreen2",
          font=("Helvetica", 30, 'bold')).grid(row=0, column=1)

# 标签
ttk.Label(window, text="选择要获取的市场数据：", background='white',
          font=("Helvetica", 15)).grid(column=0,
                                       row=5, padx=10, pady=25)

# 创建组合框
category = ttk.Combobox(
    window, width=60, state='readonly', font="Helvetica 15")

submit_btn = ttk.Button(window, text="获取股票数据！", style='my.TButton', command=scraper)

# 添加组合框下拉列表
category['values'] = ('Most Active equities - Main Board', 'Most Active equities - SME', 'Most Active equities - ETFs', 'Most Active equities - Price Spurts',
                       'Most Active equities - Volume Spurts', 'NIFTY 50 Top 20 Gainers', 'NIFTY 50 Top 20 Losers')

category.grid(column=1, row=5, padx=10)
category.current(0)

submit_btn.grid(row=5, column=3, pady=5, padx=15, ipadx=5)

frame = ttk.Frame(window, style='my.TFrame')
frame.place(relx=0.50, rely=0.12, relwidth=0.98, relheight=0.90, anchor="n")

# 用于显示股票数据
query_label = tk.Text(frame, height="52", width="500", bg="alice blue")
query_label.grid(row=7, columnspan=2)

window.mainloop()
```

## 15. 猜数字游戏

这个游戏可以让你检验自己的运气和直觉 :)
你需要找出计算机猜测的数字。

### 使用方法

设置好项目目录后，只需在cmd命令行中运行 "python main.py"。

### 这里你可以看到示例

![](img/2d09155385e2d53be8e53bc9b9a3735b_41_0.png)

**源代码：**

```
import random

print("Number guessing game")

# 使用randint函数生成1到9之间的随机数
number = random.randint(1, 9)

# 给用户猜测数字的机会次数
# 或者说是用户在输入框中输入的次数
# 这里机会次数是5
chances = 0

print("Guess a number (between 1 and 9):")

# 使用while循环来计算机会次数
while True:

    # 输入一个1到9之间的数字
    guess = int(input())

    # 比较用户输入的数字
```

# 与待猜数字进行比较
if guess == number:
    # 如果用户输入的数字
    # 与randint函数生成的
    # 数字相同，则使用
    # 循环控制语句"break"
    # 跳出循环
    print(
        f'恭喜！你在 {chances} 次尝试中猜中了数字 {number}！')
    # 使用f-strings方法打印最终语句；
    break

# 检查用户输入的数字
# 是否小于生成的数字
elif guess < number:
    print("你的猜测太低了：请猜一个大于", guess, "的数字")

# 用户输入的数字
# 大于生成的数字
else:
    print("你的猜测太高了：请猜一个小于", guess, "的数字")

# 将尝试次数增加1
chances += 1

## 16. 文件分类器

这是一个Python脚本，用于根据文件扩展名将下载目录中的文件分类到其他目录。

**源代码：**

```python
import os
import shutil
os.chdir("E:\downloads")
#print(os.getcwd())

#检查目录中的文件数量
files = os.listdir()

#扩展名列表（你可以根据需要添加更多）
extentions = {
    "images": [".jpg", ".png", ".jpeg", ".gif"],
    "videos": [".mp4", ".mkv"],
    "musics": [".mp3", ".wav"],
    "zip": [".zip", ".tgz", ".rar", ".tar"],
    "documents": [".pdf", ".docx", ".csv", ".xlsx", ".pptx", ".doc", ".ppt", ".xls"],
    "setup": [".msi", ".exe"],
    "programs": [".py", ".c", ".cpp", ".php", ".C", ".CPP"],
    "design": [".xd", ".psd"]

}

#根据扩展名分类到特定文件夹
def sorting(file):
    keys = list(extentions.keys())
    for key in keys:
        for ext in extentions[key]:
            # print(ext)
            if file.endswith(ext):
                return key

#遍历每个文件
for file in files:
    dist = sorting(file)
    if dist:
        try:
            shutil.move(file, "../download-sorting/" + dist)
        except:
            print(file + " 已存在")
    else:
        try:
            shutil.move(file, "../download-sorting/others")
        except:
            print(file + " 已存在")
```

## 17. PageSpeed

此脚本为网站生成PageSpeed API结果。

### 所需包：

- requests
- json

### 使用说明

要使用该包，只需查看test.py文件。

### 输出

根据脚本的使用情况，它可以生成一个包含PageSpeed结果的json文件，并且还会返回常规的响应对象。

**PageSpeed.py 源代码：**

```python
import requests
import json
from responses import PageSpeedResponse

class PageSpeed(object):
    """
    Google PageSpeed分析客户端

    属性:
        api_key (str): 可选的客户端账户API密钥。
        endpoint (str): HTTP请求的端点
    """

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.endpoint = 'https://www.googleapis.com/pagespeedonline/v5/runPagespeed'

    def analyse(self, url, strategy='desktop', category='performance'):
        """
        运行PageSpeed测试

        参数:
            url (str): 要获取和分析的URL。
            strategy (str, optional): 要使用的分析策略。可接受的值：'desktop', 'mobile'
            category (str, optional): 要运行的Lighthouse类别；如果未指定，将只运行Performance类别

        返回:
            response: PageSpeed API结果
        """
        strategy = strategy.lower()

        params = {
            'strategy': strategy,
            'url': url,
            'category': category,
        }

        if self.api_key:
            params['key'] = self.api_key

        # 合理性检查
        if strategy not in ('mobile', 'desktop'):
            raise ValueError('无效的策略: {0}'.format(strategy))

        # 返回原始数据
        raw = requests.get(self.endpoint, params=params)

        response = PageSpeedResponse(raw)

        return response

    def save(self, response, path='./'):
        json_data = response._json
        with open(path + "json_data.json", 'w+') as f:
            json.dump(json_data, f, indent=2)

if __name__ == "__main__":
    ps = PageSpeed()
    response = ps.analyse('https://www.example.com', strategy='mobile')
    ls = [
        response.url, response.loadingExperience,
        response.originLoadingExperience,
        response.originLoadingExperienceDetailed,
        response.loadingExperienceDetailed, response.finalUrl,
        response.requestedUrl, response.version, response.userAgent
    ] # , response.lighthouseResults]
    ps.save(response)
    print(ls)
```

**Responses.py 源代码：**

```python
import json

class Response(object):
    """
    基础响应对象

    属性:
        self.json (dict): 响应的JSON表示
        self._request (str): URL
        self._response (`requests.models.Response` 对象): 来自requests模块的响应对象
    """

    def __init__(self, response):
        response.raise_for_status()

        self._response = response
        self._request = response.url
        self._json = json.loads(response.content)

class PageSpeedResponse(Response):
    """
    PageSpeed响应对象

    属性:
        self.url (str):
        self.speed (int):
        self.statistics (`Statistics` 对象):
    """
    @property
    def url(self):
        return self._json['id']

    @property
    def loadingExperience(self):
        return self._json['loadingExperience']['overall_category']

    @property
    def originLoadingExperience(self):
        return self._json['originLoadingExperience']['overall_category']

    @property
    def originLoadingExperienceDetailed(self):
        metrics = self._json['originLoadingExperience']['metrics']
        keys_ = list(metrics.keys())

        originLoadingExperienceDetailed_ = {}

        for each in keys_:
            originLoadingExperienceDetailed_[each] = metrics[each]['category']

        return originLoadingExperienceDetailed_

    @property
    def loadingExperienceDetailed(self):
        metrics = self._json['loadingExperience']['metrics']
        keys_ = list(metrics.keys())

        loadingExperienceDetailed_ = {}

        for each in keys_:
            loadingExperienceDetailed_[each] = metrics[each]['category']

        return loadingExperienceDetailed_

    # 在重定向的情况下
    @property
    def requestedUrl(self):
        return self._json['lighthouseResult']['requestedUrl']

    @property
    def finalUrl(self):
        return self._json['lighthouseResult']['finalUrl']

    @property
    def version(self):
        return self._json['lighthouseResult']['lighthouseVersion']

    @property
    def userAgent(self):
        return self._json['lighthouseResult']['userAgent']

    @property
    def lighthouseResults(self):
        return self._json['lighthouseResult']
```

**Test.py 源代码：**

```python
import pagespeed
from pagespeed import PageSpeed

ps = PageSpeed()

response = ps.analyse('https://www.example.com', strategy='mobile')
ls = [
    response.url, response.loadingExperience, response.originLoadingExperience,
    response.originLoadingExperienceDetailed,
    response.loadingExperienceDetailed, response.finalUrl,
    response.requestedUrl, response.version, response.userAgent
] # , response.lighthouseResults]
ps.save(response)
print(ls)
```

## 18. 绘图应用

```python
from tkinter import *
import tkinter.font

class PaintApp:
    drawing_tool = "pencil"
    left_button = "up"

    x_position, y_position = None, None

    x1_line_pt, y1_line_pt, x2_line_pt, y2_line_pt = None, None, None, None

    @staticmethod
    def quit_app(event=None):
        root.quit()

    def __init__(self, root):
        drawing_area = Canvas(root)
        drawing_area.pack()

        drawing_area.bind("<Motion>", self.motion)
        drawing_area.bind("<ButtonPress-1>", self.left_button_down)
        drawing_area.bind("<ButtonRelease-1>", self.left_button_up)

        the_menu = Menu(root)

        file_menu = Menu(the_menu, tearoff=0)
        file_menu.add_command(label="直线", command=self.set_line_drawing_tool)
        file_menu.add_command(label="铅笔", command=self.set_pencil_drawing_tool)
        file_menu.add_command(label="弧线", command=self.set_arc_drawing_tool)
        file_menu.add_command(label="矩形", command=self.set_rectangle_drawing_tool)
        file_menu.add_command(label="椭圆", command=self.set_oval_drawing_tool)
        file_menu.add_command(label="文本", command=self.set_text_drawing_tool)

        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.quit_app)

        the_menu.add_cascade(label="选项", menu=file_menu)
        root.config(menu=the_menu)

    def set_line_drawing_tool(self):
        self.drawing_tool = "line"

    def set_pencil_drawing_tool(self):
        self.drawing_tool = "pencil"
```

def set_arc_drawing_tool(self):
    self.drawing_tool = "arc"

def set_rectangle_drawing_tool(self):
    self.drawing_tool = "rectangle"

def set_oval_drawing_tool(self):
    self.drawing_tool = "oval"

def set_text_drawing_tool(self):
    self.drawing_tool = "text"

def left_button_down(self, event=None):
    self.left_button = "down"
    self.x1_line_pt = event.x
    self.y1_line_pt = event.y

def left_button_up(self, event=None):
    self.left_button = "up"
    self.x_position = None
    self.y_position = None
    self.x2_line_pt = event.x
    self.y2_line_pt = event.y
    if self.drawing_tool=="line":
        self.line_draw(event)
    if self.drawing_tool=="pencil":
        self.pencil_draw(event)
    if self.drawing_tool=="arc":
        self.arc_draw(event)
    if self.drawing_tool=="oval":
        self.oval_draw(event)
    if self.drawing_tool=="rectangle":
        self.rect_draw(event)
    if self.drawing_tool=="text":
        self.text_draw(event)

def motion(self, event=None):
    if self.drawing_tool=="pencil":
        self.pencil_draw(event)
    self.x_position = event.x
    self.y_position = event.y

def pencil_draw(self, event=None):
    if self.left_button =="down":
        if self.x_position is not None and self.y_position is not None:
            event.widget.create_line(self.x_position, self.y_position, event.x, event.y, smooth=True)

def line_draw(self, event=None):
    if  None not in (self.x1_line_pt, self.x2_line_pt, self.y1_line_pt, self.y2_line_pt):
        event.widget.create_line(self.x1_line_pt, self.x2_line_pt, self.y1_line_pt, self.y2_line_pt, smooth=True,
                                fill="green")

def arc_draw(self, event=None):
    if  None not in (self.x1_line_pt, self.x2_line_pt, self.y1_line_pt, self.y2_line_pt):
        coords = self.x1_line_pt, self.x2_line_pt, self.y1_line_pt, self.y2_line_pt
        event.widget.create_arc(coords, start=0, extent=150, style=ARC, fill="blue")

def oval_draw(self, event=None):
    if  None not in (self.x1_line_pt, self.x2_line_pt, self.y1_line_pt, self.y2_line_pt):
        event.widget.create_oval(self.x1_line_pt, self.x2_line_pt, self.y1_line_pt, self.y2_line_pt, fill="midnight blue", outline="yellow", width=2)

def rect_draw(self, event=None):
    if  None not in (self.x1_line_pt, self.x2_line_pt, self.y1_line_pt, self.y2_line_pt):
        event.widget.create_rectangle(self.x1_line_pt, self.x2_line_pt, self.y1_line_pt, self.y2_line_pt, fill="red", outline="pink", width=2)

def text_draw(self, event=None):
    if  None not in (self.x1_line_pt, self.y1_line_pt):
        text_font = tkinter.font.Font(family="Helvetica", size=20, weight="bold", slant="italic")
        event.widget.create_text(self.x1_line_pt, self.y1_line_pt, fill="lightblue", font=text_font, text="helloooo!")
root = Tk()
paint_app = PaintApp(root)
root.mainloop()

## 19. 密码管理器

import os.path
# 使用 Tkinter 模块生成随机密码的 Python 程序
import random
import pyperclip
from tkinter import *
from tkinter.ttk import *

# 计算密码的函数
def low():
    entry.delete(0, END)

    # 获取密码长度
    length = var1.get()

    lower = "abcdefghijklmnopqrstuvwxyz"
    upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    digits = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()"
    password = ""

    # 如果选择的强度为低
    if var.get() == 1:
        for i in range(0, length):
            password = password + random.choice(lower)
        return password

    # 如果选择的强度为中
    elif var.get() == 0:
        for i in range(0, length):
            password = password + random.choice(upper)
        return password

    # 如果选择的强度为强
    elif var.get() == 3:
        for i in range(0, length):
            password = password + random.choice(digits)
        return password
    else:
        print("Please choose an option")

# 生成密码的函数
def generate():
    password1 = low()
    entry.insert(10, password1)

# 将密码复制到剪贴板的函数
def copy1():
    random_password = entry.get()
    pyperclip.copy(random_password)

def checkExistence():
    if os.path.exists("info.txt"):
        pass
    else:
        file = open("info.txt", 'w')
        file.close()

def appendNew():
    file = open("info.txt", 'a')
    userName = entry1.get()
    website= entry2.get()
    Random_password=entry.get()
    usrnm = "UserName: " + userName + "\n"
    pwd = "Password: " + Random_password + "\n"
    web = "Website: " + website + "\n"
    file.write("-----------------------------------\n")
    file.write(usrnm)
    file.write(pwd)
    file.write(web)
    file.write("-----------------------------------\n")
    file.write("\n")
    file.close
    # 此函数将在 txt 文件中追加新密码
    file = open("info.txt", 'a')

def readPasswords():
    file = open('info.txt', 'r')
    content = file.read()
    file.close()
    print(content)

# 主函数
checkExistence()
# 创建 GUI 窗口
root = Tk()
var = IntVar()
var1 = IntVar()

# GUI 窗口的标题
root.title("Python 密码管理器")

# 创建密码长度的标签
c_label = Label(root, text="长度")
c_label.grid(row=1)

# 创建复制按钮，用于将密码复制到剪贴板，
# 以及生成按钮，用于生成密码
copy_button = Button(root, text="复制", command=copy1)
copy_button.grid(row=0, column=2)
generate_button = Button(root, text="生成", command=generate)
generate_button.grid(row=0, column=3)

# 用于决定密码强度的单选按钮
# 默认强度为中
radio_low = Radiobutton(root, text="低", variable=var, value=1)
radio_low.grid(row=1, column=2, sticky='E')
radio_middle = Radiobutton(root, text="中", variable=var, value=0)
radio_middle.grid(row=1, column=3, sticky='E')
radio_strong = Radiobutton(root, text="强", variable=var, value=3)
radio_strong.grid(row=1, column=4, sticky='E')
combo = Combobox(root, textvariable=var1)

# 用于选择密码长度的组合框
combo['values'] = (8, 9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25,
                   26, 27, 28, 29, 30, 31, 32, "长度")
combo.current(0)
combo.bind('<<ComboboxSelected>>')
combo.grid(column=1, row=1)

# 创建标签和输入框，用于显示生成的密码
userName = Label(root, text="在此输入用户名")
userName.grid(row=2)
entry1 = Entry(root)
entry1.grid(row=2, column=1)

# 创建标签和输入框，用于显示生成的密码
website = Label(root, text="在此输入网站地址")
website.grid(row=3)
entry2 = Entry(root)
entry2.grid(row=3, column=1)

Random_password = Label(root, text="生成的密码")
Random_password.grid(row=4)
entry = Entry(root)
entry.grid(row=4, column=1)

save_button = Button(root, text="保存", command=appendNew)
save_button.grid(row=2, column=2)
show_button = Button(root, text="显示所有密码", command=readPasswords)
show_button.grid(row=2, column=3)

# 启动 GUI
root.mainloop()

## 20. 密码管理器图形用户界面

此脚本执行时会打开一个密码管理器图形用户界面。它允许用户将所有密码存储在本地 SQL 数据库中。

### 设置说明

要运行此脚本，您的系统需要安装 Python 和 pip。
安装完 Python 和 pip 后，在项目文件夹中打开终端并运行

```
python passwords.py <主密码>
```

或

```
python3 passwords.py <主密码>
```

具体取决于 Python 版本。请确保您从安装了所需模块的同一虚拟环境中运行该命令。

### 源代码：

```
from tkinter import *
from tkinter import messagebox, simpledialog
import sqlite3
from sqlite3 import Error
import sys

# 存储主密码
master_password = sys.argv[1]

# 连接到 SQL 数据库的函数

def sql_connection():
    try:
        con = sqlite3.connect('passwordManager.db')
        return con
    except Error:
        print(Error)

# 创建表的函数
```

def sql_table(con):
    cursorObj = con.cursor()
    cursorObj.execute(
        "CREATE TABLE IF NOT EXISTS passwords(website text, username text, pass text)")
    con.commit()

# 调用函数连接数据库并创建表
con = sql_connection()
sql_table(con)

# 为数据库创建提交函数

def submit(con):
    cursor = con.cursor()
    # 插入表中
    if website.get() != "" and username.get() != "" and password.get() != "":
        cursor.execute("INSERT INTO passwords VALUES (:website, :username, :password)",
                       {
                           'website': website.get(),
                           'username': username.get(),
                           'password': password.get()
                       }
                       )
        con.commit()
        # 消息框
        messagebox.showinfo("信息", "记录已添加到数据库！")

        # 数据输入后清空文本框
        website.delete(0, END)
        username.delete(0, END)
        password.delete(0, END)
    else:
        messagebox.showinfo("警告", "请填写所有详细信息！")

# 创建查询函数

def query(con):
    password = simpledialog.askstring("密码", "输入主密码")
    if(password == master_password):
        # 设置按钮文本
        query_btn.configure(text="隐藏记录", command=hide)
        cursor = con.cursor()
        # 查询数据库
        cursor.execute("SELECT *, oid FROM passwords")
        records = cursor.fetchall()

        p_records = 'ID'.ljust(10) + '网站'.ljust(40) + \
            '用户名'.ljust(70)+'密码'.ljust(100)+'\n'

        for record in records:
            single_record = ""
            single_record += (str(record[3]).ljust(10) +
                              str(record[0]).ljust(40)+str(record[1]).ljust(70)+str(record[2]).ljust(10)
            single_record += '\n'
            # print(single_record)
            p_records += single_record
        query_label['text'] = p_records
        # 提交更改
        con.commit()
        p_records = ""

    else:
        messagebox.showinfo("失败！", "身份验证失败！")

# 创建隐藏记录的函数

def hide():
    query_label['text'] = ""
    query_btn.configure(text="显示记录", command=lambda: query(con))

root = Tk()
root.title("密码管理器")
root.geometry("500x400")
root.minsize(600, 400)
root.maxsize(600, 400)

frame = Frame(root, bg="#774A9F", bd=5)
frame.place(relx=0.50, rely=0.50, relwidth=0.98, relheight=0.45, anchor="n")

# 创建文本框
website = Entry(root, width=30)
website.grid(row=1, column=1, padx=20, pady=5)
username = Entry(root, width=30)
username.grid(row=2, column=1, padx=20, pady=5)
password = Entry(root, width=30)
password.grid(row=3, column=1, padx=20, pady=5)

# 创建文本框标签
website_label = Label(root, text="网站:")
website_label.grid(row=1, column=0)
username_label = Label(root, text=" 用户名:")
username_label.grid(row=2, column=0)
password_label = Label(root, text="密码:")
password_label.grid(row=3, column=0)

# 创建按钮
submit_btn = Button(root, text="添加密码", command=lambda: submit(con))
submit_btn.grid(row=5, column=1, pady=5, padx=15, ipadx=35)
query_btn = Button(root, text="显示全部", command=lambda: query(con))
query_btn.grid(row=6, column=1, pady=5, padx=5, ipadx=35)

# 创建一个标签来显示存储的密码
global query_label
query_label = Label(frame, anchor="nw", justify="left")
query_label.place(relwidth=1, relheight=1)

def main():
    root.mainloop()

if __name__ == '__main__':
    main()

## 模块3 项目 21-30

## 21. 可朗读的PDF与图像文本阅读器

使用python、pyttsx3和Tesseract实现可朗读的PDF与图像文本阅读器

### 安装：

- 使用此命令安装tesseract-ocr：
  - 在Ubuntu上
    ```
    sudo apt-get install tesseract-ocr
    ```
  - 在Mac上
    ```
    brew install tesseract
    ```
  - 在Windows上，从[此处](https://github.com/UB-Mannheim/tesseract/wiki)下载安装程序

- 使用此pip命令安装tesseract的python绑定pytesseract：
  ```
  pip install pytesseract
  ```

- 使用此pip命令安装python图像处理库pillow：
  ```
  pip install pillow
  ```

**处理PDF文件：**

- 使用此命令安装imagemagick：
  - 在Ubuntu上
    ```
    sudo apt-get install imagemagick
    ```
  - 对于其他平台，从[此处](https://imagemagick.org/script/download.php)下载安装程序

- 使用此pip命令安装imagemagick的python绑定wand：
  ```
  pip install wand
  ```

- 安装Pyttsx3：
  ```
  pip install pyttsx3
  ```

### PDF阅读器源代码：

```python
import io
from PIL import Image
import pytesseract
from wand.image import Image as wi
import pyttsx3
import speech_recognition as sr

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
#print(voices[1].id)
engine.setProperty('voice',voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" #Path to the tesseract

pdf = wi(filename = "sample.pdf", resolution = 300)
pdfImage = pdf.convert('jpeg')

imageBlobs = []

for img in pdfImage.sequence:
    imgPage = wi(image = img)
    imageBlobs.append(imgPage.make_blob('jpeg'))

recognized_text = []

for imgBlob in imageBlobs:
    im = Image.open(io.BytesIO(imgBlob))
    text = pytesseract.image_to_string(im, lang = 'eng')
    recognized_text.append(text)

imageBlobs = str(text)
recognized_text = text
print(recognized_text)
speak(recognized_text)
remember = open('remember.txt','w')
remember.write(text)
remember.close()
```

### 图像阅读器源代码：

```python
import pytesseract #pip install tesseract
import os
from PIL import Image
import pyttsx3

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
#print(voices[1].id)
engine.setProperty('voice',voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()


    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #Path to the tesseract

    img = Image.open('img2.jpg')# 在此添加带文件扩展名的图像名称
    text = pytesseract.image_to_string(img)
    print(text)
    remember = open('remember.txt','w')
    remember.write(text)
    remember.close()
    speak(text)
```

## 22. PDF转CSV转换器

要求 - tabula-py

### 源代码：

```python
import tabula # simple wrapper for tabula-java, read tables from PDF into csv
import os
print("[-+-] starting pdf_csv.py...")
print("[-+-] import a pdf and convert it to a csv")
# -----------------------------------------------------------------
print("[-+-] importing required packages for pdf_csv.py...")
# from modules.defaults import df # local module
print("[-+-] pdf_csv.py packages imported! \n")
# -----------------------------------------------------------------

# -----------------------------------------------------------------

def pdf_csv(): # convert pdf to csv
    print("[-+-] default filenames:")
    filename = "sample1"
    pdf = filename + ".pdf"
    csv = filename + ".csv"
    print(pdf)
    print(csv + "\n")

    print("[-+-] default directory:")
    print("[-+-] (based on current working directory of python file)")
    defaultdir = os.getcwd()
    print(defaultdir + "\n")

    print("[-+-] default file paths:")
    pdf_path = os.path.join(defaultdir, pdf)
    csv_path = os.path.join(defaultdir, csv)
    print(pdf_path)
    print(csv_path + "\n")

    print("[-+-] looking for default pdf...")
    if os.path.exists(pdf_path) == True: # check if the default pdf exists
        print("[-+-] pdf found: " + pdf + "\n")
        pdf_flag = True
    else:
        print("[-+-] looking for another pdf...")
        arr_pdf = [
            defaultdir for defaultdir in os.listdir()
            if defaultdir.endswith(".pdf")
        ]
        if len(arr_pdf) == 1: # there has to be only 1 pdf in the directory
            print("[+-] pdf found: " + arr_pdf[0] + "\n")
            pdf_path = os.path.join(defaultdir, arr_pdf[0])
            pdf_flag = True
        elif len(arr_pdf) > 1: # there are more than 1 pdf in the directory
            print("[+-] more than 1 pdf found, exiting script!")
            pdf_flag = False
            # TODO add option to select from available pdfs
        else:
            print("[+-] pdf cannot be found, exiting script!")
            pdf_flag = False

    if pdf_flag == True:
        # check if csv exists at the default file path
        # if csv does not exist create a blank file at the default path
        try:
            print("[+-] looking for default csv...")
            open(csv_path, "r")
            print("[+-] csv found: " + csv + "\n")
        except IOError:
            print("[+-] did not find csv at default file path!")
            print("[+-] creating a blank csv file: " + csv + "... \n")
            open(csv_path, "w")

        print("[+-] converting pdf to csv...")
        # print("[+-] pdf to csv conversion suppressed! \n")
        try:
            tabula.convert_into(pdf_path,
                                csv_path,
                                output_format="csv",
                                pages="all")
            print("[+-] pdf to csv conversion complete!\n")
        except IOError:
            print("[+-] pdf to csv conversion failed!")

        print("[+-] converted csv file can be found here: " + csv_path + "\n")

        print("[+-] finished pdf_csv.py successfully!")

# ---------------------------------------------------------------------------
```

## 23. 查重工具

查重工具

### 工作原理

- 为了计算两个文本文档之间的相似度，文本原始数据会被转换为向量。
- 然后将其转换为数字数组，再利用基础的知识向量来计算它们之间的相似度。

### 依赖项

- 通过以下命令安装 scikit-learn：

```
$ pip install scikit-learn
```

### 运行应用

- 仓库中有四个文本文档。
- 基本上，代码会比较所有 `.txt` 文件并检查是否存在任何相似性。

```
$ python plagiarism.py
```

源代码：

```python
# OS Module for loading paths of textfiles. TfidfVectorizer to perform word embedding on the textual data and cosine similarity to compute the plagiarism.
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(File).read() for File in student_files]
# Two lambda functions, one to convert the text to arrays of numbers and the other one to compute the similarity between them.

def vectorize(Text): return TfidfVectorizer().fit_transform(Text).toarray()

def similarity(doc1, doc2): return cosine_similarity([doc1, doc2])

# Vectorize the Textual Data
vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))

# computing the similarity among students
def check_plagiarism():
    plagiarism_results = set()
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results
for data in check_plagiarism():
    print(data)
```

## 24. 带图形界面的番茄钟

- 给定的 Python 脚本可以创建一个带有用户友好图形界面的个人番茄计时器/番茄钟。
- 番茄钟是一种经过科学验证的生产力计时器，它将你的工作划分为25分钟的高专注时段和5分钟的休息间隔。

### 安装依赖：

```
$ pip install -r requirements.txt
```

### 运行脚本：

```
$ python Pomodoro_gui.py
```

**依赖项：**
**pygame==2.0.1**
**tkinter==8.6**

**源代码：**

```python
import time
import tkinter as tk
from tkinter import messagebox
import pygame
from datetime import timedelta

pygame.mixer.init()
pomo_count = 0
break_count = 0
enable = 0

# path of host file in windows
host_path = r"C:\Windows\System32\drivers\etc\hosts"

# URL of websites to block
block_list = []

# redirecting above URLs to this localhost to ensure blocking
redirect = "127.0.0.1"

def block_websites():
    """
    The function will open the host file and add the block-list websites to
    the file if it is not already present and redirect it to the localhost
    for blocking
    """
    global web_var
    global enable
    global block_list
    global host_path
    url = web_var.get()
    block_list.append(url)
    try:
        # Opening the host file in reading and writing mode
        with open(host_path, 'r+') as h_file:
            content = h_file.read()

            for website in block_list:

                # Website is already blocked
                if website in content:
                    pass

                # To redirect the website to be blocked
                else:
                    h_file.write(redirect + "\t" + website + "\n")

        tk.messagebox.showinfo("Blocked", f"{url} successfully blocked!")
        enable = 1
        web_var.set("")

    except PermissionError:
        tk.messagebox.showinfo("Error", "Run cmd in the admin mode and then try again!")
        web_var.set("")

    except (FileNotFoundError, NameError):
        tk.messagebox.showinfo("Error", "Functionality not supported in your OS!")
        web_var.set("")

def remove_websites():
    """
    The function will unblock the block_list websites by opening the file
    and removing the changes we made before
    """
    global block_list
    global host_path
    try:
        if enable:
            # Opening the host file in reading and writing mode
            with open(host_path, "r+") as file:

                # making each line of file into a list
                content = file.readlines()

                # sets the file pointer at the beginning of the file
                file.seek(0)

                # Traversing through each line of the host file and
                # checking for the websites to be blocked
                for lines in content:
                    if not any(website in lines for website in block_list):
                        file.write(lines)

                # Truncating the file to its original size
                file.truncate()

        block_list.clear()
        enable = 0
    except:
        pass
    finally:
        pass

def blocker():
    """
    The function asks input from user to block websites for high focus mode.
    """

    global enable
    global popup_4
    popup_4 = tk.Toplevel(root)
    popup_4.title("Website Blocker!")
    popup_4.geometry("360x220")
    popup_4.config( bg = 'DodgerBlue4')

    global block_list
    global web_var
    web_var=tk.StringVar()

    pass_label = tk.Label(popup_4, text = 'Enter URL to block:', font = ('Arial',12, 'bold'), bg = 'DodgerBlue4', fg = 'white')
    pass_entry = tk.Entry(popup_4, textvariable = web_var, font = ('Arial',12, 'bold'))

    sub_btn = tk.Button(popup_4, text = 'Block', font = ('Arial',12, 'bold'), command = block_websites, bg='gold', activebackground='yellow')

    text_to_put = '*Supported for windows ONLY\n*You can add multiple urls\n*Don\'t forget to unblock after'

    instructions = tk.Label(popup_4, text = text_to_put, font = ('Arial',12, 'bold'), justify='left', bg = 'sky blue')

    unblock_btn = tk.Button(popup_4, text = 'Unblock all', font = ('Arial',12, 'bold'), command = remove_websites,
    state='disabled', width = 23, height = 2, bg='gold', activebackground='yellow')

    if enable:
        unblock_btn.config(state='normal')

    pass_label.place(x=25, y=10)
    pass_entry.place(x=25, y=34)
    sub_btn.place(x=255, y=30)
    instructions.place(x=25, y=80)
    unblock_btn.place(x=50, y=150)

def break_timer():
    """
    5 min timer popup window acting as a callback function to the break timer button
    """
    global enable
    global popup_2
    popup_2 = tk.Toplevel(root)
    popup_2.title("Break Timer!")
    popup_2.geometry("370x120")
    round = 0

    try:
        # Creating a continous loop of text of time on the screen for 25 mins
        t = 5*60
        while t>-1:
            minute_count = t // 60
            second_count = t % 60
            timer = '{:02d}:{:02d}'.format(minute_count, second_count)
            time_display = tk.Label(popup_2, text = timer, bg = 'DodgerBlue4', fg = 'white', font = ('STIX', 90, 'bold'))
            time_display.place(x=0,y=0)
            popup_2.update()
            time.sleep(1)
            t -= 1
    except:
        pass

    # Setting up an alarm sound and popup window to let user know when the time is up
    if t == -1:
        tk.messagebox.showinfo("Time's up!", "Break is over!\nTime to get to work!")
        popup_2.destroy()
        global break_count
        pygame.mixer.music.load("./Pomodoro_GUI/beep.wav")
        pygame.mixer.music.play(loops=1)
        break_count += 1

def show_report():
    """
    The function acts as a callback for show report button and shows the report the hours
    of work they have put in.
    """
    global popup_3
    popup_3 = tk.Toplevel(root)
    popup_3.title("Report")
    popup_3.geometry("370x170")
    popup_3.config( bg = 'DodgerBlue4')

    pomo_time = str(timedelta(minutes=pomo_count*25))[:-3]
    break_time = str(timedelta(minutes=pomo_count*5))[:-3]
    tk.Label(popup_3, text=f"Number of Pomodoros completed: {pomo_count}", justify=tk.LEFT, bg = 'DodgerBlue4', fg = 'white', font=('Arial',12,'bold')).place(x = 10, y = 10)
    tk.Label(popup_3, text=f"Number of breaks completed: {break_count}", justify=tk.LEFT, bg = 'DodgerBlue4', fg = 'white', font=('Arial',12,'bold')).place(x = 10, y = 50)
    tk.Label(popup_3, text=f"Hours of work done: {pomo_time} hrs", justify=tk.LEFT,  bg = 'DodgerBlue4', fg = 'white', font=('Arial',12,'bold')).place(x = 10, y = 90)
    tk.Label(popup_3, text=f"Hours of break taken: {break_time} hrs", justify=tk.LEFT,  bg = 'DodgerBlue4', fg = 'white', font=('Arial',12,'bold')).place(x = 10, y = 130)

def pomodoro_timer():
    """
    25 min timer popup window acting as a callback function to the work timer button
    """
    global popup_1
    popup_1 = tk.Toplevel(root)
    popup_1.title("Work Timer!")
    popup_1.geometry("370x120")
    round = 0

    try:
        # Creating a continous loop of text of time on the screen for 25 mins
        t = 25*60
        while t>-1:
            minute_count = t // 60
```

second_count = t % 60
timer = '{:02d}:{:02d}'.format(minute_count, second_count)
time_display = tk.Label(popup_1, text = timer, bg = 'DodgerBlue4', fg = 'white', font = ('STIX', 90, 'bold'))
time_display.place(x=0,y=0)
popup_1.update()
time.sleep(1)
t -= 1
except:
    pass

# 设置闹钟声音和弹出窗口，以便在时间结束时通知用户
if t == -1:
    tk.messagebox.showinfo("时间到！", "番茄钟成功完成！\n你值得休息一下！")
    popup_1.destroy()
    global pomo_count
    pomo_count += 1
    pygame.mixer.music.load("./Pomodoro_GUI/beep.wav")
    pygame.mixer.music.play(loops=0)

def main():
    """
    此函数生成番茄钟计时器的主屏幕，提供选择25分钟工作计时器、5分钟休息计时器、
    用于额外专注的网站屏蔽功能，以及查看你投入工作时间的统计信息的选项。
    """

    # 创建根窗口（主屏幕）
    global root
    root = tk.Tk()
    root.title('计时器')
    root.geometry('470x608')

    # 设置屏幕背景
    bg = tk.PhotoImage(file = "./Pomodoro_GUI/bg.png")
    label1 = tk.Label( root, image = bg)
    label1.place(x = 0, y = 0)

    intro1 = tk.Label(root, text = '番茄钟计时器', bg = 'snow', fg = 'maroon', font = ('Arial', 25, 'bold'))
    intro1.place(x=100, y=100)

    blocker_btn = tk.Button(root, text = '网站屏蔽器', command = blocker, font = ('Arial', 12, 'bold'),
    bg='gold', activebackground='yellow', height = 3, width = 25)
    blocker_btn.place(x=100, y=150)

    start_btn = tk.Button(root, text = '开始工作计时', command = pomodoro_timer, font = ('Arial', 12,
    'bold'), bg='gold', activebackground='yellow', height = 3, width = 25)
    start_btn.place(x=100, y=250)

    break_btn = tk.Button(root, text = '开始休息计时', command = break_timer, font = ('Arial', 12,
    'bold'), bg='gold', activebackground='yellow', height = 3, width = 25)
    break_btn.place(x=100, y=350)

    report_btn = tk.Button(root, text = '显示报告', command = show_report, font = ('Arial', 12, 'bold'),
    bg='gold', activebackground='yellow', height = 3, width = 25)
    report_btn.place(x=100, y=450)

    root.mainloop()

if __name__ == '__main__':
    main()

## 25. Pyduku

解决数独，或让Python为你解决！

- [X] 自己玩数独
- [X] 让Python为你玩并解决它！
- [X] 生成随机数独谜题

### 构建工具

- [Python3](https://www.python.org/)

**源代码：**

```python
import tkinter as tk
from tkinter import font
# from time import sleep
import random

count = 0

class Sudoku:
    #Canvas background
    canvas_bg = "#fafafa" #impure white
    #Grid lines
    line_normal = "#4f4f4f" #dark grey
    line_thick = "#000000" #pure black
    #cell highlight box
    hbox_green = "#15fa00" #light green
    hbox_red = "#d61111" #red

    def __init__(self, master):
        #A record of all cells and their attributes
        self.grid = {}
        #A small edit window which will be initilized and displayed on click
        self.e = None
        self.canvas_width = 300
        self.canvas_height = 300
        #The sudoku grid
        self.canvas = tk.Canvas(master,bg=self.canvas_bg, width=self.canvas_width, height=self.canvas_height)
        self.t = tk.Entry(self.canvas)
        self.t.bind("<KeyRelease>",self.keyPressed)
        self.canvas.grid(columnspan=3)
        self.canvas.bind("<Button 1>",self.click)
        #Solve button
        self.btn_solve = tk.Button(master,text='Solve', command=self.wrapper, width=8)
        self.btn_solve.grid(row=1, padx=5, pady=5)
        #Generate button
        self.btn_gen = tk.Button(master,text='Generate', command=self.Generate, width=8)
        self.btn_gen.grid(row=1, column=1, padx=5, pady=5, sticky=tk.E)
        #Difficulty selector
        self.set_difficulty = tk.IntVar(master,1)
        self.difficulty_selector = tk.OptionMenu(master,self.set_difficulty,1,2,3,4,5)
        self.difficulty_selector.grid(row=1, column=2, pady=5, sticky=tk.W)
        #Individual cell width and height
        self.cell_width = self.canvas_width/9
        self.cell_height = self.canvas_height/9
        #Draw vertical lines
        for x in range(1,9):
            width=1
            fill=self.line_normal
            if(x%3==0):
                #Draw thicker black lines for seperating 3x3 boxes
                width=2
                fill=self.line_thick
            else:
                #Draw normal thin dark-grey lines
                width=1
                fill=self.line_normal
            self.canvas.create_line(self.cell_width*x, 0, self.cell_width*x, self.canvas_height, width=width, fill=fill)
        #Draw horizontal lines in the same way
        for y in range(1,9):
            width=1
            fill=self.line_normal
            if(y%3==0):
                width=2
                fill=self.line_thick
            else:
                width=1
                fill=self.line_normal
            self.canvas.create_line(0, self.cell_height*y, self.canvas_width, self.cell_height*y, width=width, fill=fill)

    def click(self, eventorigin):
        x = eventorigin.x
        y = eventorigin.y
        #Calculate top-left x,y coords of cell clicked by mouse
        rect_x = int(x/self.cell_width)*self.cell_width
        rect_y = int(y/self.cell_height)*self.cell_height
        #Coords for drawing a square to highlight clicked cell
        coords = [rect_x,rect_y,rect_x+self.cell_width,rect_y,rect_x+self.cell_width,rect_y+self.cell_height,rect_x,rect_y+self.cell_height]
        # For some stupid reason, this line below didn't work as expected. So I had to choose the hard way.
        # h_box = self.canvas.create_rectangle(rect_x, rect_y, self.cell_width, self.cell_height, outline="#15fa00", width=3)
        #Get cell info
        editable = self.getCell(x/self.cell_width,y/self.cell_height)[1]
        if editable:
            #It's a cell you can edit
            #Show a green box highlight and edit
            h_box = self.canvas.create_polygon(coords, outline=self.hbox_green, fill="", width=3)
            self.edit(rect_x, rect_y)
        else:
            #It's a cell containing a clue number, cannot edit
            #Show a red box highlight
            h_box = self.canvas.create_polygon(coords, outline=self.hbox_red, fill="", width=3)
        self.canvas.after(200,lambda : self.canvas.delete(h_box))

    def edit(self,cordx:int,cordy:int):
        #Create a entry inside a small canvas window
        #make sure it's actuall initilized before deleting it
        if self.e is None:
            #Not initilized, else block skipped
            pass
        else:
            #Canvas window initilized, delete and reset it to current position
            self.canvas.delete(self.e)
        #Create a mini edit window that just fits the cell
        self.e = self.canvas.create_window(cordx+1,cordy+1,window=self.t,width=self.cell_width-1,height=self.cell_height-2,anchor=tk.NW)
        #Clean up
        self.t.delete(0,tk.END)
        self.t.focus_set()

    def keyPressed(self, event):
        val = self.t.get().strip()
        try:
            #If input is a number between 1-9, this won't raise any errors
            val = int(val)
            if(val>9 or val<0):
                raise ValueError
        except ValueError:
            print("Invalid input!")
            self.t.delete(0,tk.END)
        else:
            #Get x,y coords of edit window and calculate cell row,column values
            x,y = (self.t.winfo_x())/self.cell_width,(self.t.winfo_y())/self.cell_height
            #Update cell with new value
            self.updateCell(val,x,y)
            self.canvas.delete(self.e)

    def updateCell(self,value,x,y,editable=True):
        #Get cell information stored in dict self.grid
        t = self.getCell(x,y)
        #Update values
        t[0] = value
        t[1] = editable
        text=value
        if value==0:
            text=' '
        #Update display value by using item id
        self.canvas.itemconfigure(t[2],text=text)
        self.canvas.update()
        #Update the dict
        self.grid[(x,y)] = t

    def getCell(self, x:int, y:int):
        #Returns info of cell at 'x' row 'y' column
        x=int(x)
        y=int(y)
        val = self.grid[(x,y)]
        return val

    def populate(self, X:[[]]):
        #Populates the sudoku grid with given 9x9 matrix and also store it in a dict
        c = self.canvas
        #The bookeeping is managed as shown below
        """Dict->(X,Y) : [value,True/Flase,id]
                    ^     ^     ^     ^
                X,Y coords value editable object id"""
        for i in range(9):
            for j in range(9):
                #Calculate x,y position of center of cell
                text_x = j*self.cell_width+self.cell_width/2
                text_y = i*self.cell_height+self.cell_height/2
                val = X[i][j]
                if val == 0:
                    t = c.create_text(text_x,text_y,text=' ',font=('Times', 14))
                    self.grid[(j,i)] = [ val, True, t]
                else:
                    t = c.create_text(text_x,text_y,text=val,font=('Times', 15, 'bold'))
                    self.grid[(j,i)] = [ val, False, t]

    def clearGrid(self):
        #Utility function to clear the grid, this will also wipe out the puzzle from memory
        for i in range(9):
            for j in range(9):
                self.updateCell(0,i,j)

    def getValue(self, row:int, col:int):
        #Return value at row, column
        return self.grid[(row,col)][0]

    def printGrid(self):
        #Utility function to print the grid
        for i in range(9):
            x=[]
            for j in range(9):
                x.append(self.getValue(j,i))
            print(x)

    def wrapper(self):
        #A small wrapper funtion that performs some small tasks before solving
        global count
        #Reset the count
        count = 0
        #Delete edit boxes if any
        self.canvas.delete(self.e)
        #Lock the buttons and start solving
```

## 26. PYQT5 密码生成器

Main.py 源代码：

```python
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QMessageBox
from PyQt5 import QtGui
import sys
import random_pass as rp
import logging

logging.basicConfig(filename="passwords.txt", format="%(message)s", level=logging.INFO)

with open("showMessage", "r") as f:
    showM = f.read()
    if showM == "1":
        showM = True
    else:
        showM = False


class window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.x = 500
        self.y = 500
        self.title = "password-gen"


    def start(self):
        self.setGeometry(100, 100, self.x, self.y)
        self.setWindowTitle(self.title)
        self.setFixedSize(self.x, self.y)
        self.setWindowIcon(QtGui.QIcon("lock.png"))

        self.label1 = QLabel(self)
        self.label1.setText("characters:")
        self.label1.move(190, 50)

        self.charsInput = QLineEdit(self)
        self.charsInput.setText("a,b,c,d")
        self.charsInput.setGeometry(190, 100, 100, 30)

        self.label2 = QLabel(self)
        self.label2.setText("length:")
        self.label2.move(190, 150)

        self.passLength = QLineEdit(self)
        self.passLength.setText("5")
        self.passLength.move(190, 200)

        self.button1 = QPushButton(self)
        self.button1.setText("generate password")
        self.button1.clicked.connect(self.generatePassword)
        self.button1.setGeometry(170, 240, 150, 30)

        self.deletePassButton = QPushButton(self)
        self.deletePassButton.setText("Delete")
        self.deletePassButton.setGeometry(10, 460, 120, 30)
        self.deletePassButton.clicked.connect(self.deletePopUp)
        self.show()


    def generatePassword(self):
        global showM
        self.chars = self.charsInput.text().split(",")
        self.passLen = int(self.passLength.text())

        self.password = rp.randomPass(self.chars, self.passLen)

        logging.info(f"password: {self.password} ")

        if showM:
            self.messageBox()


    def messageBox(self):
        message = QMessageBox()
        message.setText("The password was written to password.txt, show this again?")
        message.setWindowTitle("password")
        message.setIcon(QMessageBox.Question)
        message.setStandardButtons(QMessageBox.No|QMessageBox.Yes)
        message.buttonClicked.connect(self.YesNo)

        x = message.exec_()


    def YesNo(self, button):
        if button.text() == "&Yes":
            pass
        elif button.text() == "&No":
            with open("showMessage", "w") as f:
                f.write("0")


    def deletePasswords(self, button):
        if button.text() == "&Yes":
            try:
                with open("passwords.txt", "w") as f:
                    f.write("")
            except:
                raise FileNotFoundError("password file not found please press generate password")


    def deletePopUp(self):
        message = QMessageBox()
        message.setText("Are you sure you want to delete all the passwords")
        message.setIcon(QMessageBox.Warning)
        message.setStandardButtons(QMessageBox.Yes|QMessageBox.Cancel)
        message.setDefaultButton(QMessageBox.Cancel)
        message.buttonClicked.connect(self.deletePasswords)

        x = message.exec_()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    win = window()
    win.start()

    sys.exit(app.exec_())
```

Random Pass.py 源代码：

```python
import random


def randChar(chars):
    ranChar = random.choice(chars)

    return ranChar


def randomPass(chars, passLen):
    password = ""
```

## 27. Python 自动绘图

### *演示*：

### 在你的电脑上运行：

-   确保你已安装 Python 3.7.x 或 Python 3.8.x，如果没有，请点击[此处](https://www.python.org/downloads/)进行安装！
-   安装 PyAutoGUI：`pip install pyautogui`
-   将此项目克隆到你的桌面：`git clone "https://github.com/tusharnankani/PythonAutoDraw"`
-   打开命令行或终端
-   切换到相应游戏的目录：`cd "Desktop\PythonAutoDraw"`
-   运行：`python python-auto-draw.py`

### 基础：

```python
>>> import pyautogui
```

```python
>>> screenWidth, screenHeight = pyautogui.size() # 获取主显示器的尺寸。
```

```python
>>> currentMouseX, currentMouseY = pyautogui.position() # 获取鼠标的XY坐标。
```

```python
>>> pyautogui.moveTo(100, 150) # 将鼠标移动到XY坐标。
```

```python
>>> pyautogui.click() # 点击鼠标。
>>> pyautogui.click(100, 200) # 将鼠标移动到XY坐标并点击。
>>> pyautogui.click('button.png') # 查找 button.png 在屏幕上的位置并点击。
```

```python
>>> pyautogui.move(0, 10) # 将鼠标从当前位置向下移动10像素。
>>> pyautogui.doubleClick() # 双击鼠标。
>>> pyautogui.moveTo(500, 500, duration=2, tween=pyautogui.easeInOutQuad) # 使用缓动函数在2秒内移动鼠标。
```

```python
>>> pyautogui.write('Hello world!', interval=0.25) # 输入文字，每个按键之间暂停四分之一秒
>>> pyautogui.press('esc') # 按下 Esc 键。所有按键名称都在 pyautogui.KEY_NAMES 中。
```

```python
>>> pyautogui.keyDown('shift') # 按下 Shift 键并保持。
>>> pyautogui.press(['left', 'left', 'left', 'left']) # 按下左箭头键4次。
>>> pyautogui.keyUp('shift') # 松开 Shift 键。
```

```python
>>> pyautogui.hotkey('ctrl', 'c') # 按下 Ctrl-C 快捷键组合。
```

```python
>>> pyautogui.alert('This is the message to display.') # 显示一个警告框，并暂停程序直到点击确定。
```

**源代码：**

```python
import pyautogui
import time

# 从编辑器切换到画图程序的时间；
time.sleep(10)

# 程序结束前将保持点击状态；
pyautogui.click()

# 可根据需要调整
distance = 250

while distance > 0:
    # 向右
    pyautogui.dragRel(distance, 0, duration = 0.1)
    distance -= 5

    # 向下
    pyautogui.dragRel(0, distance, duration = 0.1)

    # 向左
    pyautogui.dragRel(-distance, 0, duration = 0.1)
    distance -= 5

    # 向上
    pyautogui.dragRel(0, -distance, duration = 0.1)
```

## 28. Pyweather

一个预测任何给定城市天气的 Python 脚本。

**源代码：**

```python
# 导入所需模块
import requests, json

# 在此处输入你的 openweathermap.org API 密钥
api_key = 'Your API key goes here'

# 存储 API 返回 URL 的基础 URL
base_url = "http://api.openweathermap.org/data/2.5/weather?"

# 在此输入城市名称
city_name = input('Enter city name: ')

complete_url = base_url + 'appid=' + api_key + '&q=' + city_name
response = requests.get(complete_url)
x = response.json()

# 检查城市名称的有效性
if x['cod'] != '404':
    y = x['main']
    current_temperature = y['temp']
    current_pressure = y['pressure']
    current_humidity = y['humidity']
    z = x['weather']
    weather_description = z[0]['description']
    q = x['wind']
    wind_speed = q['speed']
    wind_direction = q['deg']
    k = x['clouds']
    cloudliness = k['all']

    print("Temperature (in Kelvin) = " + str(current_temperature) + "\n Atmospheric Pressure (in hPa) = " + str(current_pressure) + "\n Humidity (in percentage) = " + str(current_humidity) + "\n Wind Speed (in m/s) = " + str(wind_speed) + "\n Wind Direction (in degrees) = " + str(wind_direction) + "\n Cloudliness (in percentage) = " + str(cloudliness) + "\n Weather Description = " + str(weather_description))
else:
    print("City Not Found")
```

## 29. 使用 Python 生成二维码

此脚本接受任何 URL 的链接并生成相应的二维码。

### 使用的库

-   qrcode

#### 安装所需的外部模块

-   运行 `pip install qrcode`

### 如何运行脚本

-   在脚本中提供你想要的 URL
-   执行 `python3 generate_qrcode.py`

**源代码：**

```python
import qrcode

input_URL = "https://www.google.com/"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=15,
    border=4,
)

qr.add_data(input_URL)
qr.make(fit=True)

img = qr.make_image(fill_color="red", back_color="white")
img.save("url_qrcode.png")

print(qr.data_list)
```

## 30. 动态条形图动画

### 所需的包

**确保你正在使用 Python 虚拟环境**

```bash
pip install jupyterlab
```

```bash
pip install pandas
```

```bash
pip install requests
```

或者

```bash
pip install -r requirements.txt
```

### 依赖项

jupyterlab==2.2.2
matplotlib==3.3.0
notebook==6.1.1
numpy==1.19.1
pandas==1.1.0
requests==2.24.0

**源代码文件：**

![](img/2d09155385e2d53be8e53bc9b9a3735b_94_0.png)

## 模块 4 项目 31-40

## 31. 随机密码生成器

##### 这个简单的项目是使用 Python 库函数如 `string` 和 `random` 制作的。

-   `string.ascii_letters`
    -   下述 ascii_lowercase 和 ascii_uppercase 常量的连接。此值不依赖于区域设置。

-   `string.ascii_lowercase`
    -   小写字母 <kbd>abcdefghijklmnopqrstuvwxyz</kbd>。此值不依赖于区域设置且不会改变。

-   `string.ascii_uppercase`
    -   大写字母 <kbd>ABCDEFGHIJKLMNOPQRSTUVWXYZ</kbd>。此值不依赖于区域设置且不会改变。

-   `string.digits`
    -   字符串 <kbd>0123456789</kbd>。

-   `string.hexdigits`
    -   字符串 <kbd>0123456789abcdefABCDEF</kbd>。

-   `string.octdigits`
    -   字符串 <kbd>01234567</kbd>。

-   `string.punctuation`
    -   在 C 语言区域设置中被视为标点字符的 ASCII 字符串：`!"#$%&'()*+,-./:;<=>?@[\]^_{|}~`

**Python-密码生成器源代码：**

```python
import random
import string

total = string.ascii_letters + string.digits + string.punctuation
length = 16

password = "".join(random.sample(total, length))

print(password)
```

**Random_password_gen 源代码：**

```python
import random
import math

alpha = "abcdefghijklmnopqrstuvwxyz"
num = "0123456789"
special = "@#$%&*"

# pass_len=random.randint(8,13) #无需用户输入
pass_len = int(input("Enter Password Length"))

# 根据 50-30-20 公式确定密码长度
alpha_len = pass_len//2
num_len = math.ceil(pass_len*30/100)
special_len = pass_len-(alpha_len+num_len)

password = []

def generate_pass(length, array, is_alpha=False):
    for i in range(length):
        index = random.randint(0, len(array) - 1)
        character = array[index]
        if is_alpha:
            case = random.randint(0, 1)
            if case == 1:
                character = character.upper()
        password.append(character)

# 字母密码
generate_pass(alpha_len, alpha, True)
# 数字密码
generate_pass(num_len, num)
# 特殊字符密码
generate_pass(special_len, special)
# 打乱生成的密码列表
random.shuffle(password)
# 将列表转换为字符串
gen_password = ""
for i in password:
    gen_password = gen_password + str(i)
print(gen_password)
```

## 32. 随机维基百科文章

一个将维基百科上的任意随机文章保存到文本文件的应用程序。

使用：
`pip install htmlparser` 和 `pip install beautifulsoup4`

**依赖项：**
**HTMLParser==0.0.2**

**源代码：**

```python
from bs4 import BeautifulSoup
import requests

# 尝试打开一个随机的维基百科文章
# Special:Random 会打开随机文章
res = requests.get("https://en.wikipedia.org/wiki/Special:Random")
res.raise_for_status()

# pip install htmlparser
wiki = BeautifulSoup(res.text, "html.parser")

r = open("random_wiki.txt", "w+", encoding='utf-8')

# 将标题添加到文本文件
heading = wiki.find("h1").text

r.write(heading + "\n")
for i in wiki.select("p"):
    # 可选打印文本
    # print(i.getText())
    r.write(i.getText())

r.close()
print("File Saved as random_wiki.txt")
```

## 33. 从列表中随机选取单词

这是一个实用的程序，可以从给定的列表中随机选择一个单词。

### 如何运行脚本

```bash
python Random_word_from_list.py
```

请确保你希望从中随机选取单词的文件位于同一目录下。

**源代码：**

```python
import sys
import random

# check if filename is supplied as a command line argument
if sys.argv[1:]:
    filename = sys.argv[1]
else:
    filename = input("What is the name of the file? (extension included): ")

try:
    file = open(filename)
except (FileNotFoundError, IOError):
    print("File doesn't exist!")
    exit()
# handle exception

# get number of lines
num_lines = sum(1 for line in file if line.rstrip())

# generate a random number between possible interval
random_line = random.randint(0, num_lines)

# re-iterate from first line
file.seek(0)

for i, line in enumerate(file):
    if i == random_line:
        print(line.rstrip()) # rstrip removes any trailing newlines :)
        break
```

## 34. 高质量YouTube视频下载器

这是一个用于生成随机电子邮件地址的Python脚本。

### 要求

要运行此脚本，你需要安装`progressbar`包。

### 在终端运行命令以安装包

```bash
$ pip install progressbar
```

### 使用命令运行程序

```bash
$ python random_email_generator.py
```

**源代码：**

```python
import random
import string
import csv
import progressbar

""" Ask user for total number of emails required"""
def getcount():
    rownums = input("How many email addresses?: ")
    try:
        rowint = int(rownums)
        return rowint
    except ValueError:
        print("Please enter an integer value")
        return getcount()

"""Below function creates a random length of email between 1-20 characters length and adds domain and extension to give the resulting email"""

def makeEmail():
    extensions = ['com', 'net', 'org', 'gov']
    domains = [
        'gmail', 'yahoo', 'comcast', 'verizon', 'charter', 'hotmail',
        'outlook', 'frontier'
    ]

    finalext = extensions[random.randint(0, len(extensions) - 1)]
    finaldom = domains[random.randint(0, len(domains) - 1)]

    accountlen = random.randint(1, 20)

    finalacc = ''.join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(accountlen))

    finale = finalacc + "@" + finaldom + "." + finalext
    return finale

# Take the total count of emails and pass them to getcount()
howmany = getcount()

# counter for While loop
counter = 0

# empty array to add emails
emailarray = []

print("Creating email addresses...")
print("Progress: ")

prebar = progressbar.ProgressBar(maxval=int(howmany))

for i in prebar(range(howmany)):
    while counter < howmany:
        emailarray.append(str(makeEmail()))
        counter += 1
        prebar.update(i)

print("Creation completed.")

for i in emailarray:
    print(i)
```

## 35. 树莓派-Sonoff

这是一个使用树莓派的Sonoff设备。

### 硬件要求：

1. 树莓派（任何版本均可）

![](img/2d09155385e2d53be8e53bc9b9a3735b_103_0.png)

2. 继电器板

![](img/2d09155385e2d53be8e53bc9b9a3735b_103_1.png)

### 运行

在树莓派上运行`Main.py`文件

```bash
python main.py
```

一个Flask服务器将在`http://0.0.0.0:8000/`上运行。将继电器连接到GPIO 2。

**源代码：**

```python
from flask import Flask, render_template, request, redirect
from gpiozero import LED
from time import sleep

led = LED(2)

app = Flask(__name__)

@app.route("/")
def home():
    if led.value == 1:
        status = 'ON'
    else:
        status = 'OFF'
    return render_template('home.html', status=status)

@app.route("/on")
def on():
    led.on()
    return "LED on"

@app.route("/off")
def off():
    led.off()
    return "LED off"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
```

## 36. 递归密码生成器

使用递归方法生成指定长度的随机密码。

### 先决条件

无

### 如何运行脚本

执行 `python3 generator.py`

**源代码：**

```python
import random
import string

def stretch(text,maxlength):
    if len(text) < maxlength:
        randomChar = get_random_char()
        return stretch(text+randomChar,maxlength)
    else:
        return text

def get_random_char():
    chars = string.printable
    randomChar = chars[random.randint(0,len(chars)-1)]
    return randomChar

while 1:
    maxlen = input(' [?] Enter a length for your password (e for exit): ')
    try:
        maxlength = int(maxlen)
        print("",stretch("",maxlength),"\n")
    except:
        if maxlen == 'e':
            break
        print('Please Enter an integer')
```

## 37. 无API的Reddit爬虫

- 使用BeautifulSoup（一个用于网络爬虫的Python库），此脚本有助于爬取所需的subreddit，以获取其帖子的所有相关数据。
- 在`fetch_reddit.py`中，我们获取用户输入的subreddit名称、标签和要爬取的最大帖子数量，然后获取并将所有这些信息存储在数据库文件中。
- 在`display_reddit.py`中，我们向用户显示数据库中的所需结果。

### 设置说明

- 可以按照以下方式安装依赖项：

```bash
$ pip install -r requirements.txt
```

**依赖项：**
beautifulsoup4==4.9.3
certifi==2020.12.5
chardet==4.0.0
idna==2.10
requests==2.25.1
soupsieve==2.2.1
urllib3==1.26.4

**源代码文件：**

![](img/2d09155385e2d53be8e53bc9b9a3735b_107_0.png)

![](img/2d09155385e2d53be8e53bc9b9a3735b_107_1.png)

## 38. 缩减图像大小

使用Python的openCV库缩减图像文件大小的脚本。

### 先决条件

openCV库

```bash
pip install opencv-python
```

### 如何运行脚本

- 将名为`input.jpg`的jpg格式图像添加到此文件夹中。
- 运行`reduce_image_size.py`脚本。
- 调整大小后的输出图像将在此文件夹中生成。

**源代码：**

```python
# import openCV library for image handling
import cv2

# read image to be resized by imread() function of openCV library
img = cv2.imread('input.jpg')
print(img.shape)

# set the ratio of resized image
k = 5
width = int((img.shape[1])/k)
height = int((img.shape[0])/k)

# resize the image by resize() function of openCV library
scaled = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
print(scaled.shape)

# show the resized image using imshow() function of openCV library
cv2.imshow("Output", scaled)
cv2.waitKey(500)
cv2.destroyAllWindows()

# get the resized image output by imwrite() function of openCV library
cv2.imwrite('resized_output_image.jpg', scaled)
```

## 39. 石头剪刀布游戏

与电脑对战。

- 你可以输入想要玩的游戏局数。
- 每轮结束后还会显示一个计分窗口。

**源代码：**

```python
#START;

import random

#DEFAULT;
my_dict={'R':'Rock','P':'Paper','S':'Scissors'}
user_count=0
comp_count=0

#INPUT;
games=int(input("\nEnter the number of games you want to play: "))

while(comp_count+user_count<games):
    #WHILE LOOP STARTS;

    flag=0

    user_input=input("\nUser's Input: ")[0]
    user_input=user_input.upper()
    #The [0] after the input() will assign the first charcter of input to the variable;
    #Hence, the user can enter anything, anyway;
    #Example: The user can enter Rock or rock or r or R or ro or any such thing which represents Rock;
    #It will always take input as a R
    #Thereby, increasing the user input window;

    for i in my_dict.keys():
        if(user_input==i):            #If the entered input is confined to Rock, Paper or Scissors;
            flag=1
            break
    if(flag!=1):            #If not, run the loop again;
        print("INVALID INPUT")
        continue

    comp_input=random.choice(list(my_dict.keys()))    #Random Key from the dictionary my_dict i.e. R,P or S;

    print("Computer's Input: ", my_dict[comp_input])
```

## 40. 使用笔记本电脑摄像头实现房间安防

### 依赖项：

*Open CV*

```
python
pip install opencv-python
```

*Flask*

```
python
pip install flask
```

### 运行：

*运行此脚本以启动本地服务器*

```
python
python main.py
```

*实时流*

```
http://0.0.0.0:5000
```

**main.py 源代码：**

```python
# main.py
# import the necessary packages
from flask import Flask, render_template, Response
from camera import VideoCamera
app = Flask(__name__)
@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')
def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=False)
```

### Camera.py 源代码：

```python
import cv2

face_cascade=cv2.CascadeClassifier("faces.xml")
ds_factor=0.6
class VideoCamera(object):
    def __init__(self):
        #capturing video
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        #releasing camera
        self.video.release()
    def get_frame(self):
        #extracting frames
        ret, frame = self.video.read()
        frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,
        interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            break
        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
```

## 模块5 项目 41-50

## 41. 从 HackerNews 网站抓取新闻

从 HackerNews 网站抓取新闻

一个从 HackerNews 抓取多个页面的脚本

### 如何运行脚本

在命令行中进入文件目录，然后运行 `python main.py`

**源代码：**

```python
import requests
import os
from bs4 import BeautifulSoup, SoupStrainer
# Makes Output Directory if it does not exist
if not os.path.exists(os.path.join(os.getcwd(), 'HackerNews')):
    os.makedirs(os.path.join(os.getcwd(), 'HackerNews'))
"""
@params page_no: The page number of HackerNews to fetch.
Adding only page number in order to add multiprocess support in future.
@params verbose: Adds verbose output to screen instead
of running the program silently.
"""

def fetch(page_no, verbose=False):
    # Should be unreachable, but just in case
    if page_no <= 0:
        raise ValueError('Number of Pages must be greater than zero')
    page_no = min(page_no, 20)
    i = page_no
    if verbose:
        print('Fetching Page {}...'.format(i))
    try:
        res = requests.get('https://news.ycombinator.com/?p=' + str(i))
        only_td = SoupStrainer('td')
        soup = BeautifulSoup(res.content, 'html.parser', parse_only=only_td)
        tdtitle = soup.find_all('td', attrs={'class': 'title'})
        tdmetrics = soup.find_all('td', attrs={'class': 'subtext'})
        with open(os.path.join('HackerNews', 'NewsPage{}.txt'.format(i)), 'w+') as f:
            f.write('-' * 80)
            f.write('\n')
            f.write('Page {}'.format(i))
            tdtitle = soup.find_all('td', attrs={'class': 'title'})
            tdrank = soup.find_all(
                'td',
                attrs={
                    'class': 'title',
                    'align': 'right'})
            tdtitleonly = [t for t in tdtitle if t not in tdrank]
            tdmetrics = soup.find_all('td', attrs={'class': 'subtext'})
            tdt = tdtitleonly
            tdr = tdrank
            tdm = tdmetrics
            num_iter = min(len(tdr), len(tdt))
            for idx in range(num_iter):
                f.write('\n' + '-' * 80 + '\n')
                rank = tdr[idx].find('span', attrs={'class': 'rank'})
                titl = tdt[idx].find('a', attrs={'class': 'storylink'})
                url = titl['href'] if titl and titl['href'].startswith(
                    'https') else 'https://news.ycombinator.com/' + titl['href']
                site = tdt[idx].find('span', attrs={'class': 'sitestr'})
                score = tdm[idx].find('span', attrs={'class': 'score'})
                time = tdm[idx].find('span', attrs={'class': 'age'})
                author = tdm[idx].find('a', attrs={'class': 'hnuser'})
                f.write(
                    '\nArticle Number: ' +
                    rank.text.replace(
                        '.',
                        '') if rank else '\nArticle Number: Could not get article number')
                f.write(
                    '\nArticle Title: ' +
                    titl.text if titl else '\nArticle Title: Could not get article title')
                f.write(
                    '\nSource Website: ' +
                    site.text if site else '\nSource Website: https://news.ycombinator.com')
                f.write(
                    '\nSource URL: ' +
                    url if url else '\nSource URL: No URL found for this article')
                f.write(
                    '\nArticle Author: ' +
                    author.text if author else '\nArticle Author: Could not get article author')
                f.write(
                    '\nArticle Score: ' +
                    score.text if score else '\nArticle Score: Not Scored')
                f.write(
                    '\nPosted: ' +
                    time.text if time else '\nPosted: Could not find when the article was posted')
                f.write('\n' + '-' * 80 + '\n')
    except (requests.ConnectionError, requests.packages.urllib3.exceptions.ConnectionError) as e:
        print('Connection Failed for page {}'.format(i))
    except requests.RequestException as e:
        print("Some ambiguous Request Exception occurred. The exception is " + str(e))

while(True):
    try:
        pages = int(
            input('Enter number of pages that you want the HackerNews for (max 20): '))
        v = input('Want verbose output y/[n] ?')
        verbose = v.lower().startswith('y')
        if pages > 20:
            print('A maximum of only 20 pages can be fetched')
        pages = min(pages, 20)
        for page_no in range(1, pages + 1):
            fetch(page_no, verbose)
        break
    except ValueError:
        print('\nInvalid input, probably not a positive integer\n')
        continue
```

## 42. 名言抓取器

### 先决条件

- beautifulsoup4
- requests

运行 `pip install -r requirements.txt` 以安装所需的外部模块。

### 如何运行脚本

执行 `python3 quote_scraper.py`

**依赖项：**
**beautifulsoup4**
**requests==2.23.0**

**源代码：**

```python
from bs4 import BeautifulSoup
import requests
import csv

# URL to the website
url='http://quotes.toscrape.com'

# Getting the html file and parsing with html.parser
html=requests.get(url)
bs=BeautifulSoup(html.text,'html.parser')

# Tries to open the file
try:
    csv_file=open('quote_list.csv','w')
    fieldnames=['quote','author','tags']
    dictwriter=csv.DictWriter(csv_file,fieldnames=fieldnames)

    # Writes the headers
    dictwriter.writeheader()

    #While next button is found in the page the loop runs
    while True:
        # Loops through quote in the page
        for quote in bs.findAll('div',{'class':'quote'}):
            #Extract the text part of quote, author and tags
            text=quote.find('span',{'class':'text'}).text
            author=quote.find('small',{'class':'author'}).text
            tags=[]
            for tag in quote.findAll('a',{'class':'tag'}):
                tags.append(tag.text)
            #Writes the current quote,author and tags to a csv file
            dictwriter.writerow({'quote':text,'author':author,'tags':tags})

        #Finds the link to next page
        next=bs.find('li',{'class':'next'})
        if not next:
            break

        #Gets and parses the html file of next page
        html=requests.get(url+next.a.attrs['href'])
        bs=BeautifulSoup(html.text,'html.parser')
except:
    print('Unknown Error!!!')
finally:
    csv_file.close()
```

## 43. 抓取 Medium 文章

[Medium](https://medium.com/) 是一个包含优质文章的网站，许多程序员都在使用它。
此脚本会向用户请求一篇 Medium 文章的网址，抓取其文本内容，并将其保存到当前目录下名为 `scraped_articles` 的文件夹中的一个文本文件里。
文件夹 `scraped_articles` 中有 3 个文本文件，作为文章抓取方式的示例。

### 前提条件

使用 `pip` 安装 `requirements.txt` 中给出的模块。
设备需要有可用的网络连接。

### 如何运行脚本

像运行任何其他 Python 文件一样运行它。

**依赖：**
**beautifulsoup4==4.9.1**
**requests==2.23.0**

**源代码：**

```
import os
import sys
import requests
import re
from bs4 import BeautifulSoup

# switching to current running python files directory
os.chdir('\'.join(__file__.split('/')[:-1]))

# function to get the html of the page
def get_page():
    global url
    url = input('Enter url of a medium article: ')
    # handling possible error
    if not re.match(r'https?://medium.com/',url):
        print('Please enter a valid website, or make sure it is a medium article')
        sys.exit(1)
    res = requests.get(url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'html.parser')
    return soup

# function to remove all the html tags and replace some with specific strings
def purify(text):
    rep = {"<br>": "\n", "<br/>": "\n", "<li>": "\n"}
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    text = re.sub('\<.*?\>', '', text)
    return text

# function to compile all of the scraped text in one string
def collect_text(soup):
    fin = f'url: {url}\n\n'
    main = (soup.head.title.text).split('|')
    global title
    title = main[0].strip()
    fin += f'Title: {title.upper()}\n{main[1].strip()}'

    header = soup.find_all('h1')
    j = 1

    try:
        fin += '\n\nINTRODUCTION\n'
        for elem in list(header[j].previous_siblings)[::-1]:
            fin += f'\n{purify(str(elem))}'
    except:
        pass

    fin += f'\n\n{header[j].text.upper()}'
    for elem in header[j].next_siblings:
        if elem.name == 'h1':
            j+=1
            fin += f'\n\n{header[j].text.upper()}'
            continue
        fin += f'\n{purify(str(elem))}'
    return fin

# function to save file in the current directory
def save_file(fin):
    if not os.path.exists('./scraped_articles'):
        os.mkdir('./scraped_articles')
    fname = './scraped_articles/' + '_'.join(title.split()) + '.txt'
    with open(fname, 'w', encoding='utf8') as outfile:
        outfile.write(fin)
    print(f'File saved in directory {fname}')

# driver code
if __name__ == '__main__':
    fin = collect_text(get_page())
    save_file(fin)
```

## 44. 屏幕录制器

它录制计算机屏幕。

### 使用的模块

- time
- PIL
- numpy
- cv2

### 工作原理

- 运行脚本时，它会捕获屏幕帧。
- 然后它会返回带有实时变化的屏幕录制。

**源代码：**

```
import cv2
import numpy as np
from PIL import ImageGrab
import time

def screenrecorder():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    name = int(round(time.time() * 1000))
    name = '{}.avi'.format(name)
    out = cv2.VideoWriter(name, fourcc, 5.0, (1920, 1080))

    while True:
        img = ImageGrab.grab()
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        cv2.imshow("Screen Recorder", frame)
        out.write(frame)

        if cv2.waitKey(1) == 27:
            break

out.release()
cv2.destroyAllWindows()

screenrecorder()
```

## 45. 使用 Python 发送电子邮件

```
import smtplib
import csv
from string import Template
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
def read_template(filename):
    with open(filename, 'r', encoding='utf-8') as template_file:
        template_file_content = template_file.read()
    return Template(template_file_content)
def main():
    message_template = read_template('template.txt')
    MY_ADDRESS = '**********@gmail.com'
    PASSWORD = '**************'
    # set up the SMTP server
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    s.login(MY_ADDRESS, PASSWORD)

    with open("details.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # the below statement will skip the first row
        next(csv_reader)
        for lines in csv_reader:
            msg = MIMEMultipart() # create a message
            # add in the actual person name to the message template
            message = message_template.substitute(PERSON_NAME=row[0],MATH=row[2],
                                                  ENG=row[3],SCI=row[4])
            print(message)
            # setup the parameters of the message
            msg['From']=MY_ADDRESS
            msg['To']=lines[1]
            msg['Subject']="Mid term grades"
            # add in the message body
            msg.attach(MIMEText(message, 'plain'))
            # send the message via the server set up earlier.
            s.send_message(msg)
            del msg
    # Terminate the SMTP session and close the connection
    s.quit()
if __name__ == '__main__':
    main()
```

## 46. 从 CSV 文件发送电子邮件

此项目包含一个简单的批量邮件脚本，它向收件人列表发送相同的消息。

### 依赖项

此项目仅需要 Python 标准库（更具体地说，是 `csv`、`email` 和 `smtplib` 模块）。

### 运行脚本

该脚本需要两个配置文件：

- `emails.csv` 应包含要发送消息的电子邮件地址。
- `credentials.txt` 应包含您的 SMTP 服务器登录凭据，用户名和密码各占一行，没有额外的空格或其他装饰。

项目的目录中包含两个示例文件，您可能都需要编辑它们。

一旦设置好这些文件，只需运行

```
python Send_emails.py
```

### 开发思路

一个合适的邮件发送器应该使用 `Cc:` 或 `Bcc:`，并且只发送一次相同的消息。

不要随意使用此功能；您的电子邮件提供商和/或收件人的提供商可能有自动过滤器，会快速屏蔽任何发送多条相同消息的人。

该脚本只是硬编码了 Gmail.com 的约定。其他提供商可能使用不同的端口号和身份验证机制。

**源代码：**

```
import csv
from email.message import EmailMessage
import smtplib

def get_credentials(filepath):
    with open("credentials.txt", "r") as f:
        email_address = f.readline()
        email_pass = f.readline()
    return (email_address, email_pass)

def login(email_address, email_pass, s):
    s.ehlo()
    # start TLS for security
    s.starttls()
    s.ehlo()
    # Authentication
    s.login(email_address, email_pass)
    print("login")

def send_mail():
    s = smtplib.SMTP("smtp.gmail.com", 587)
    email_address, email_pass = get_credentials("./credentials.txt")
    login(email_address, email_pass, s)

    # message to be sent
    subject = "Welcome to Python"
    body = """Python is an interpreted, high-level,
    general-purpose programming language.\n
    Created by Guido van Rossum and first released in 1991,
    Python's design philosophy emphasizes code readability\n
    with its notable use of significant whitespace"""

    message = EmailMessage()
    message.set_content(body)
    message['Subject'] = subject

    with open("emails.csv", newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        for email in spamreader:
            s.send_message(email_address, email[0], message)
            print("Send To " + email[0])

    # terminating the session
```

## 47. 发送短信

如果你执行 `import sys`，就可以通过 `sys.foo` 或 `sys.bar()` 来访问 `sys` 模块中的函数和变量。这可能会导致大量输入，尤其是当使用子模块中的内容时（例如，我经常需要访问 `django.contrib.auth.models.User`）。为了避免这种冗余，你可以将一个、多个或所有变量和函数引入全局作用域。`from os.path import exists` 允许你直接使用 `exists()` 函数，而无需每次都加上 `os.path` 前缀。

如果你想从 `os.path` 导入多个变量或函数，可以使用 `from os.path import foo, bar`。

**源代码：**

```python
from twilio.rest import Client

# Your Account SID from twilio.com/console
account_sid = "0000"
# Your Auth Token from twilio.com/console
auth_token = "0000"

client = Client(account_sid, auth_token)

message = client.messages.create(
    to="0000",
    from_="0000",
    body="Hello from Python!")

print(message.sid)
```

## 48. 设置闹钟

此脚本允许你设置一个闹钟，并在选定的时间后播放你选择的音乐。
**此脚本仅适用于 Windows**

### 用法

```
$ python alarm.py
```

### 示例输出

```
$ python3 alarm.py

##########################
####### Alarm Program #######
##########################

Set the alarm time (e.g. 01:10): 00:01

        Select any alarm music:

        1. The Four Seasons
        2. Carnival
        3. Renaissance
        4. Variations
        5. Dreamy Nights
        6. Lakhau Hajarau
        7. New Horizon
        8. Crusade
        9. Mozart Wakes
        10. Morning Calm

Enter the index of the listed musics (e.g. 1): 1
>> Alarm music has been set --> The Four Seasons

>>> Alarm has been set successfully for 00:01! Please dont close the program! <<<
```

**源代码：**

```python
import datetime
import os
import re
import subprocess

def rename_files_with_whitespaces(cd, files, extra_path=""):
    for file in files:
        if " " in file:
            renamed_file = file.replace(" ", "_")
            os.rename(os.path.join(cd, extra_path, file), os.path.join(cd, extra_path, renamed_file))

def clean_filename(file):
    return ' '.join(map(str.capitalize, file[:-4].split('_')))

def set_alarm():
    stop = False
    error = True
    while error:
        user_set_time = ":".join(map(lambda x: str(x).zfill(2), input("\nSet the alarm time (e.g. 01:10): ").split(":")))

        if re.match(r"^[0-9]{2}:[0-9]{2}$", user_set_time):
            playback_time = f"{user_set_time}:00.000000"
            error = False
        else:
            print(">>> Error: Time format invalid! Please try again!\n")

cd = os.path.dirname(os.path.realpath(__file__))
musics_path = os.path.join(cd, "musics")

rename_files_with_whitespaces(cd, os.listdir(musics_path), "musics")

musics = os.listdir(musics_path)
if len(musics) < 1:
    print(">>> Error: No music in the musics folder! Please add music first!\n")
    exit()

elif len(musics) == 1:
    print(">> Alarm music has been set default --> " + clean_filename(musics[0]))
    selected_music = musics[0]

else:
    error = True
    while error:
        try:
            print("\nSelect any alarm music:\n")
            for i in range(1, len(musics) + 1):
                print(f"{i}. {clean_filename(musics[i - 1])}")

            user_input = int(input("\nEnter the index of the listed musics (e.g. 1): "))
            selected_music = musics[user_input - 1]
            print(">> Alarm music has been set --> " + clean_filename(selected_music))
            error = False

        except:
            print(">>> Error: Invalid Index! Please try again!\n")

    print(f"\n>>> Alarm has been set successfully for {user_set_time}! Please dont close the program! <<<")
    while stop == False:
        current_time = str(datetime.datetime.now().time())
        if current_time >= playback_time:
            stop = True
            subprocess.run(('cmd', '/C', 'start', f"{cd}\musics\{selected_music}"))
            print(">>> Alarm ringing! Closing the program!! Bye Bye!!! <<<")

def display_header(header):
    print("")
    print("###########################".center(os.get_terminal_size().columns))
    print(f"####### {header} #######".center(os.get_terminal_size().columns))
    print("###########################".center(os.get_terminal_size().columns))

if __name__ == "__main__":
    display_header("Alarm Program")
    set_alarm()
```

## 49. 关闭或重启你的设备

### 电源选项

此脚本用于关闭或重启你的计算机。

### 前提条件

无

### 如何运行脚本

运行脚本的步骤及合适的示例。

1. 在命令行中输入以下内容：
   ```
   python PowerOptions.py
   ```
2. 按回车键并等待提示。输入 “r” 重启或 “s” 关机。

示例：
```
python PowerOptions.py
Use 'r' for restart and 's' for shutdown: r
```

**源代码：**

```python
import os
import platform

def shutdown():
    if platform.system() == "Windows":
        os.system('shutdown -s')
    elif platform.system() == "Linux" or platform.system() == "Darwin":
        os.system("shutdown -h now")
    else:
        print("Os not supported!")

def restart():
    if platform.system() == "Windows":
        os.system("shutdown -t 0 -r -f")
    elif platform.system() == "Linux" or platform.system() == "Darwin":
        os.system('reboot now')
    else:
        print("Os not supported!")

command = input("Use 'r' for restart and 's' for shutdown: ").lower()

if command == "r":
    restart()
elif command == "s":
    shutdown()
else:
    print("Wrong letter")
```

## 50. 正弦 vs 余弦

```python
import numpy as np
import matplotlib.pyplot as plot
# Get x values of the sine wave
time = np.linspace(-2*np.pi, 2*np.pi, 256, endpoint=True)
# Amplitude of the sine wave is sine of a variable like time
amplitude_sin = np.sin(time)
amplitude_cos = np.cos(time)
# Plot a sine wave using time and amplitude obtained for the sine wave
plot.plot(time, amplitude_sin)
plot.plot(time, amplitude_cos)
# Give a title for the sine wave plot
plot.title('Sine & Cos wave')
# Give x axis label for the sine wave plot
plot.xlabel('Time')
# Give y axis label for the sine wave plot
plot.ylabel('Amplitude')
plot.grid(True, which='both')
plot.axhline(y=0, color='k')
plot.show()
```

## 51. 网站屏蔽器

### 描述

这是一个旨在为基于 Windows 的系统实现网站屏蔽功能的脚本。它利用计算机的 hosts 文件，并将其作为后台进程运行，阻止用户以数组格式输入的网站访问。

### 所需的第三方库：

该项目仅需要 Python 的 `datetime` 库。

### 导入库：

在你的计算机上打开命令提示符并输入以下内容：
在脚本的控制台中，输入：
```python
import time
from datetime import datetime as dt
```

### 运行脚本：

在你的 Python IDE 中打开脚本后，执行代码以获得控制台输出窗口。打开你的浏览器并尝试访问你屏蔽的网站。当脚本成功运行时，你将在浏览器中看到 `This site can't be reached` 错误。

**注意：**

> 在某些系统中，为了防止恶意软件攻击，默认情况下可能禁止访问计算机的 hosts 文件。因此，脚本在执行时修改 hosts 文件可能会显示错误。
> 请访问[此处](https://www.technipages.com/windows-access-denied-when-modifying-hosts-or-lmhosts-file)简要了解如何解决此问题。

### 输出：

当你尝试访问被屏蔽的网站时，浏览器的行为如下：

![](img/2d09155385e2d53be8e53bc9b9a3735b_135_0.png)

*根据你的列表更改，所有提到的网站的访问都将被拒绝。*

**源代码：**

```python
import time
from datetime import datetime as dt

# Windows host file path
hostsPath = r"C:\Windows\System32\drivers\etc\hosts"
redirect = "127.0.0.1"

# Add the website you want to block, in this list
websites = [
    "www.youtube.com", "youtube.com", "www.facebook.com",
    "facebook.com"
]
while True:
    # Duration during which, website blocker will work
    if dt(dt.now().year,
          dt.now().month,
          dt.now().day, 9) < dt.now() < dt(dt.now().year,
                                          dt.now().month,
                                          dt.now().day, 18):
        print("Access denied to Website")
        with open(hostsPath, 'r+') as file:
            content = file.read()
            for site in websites:
                if site in content:
                    pass
                else:
                    file.write(redirect + " " + site + "\n")
```

else:
    with open(hostsPath, 'r+') as file:
        content = file.readlines()
    file.seek(0)
for line in content:
    if not any(site in line for site in websites):
        file.write(line)
    file.truncate()
print("Allowed access!")
time.sleep(5)

## 52. 短信自动化

### 短信自动化功能：

- 首先在Twilio注册，并添加需要发送消息的电话号码。
- 运行脚本并输入API密钥和电话号码后，消息将被发送。

### 设置步骤：

- 首先在Twilio注册。
- 然后验证您要从中发送消息的电话号码。
- 现在添加接收者的电话号码并进行验证。
- 同时允许地理位置[权限](https://www.twilio.com/console/sms/settings/geo-permissions)。

### 自动化操作说明：

#### 步骤 1：

打开终端。

#### 步骤 2：

定位到Python文件所在的目录。

#### 步骤 3：

运行命令：python script.py/python3 script.py。

#### 步骤 4：

坐下来放松。让脚本完成工作。

### 依赖要求

- twilio

### 源代码：

```python
from twilio.rest import Client

api = input("Enter your ACCOUNT SID: ")
auth = input("Enter your AUTH TOKEN: ")
from_number = input("Enter number from which you want to send the SMS: ")
message = input("Enter the message: ")
to_number = input(
    "Enter comma separated numbers to which you want to send the SMS: ")
lists = to_number.split(",")
groupnum = []
for i in lists:
    groupnum.append(i)

account_sid = api
auth_token = auth
client = Client(account_sid, auth_token)

for i in range(len(groupnum)):
    client.messages.create(from_=from_number, body=message, to=groupnum[i])
```

### 如何下载此项目：

作为我们的特别读者，您值得享有特殊权限。请按照以下步骤下载所有这些项目以供进一步练习。

1. 前往 - https://www.edcredibly.com/s/store/courses/description/52-Python-Projects，这是我们的官方网站。
2. 应用优惠券代码 – SPECIAL，即可免费获取此课程。
3. 无需支付任何费用即可结账并注册。
4. 下载文件并享受学习。

干杯，学习愉快！！

### 关于作者

“Edcorner Learning”在Udemy上拥有大量学生，超过90000名学生，评分在4.1或以上。

Edcorner Learning是Edcredibly的一部分。

Edcredibly是一个在线电子学习平台，提供所有热门技术的课程，旨在最大化学习成果，并为专业人士和学生提供职业机会。Edcredibly在其自有平台上拥有超过100000名学生，并在Google Play商店 – Edcredibly App上获得了4.9的评分。

欢迎查看或加入我们的课程：

Edcredibly 官网 - https://www.edcredibly.com/

Edcredibly 应用 – https://play.google.com/store/apps/details?id=com.edcredibly.courses

Edcorner Learning Udemy - https://www.udemy.com/user/edcorner/

请查看我们在Kindle商店提供的其他电子书。