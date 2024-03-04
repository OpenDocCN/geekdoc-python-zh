# 美丽的汤 4 蟒蛇

> 原文：<https://www.pythonforbeginners.com/beautifulsoup/beautifulsoup-4-python>

## 概观

本文是对 Python 中 BeautifulSoup 4 的介绍。如果你想了解更多，我推荐你阅读官方文档，在这里找到。

## 什么是美汤？

Beautiful Soup 是一个 Python 库，用于从 HTML 和 XML 文件中提取数据。

## BeautifulSoup 3 还是 4？

美汤 3 已经被美汤 4 取代了。Beautiful Soup 3 只在 Python 2.x 上工作，但 Beautiful Soup 4 也在 Python 3.x 上工作，Beautiful Soup 4 速度更快，功能更多，可以与 lxml 和 html5lib 这样的第三方解析器一起工作。所有新项目都应该使用 Beautiful Soup 4。

## 安装美丽的汤

如果运行 Debian 或者 Ubuntu，可以用系统包管理器安装美汤

```py
apt-get install python-bs4 
```

美汤 4 是通过 PyPi 发布的，如果不能用系统打包器安装，可以用 easy_install 或者 pip 安装。这个包的名字是 beautifulsoup4，同样的包在 Python 2 和 Python 3 上工作。

```py
easy_install beautifulsoup4

pip install beautifulsoup4 
```

如果没有安装 easy_install 或者 pip，可以[下载](http://www.crummy.com/software/BeautifulSoup/bs4/download/4.0/ "crummy_bs4")漂亮的 Soup 4 源码 tarball，用 setup.py 安装，python setup.py install

## 漂亮的一组用法

安装完成后，您可以开始使用 BeautifulSoup。在 Python 脚本的开始，导入库，现在你必须传递一些东西给 BeautifulSoup 来创建一个 Soup 对象。这可能是一个文档或一个 URL。BeautifulSoup 不会为你获取网页，你必须自己去做。这就是我结合使用 urllib2 和 BeautifulSoup 库的原因。

## 过滤

搜索 API 可以使用一些不同的过滤器。下面我将展示一些例子，告诉你如何将这些过滤器传递给 find_all 这样的方法。你可以根据标签的名称、属性、字符串的文本或者这些的组合来使用这些过滤器。

### 一根绳子

最简单的过滤器是字符串。将一个字符串传递给一个搜索方法，Beautiful Soup 将根据该字符串执行匹配。这段代码查找文档中所有的‘b’标签(你可以用任何你想查找的标签替换 b)

```py
soup.find_all('b') 
```

如果你传入一个字节串，Beautiful Soup 会假设这个字符串的编码是 UTF 8。您可以通过传入 Unicode 字符串来避免这种情况。

### 正则表达式

如果您传入一个[正则表达式](https://www.pythonforbeginners.com/regex/regular-expressions-in-python)对象，Beautiful Soup 将使用 match()方法过滤该正则表达式。此代码查找名称以字母“b”开头的所有标签，在本例中是“body”标签和“b”标签:

```py
import re
for tag in soup.find_all(re.compile("^b")):
    print(tag.name) 
```

此代码查找名称包含字母“t”的所有标签:

```py
for tag in soup.find_all(re.compile("t")):
    print(tag.name) 
```

### 一份名单

如果您传入一个列表，Beautiful Soup 将允许对列表中的任何项目进行字符串匹配。这段代码查找所有的' a '标签和所有的' b '标签

```py
print soup.find_all(["a", "b"]) 
```

### 真实的

值 True 匹配它所能匹配的一切。此代码查找文档中的所有标签，但不查找文本字符串:

```py
for tag in soup.find_all(True):
    print(tag.name) 
```

### 一项功能

如果其他匹配都不适合您，请定义一个将元素作为唯一参数的函数。如果你想这么做，请查看官方文档。

## 漂亮的一组物体

例如，我们将使用您当前所在的网站(https://www.pythonforbeginners.com)来解析内容中的数据，我们只需为它创建一个 BeautifulSoup 对象，它将为我们传入的 url 内容创建一个 Soup 对象。从这一点来看，我们现在可以在 soup 对象上使用漂亮的 Soup 方法。我们可以使用 prettify 方法将 BS 解析树转换成格式良好的 Unicode 字符串

### Find_all 方法

find_all 方法是 BeautifulSoup 中最常用的方法之一。它会查看标签的后代，并检索与您的过滤器匹配的所有后代。

```py
soup.find_all("title")

soup.find_all("p", "title")

soup.find_all("a")

soup.find_all(id="link2") 
```

让我们看一些如何使用 BS 4 的例子

```py
from bs4 import BeautifulSoup
import urllib2

url = "https://www.pythonforbeginners.com"

content = urllib2.urlopen(url).read()

soup = BeautifulSoup(content)

print soup.prettify()

print title
>> 'title'? Python For Beginners

print soup.title.string
>> ? Python For Beginners

print soup.p
```

```py
print soup.a
[Python For Beginners](https://www.pythonforbeginners.com) 
```

## 导航解析树

如果您想知道如何浏览该树，请参见官方[文档](http://www.crummy.com/software/BeautifulSoup/bs4/doc/#navigating-the-tree "navigation_bs4")。在那里你可以读到以下内容:

**往下走**

*   使用标记名导航
*   。内容和。儿童
*   。后代
*   。字符串。字符串和剥离的 _ 字符串

**上升**

*   。父母
*   。双亲

**横着走**

*   。next_sibling 和。上一个 _ 兄弟姐妹
*   。下一个 _ 兄弟姐妹和。上一个 _ 兄弟姐妹

**来来回回**

*   。next_element 和。前一个元素
*   。next_elements 和。前 _ 个元素

## 提取在页面“a”标签中找到的所有 URL

一个常见的任务是使用 find_all 方法提取在页面的‘a’标签中找到的所有 URL，给我们一个带有标签‘a’的元素的完整列表。

```py
for link in soup.find_all('a'):
    print(link.get('href')) 
```

```py
Output: 
```

```py
..https://www.pythonforbeginners.com
..https://www.pythonforbeginners.com/python-overview-start-here/
..https://www.pythonforbeginners.com/dictionary/
..https://www.pythonforbeginners.com/python-functions-cheat-sheet/
..https://www.pythonforbeginners.com/lists/python-lists-cheat-sheet/
..https://www.pythonforbeginners.com/loops/
..https://www.pythonforbeginners.com/python-modules/
..https://www.pythonforbeginners.com/strings/
..https://www.pythonforbeginners.com/sitemap/
...
... 
```

## 从页面中提取所有文本

另一个常见任务是从页面中提取所有文本:

```py
print(soup.get_text()) 
```

```py
Output: 
```

```py
Python For Beginners
Python Basics
Dictionary
Functions
Lists
Loops
Modules
Strings
Sitemap
...
... 
```

## 从 Reddit 获取所有链接

作为最后一个例子，让我们从 Reddit 获取所有链接

```py
from bs4 import BeautifulSoup
import urllib2

redditFile = urllib2.urlopen("http://www.reddit.com")
redditHtml = redditFile.read()
redditFile.close()

soup = BeautifulSoup(redditHtml)
redditAll = soup.find_all("a")
for links in soup.find_all('a'):
    print (links.get('href')) 
```

```py
Output: 
```

```py
#content
..http://www.reddit.com/r/AdviceAnimals/
..http://www.reddit.com/r/announcements/
..http://www.reddit.com/r/AskReddit/
..http://www.reddit.com/r/atheism/
..http://www.reddit.com/r/aww/
..http://www.reddit.com/r/bestof/
..http://www.reddit.com/r/blog/
..http://www.reddit.com/r/funny/
..http://www.reddit.com/r/gaming/
..http://www.reddit.com/r/IAmA/
..http://www.reddit.com/r/movies/
..http://www.reddit.com/r/Music/
..http://www.reddit.com/r/pics/
..http://www.reddit.com/r/politics/
... 
```

更多信息，请参见[官方](https://beautiful-soup-4.readthedocs.org/en/latest/ "bs4")文档。