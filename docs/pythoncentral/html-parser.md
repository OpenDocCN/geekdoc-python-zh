# HTML 解析器:如何抓取 HTML 内容

> 原文：<https://www.pythoncentral.io/html-parser/>

## 先决条件

需要了解以下内容:

1.  Python 3
2.  基本 HTML
3.  Urllib2(非强制但推荐)
4.  基本的 OOP 概念
5.  Python 数据结构-列表、元组

## 为什么要解析 HTML？

Python 是广泛用于从网页中抓取数据的语言之一。这是收集信息的一种非常简单的方法。例如，它对于快速提取网页中的所有链接并检查它们的有效性非常有帮助。这只是许多潜在用途的一个例子...所以继续读下去吧！

下一个问题是:这些信息是从哪里提取的？要回答这个，我们来举个例子。进入网站[【NYTimes】](https://www.nytimes.com/)点击页面右键。选择 *查看页面来源* 或者直接按键盘上的*Ctrl+u*键。将打开一个新页面，其中包含许多链接、HTML 标记和内容。这是 HTML 解析器为纽约时报抓取内容的来源！

## 什么是 HTML 解析器？

HTML 解析器，顾名思义，就是简单地解析一个网页的 HTML/XHTML 内容，并提供我们要找的信息。这是一个用各种方法定义的类，可以覆盖这些方法来满足我们的需求。注意，要使用 HTML 解析器，必须获取网页。正因如此，HTML 解析器经常与 [urllib2](https://docs.python.org/2/library/urllib2.html) 一起使用。

要使用 HTML 解析器，你必须导入这个模块:

```py
from html.parser import HTMLParser
```

## HTML 解析器中的方法

1.  HTML Parser . feed(data)-HTML 解析器读取数据就是通过这个方法。此方法接受 unicode 和字符串格式的数据。它不断处理获得的数据，并等待缓冲不完整的数据。只有在使用这个方法输入数据之后，才能调用 HTML 解析器的其他方法。
2.  HTMLParser.close() - 调用这个方法来标记 HTML 解析器输入提要的结束。
3.  HTMLParser.reset() - 该方法重置实例，所有未处理的数据都将丢失。
4.  html parser . handle _ starttag(tag，attrs) - 这个方法只处理开始标签，像 *< title >* 。tag 参数指的是开始标记的名称，而 attrs 指的是开始标记内的内容。例如，对于标签 *< Meta name="PT" >* ，方法调用将是 handle_starttag('meta '，[('name '，' PT')])。注意，标记名被转换成小写，标记的内容被转换成键、值对。如果一个标签有属性，它们将被转换成一个键、值对元组并添加到列表中。例如，在标签*<meta name = " application-name " content = " The New York Times "/>*中，方法调用将是 handle_starttag('meta '，[('name '，' application-name ')，(' content '。《纽约时报》)])。
5.  html parser . handle _ end tag(tag)-这个方法与上面的方法非常相似，除了它只处理像 *< /body >* 这样的结束标记。因为在结束标记中没有内容，所以这个方法只有一个参数，就是标记本身。比如 *< /body >* 的方法调用会是:handle_endtag('body ')。类似于 handle_starttag(tag，attrs)方法，这也将标记名转换为小写。
6.  html parser . handle _ startendtag(tag，attrs) - 顾名思义，这个方法处理的是开始结束标签，比如，<a href = http://nytimes . com/>。参数 tag 和 attrs 类似于 HTMLParser.handle_starttag(tag，attrs)方法。
7.  html parser . handle _ data(data)-该方法用于处理类似 *< p > ……的数据/内容。< /p >* 。当您想要查找特定的单词或短语时，这尤其有用。这种方法结合正则表达式可以创造奇迹。
8.  html parser . handle _ comment(data)-顾名思义，这种方法是用来处理类似 *<的评论的！-* ny *times - >* 并且方法调用类似于 html parser . handle _ comment(‘ny times’)。

咻！这需要处理很多东西，但是这些是 HTML 解析器的一些主要(也是最有用的)方法。如果你感到头晕，不要担心，让我们来看一个例子，让事情变得更清楚一点。

## 【HTML 解析器是如何工作的？

既然你已经具备了理论知识，让我们来实际测试一下。要尝试以下示例，您必须安装 urllib2 或按照以下步骤安装它:

1.  安装[pip](https://pip.pypa.io/en/stable/installation/)
2.  安装 urllib - pip 安装 urllib2

```py
from html.parser import HTMLParser
import urllib.request as urllib2

class MyHTMLParser(HTMLParser):

   #Initializing lists
   lsStartTags = list()
   lsEndTags = list()
   lsStartEndTags = list()
   lsComments = list()

   #HTML Parser Methods
   def handle_starttag(self, startTag, attrs):
       self.lsStartTags.append(startTag)

   def handle_endtag(self, endTag):
       self.lsEndTags.append(endTag)

   def handle_startendtag(self,startendTag, attrs):
       self.lsStartEndTags.append(startendTag)

   def handle_comment(self,data):
       self.lsComments.append(data)

#creating an object of the overridden class
parser = MyHTMLParser()

#Opening NYTimes site using urllib2
html_page = html_page = urllib2.urlopen("https://www.nytimes.com/")

#Feeding the content
parser.feed(str(html_page.read()))

#printing the extracted values
print(“Start tags”, parser.lsStartTags)
#print(“End tags”, parser.lsEndTags)
#print(“Start End tags”, parser.lsStartEndTags)
#print(“Comments”, parser.lsComments)
```

或者，如果您不想安装 urllib2，可以直接将一串 HTML 标记提供给解析器，如下所示:

```py
parser = MyHTMLParser()
parser.feed('<html><body><title>Test</title></body>')

```

当您处理大量数据时，一次打印一个输出以避免崩溃！

注意:如果你得到错误: *IDLE 不能启动进程*，在管理员模式下启动你的 Python IDLE。这应该能解决问题。

## 异常情况

HTMLParser。当 HTML 解析器遇到损坏的数据时，这个异常被抛出。这个异常以三种属性的形式给出信息。 *msg* 属性告诉您错误的原因， *lineno* 属性指定错误发生的行号， *offset* 属性给出构造开始的确切字符。

## 结论

关于 HTML 解析器的这篇文章到此结束。一定要自己尝试更多的例子来提高自己的理解能力！如果您需要开箱即用的电子邮件解析器或 pdf 表格解析解决方案，我们的姐妹网站可以满足您的需求，直到您掌握 python mojo。请务必阅读一下 [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) ，它是Python 中另一个帮助抓取 HTML 的神奇模块。但是，要使用这个模块，您必须安装它。坚持学习，快乐 Pythoning 化！