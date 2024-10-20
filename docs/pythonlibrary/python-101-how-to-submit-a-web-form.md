# Python 101:如何提交 web 表单

> 原文：<https://www.blog.pythonlibrary.org/2012/06/08/python-101-how-to-submit-a-web-form/>

今天我们将花一些时间来看看让 Python 提交 web 表单的三种不同方式。在这种情况下，我们将使用[duckduckgo.com](http://duckduckgo.com/)对术语“python”进行网络搜索，并将结果保存为 HTML 文件。我们将使用 Python 包含的 urllib 模块和两个第三方包:[请求](http://docs.python-requests.org/en/latest/)和[机械化](http://wwwsearch.sourceforge.net/mechanize/)。我们有三个小脚本要讲，所以让我们开始吧！

### 使用 urllib 提交 web 表单

我们将从 **urllib** 和 **urllib2** 开始，因为它们包含在 Python 的标准库中。我们还将导入**网络浏览器**来打开搜索结果进行查看。代码如下:

```py

import urllib
import urllib2
import webbrowser

data = urllib.urlencode({'q': 'Python'})
url = 'http://duckduckgo.com/html/'
full_url = url + '?' + data
response = urllib2.urlopen(full_url)
with open("results.html", "w") as f:
    f.write(response.read())

webbrowser.open("results.html")

```

当你想提交一个网络表单时，你要做的第一件事就是弄清楚这个表单叫什么，以及你要发布的 url 是什么。如果你去 duckduckgo 的网站查看源代码，你会注意到它的动作指向一个相对链接，“/html”。所以我们的网址是“http://duckduckgo.com/html”。输入字段被命名为“q ”,因此要向 duckduckgo 传递一个搜索项，我们必须将 url 连接到“q”字段。结果被读取并写入磁盘。最后，我们使用 **webbrowser** 模块打开保存的结果。现在让我们看看在使用**请求**包时这个过程有什么不同。

### 提交带有请求的 web 表单

requests 包确实使提交的形式更优雅了一点。让我们来看看:

```py

# Python 2.x example
import requests

url = 'https://duckduckgo.com/html/'
payload = {'q':'python'}
r = requests.get(url, params=payload)
with open("requests_results.html", "w") as f:
    f.write(r.content)

```

对于请求，您只需要创建一个字典，将字段名作为键，将搜索词作为值。然后使用 **requests.get** 进行搜索。最后，使用产生的请求对象“r”，并访问保存到磁盘的其**内容**属性。为了简洁起见，我们跳过了这个例子(以及下一个例子)中的 webbrowser 部分。

在 Python 3 中，需要注意的是 **r.content** 现在返回字节而不是字符串。如果我们试图把它写到磁盘上，这将导致一个**类型的错误**被引发。要修复它，我们需要做的就是将文件标志从“w”更改为“wb”，就像这样:

```py

with open("requests_results.html", "wb") as f:
    f.write(r.content)

```

现在我们应该准备好看看**机械化**如何完成它的工作。

### 用 mechanize 提交 web 表单

mechanize 模块有很多有趣的特性，可以用 Python 浏览互联网。遗憾的是，它不支持 javascript。不管怎样，让我们继续表演吧！

```py

import mechanize

url = "http://duckduckgo.com/html"
br = mechanize.Browser()
br.set_handle_robots(False) # ignore robots
br.open(url)
br.select_form(name="x")
br["q"] = "python"
res = br.submit()
content = res.read()
with open("mechanize_results.html", "w") as f:
    f.write(content)

```

正如您所看到的，mechanize 比其他两种方法更加冗长。我们还需要告诉它忽略 robots.txt 指令，否则它会失败。当然，如果你想成为一个好网民，那么你就不应该忽视它。无论如何，首先，您需要一个浏览器对象。然后打开 url，选择表单(在本例中是“x”)，像以前一样用搜索参数设置一个字典。请注意，在每种方法中，dict 设置都略有不同。接下来，您提交查询并阅读结果。最后，你把结果保存到磁盘，你就完成了！

### 包扎

在这三者中，requests 可能是最简单的，紧接着是 urllib。尽管 Mechanize 比其他两个做得更多。它是为屏幕抓取和网站测试而设计的，所以有点冗长也就不足为奇了。你也可以用 selenium 提交表单，但是你可以在这个博客的[档案](https://www.blog.pythonlibrary.org/2012/05/06/website-automation-with-python-firefox-and-selenium/)中读到。我希望你觉得这篇文章很有趣，也许会有启发。下次见！

### 进一步阅读

*   [屏幕刮样章](http://rhodesmill.org/brandon/chapters/screen-scraping/)

### 源代码

*   [form_submission.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/06/form_submission.zip)