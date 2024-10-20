# 如何在 Selenium 中打开网页 URL

> 原文：<https://www.askpython.com/python-modules/open-webpage-urls-selenium>

在本文中，我们将学习如何在 Selenium 中访问和打开网页 URL。Python Selenium 是以编程方式操作 web 浏览器的强大工具。它兼容所有浏览器，运行在所有主流操作系统上，其脚本是用各种语言编写的，包括 Python、Java、C#等。其中我们将使用 Python。

Selenium Python 绑定提供了一个简单的 API 来访问 Selenium WebDrivers，比如 Firefox、Internet Explorer、Chrome、Remote 等等。Selenium 目前支持 Python、3.5 和更高版本。

***推荐阅读:[Python Selenium 入门——安装设置](https://www.askpython.com/python-modules/selenium-introduction-and-setup)***

下面给出的代码示例一定会帮助你用 Python 打开网页 URL:

## 使用 Selenium 打开 URL

现在让我们学习如何在 Python Selenium 中访问网页和打开 URL。这是使用硒的最基本要求。一旦理解了这一点，您只需使用 XPaths 并确定如何使用您用 Python Selenium 收集的数据

### 1.安装 Python Selenium

我们将使用 [pip 命令](https://www.askpython.com/python-modules/python-pip)来安装 selenium 包。

```py
python -m pip install selenium

```

### 2.导入模块

现在让我们在 Python 代码中导入 selenium 模块，开始使用它。

```py
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome('./chromedriver.exe')

```

注意:现在我们已经安装了 Selenium，但是要访问开放的 web 浏览器并使我们的代码可以访问它们，我们需要下载浏览器的官方驱动程序并记下它的路径

这里我们给路径命名为。/chromedriver.exe '因为我们已经将驱动程序放在了 Python 脚本的同一个目录中，如果您将它保存在任何其他地方，那么您必须提供它的完整路径。

### 3.打开 URL 示例

Python Selenium 中的 URL 是使用 Selenium 模块的 **get()** 方法打开或获取的

```py
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
driver = webdriver.Chrome('./chromedriver.exe')

driver.get("https://www.google.com")
driver.close()

```

这将在一个新的测试浏览器窗口中打开谷歌的 Chrome 网站。

close()方法用于关闭浏览器窗口。

### 4.网页的标题

通过使用以下 python 命令，我们可以在控制台/终端窗口中将网页标题作为文本输出打开:

```py
print(driver.title)

```

### 5.在谷歌上搜索查询

**语法:**网站/搜索？q= '要搜索的主题'

这将在一个新窗口中显示 Chrome 浏览器中谷歌搜索 python 的结果。

```py
driver.get("https://www.google.com/search?q =Python")

```

### 6.在浏览器历史记录中来回移动

**后退驱动**在浏览器历史中后退一步。

语法:driver.back()

**前进驱动**在浏览器历史中前进了一步

语法:driver.forward()

**示例实现:**

```py
from selenium import webdriver

driver = webdriver.Chrome("./chromedriver.exe")

# opens Google
driver.get("https://www.google.com")

# open python official website
driver.get("https://www.python.org")

```

现在，这里首先谷歌将在一个新的窗口中打开，然后在同一窗口和谷歌网站上打开 python 官方网站

```py
driver.back()
# will go to Google

driver.forward()
# will go to python official website

```

您需要在 back 和 forward 方法之间使用类似 time.sleep(5)的东西来真正注意到转换。

## 结论

希望您已经学会了使用 Selenium 库在 Python 中打开网页 URL，并准备亲自尝试一下。