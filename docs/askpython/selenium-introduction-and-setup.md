# Python Selenium 简介和设置

> 原文：<https://www.askpython.com/python-modules/selenium-introduction-and-setup>

本教程的目的是向您介绍 selenium，并展示安装 Selenium 和 webdriver 以实现浏览器自动化的过程。在本文中，我们假设您的机器上已经安装了 python。

需要注意的重要一点是，浏览器自动化和 web 废弃在他们的方法中完全是白帽子，Web 浏览器本身正式支持它，为自动化和测试提供 Web 驱动程序，只是使用代码打开的浏览器窗口被标记为“该浏览器由自动化测试软件控制”

## 硒是什么？

Selenium 是一个开源项目，它提供了一系列工具来自动化网络浏览器。它还用于创建 web scrapers，以从网页中获取(抓取)所需的数据。

使用 Python Selenium 可以完成的一些任务有:

*   自动化浏览器任务，如登录、加入会议、滚动、网上冲浪等。
*   从网站/网页获取文本、excel 文件、代码等形式的数据。

浏览器自动化的一个关键组件是 web 驱动程序。Webdriver 是一个 API 的集合，它使得与浏览器的交互变得容易。将 Selenium 和 webdriver 结合在一起，可以非常容易地将枯燥的 web 任务自动化。

## 安装 Selenium

要开始为浏览器自动化和 web 抓取设置我们的计算机，我们需要从安装一些工具和库开始。

### 1.安装 Selenium

首先，我们将使用 [pip](https://www.askpython.com/python-modules/python-pip) 安装 selenium 包。使用 **pip install package_name** 命令可以很容易地安装任何 python 包。

打开计算机的命令提示符，输入以下命令。您也可以在系统或 IDE 的终端中运行该命令。

```py
pip install selenium

```

它将在我们的机器上安装最新版本的 selenium。

### 2.安装 Selenium 驱动程序

在设置过程中，我们的第二个任务是按照我们的浏览器安装 webdriver，我们打算用它来实现自动化。

安装网络驱动程序时，我们需要确保它与我们的网络浏览器有相同的版本。每个浏览器都有自己的网络驱动程序，由母公司维护。

下面是下载流行浏览器的驱动程序的链接，分别是 Mozilla Firefox、Google Chrome 和 Microsoft Edge。

下载 Mozilla Firefox Webdriver: [此处](https://github.com/mozilla/geckodriver/releases)
下载谷歌 Chrome Webdriver: [此处](https://chromedriver.chromium.org/downloads)下载微软 Webdriver: [此处](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/)

下载 selenium 和所需的 web 驱动程序后，您就可以编写 python 脚本来自动化 web 浏览器了。

### 3.在 Python 中导入硒

由于我们已经下载了所需的工具和库，最后一步我们需要导入所需的，如下所示:

**注意**:我们需要将安装的 web 驱动文件的位置(保存在我们的计算机上)传递给 web driver 方法。

```py
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome('C://software/chromedriver.exe')

```

**提示**:在每段代码中，我们可以聪明地跳过这一步，将下载的 web 驱动文件的位置声明(保存)为环境变量，而不是每次都将位置作为参数传递。

## 硒的推荐读物

最后，我们完成了设置，您可以按照我们关于 Python Selenium 的教程开始抓取网页并自动化您的网页浏览器任务

*   [使用 Python Selenium 打开网页 URLs】](https://www.askpython.com/python-modules/open-webpage-urls-selenium)
*   [从网页获取数据](https://www.askpython.com/python-modules/fetch-website-data-selenium)
*   [使用 Selenium 获取股票市场数据](https://www.askpython.com/python-modules/fetch-stock-market-data-selenium)