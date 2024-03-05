# 如何用 Python 和 InstaPy 制作一个 Instagram Bot

> 原文：<https://realpython.com/instagram-bot-python-instapy/>

[SocialCaptain](https://socialcaptain.com/pricing/) 、 [Kicksta](https://kicksta.co/pricing) 、 [Instavast](https://instavast.com/pricing/) 和[其他很多公司](https://socialmediaexplorer.com/social-media-marketing/instagram-automation-tools/)有什么共同点？它们都有助于你在 Instagram 上接触到更多的观众，获得更多的关注者，获得更多的喜欢，而你几乎不用动一根手指。他们都是通过自动化来完成的，人们为此付给他们很多钱。但是你可以免费使用 InstaPy 做同样的事情！

在本教程中，你将学习如何用 [Python](https://realpython.com/learning-paths/python-basics-book/) 和 [InstaPy](https://github.com/timgrossmann/InstaPy) 构建一个机器人，这是一个由[蒂姆·格罗曼](https://twitter.com/timigrossmann)创建的库，它**自动化**你的 Instagram 活动，以便你用最少的手动输入获得更多的关注者和喜欢。在这一过程中，您将学习使用 [Selenium](https://realpython.com/modern-web-automation-with-python-and-selenium/) 和**页面对象模式**实现浏览器自动化，它们共同作为 InstaPy 的基础。

在本教程中，您将学习:

*   Instagram 机器人如何工作
*   如何用 [Selenium](https://realpython.com/headless-selenium-testing-with-python-and-phantomjs/) 自动化浏览器
*   如何使用**页面对象模式**获得更好的可读性和可测试性
*   如何用 **InstaPy** 搭建 Instagram bot

在你创建一个 Instagram 机器人之前，你首先要学习它是如何工作的。

**重要提示:**在实施任何自动化或抓取技术之前，请务必查看 [Instagram 的使用条款](https://help.instagram.com/581066165581870)。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## Instagram 机器人如何工作

一个自动化脚本如何让你获得更多的关注者和喜欢？在回答这个问题之前，先想想一个实际存在的人是如何获得更多的关注者和喜欢的。

他们通过在平台上的持续活跃来做到这一点。他们经常发帖，关注其他人，喜欢并评论其他人的帖子。机器人的工作方式完全相同:它们根据你设定的标准，在一致的基础上跟随、喜欢和评论。

你设定的标准越好，你的结果就越好。你要确保你瞄准了正确的群体，因为你的机器人在 Instagram 上互动的人更有可能与你的内容互动。

例如，如果你在 Instagram 上销售女装，那么你可以指示你的机器人喜欢、评论和关注大多数女性或其帖子包含标签如`#beauty`、`#fashion`或`#clothes`的个人资料。这使得你的目标受众更有可能注意到你的个人资料，关注你，并开始与你的帖子互动。

但是，它在技术方面是如何工作的呢？你不能使用 [Instagram 开发者 API](https://www.instagram.com/developer/) ,因为它在这方面相当有限。进入**浏览器自动化**。它的工作方式如下:

1.  你给它你的凭证。
2.  您可以设置关注谁、留下什么评论以及喜欢哪种类型的帖子的标准。
3.  你的机器人打开浏览器，在地址栏输入`https://instagram.com`，用你的凭证登录，然后开始做你指示它做的事情。

接下来，您将构建 Instagram bot 的初始版本，它会自动登录到您的个人资料。请注意，您现在还不会使用 InstaPy。

[*Remove ads*](/account/join/)

## 如何自动化浏览器

对于这个版本的 Instagram 机器人，你将使用 [Selenium](https://selenium.dev/) ，这是 InstaPy 在引擎盖下使用的工具。

首先，[装硒](https://selenium-python.readthedocs.io/installation.html)。在安装过程中，确保你也安装了 [Firefox WebDriver](https://selenium-python.readthedocs.io/installation.html#drivers) ，因为最新版本的 [InstaPy 放弃了对 Chrome](https://github.com/timgrossmann/InstaPy/blob/master/CHANGELOG.md#breaking-changes) 的支持。这也意味着你需要在电脑上安装 [Firefox 浏览器](https://www.mozilla.org/en-US/firefox/new/)。

现在，创建一个 Python 文件，并在其中编写以下代码:

```py
 1from time import sleep
 2from selenium import webdriver
 3
 4browser = webdriver.Firefox()
 5
 6browser.get('https://www.instagram.com/')
 7
 8sleep(5)
 9
10browser.close()
```

运行代码，你会看到一个 Firefox 浏览器打开，把你导向 Instagram 登录页面。下面是代码的逐行分解:

*   **1、2 号线**进口`sleep`和`webdriver`。
*   **第 4 行**初始化火狐驱动，并设置为`browser`。
*   **第 6 行**在地址栏键入`https://www.instagram.com/`，点击 `Enter` 。
*   **第 8 行**等待五秒钟，以便您可以看到结果。否则，它会立即关闭浏览器。
*   **第 10 行**关闭浏览器。

这是`Hello, World`的硒版。现在，您可以添加登录 Instagram 个人资料的代码了。但是首先，想想你将如何手动登录到你的个人资料。您将执行以下操作:

1.  转到`https://www.instagram.com/`。
2.  单击登录链接。
3.  输入您的凭据。
4.  点击登录按钮。

上面的代码已经完成了第一步。现在修改一下，让它点击 Instagram 主页上的登录链接:

```py
 1from time import sleep
 2from selenium import webdriver
 3
 4browser = webdriver.Firefox()
 5browser.implicitly_wait(5) 6
 7browser.get('https://www.instagram.com/')
 8
 9login_link = browser.find_element_by_xpath("//a[text()='Log in']") 10login_link.click() 11
12sleep(5)
13
14browser.close()
```

注意突出显示的行:

*   **第 5 行**设定五秒钟的等待时间。如果 Selenium 找不到某个元素，那么它会等待五秒钟来加载所有内容，然后再次尝试。
*   **第 9 行**找到文本等于`Log in`的元素`<a>`。它使用了 [XPath](https://developer.mozilla.org/en-US/docs/Web/XPath) 来实现这个功能，但是还有一些其他的方法可以使用。
*   **第 10 行**为登录链接点击找到的元素`<a>`。

运行脚本，您将看到您的脚本在运行。它会打开浏览器，进入 Instagram，点击登录链接，进入登录页面。

在登录页面上，有三个重要元素:

1.  用户名输入
2.  密码输入
3.  登录按钮

接下来，更改脚本，使其找到这些元素，输入您的凭证，然后单击登录按钮:

```py
 1from time import sleep
 2from selenium import webdriver
 3
 4browser = webdriver.Firefox()
 5browser.implicitly_wait(5)
 6
 7browser.get('https://www.instagram.com/')
 8
 9login_link = browser.find_element_by_xpath("//a[text()='Log in']")
10login_link.click()
11
12sleep(2) 13
14username_input = browser.find_element_by_css_selector("input[name='username']") 15password_input = browser.find_element_by_css_selector("input[name='password']") 16
17username_input.send_keys("<your username>") 18password_input.send_keys("<your password>") 19
20login_button = browser.find_element_by_xpath("//button[@type='submit']") 21login_button.click() 22
23sleep(5)
24
25browser.close()
```

以下是这些变化的详细情况:

1.  **第 12 行**休眠两秒钟让页面加载。
2.  **第 14 行和第 15 行**查找 CSS 输入的用户名和密码。你可以使用[或者任何你喜欢的方法](https://selenium-python.readthedocs.io/locating-elements.html)。
3.  **第 17 行和第 18 行**在各自的输入框中输入你的用户名和密码。别忘了填`<your username>`和`<your password>`！
4.  **第 20 行**通过 XPath 找到登录按钮。
5.  **第 21 行**点击登录按钮。

运行该脚本，您将自动登录到您的 Instagram 个人资料。

你的 Instagram 机器人有了一个良好的开端。如果您继续编写这个脚本，那么其余部分看起来会非常相似。你可以通过向下滚动 feed 找到你喜欢的帖子，通过 CSS 找到 like 按钮，点击它，找到评论区，留下评论，然后继续。

好消息是，所有这些步骤都可以由 InstaPy 来处理。但是在您开始使用 InstaPy 之前，还有一件事您应该知道，以便更好地理解 Instapy 是如何工作的:页面对象模式。

[*Remove ads*](/account/join/)

## 如何使用页面对象模式

现在您已经编写了登录代码，您将如何为它编写一个测试呢？它看起来会像下面这样:

```py
def test_login_page(browser):
    browser.get('https://www.instagram.com/accounts/login/')
    username_input = browser.find_element_by_css_selector("input[name='username']")
    password_input = browser.find_element_by_css_selector("input[name='password']")
    username_input.send_keys("<your username>")
    password_input.send_keys("<your password>")
    login_button = browser.find_element_by_xpath("//button[@type='submit']")
    login_button.click()

    errors = browser.find_elements_by_css_selector('#error_message')
    assert len(errors) == 0
```

你能看出这段代码有什么问题吗？它不遵循[干原理](https://realpython.com/lessons/dry-principle/)。也就是说，代码在应用程序和测试代码中都是重复的。

复制代码在这种情况下尤其糟糕，因为 Selenium 代码依赖于 UI 元素，而 UI 元素往往会发生变化。当它们发生变化时，您希望在一个地方更新您的代码。这就是[页面对象模式](https://selenium.dev/documentation/en/guidelines_and_recommendations/page_object_models/)的用武之地。

使用这种模式，您可以为最重要的页面或片段创建**页面对象类**，这些页面或片段提供易于编程的接口，并隐藏窗口中的底层 widgetry。考虑到这一点，您可以重写上面的代码并创建一个`HomePage`类和一个`LoginPage`类:

```py
from time import sleep

class LoginPage:
    def __init__(self, browser):
        self.browser = browser

    def login(self, username, password):
        username_input = self.browser.find_element_by_css_selector("input[name='username']")
        password_input = self.browser.find_element_by_css_selector("input[name='password']")
        username_input.send_keys(username)
        password_input.send_keys(password)
        login_button = browser.find_element_by_xpath("//button[@type='submit']")
        login_button.click()
        sleep(5)

class HomePage:
    def __init__(self, browser):
        self.browser = browser
        self.browser.get('https://www.instagram.com/')

    def go_to_login_page(self):
        self.browser.find_element_by_xpath("//a[text()='Log in']").click()
        sleep(2)
        return LoginPage(self.browser)
```

除了主页和登录页面被表示为类之外，代码是相同的。这些类封装了在 UI 中查找和操作数据所需的机制。也就是说，有方法和[访问器](https://realpython.com/lessons/take-advantage-accessor-methods/)允许软件做任何人类能做的事情。

另一件要注意的事情是，当您使用 page 对象导航到另一个页面时，它会为新页面返回一个 page 对象。注意`go_to_log_in_page()`的返回值。如果你有另一个名为`FeedPage`的类，那么`LoginPage`类的`login()`将返回一个实例:`return FeedPage()`。

下面是使用页面对象模式的方法:

```py
from selenium import webdriver

browser = webdriver.Firefox()
browser.implicitly_wait(5)

home_page = HomePage(browser)
login_page = home_page.go_to_login_page()
login_page.login("<your username>", "<your password>")

browser.close()
```

它看起来好多了，上面的测试现在可以重写为这样:

```py
def test_login_page(browser):
    home_page = HomePage(browser)
    login_page = home_page.go_to_login_page()
    login_page.login("<your username>", "<your password>")

    errors = browser.find_elements_by_css_selector('#error_message')
    assert len(errors) == 0
```

有了这些变化，如果 UI 中有什么变化，您将不必修改您的测试。

关于页面对象模式的更多信息，请参考官方文档和[马丁·福勒的文章](https://martinfowler.com/bliki/PageObject.html)。

既然您已经熟悉了 Selenium 和页面对象模式，那么您将会对 InstaPy 如鱼得水。接下来，您将使用它构建一个基本的机器人。

**注意**:Selenium 和 Page Object 模式都广泛用于其他网站，而不仅仅是 Instagram。

## 如何用 InstaPy 搭建 insta gram Bot

在本节中，您将使用 InstaPy 构建一个 Instagram 机器人，它会自动喜欢、关注和评论不同的帖子。首先，您需要安装 InstaPy:

```py
$ python3 -m pip install instapy
```

这将在您的系统中安装`instapy`。

**注意**:最佳实践是为每个项目使用[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)，这样依赖关系就被隔离了。

[*Remove ads*](/account/join/)

### 基本特征

现在，您可以用 InstaPy 重写上面的代码，以便比较这两个选项。首先，创建另一个 Python 文件，并将以下代码放入其中:

```py
from instapy import InstaPy

InstaPy(username="<your_username>", password="<your_password>").login()
```

用你自己的替换用户名和密码，运行脚本，瞧！仅仅用**一行代码**，你就实现了**同样的结果**。

尽管结果相同，但您可以看到行为并不完全相同。除了简单地登录你的个人资料，InstaPy 还会做一些其他的事情，比如检查你的互联网连接和 Instagram 服务器的状态。这可以在浏览器或日志中直接观察到:

```py
INFO [2019-12-17 22:03:19] [username]  -- Connection Checklist [1/3] (Internet Connection Status)
INFO [2019-12-17 22:03:20] [username]  - Internet Connection Status: ok
INFO [2019-12-17 22:03:20] [username]  - Current IP is "17.283.46.379" and it's from "Germany/DE"
INFO [2019-12-17 22:03:20] [username]  -- Connection Checklist [2/3] (Instagram Server Status)
INFO [2019-12-17 22:03:26] [username]  - Instagram WebSite Status: Currently Up
```

对于一行代码来说已经很不错了，不是吗？现在是时候让脚本做比登录更有趣的事情了。

出于本例的目的，假设您的个人资料都是关于汽车的，并且您的机器人打算与对汽车感兴趣的人的个人资料进行交互。

首先，你可以喜欢一些使用`like_by_tags()`标记为`#bmw`或`#mercedes`的帖子:

```py
 1from instapy import InstaPy
 2
 3session = InstaPy(username="<your_username>", password="<your_password>")
 4session.login()
 5session.like_by_tags(["bmw", "mercedes"], amount=5)
```

这里，您为该方法提供了一个标签列表，以及每个标签的点赞数量。在本例中，您指示它喜欢十篇文章，两个标签各五篇。但是看看运行脚本后会发生什么:

```py
INFO [2019-12-17 22:15:58] [username]  Tag [1/2]
INFO [2019-12-17 22:15:58] [username]  --> b'bmw'
INFO [2019-12-17 22:16:07] [username]  desired amount: 14  |  top posts [disabled]: 9  |  possible posts: 43726739
INFO [2019-12-17 22:16:13] [username]  Like# [1/14]
INFO [2019-12-17 22:16:13] [username]  https://www.instagram.com/p/B6MCcGcC3tU/
INFO [2019-12-17 22:16:15] [username]  Image from: b'mattyproduction'
INFO [2019-12-17 22:16:15] [username]  Link: b'https://www.instagram.com/p/B6MCcGcC3tU/'
INFO [2019-12-17 22:16:15] [username]  Description: b'Mal etwas anderes \xf0\x9f\x91\x80\xe2\x98\xba\xef\xb8\x8f Bald ist das komplette Video auf YouTube zu finden (n\xc3\xa4here Infos werden folgen). Vielen Dank an @patrick_jwki @thehuthlife  und @christic_  f\xc3\xbcr das bereitstellen der Autos \xf0\x9f\x94\xa5\xf0\x9f\x98\x8d#carporn#cars#tuning#bagged#bmw#m2#m2competition#focusrs#ford#mk3#e92#m3#panasonic#cinematic#gh5s#dji#roninm#adobe#videography#music#bimmer#fordperformance#night#shooting#'
INFO [2019-12-17 22:16:15] [username]  Location: b'K\xc3\xb6ln, Germany'
INFO [2019-12-17 22:16:51] [username]  --> Image Liked!
INFO [2019-12-17 22:16:56] [username]  --> Not commented
INFO [2019-12-17 22:16:57] [username]  --> Not following
INFO [2019-12-17 22:16:58] [username]  Like# [2/14]
INFO [2019-12-17 22:16:58] [username]  https://www.instagram.com/p/B6MDK1wJ-Kb/
INFO [2019-12-17 22:17:01] [username]  Image from: b'davs0'
INFO [2019-12-17 22:17:01] [username]  Link: b'https://www.instagram.com/p/B6MDK1wJ-Kb/'
INFO [2019-12-17 22:17:01] [username]  Description: b'Someone said cloud? \xf0\x9f\xa4\x94\xf0\x9f\xa4\xad\xf0\x9f\x98\x88 \xe2\x80\xa2\n\xe2\x80\xa2\n\xe2\x80\xa2\n\xe2\x80\xa2\n#bmw #bmwrepost #bmwm4 #bmwm4gts #f82 #bmwmrepost #bmwmsport #bmwmperformance #bmwmpower #bmwm4cs #austinyellow #davs0 #mpower_official #bmw_world_ua #bimmerworld #bmwfans #bmwfamily #bimmers #bmwpost #ultimatedrivingmachine #bmwgang #m3f80 #m5f90 #m4f82 #bmwmafia #bmwcrew #bmwlifestyle'
INFO [2019-12-17 22:17:34] [username]  --> Image Liked!
INFO [2019-12-17 22:17:37] [username]  --> Not commented
INFO [2019-12-17 22:17:38] [username]  --> Not following
```

默认情况下，除了你的`amount`值，InstaPy 还会喜欢前九个热门帖子。在这种情况下，每个标签的总赞数为 14(9 个热门帖子加上您在`amount`中指定的 5 个)。

还要注意，InstaPy 会记录它采取的每一个动作。正如你在上面看到的，它提到了它喜欢的帖子以及它的链接，描述，位置，以及机器人是否对帖子发表了评论或关注了作者。

你可能已经注意到，几乎每个动作之后都有延迟。那是故意的。它可以防止你的个人资料在 Instagram 上被封禁。

现在，你可能不希望你的机器人喜欢不合适的帖子。为了防止这种情况发生，您可以使用`set_dont_like()`:

```py
from instapy import InstaPy

session = InstaPy(username="<your_username>", password="<your_password>")
session.login()
session.like_by_tags(["bmw", "mercedes"], amount=5)
session.set_dont_like(["naked", "nsfw"])
```

随着这一改变，描述中带有`naked`或`nsfw`字样的帖子将不会被喜欢。你可以标记任何你希望你的机器人避免使用的单词。

接下来，你可以告诉机器人不仅要喜欢这些帖子，还要关注这些帖子的作者。你可以用`set_do_follow()`来做:

```py
from instapy import InstaPy

session = InstaPy(username="<your_username>", password="<your_password>")
session.login()
session.like_by_tags(["bmw", "mercedes"], amount=5)
session.set_dont_like(["naked", "nsfw"])
session.set_do_follow(True, percentage=50)
```

如果你现在运行这个脚本，那么这个机器人将会关注 50%的用户，他们的帖子是它喜欢的。像往常一样，每个动作都会被记录。

也可以在帖子上留下一些评论。你需要做两件事。首先，用`set_do_comment()`启用注释:

```py
from instapy import InstaPy

session = InstaPy(username="<your_username>", password="<your_password>")
session.login()
session.like_by_tags(["bmw", "mercedes"], amount=5)
session.set_dont_like(["naked", "nsfw"])
session.set_do_follow(True, percentage=50)
session.set_do_comment(True, percentage=50)
```

接下来，告诉机器人给`set_comments()`留下什么评论:

```py
from instapy import InstaPy

session = InstaPy(username="<your_username>", password="<your_password>")
session.login()
session.like_by_tags(["bmw", "mercedes"], amount=5)
session.set_dont_like(["naked", "nsfw"])
session.set_do_follow(True, percentage=50)
session.set_do_comment(True, percentage=50)
session.set_comments(["Nice!", "Sweet!", "Beautiful :heart_eyes:"])
```

运行这个脚本，机器人会在它交互的一半帖子上留下这三条评论中的一条。

现在您已经完成了基本设置，最好用`end()`结束会话:

```py
from instapy import InstaPy

session = InstaPy(username="<your_username>", password="<your_password>")
session.login()
session.like_by_tags(["bmw", "mercedes"], amount=5)
session.set_dont_like(["naked", "nsfw"])
session.set_do_follow(True, percentage=50)
session.set_do_comment(True, percentage=50)
session.set_comments(["Nice!", "Sweet!", "Beautiful :heart_eyes:"])
session.end()
```

这将关闭浏览器，保存日志，并准备一份您可以在控制台输出中看到的报告。

[*Remove ads*](/account/join/)

### InstaPy 中的附加功能

InstaPy 是一个相当大的项目，有很多[完整记录的特性](https://github.com/timgrossmann/InstaPy/blob/master/DOCUMENTATION.md)。好消息是，如果您对上面使用的特性感到满意，那么其余的应该感觉非常相似。本节将概述 InstaPy 的一些更有用的功能。

#### 定额主管

你不能整天刮 Instagram，天天刮。该服务将很快注意到你正在运行一个机器人，并将禁止它的一些行动。这就是为什么对你的机器人的一些行为设置限额是个好主意。以下面的例子为例:

```py
session.set_quota_supervisor(enabled=True, peak_comments_daily=240, peak_comments_hourly=21)
```

该机器人将继续评论，直到它达到每小时和每天的限制。配额期过后，它将恢复评论。

#### 无头浏览器

该特性允许您在没有浏览器 GUI 的情况下运行 bot。如果你想把你的机器人部署到一个没有或者不需要图形界面的服务器上，这是非常有用的。它对 CPU 的占用也更少，因此可以提高性能。你可以这样使用它:

```py
session = InstaPy(username='test', password='test', headless_browser=True)
```

请注意，您在初始化`InstaPy`对象时设置了该标志。

#### 利用人工智能分析帖子

前面你已经看到了如何忽略描述中包含不恰当词语的帖子。描述的很好但是图像本身不合适怎么办？您可以将 InstaPy bot 与提供图像和视频识别服务的 ClarifAI 集成在一起:

```py
session.set_use_clarifai(enabled=True, api_key='<your_api_key>')
session.clarifai_check_img_for(['nsfw'])
```

现在你的机器人不会喜欢或评论任何 ClarifAI 认为 [NSFW](https://en.wikipedia.org/wiki/Not_safe_for_work) 的图像。你每月可以获得 5000 次免费的 API 调用。

#### 关系界限

有很多粉丝的人跟帖子互动往往是浪费时间。在这种情况下，设置一些关系界限是个好主意，这样你的机器人就不会浪费你宝贵的计算资源:

```py
session.set_relationship_bounds(enabled=True, max_followers=8500)
```

这样，你的机器人就不会与拥有超过 8500 名粉丝的用户的帖子进行互动。

关于 InstaPy 的更多功能和配置，请查看[文档](https://github.com/timgrossmann/InstaPy/blob/master/DOCUMENTATION.md)。

## 结论

InstaPy 让你可以毫不费力地自动化你的 Instagram 活动。这是一个非常灵活的工具，有很多有用的功能。

**在本教程中，您学习了:**

*   Instagram 机器人如何工作
*   如何用 **Selenium** 自动化浏览器
*   如何使用**页面对象模式**使你的代码更易维护和测试
*   如何使用 **InstaPy** 构建一个基本的 Instagram bot

阅读 InstaPy 文档并对你的机器人进行一点点试验。很快你就会开始用最少的努力获得新的关注者和喜欢。在写这篇教程的时候，我自己也获得了一些新的追随者。如果你更喜欢视频教程，还有一个由 InstaPy [的创建者 Tim gro Mann](https://github.com/timgrossmann/)提供的 [Udemy 课程](https://www.udemy.com/course/instapy-guide/)。

您还可以探索[聊天机器人](https://realpython.com/build-a-chatbot-python-chatterbot/)、 [Tweepy](https://realpython.com/twitter-bot-python-tweepy/) 、 [Discord](https://realpython.com/how-to-make-a-discord-bot-python/) 和 [Alexa Skills](hhttps://realpython.com/alexa-python-skill/) 的可能性，以了解如何使用 Python 为不同平台制作机器人。

如果你有什么想问或分享的，请在下面的评论中提出。****