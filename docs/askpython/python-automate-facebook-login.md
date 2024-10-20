# 在 Python 中使用 Selenium 自动化脸书登录

> 原文：<https://www.askpython.com/python/examples/python-automate-facebook-login>

大家好！在今天的文章中，我们将学习使用 Python 自动登录脸书。

这将是一个有趣的实验，让您一瞥使用 Python 的 Selenium web 驱动程序的 web 浏览器自动化。所以让我们直接进入主题，创建一个访问脸书页面的脚本，输入凭证，然后登录！

* * *

## 先决条件

现在，在阅读本教程之前，您需要在 Python 中安装某些库。这些库将使我们很容易登录到浏览器。

我们将使用 Python 中的 *Selenium* webdriver 模块。这个模块使我们能够使用驱动程序控制我们的网络浏览器(Chrome / Firefox)。

但是，要将 Selenium 与我们的浏览器一起使用，我们需要安装该浏览器的驱动程序(Chrome/Firefox)。为了安装它们，我们将借助另一个 Python 模块:`webdriver_manager`

不需要手动下载 selenium webdriver，您可以简单地导入这个模块！这将为您自动获取所有需求。

现在，让我们`pip install`必要的包，使用 [pip](https://www.askpython.com/python-modules/python-pip) 管理器:

```py
pip install selenium
pip install webdriver_manager

```

现在我们已经安装了我们的需求，让我们开始编写代码吧！

* * *

## 编写我们的脚本来自动化脸书登录

我们先导入必要的模块。我们需要`selenium`和`webdriver_manager`。

```py
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.chrome import ChromeDriverManager
import time

```

这里，我需要核心 Selenium 模块的`webdriver`类。此外，由于我们将在 firefox/chrome 上使用它，我们需要加载必要的网络驱动程序。

现在，我们将使用以下 url 登录:

```py
LOGIN_URL = 'https://www.facebook.com/login.php'

```

现在，我们将登录功能实现为一个类。姑且称之为`FacebookLogin`。

当我们调用`__init__()`时，我们将初始化 selenium webdriver 会话。我们需要将电子邮件和密码字段发送到我们的 webdriver 会话，所以我们将它们作为输入。

最后，我们将从 webdriver 获取带有 GET 请求的`LOGIN_URL`。

```py
class FacebookLogin():
    def __init__(self, email, password, browser='Chrome'):
        # Store credentials for login
        self.email = email
        self.password = password
        if browser == 'Chrome':
            # Use chrome
            self.driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())
        elif browser == 'Firefox':
            # Set it to Firefox
            self.driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
        self.driver.get(LOGIN_URL)
        time.sleep(1) # Wait for some time to load

```

好了，现在我们已经初始化了类实例。现在，为了登录，我们将创建另一个名为`login()`的方法来完成这项工作。

要登录，我们需要向登录元素(html 页面上的`email`和`pass`)提供输入

Selenium 有`find_element_by_id()`方法，会自动给你定位对应的元素！

要发送键盘输入，我们可以直接用`element.send_keys(input)`！

```py
    def login(self):
        email_element = self.driver.find_element_by_id('email')
        email_element.send_keys(self.email) # Give keyboard input

        password_element = self.driver.find_element_by_id('pass')
        password_element.send_keys(self.password) # Give password as input too

        login_button = self.driver.find_element_by_id('loginbutton')
        login_button.click() # Send mouse click

        time.sleep(2) # Wait for 2 seconds for the page to show up

```

注意这个 API 有多简单！我们可以直接做`element.send_keys()`和`element.click()`！

最后，用`time.sleep()`给程序一些时间来加载网页

下面我给你完整的代码。请确保在`main`模块中使用正确的登录凭证。

```py
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.chrome import ChromeDriverManager
import time

LOGIN_URL = 'https://www.facebook.com/login.php'

class FacebookLogin():
    def __init__(self, email, password, browser='Chrome'):
        # Store credentials for login
        self.email = email
        self.password = password
        if browser == 'Chrome':
            # Use chrome
            self.driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())
        elif browser == 'Firefox':
            # Set it to Firefox
            self.driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
        self.driver.get(LOGIN_URL)
        time.sleep(1) # Wait for some time to load

    def login(self):
        email_element = self.driver.find_element_by_id('email')
        email_element.send_keys(self.email) # Give keyboard input

        password_element = self.driver.find_element_by_id('pass')
        password_element.send_keys(self.password) # Give password as input too

        login_button = self.driver.find_element_by_id('loginbutton')
        login_button.click() # Send mouse click

        time.sleep(2) # Wait for 2 seconds for the page to show up

if __name__ == '__main__':
    # Enter your login credentials here
    fb_login = FacebookLogin(email='[email protected]', password='PASSWORD', browser='Firefox')
    fb_login.login()

```

希望你的浏览器现在会显示你的主页。万岁，你已成功登录 facebook！

* * *

## 结论

在本文中，我们学习了如何使用 Python 和 Selenium 快速自动登录脸书！

* * *