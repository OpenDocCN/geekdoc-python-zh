# 用 Python 和 PhantomJS 进行无头 Selenium 测试

> 原文：<https://realpython.com/headless-selenium-testing-with-python-and-phantomjs/>

PhantomJS 是一个无头的 [Webkit](https://www.webkit.org/) ，它有很多用途。在本例中，我们将结合 Selenium WebDriver 使用它，直接从命令行进行基本的系统测试。由于 PhantomJS 消除了对图形浏览器的需求，测试运行得更快。

点击[此处](#video)观看附带视频。

## 设置

用 Pip 安装 Selenium，用[自制软件](http://brew.sh/)安装 PhantomJS:

```py
$ pip install selenium
$ brew install phantomjs
```

> 用 Brew 安装 PhantomJS 有问题吗？在这里获取最新版本。

[*Remove ads*](/account/join/)

## 示例

现在让我们看两个简单的例子。

### DuckDuckGo

在第一个示例中，我们将在 DuckDuckGo 中搜索关键字“realpython ”,以找到搜索结果的 URL。

```py
from selenium import webdriver
driver = webdriver.PhantomJS()
driver.set_window_size(1120, 550)
driver.get("https://duckduckgo.com/")
driver.find_element_by_id('search_form_input_homepage').send_keys("realpython")
driver.find_element_by_id("search_button_homepage").click()
print(driver.current_url)
driver.quit()
```

您可以在终端中看到输出的 URL。

下面是用 Firefox 显示结果的例子。

```py
from selenium import webdriver
driver = webdriver.Firefox()
driver.get("https://duckduckgo.com/")
driver.find_element_by_id('search_form_input_homepage').send_keys("realpython")
driver.find_element_by_id("search_button_homepage").click()
driver.quit()
```

你有没有注意到我们不得不在幻影脚本上创建一个虚拟的浏览器尺寸？这是目前在 [Github](https://github.com/ariya/phantomjs/issues/11637) 中存在的一个问题的解决方法。尝试没有它的脚本:您将得到一个`ElementNotVisibleException`异常。

现在我们可以编写一个[快速测试](https://realpython.com/python-testing/)来断言搜索结果显示的 URL 是正确的。

```py
import unittest
from selenium import webdriver

class TestOne(unittest.TestCase):

    def setUp(self):
        self.driver = webdriver.PhantomJS()
        self.driver.set_window_size(1120, 550)

    def test_url(self):
        self.driver.get("http://duckduckgo.com/")
        self.driver.find_element_by_id(
            'search_form_input_homepage').send_keys("realpython")
        self.driver.find_element_by_id("search_button_homepage").click()
        self.assertIn(
            "https://duckduckgo.com/?q=realpython", self.driver.current_url
        )

    def tearDown(self):
        self.driver.quit()

if __name__ == '__main__':
    unittest.main()
```

测试通过了。

### real ython . com

最后，让我们看一个我每天运行的真实世界的例子。导航到[RealPython.com](https://realpython.com)，我会告诉你我们将测试什么。本质上，我想确保底部的“立即下载”按钮有正确的相关产品。

下面是基本的单元测试:

```py
import unittest
from selenium import webdriver

class TestTwo(unittest.TestCase):

    def setUp(self):
        self.driver = webdriver.PhantomJS()

    def test_url(self):
        self.driver.get("https://app.simplegoods.co/i/IQCZADOY") # url associated with button click
        button = self.driver.find_element_by_id("payment-submit").get_attribute("value")
        self.assertEquals(u'Pay - $60.00', button)

    def tearDown(self):
        self.driver.quit()

if __name__ == '__main__':
    unittest.main()
```

## 基准测试

与浏览器相比，使用幻想曲的一个主要优势是测试通常要快得多。在下一个例子中，我们将使用 PhantomJS 和 Firefox 对之前的测试进行基准测试。

```py
import unittest
from selenium import webdriver
import time

class TestThree(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def test_url_fire(self):
        time.sleep(2)
        self.driver = webdriver.Firefox()
        self.driver.get("https://app.simplegoods.co/i/IQCZADOY") # url associated with button click
        button = self.driver.find_element_by_id("payment-submit").get_attribute("value")
        self.assertEquals(u'Pay - $60.00', button)

    def test_url_phantom(self):
        time.sleep(1)
        self.driver = webdriver.PhantomJS()
        self.driver.get("https://app.simplegoods.co/i/IQCZADOY") # url associated with button click
        button = self.driver.find_element_by_id("payment-submit").get_attribute("value")
        self.assertEquals(u'Pay - $60.00', button)

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f" % (self.id(), t))
        self.driver.quit()

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestThree)
    unittest.TextTestRunner(verbosity=0).run(suite)
```

你可以看到幻想曲有多快:

```py
$ python test.py -v
__main__.TestThree.test_url_fire: 19.801
__main__.TestThree.test_url_phantom: 10.676
----------------------------------------------------------------------
Ran 2 tests in 30.683s

OK
```

[*Remove ads*](/account/join/)

## 视频

> **注意:**尽管这个视频已经过时了(由于脚本的改变)，但它仍然值得一看，因为为你的网站实现无头系统测试的基本方法基本上保持不变。

[https://www.youtube.com/embed/X0b0xM2Ddh8?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com](https://www.youtube.com/embed/X0b0xM2Ddh8?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com)**