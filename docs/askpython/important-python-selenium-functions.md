# 您必须知道的重要 Python Selenium 函数

> 原文：<https://www.askpython.com/python-modules/important-python-selenium-functions>

在本教程中，我们将探索 selenium 最有用的方法。Selenium 提供了许多使自动化过程变得容易的方法，其中一些方法并不常见，但是对于使自动化/抓取过程变得容易和有效很有用。

***也读作:[安装设置硒](https://www.askpython.com/python-modules/selenium-introduction-and-setup)***

## 导入模块并初始化 Webdriver

Selenium webdriver 是所有类和方法的父类。使用 webdriver，我们可以访问 selenium 提供的所有方法。

```py
from selenium import webdriver

driver = webdriver.Chrome("path of webdriver")

```

代替“webdriver 的路径”，我们甚至可以只使用 Chrome()方法，而不传递 webdriver 的路径的任何位置参数，只要我们已经在我们的计算机中全局地将浏览器的 webdriver 的**路径**位置声明为**环境变量**。

## Python Selenium 中的浏览器方法

让我们从您最常使用并且应该很好掌握的浏览器方法开始。

### 1.获取网页

为了获取一个特定的网页，我们使用 **get** 方法，并将网页 URL 作为参数传递。

```py
driver.get("https://m.youtube.com")

```

### 2.获取网页标题

**title** 方法返回当前页面(网页)的标题，该标题打印在执行代码的控制台/终端上。

```py
driver.title()

```

### 3.在页面历史记录之间来回移动

通过使用**后退**和**前进**方法，我们可以在浏览器中自动执行后退或前进网页的任务

```py
driver.back()
driver.forward()

```

### 4.全屏方法

调用此方法将使浏览器(chrome)窗口全屏显示。

```py
driver.fullscreen_window()

```

### 5.设置窗口在屏幕上的位置

通过使用这种方法，您将能够设置当前窗口的坐标。该方法获取 x 和 y 坐标，并根据屏幕上的坐标在指定位置设置当前窗口(使用 python-selenium 代码打开)。

```py
driver.set_window_position(500,500)

```

屏幕位置的原点(0，0)位于屏幕的最底部左侧。

### 6.打开一个新标签

要在新标签页上打开一个新网站，我们将使用 **execute_script()** 方法。在下面的例子中，代码将在我们的第一个网站 YouTube 旁边的新标签页中打开 Twitter。这个方法**需要 JavaScript** 来执行。

```py
driver.execute_script("window.open('https://twitter.com')")

```

### 7.截图

Selenium 提供了一种获取当前窗口截图的方法。这是通过使用以下方法完成的

```py
driver.get_screenshot_as_file('ss.png')

```

运行此代码后，名为“ss”的图像将存储在同一目录中。

### 8.刷新页面

通过使用 refresh 方法，我们可以刷新当前框架/网页。

```py
driver.refresh()

```

### 9.选择一个元素

有各种方法和技术来选择元素、图像、文本字段、视频、标签等。在网页中。因此，我们已经在一篇单独的、详细的文章中介绍了选择元素的所有方法。

推荐阅读:[Selenium-Python 中选择元素的所有不同方式](#)

### 10.单击一个元素

该方法用于点击 web 元素，如按钮或链接。

在下面的代码中，我们首先使用其 ID 或任何其他选择器找到链接或按钮，然后在这个 web 元素上调用 click 方法。

```py
elements = driver.find_element_by_id("id of link or button")
elements.click()

```

### 11.发送密钥(文本)

使用这种方法，我们可以将一些文本发送到网页中的任何文本字段。我们通过使用它的 id 找到 Gmail 的文本框，然后用 send_keys 方法发送我们的文本。

```py
elements = driver.find_element_by_id("gmail")
elements.send_keys(“some text we want to send”)

```

### 12.清除文本

它用于清除任何输入字段的文本。

```py
elements = driver.find_element_by_id("gmail")
elements.send_keys("some text we want to send")
elements.clear()

```

### 13.使用自定义 JavaScript

使用这种方法，我们可以发送定制的 JavaScript 代码，并对事件、**提示**等执行 JavaScript 支持的各种操作。

```py
driver.execute_script()

#practical code implementation
driver.execute_script("window.open('https://google.com')")

```

### 14.关闭当前选项卡，但不关闭浏览器

通过使用 close 方法，我们可以在不关闭浏览器的情况下关闭**当前选项卡。**

```py
driver.close()

```

### 15.关闭浏览器

我们可以用**退出**的方法关闭浏览器。显然，整个浏览器窗口关闭，关闭了所有打开的标签页。

```py
driver.quit()

```

## 16.时间-睡眠(Imp。)

这实际上不是与 Selenium 库相关联的功能或方法，但它是一个非常有用的技巧，对于各种目的都很方便，比如等待执行一些任务——加载站点、执行一些其他代码等。

这是时间模块的一部分，它已经与我们的 python 安装捆绑在一起:

```py
import time

#time.sleep(time to wait (in seconds))

time.sleep(5) #this will let the interpreter wait at this line for 5 seconds and then proceed with execution.

```

## 结论

教程到此为止。有各种各样的方法和功能与 Selenium 相关，其中一些您将在使用 Selenium 时自己学习，而另一些我们坚持要查看 Selenium 的官方[文档](https://www.selenium.dev/documentation/)。