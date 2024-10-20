# PySide/PyQt 教程:QWebView

> 原文：<https://www.pythoncentral.io/pyside-pyqt-tutorial-qwebview/>

QWebView 是一个非常有用的控件；它允许您显示来自 URL、任意 HTML、带有 XSLT 样式表的 XML 的网页、构造为 q 网页的网页以及它知道如何解释其 MIME 类型的其他数据。它使用 [WebKit](https://webkit.org/) 网络浏览器引擎。WebKit 是一个最新的、符合标准的渲染引擎，被谷歌的 Chrome、苹果的 Safari 以及不久的 Opera 浏览器所使用。

## 创建和填写 QWebView

### 来自 QWebView 网址的内容

您可以像实例化其他`QWidget`一样实例化一个`QWebView`，并带有一个可选的父对象。然而，有许多方法可以将内容放入其中。最简单的——也可能是最明显的——是它的`load`方法，它需要一个`QUrl`；构建`QUrl`最简单的方法是使用 unicode URL 字符串:

```py

web_view.load(QUrl('https://www.pythoncentral.io/'))

```

这将在`QWebView`控件中加载  的主页。等同于使用`setUrl`方法，就像这样:

```py

web_view.setUrl(QUrl('https://www.pythoncentral.io/'))

```

### 用 QWebView 加载任意 HTML

还有其他有趣的方式将内容加载到`QWebView`中。您可以使用`setHtml`方法将生成的 HTML 加载到其中。例如，您可以这样做:

```py

html = '''<html>

<head>

<title>A Sample Page</title>

</head>

<body>

<h1>Hello, World!</h1>

<hr />

I have nothing to say.

</body>

</html>'''
web_view.setHtml(html) 

```

`setHtml`方法可以接受可选的第二个参数，即文档的基本 URL——基于这个 URL 解析文档中包含的任何相对链接。

### 其他 QWebView 内容类型

一个`QWebView`的内容不一定是 HTML 如果您有其他浏览器可查看的内容，您可以使用它的`setContent`方法将其放在一个`QWebView`中，该方法接受内容和可选的 MIME 类型以及一个基本 URL。如果省略 MIME 类型，会假设是`text/html`；MIME 类型的自动检测还没有实现，尽管它至少是初步计划的。我还没有找到 QWebView 可以处理的 MIME 类型列表，但是这里有一个处理 PNG 文件的例子:

```py

app = QApplication([])

win = QWebView()
img = open('myImage.png '，' rb ')。read() 
 win.setContent(img，' image/png ')。
win.show() 
 app.exec_() 

```

## 用 JavaScript 与 QWebView 交互

向用户呈现 web 风格的内容本身是有用的，但是这些内容可以与 JavaScript 交互，JavaScript 可以从 Python 代码启动。

一个`QWebView`包含一个`QWebFrame`对象，它的`evaluateJavaScript`方法现在对我们很有用。该方法接受一个 JavaScript 字符串，在`QWebView`内容的上下文中对其求值，并返回其值。

可以返回哪些值？QWebFrame 的 [PySide 文档，就像 PyQt 和 Qt 本身的文档一样，在这一点上并不清楚。事实上，这些信息在网上根本看不到，所以我做了一些测试。](https://www.pythoncentral.io/pyside-pyqt-tutorial-qwebview/)

看起来字符串和布尔值只是简单的工作，而数字、对象、`undefined`和`null`的工作需要注意:

*   因为 JavaScript 缺少独立的整数和浮点数据类型，所以数字总是被转换成 Python 浮点。
*   对象作为 Python 字典返回，除非它们是函数或数组；函数作为无用的空字典返回，数组成为 Python 列表。
*   `undefined`变成了`None`，足够理智。
*   `null`变得，不太理智地，“。没错——空字符串。

特别注意关于`null`和函数的行为，因为两者都可能导致看起来正确的代码行为错误。我看不到更好的函数选项，但是`null`尤其令人困惑；检测来自`evaluateJavaScript`的空值的*唯一的*方法是在将它返回给 Python 之前，在 JavaScript 端*进行比较`val === null`。(正是在这一点上，我们集体为 JavaScript 考虑不周的类型感到悲伤。)*

关于`evaluateJavaScript`的一个重要警告:它具有 JavaScript 内置`eval`的所有安全含义，应该谨慎使用，前端 JavaScript 编码人员很少表现出这种谨慎。例如，通过天真地构建一个字符串并将其发送到`evaluateJavaScript`来执行任意的 JavaScript 就太简单了。小心，验证用户输入，阻止任何看起来太聪明的东西。

### 在 QWebView 中评估 JavaScript 的示例

现在，让我们抛开谨慎，看一个简单的例子。它将显示一个允许用户输入名和姓的表单。将有一个被禁用的全名条目；用户不能编辑它。有一个提交按钮，但它是隐藏的。(现实生活中不要这样好吗？这种形式是可用性和文化敏感性的灾难，在公共场合展示几乎是一种侮辱。)我们将提供一个 Qt 按钮来填写全名条目，显示 submit 按钮，并将全名打印到控制台。下面是这个例子的来源:

```py
# Create an application
app = QApplication([])

# And a window
win = QWidget()
win.setWindowTitle('QWebView Interactive Demo')

# And give it a layout
layout = QVBoxLayout()
win.setLayout(layout)

# Create and fill a QWebView
view = QWebView()
view.setHtml('''
<html>
<head>
<title>A Demo Page</title>

<script language="javascript">
// Completes the full-name control and
// shows the submit button
function completeAndReturnName() {
var fname = document.getElementById('fname').value;
var lname = document.getElementById('lname').value;
var full = fname + ' ' + lname;

document.getElementById('fullname').value = full;
document.getElementById('submit-btn').style.display = 'block';

return full;
}
</script>
</head>

<body>
<form>
<label for="fname">First name:</label>
<input type="text" name="fname" id="fname"></input>
<br />
<label for="lname">Last name:</label>
<input type="text" name="lname" id="lname"></input>
<br />
<label for="fullname">Full name:</label>
<input disabled type="text" name="fullname" id="fullname"></input>
<br />
<input style="display: none;" type="submit" id="submit-btn"></input>
</form>
</body>
</html>
''')

# A button to call our JavaScript
button = QPushButton('Set Full Name')

# Interact with the HTML page by calling the completeAndReturnName
# function; print its return value to the console
def complete_name():
frame = view.page().mainFrame()
print frame.evaluateJavaScript('completeAndReturnName();')

# Connect 'complete_name' to the button's 'clicked' signal
button.clicked.connect(complete_name)

# Add the QWebView and button to the layout
layout.addWidget(view)
layout.addWidget(button)

# Show the window and run the app
win.show()
app.exec_()
```

试着运行它。填写名字和姓氏，然后单击按钮。您应该会看到一个全名和一个提交按钮。查看控制台，您也应该看到全名打印在那里。

这是一个非常复杂的例子，但是我们可以做更多有趣的事情:在下一期文章中，我们将构建一个简单的应用程序，它将 HTML 和一些其他 Qt 小部件结合起来，有效地与 web API 一起工作。