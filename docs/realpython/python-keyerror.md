# Python KeyError 异常以及如何处理它们

> 原文：<https://realpython.com/python-keyerror/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Python KeyError 异常及如何处理**](/courses/python-keyerror/)

Python 的`KeyError`异常是初学者经常遇到的异常。了解为什么会引发一个`KeyError`,以及一些防止它停止你的程序的解决方案，是提高 Python 程序员水平的基本步骤。

本教程结束时，你会知道:

*   一条蟒蛇通常意味着什么
*   在标准库中还有什么地方可以看到一个`KeyError`
*   看到一个`KeyError`怎么处理

**免费奖励:** ，它向您展示 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 什么是 Python `KeyError`通常指的是

Python `KeyError` [异常](https://realpython.com/python-exceptions/)是当你试图访问一个不在[字典](https://realpython.com/python-dicts/) ( `dict`)中的键时引发的。

Python 的[官方文档](https://docs.python.org/3/library/exceptions.html#KeyError)称，当访问映射键但在映射中找不到时，会引发`KeyError`。映射是将一组值映射到另一组值的数据结构。Python 中最常见的映射是字典。

Python `KeyError`是一种 [`LookupError`](https://docs.python.org/3/library/exceptions.html#LookupError) 异常，表示在检索您正在寻找的密钥时出现了问题。当你看到一个`KeyError`时，语义是找不到要找的钥匙。

在下面的例子中，你可以看到一个用三个人的年龄定义的字典(`ages`)。当您试图访问一个不在字典中的键时，会引发一个`KeyError`:

>>>

```py
>>> ages = {'Jim': 30, 'Pam': 28, 'Kevin': 33}
>>> ages['Michael']
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'Michael'
```

这里，试图访问`ages`字典中的键`'Michael'`会导致一个`KeyError`被引发。在追溯的底部，您可以获得相关信息:

*   一个`KeyError`被提出的事实
*   找不到的钥匙是`'Michael'`

倒数第二行告诉您哪一行引发了异常。当您从文件中执行 Python 代码时，这些信息会更有帮助。

**注意:**当一个异常在 Python 中出现时，它是通过一个**回溯**来完成的。回溯为您提供了所有相关信息，以便您能够确定异常出现的原因以及导致异常的原因。

学习如何阅读 Python traceback 并理解它告诉你什么对于提高 Python 程序员来说至关重要。要了解更多关于 Python 回溯的信息，请查看[了解 Python 回溯](https://realpython.com/python-traceback/)

在下面的程序中，可以看到再次定义的`ages`字典。这一次，系统会提示您提供要检索年龄的人的姓名:

```py
 1# ages.py
 2
 3ages = {'Jim': 30, 'Pam': 28, 'Kevin': 33}
 4person = input('Get age for: ')
 5print(f'{person} is {ages[person]} years old.')
```

该代码将采用您在提示符下提供的姓名，并尝试检索该人的年龄。您在提示符下输入的任何内容都将被用作第 4 行的`ages`字典的关键字。

重复上面失败的例子，我们得到另一个[回溯](https://realpython.com/courses/python-traceback/)，这次是关于文件中产生`KeyError`的行的信息:

```py
$ python ages.py
Get age for: Michael
Traceback (most recent call last):
File "ages.py", line 4, in <module>
 print(f'{person} is {ages[person]} years old.')
KeyError: 'Michael'
```

当你给出一个不在字典中的键时，程序会失败。这里，回溯的最后几行指出了问题所在。`File "ages.py", line 4, in <module>`告诉您哪个文件的哪一行引发了结果`KeyError`异常。然后您会看到这一行。最后，`KeyError`异常提供了丢失的密钥。

所以你可以看到`KeyError` traceback 的最后一行本身并没有给你足够的信息，但是它之前的几行可以让你更好地理解哪里出错了。

**注意:**和上面的例子一样，本教程中的大多数例子都使用了在 Python 3.6 中引入的 [f 字符串](https://realpython.com/python-f-strings/)。

[*Remove ads*](/account/join/)

## 在标准库中，你还能在哪里看到 Python `KeyError`

大多数情况下，Python `KeyError`被引发是因为在字典或字典子类中找不到键(比如`os.environ`)。

在极少数情况下，如果在 ZIP 存档中找不到某个项目，您可能还会在 Python 的标准库中的其他地方看到它，例如在 [`zipfile`](https://realpython.com/python-zipfile/) 模块中。然而，这些地方保留了 Python `KeyError`的相同语义，即没有找到请求的键。

在下面的例子中，您可以看到使用`zipfile.ZipFile`类提取关于使用`.getinfo()`的 ZIP 存档的信息:

>>>

```py
>>> from zipfile import ZipFile
>>> zip_file = ZipFile('the_zip_file.zip')
>>> zip_file.getinfo('something')
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File "/path/to/python/installation/zipfile.py", line 1304, in getinfo
 'There is no item named %r in the archive' % name)
KeyError: "There is no item named 'something' in the archive"
```

这看起来不太像字典键查找。相反，是对`zipfile.ZipFile.getinfo()`的调用引发了异常。

回溯看起来也有一点不同，它给出了比丢失的键更多的信息:`KeyError: "There is no item named 'something' in the archive"`。

这里要注意的最后一点是，引发`KeyError`的代码行不在您的代码中。它在`zipfile`代码中，但是回溯的前几行指出了代码中的哪几行导致了问题。

## 当你需要在自己的代码中提出一个 Python `KeyError`

有时候[您在自己的代码中引发](https://realpython.com/python-exceptions/#raising-an-exception)一个 Python `KeyError`异常是有意义的。这可以通过使用`raise` [关键字](https://realpython.com/python-keywords/)并调用`KeyError`异常来完成:

```py
raise KeyError(message)
```

通常，`message`将是丢失的键。然而，就像在`zipfile`包的情况下，您可以选择提供更多的信息来帮助下一个开发者更好地理解哪里出错了。

如果您决定在自己的代码中使用 Python `KeyError`,只需确保您的用例与异常背后的语义相匹配。它应该表示找不到正在寻找的密钥。

## 当你看到一条 Python `KeyError`时如何处理它

当你遇到一个`KeyError`时，有几个标准的处理方法。根据您的使用情况，这些解决方案中的一些可能比其他的更好。最终目标是阻止意外的`KeyError`异常被引发。

### 通常的解决方案:`.get()`

如果在您自己的代码中由于字典键查找失败而引发了`KeyError`，您可以使用`.get()`返回在指定键中找到的值或默认值。

与前面的年龄检索示例非常相似，下面的示例展示了使用提示符下提供的键从字典中获取年龄的更好方法:

```py
 1# ages.py
 2
 3ages = {'Jim': 30, 'Pam': 28, 'Kevin': 33}
 4person = input('Get age for: ')
 5age = ages.get(person) 6
 7if age:
 8    print(f'{person} is {age} years old.')
 9else:
10    print(f"{person}'s age is unknown.")
```

在这里，第 5 行显示了如何使用`.get()`从`ages`获得年龄值。这将导致`age` [变量](https://realpython.com/python-variables/)具有在字典中为所提供的键找到的年龄值或默认值，在本例中为 [`None`](https://realpython.com/null-in-python/) 。

这一次，您将不会得到引发的`KeyError`异常，因为使用了更安全的`.get()`方法来获取年龄，而不是尝试直接访问密钥:

```py
$ python ages.py
Get age for: Michael
Michael's age is unknown.
```

在上面的执行示例中，当提供了一个错误的键时，不再引发`KeyError`。键`'Michael'`在字典中找不到，但是通过使用`.get()`，我们得到一个返回的`None`，而不是一个提升的`KeyError`。

`age`变量要么是在字典中找到的人的年龄，要么是默认值(默认为`None`)。您还可以通过传递第二个参数在`.get()`调用中指定一个不同的默认值。

这是上例中的第 5 行，使用`.get()`指定了不同的默认年龄:

```py
age = ages.get(person, 0)
```

这里，不是`'Michael'`返回`None`，而是返回`0`，因为没有找到键，现在返回的默认值是`0`。

[*Remove ads*](/account/join/)

### 罕见的解决方案:检查键

有时候，您需要确定字典中是否存在某个键。在这些情况下，使用`.get()`可能不会给你正确的信息。从对`.get()`的调用中返回一个`None`可能意味着没有找到键，或者在字典中找到的键的值实际上是`None`。

对于字典或类似字典的对象，可以使用`in`操作符来确定一个键是否在映射中。该操作符将返回一个[布尔](https://realpython.com/python-boolean/) ( `True`或`False`)值，指示是否在字典中找到了该键。

在这个例子中，您将从调用 API 的[获得一个`response`字典。该响应可能在响应中定义了一个`error`键值，这将表明该响应处于错误状态:](https://realpython.com/python-api/)

```py
 1# parse_api_response.py
 2...
 3# Assuming you got a `response` from calling an API that might
 4# have an error key in the `response` if something went wrong
 5
 6if 'error' in response: 7    ...  # Parse the error state
 8else:
 9    ...  # Parse the success state
```

这里，检查`error`键是否存在于`response`中并从该键获得默认值是有区别的。这是一种罕见的情况，你真正要找的是这个键是否在字典中，而不是这个键的值是什么。

### 一般解法:`try` `except`

对于任何异常，您都可以使用`try` `except`块来隔离潜在的引发异常的代码，并提供备份解决方案。

您可以在与前面类似的示例中使用`try` `except`块，但是这一次提供了一个默认的要打印的消息，如果在正常情况下引发了一个`KeyError`:

```py
 1# ages.py
 2
 3ages = {'Jim': 30, 'Pam': 28, 'Kevin': 33}
 4person = input('Get age for: ')
 5
 6try:
 7    print(f'{person} is {ages[person]} years old.')
 8except KeyError:
 9    print(f"{person}'s age is unknown.")
```

在这里，您可以在打印人名和年龄的`try`块中看到正常情况。备份实例在`except`块中，如果在正常情况下`KeyError`被引发，那么备份实例将打印不同的消息。

对于其他可能不支持`.get()`或`in`操作符的地方来说，`try` `except`阻塞解决方案也是一个很好的解决方案。如果`KeyError`是从另一个人的代码中产生的，这也是最好的解决方案。

这里是一个再次使用`zipfile`包的例子。这一次，`try` `except`块为我们提供了一种阻止`KeyError`异常停止程序的方法:

>>>

```py
>>> from zipfile import ZipFile
>>> zip = ZipFile('the_zip_file.zip')
>>> try:
...     zip.getinfo('something')
... except KeyError:
...     print('Can not find "something"')
...
Can not find "something"
```

因为`ZipFile`类不像字典那样提供`.get()`，所以您需要使用`try` `except`解决方案。在这个例子中，您不需要提前知道哪些值可以传递给`.getinfo()`。

## 结论

您现在知道了 Python 的`KeyError`异常可能出现的一些常见地方，以及可以用来防止它们停止您的程序的一些很好的解决方案。

现在，下次你看到一个`KeyError`被提出来，你就知道这很可能只是一个不好的字典键查找。通过查看回溯的最后几行，您还可以找到确定错误来源所需的所有信息。

如果问题是在您自己的代码中查找字典键，那么您可以从直接在字典上访问键切换到使用带有默认返回值的更安全的`.get()`方法。如果问题不是来自您自己的代码，那么使用`try` `except`块是您控制代码流的最佳选择。

例外不一定是可怕的。一旦你知道如何理解回溯中提供给你的信息和异常的根本原因，那么你就可以使用这些解决方案来使你的程序流程更加可预测。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Python KeyError 异常及如何处理**](/courses/python-keyerror/)****