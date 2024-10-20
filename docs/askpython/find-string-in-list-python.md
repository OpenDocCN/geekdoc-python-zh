# 在 Python 中查找列表中的字符串

> 原文：<https://www.askpython.com/python/list/find-string-in-list-python>

在本文中，我们将看看如何在 Python 中查找列表中的字符串。

* * *

## 在 Python 中查找列表中的字符串

对于这个问题有各种各样的方法，从易用性到效率。

### 使用“in”运算符

我们可以使用 Python 的 **in** 操作符在 Python 中查找一个列表中的字符串。它接受两个操作数`a`和`b`，其形式如下:

```py
ret_value = a in b

```

这里，`ret_value`是一个布尔值，如果`a`位于`b`内，则该值为`True`，否则为`False`。

我们可以通过以下方式直接使用该运算符:

```py
a = [1, 2, 3]

b = 4

if b in a:
    print('4 is present!')
else:
    print('4 is not present')

```

**输出**

```py
4 is not present

```

为了方便使用，我们也可以将它转换成一个函数。

```py
def check_if_exists(x, ls):
    if x in ls:
        print(str(x) + ' is inside the list')
    else:
        print(str(x) + ' is not present in the list')

ls = [1, 2, 3, 4, 'Hello', 'from', 'AskPython']

check_if_exists(2, ls)
check_if_exists('Hello', ls)
check_if_exists('Hi', ls)

```

**输出**

```py
2 is inside the list
Hello is inside the list
Hi is not present in the list

```

这是在列表中搜索字符串的最常用和推荐的方法。但是，为了便于说明，我们还将向您展示其他方法。

* * *

### 使用列表理解

让我们看另一个例子，您希望只检查字符串是否是列表中另一个单词的一部分，并返回所有这些单词，其中您的单词是列表项的子字符串。

考虑下面的列表:

```py
ls = ['Hello from AskPython', 'Hello', 'Hello boy!', 'Hi']

```

如果您想在列表的所有元素中搜索子串`Hello`，我们可以使用以下格式的列表理解:

```py
ls = ['Hello from AskPython', 'Hello', 'Hello boy!', 'Hi']

matches = [match for match in ls if "Hello" in match]

print(matches)

```

这相当于下面的代码，它只有两个循环并检查条件。

```py
ls = ['Hello from AskPython', 'Hello', 'Hello boy!', 'Hi']

matches = []

for match in ls:
    if "Hello" in match:
        matches.append(match)

print(matches)

```

在这两种情况下，输出都是:

```py
['Hello from AskPython', 'Hello', 'Hello boy!']

```

如您所见，在输出中，所有匹配都包含字符串`Hello`作为字符串的一部分。很简单，不是吗？

* * *

### 使用“any()”方法

如果你想检查输入字符串是否存在于列表的 **any** 项中，我们可以使用 [any()方法](https://www.askpython.com/python/built-in-methods/any-method-in-python)来检查这是否成立。

例如，如果您希望测试' **AskPython'** '是否是列表中任何项目的一部分，我们可以执行以下操作:

```py
ls = ['Hello from AskPython', 'Hello', 'Hello boy!', 'Hi']

if any("AskPython" in word for word in ls):
    print('\'AskPython\' is there inside the list!')
else:
    print('\'AskPython\' is not there inside the list')

```

**输出**

```py
'AskPython' is there inside the list!

```

* * *

### 使用过滤器和 lambdas

我们也可以在一个 [lambda 函数](https://www.askpython.com/python/python-lambda-anonymous-function)上使用`filter()`方法，这是一个简单的函数，只在那一行定义。把 lambda 想象成一个迷你函数，在调用后不能被重用。

```py
ls = ['Hello from AskPython', 'Hello', 'Hello boy!', 'Hi']

# The second parameter is the input iterable
# The filter() applies the lambda to the iterable
# and only returns all matches where the lambda evaluates
# to true
filter_object = filter(lambda a: 'AskPython' in a, ls)

# Convert the filter object to list
print(list(filter_object))

```

**输出**

```py
['Hello from AskPython']

```

我们确实得到了我们所期望的！只有一个字符串与我们的过滤函数匹配，这就是我们得到的结果！

* * *

## 结论

在本文中，我们学习了如何用不同的方法找到带有输入列表的字符串。希望这对你的问题有所帮助！

* * *

## 参考

*   JournalDev 关于在列表中查找字符串的文章
*   在一个列表中寻找一个字符串

* * *