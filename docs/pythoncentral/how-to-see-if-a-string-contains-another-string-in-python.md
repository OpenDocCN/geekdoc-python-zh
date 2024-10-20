# 如何在 Python 中查看一个字符串是否包含另一个字符串

> 原文：<https://www.pythoncentral.io/how-to-see-if-a-string-contains-another-string-in-python/>

曾经想看看一个字符串是否包含 Python 中的一个字符串？以为会像 C 一样复杂？再想想！

Python 以一种非常易读和易于实现的方式实现了这个特性。有两种方法，有些人会更喜欢其中一种，所以我会让你决定你更喜欢哪一种。

## **第一种方式:使用 Python 的 in 关键字**

检查一个字符串是否包含另一个字符串的第一种方法是使用`in`语法。`in`接受两个“参数”，一个在左边，一个在右边，如果左边的参数包含在右边的参数中，则返回`True`。

这里有一个例子:

```py

>>> s = "It's not safe to go alone. Take this."

>>> 'safe' in s

True

>>> 'blah' in s

False

>>> if 'safe' in s:

... print('The message is safe.')

The message is safe.

```

你明白了。这就是全部了。关键字`in`在后台为你完成所有神奇的工作，所以不用担心`for`循环或任何类似的事情。

## **第二种方式:使用 Python 的 str.find**

这种方式是不太 Pythonic 化的方式，但是仍然被接受。它更长，也更令人困惑，但它仍然完成了任务。

这种方式要求我们在字符串上调用`find`方法，并检查它的返回代码。

下面是我们的例子，使用上面定义的字符串:

```py

>>> if s.find('safe') != -1:

... print('This message is safe.')

This message is safe.

```

就像我上面说的，这种方式不太清晰，但它仍然完成了工作。`find`方法返回字符串在字符串中的位置，如果没有找到则返回-1。所以我们简单地检查位置是否不是-1，然后继续我们的快乐之路。

本文的长度表明了用 Python 检查字符串中的字符串是多么容易，我希望它能让您相信，就编写更多 Python 代码而言，第一种方法远远好于第二种方法。

暂时先这样。