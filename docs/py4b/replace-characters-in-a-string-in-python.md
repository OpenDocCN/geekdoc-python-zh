# 在 Python 中替换字符串中的字符

> 原文：<https://www.pythonforbeginners.com/basics/replace-characters-in-a-string-in-python>

在程序中处理文本数据时，有时我们可能需要修改数据。在本文中，我们将看看在 Python 中替换字符串中的字符的各种方法。我们将讨论各种用例，以便更好地理解这个过程。

## 如何替换字符串中的一个字符？

要用另一个字符替换字符串中的一个字符，我们可以使用 replace()方法。在字符串上调用 replace()方法时，将被替换的字符作为第一个输入，新字符作为第二个输入，将被替换的字符数作为可选输入。replace()方法的语法如下:

`str.replace(old_character, new_character, n)`

这里，old_character 是将用 new_character 替换的字符。输入 n 是可选参数，指定必须用 new_character 替换的 old_character 的出现次数。

现在让我们讨论如何使用各种用例替换字符串中的字符。

## 替换字符串中出现的所有字符

要用新字符替换所有出现的字符，我们只需在字符串上调用 replace()方法时，将旧字符和新字符作为输入传递给该方法。您可以在下面的示例中观察到这一点。

```py
input_string = "This is PythonForBeginners.com. Here, you   can read python tutorials for free."
new_string = input_string.replace('i', "I")
print("The original string is:")
print(input_string)
print("Output String is:")
print(new_string) 
```

输出:

```py
The original string is:
This is PythonForBeginners.com. Here, you   can read python tutorials for free.
Output String is:
ThIs Is PythonForBegInners.com. Here, you   can read python tutorIals for free.
```

这里，我们使用 replace()方法将所有出现的字符“I”替换为字符“I”。

我们也可以用一组新的字符替换一组连续的字符，而不是替换单个字符。替换连续字符组的语法保持不变。我们只需将旧字符和新字符传递给 replace()方法，如下所示。

```py
input_string = "This is PythonForBeginners.com. Here, you   can read python tutorials for free."
new_string = input_string.replace('is', "IS")
print("The original string is:")
print(input_string)
print("Output String is:")
print(new_string) 
```

输出:

```py
The original string is:
This is PythonForBeginners.com. Here, you   can read python tutorials for free.
Output String is:
ThIS IS PythonForBeginners.com. Here, you   can read python tutorials for free. 
```

在本例中，您可以看到我们在输出字符串中用“is”替换了“is”。

## 替换字符串中第 n 个出现的字符

要用另一个字符替换字符串中出现的前 n 个字符，我们可以在 replace()方法中指定可选的输入参数。指定要替换的字符的出现次数后，从字符串开始的 n 个字符将被替换。

例如，我们可以用“A”替换给定文本中出现的前 3 个字符“A ”,如下所示。

```py
input_string = "An owl sat on the back of an elephant and started to tease him."
new_string = input_string.replace('a', "A",3)
print("The original string is:")
print(input_string)
print("Output String is:")
print(new_string) 
```

输出:

```py
The original string is:
An owl sat on the back of an elephant and started to tease him.
Output String is:
An owl sAt on the bAck of An elephant and started to tease him. 
```

## 使用 replace()方法的好处

一般情况下，我们需要使用正则表达式来查找字符串中的字符，以替换它们。正则表达式很难理解，需要小心使用以匹配要正确替换的字符。此外，使用正则表达式在计算上也是低效的。因此，replace()方法为我们提供了一种简单且计算高效的方法来替换字符串中的字符。

## 结论

在本文中，我们讨论了 Python 中的 replace()方法。我们还看到了用它来处理文本或原始数据的各种方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)