# 如何在 Python 3 中连接字符串

> 原文：<https://www.pythonforbeginners.com/python-strings/how-to-join-strings-in-python-3>

程序员注定要处理大量的字符串数据。这部分是因为计算机语言与人类语言紧密相连，我们用一种语言创造另一种语言，反之亦然。

因此，尽早掌握使用字符串的细节是一个好主意。在 Python 中，这包括学习如何连接字符串。

操纵字符串似乎令人生畏，但是 Python 语言包含了使这一复杂任务变得更容易的工具。在深入 Python 的工具集之前，让我们花点时间研究一下 Python 中字符串的属性。

## 一点弦理论

您可能还记得，在 Python 中，字符串是字符数据的数组。

关于字符串的重要一点是，它们在 Python 语言中是不可变的。这意味着一旦创建了 Python 字符串，就不能更改。更改字符串需要创建一个全新的字符串，或者覆盖旧的字符串。

我们可以通过创建一个新的字符串变量来验证 Python 的这个特性。如果我们试图改变字符串中的一个字符，Python 会给我们一个回溯错误。

```py
>>> my_string = "Python For Beginners"
>>> my_string[0]
'P'
>>> my_string[0] = 'p'
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'str' object does not support item assignment 
```

在编写 Python 代码时，牢记字符串的不可改变性是一个好主意。

虽然您不能在 Python 中更改字符串，但是您可以连接它们，或者追加它们。Python 附带了许多工具来简化字符串的处理。

在这一课中，我们将讲述连接字符串的各种方法，包括[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)。说到连接字符串，我们可以利用 Python 操作符和内置方法。

随着学生的进步，他们可能会以这样或那样的方式使用这些技术。各有各的目的。

## 用“+”运算符连接字符串

串联是将两个或多个字符串连接在一起以创建一个新的字符串的行为。

在 Python 中，可以使用“+”运算符连接字符串。类似于数学等式，这种连接字符串的方式非常简单，允许将许多字符串“相加”在一起。

让我们来看一些例子:

```py
# joining strings with the '+' operator
first_name = "Bilbo"
last_name = "Baggins"

# join the names, separated by a space
full_name = first_name + " " + last_name

print("Hello, " + full_name + ".")
```

在我们的第一个例子中，我们创建了两个字符串，first_name 和 last_name，然后使用'+'操作符将它们连接起来。为了清楚起见，我们在名称之间添加了空格。

运行该文件时，我们会在命令提示符下看到以下文本:

```py
Hello, Bilbo Baggins.
```

示例末尾的 print 语句显示了连接字符串如何生成更易读的文本。通过串联的力量添加标点符号，我们可以创建更容易理解、更容易更新、更容易被其他人使用的 Python 程序。

让我们看另一个例子。这次我们将利用 for 循环来连接字符串数据。

```py
# some characters from Lord of the Rings
characters = ["Frodo", "Gandalf", "Sam", "Aragorn", "Eowyn"]

storyline = ""

# loop through each character and add them to the storyline
for i in range(len(characters)):
    # include "and" before the last character in the list
    if i == len(characters)-1:
        storyline += "and " + characters[i]
    else:
        storyline += characters[i] + ", "

storyline += " are on their way to Mordor to destroy the ring."

print(storyline) 
```

这个更高级的例子展示了如何使用串联从 Python 列表中生成人类可读的文本。使用 for 循环，角色列表(取自《指环王》系列小说)被一个接一个地加入到故事情节字符串中。

这个循环中包含了一个条件语句，用来检查我们是否到达了字符列表中的最后一个对象。如果有的话，还要加上一个“和”,以便最终文本更加清晰易读。我们也一定会包括我们的牛津逗号，以增加可读性。

以下是最终输出:

```py
Frodo, Gandalf, Sam, Aragorn, and Eowyn, are on their way to Mordor to destroy the ring.
```

除非两个对象都是字符串，否则这个方法不起作用。例如，试图将一个字符串与一个数字连接会产生一个错误。

```py
>>> string = "one" + 2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: can only concatenate str (not "int") to str 
```

如您所见，Python 只允许我们将一个字符串连接到另一个字符串。作为程序员，我们的工作就是理解我们正在使用的语言的局限性。在 Python 中，如果我们希望避免任何错误，我们需要确保我们连接了正确类型的对象。

## 用“+”运算符连接列表

“+”运算符也可以用于连接一个或多个字符串数据列表。例如，如果我们有三个列表，每个列表都有自己唯一的字符串，我们可以使用'+'操作符来创建一个新的列表，组合所有三个列表中的元素。

```py
hobbits = ["Frodo", "Sam"]
elves = ["Legolas"]
humans = ["Aragorn"]

print(hobbits + elves + humans) 
```

如您所见，运算符“+”有许多用途。有了它，Python 程序员可以轻松地组合字符串数据和字符串列表。

## 将字符串与。join()方法

如果你在 Python 中处理一个 *iterable* 对象，你可能会想要使用**。加入()**方法。一个*可迭代的*对象，比如一个字符串或者一个列表，可以很容易地用**连接起来。join()** 方法。

任何 Python iterable 或序列都可以使用**来连接。join()** 方法。这包括列表和字典。

**。join()** 方法是一个字符串实例方法。语法如下:

*string _ name . join(iterable)*

这里有一个使用**的例子。join()** 方法连接一列字符串:

```py
numbers = ["one", "two", "three", "four", "five"]

print(','.join(numbers)) 
```

在命令提示符下运行程序，我们将看到以下输出:

```py
one,two,three,four,five
```

**。join()** 方法将返回一个新字符串，该字符串包含 iterable 中的所有元素，由一个*分隔符*连接。在前面的例子中，分隔符是逗号，但是任何字符串都可以用来连接数据。

```py
numbers = ["one", "two", "three", "four", "five"]
print(' and '.join(numbers))
```

我们也可以使用这种方法连接字母数字数据列表，使用空字符串作为分隔符。

```py
title = ['L','o','r','d',' ','o','f',' ','t','h','e',' ','R','i','n','g','s']
print(“”.join(title)) 
```

**。join()** 方法也可以用来获得一个带有字典内容的字符串。使用**时。join()** 这样，该方法将只返回字典中的键，而不是它们的值。

```py
number_dictionary = {"one":1, "two":2, "three":3,"four":4,"five":5}
print(', '.join(number_dictionary)) 
```

用**连接序列时。join()** 方法，结果将是一个包含来自两个序列的元素的字符串。

## 使用“*”运算符复制字符串

如果需要连接两个或更多相同的字符串，可以使用' * '操作符。

使用' * '操作符，您可以将一个字符串重复任意次。

```py
fruit = “apple”
print(fruit * 2)
```

“*”运算符可以与“+”运算符结合使用，以连接字符串。结合这些方法使我们能够利用 Python 的许多高级特性。

```py
fruit1 = "apple"
fruit2 = "orange"

fruit1 += " "
fruit2 += " "

print(fruit1 * 2 + " " + fruit2 * 3) 
```

## 拆分和重新连接字符串

因为 Python 中的字符串是不可变的，所以拆分和重新组合它们是很常见的。

的。 **split()** 方法是另一个字符串实例方法。这意味着我们可以在任何字符串对象的末尾调用它。

就像**一样。加入()**方法，**。split()** 方法使用一个*分隔符*来解析字符串数据。默认情况下，空白用作该方法的分隔符。

我们来看看**。【split()方法在起作用。**

```py
names = "Frodo Sam Gandalf Aragorn"

print(names.split()) 
```

这段代码输出一个字符串列表。

```py
['Frodo', 'Sam', 'Gandalf', 'Aragorn']
```

再举一个例子，我们来看看如何将一个句子分割成独立的部分。

```py
story = "Frodo took the ring of power to the mountain of doom."
words = story.split()
print(words) 
```

使用。 **split()** 方法返回新的 iterable 对象。因为对象是可迭代的，我们可以使用**。join()** 我们之前学过的将字符串“粘合”在一起的方法。

```py
original = "Frodo took the ring of power to the mountain of doom."
words = original.split()

remake = ' '.join(words)
print(remake) 
```

通过使用 Python 中的字符串方法，我们可以轻松地拆分和连接字符串。这些方法对于处理字符串和可迭代对象至关重要。

## 把松散的部分绑起来

到目前为止，您应该对字符串以及如何在 Python 3 中使用它们有了更深入的了解。完成本教程中提供的示例将是您掌握 Python 之旅的良好开端。

然而，没有学生能独自成功。这就是为什么我们编辑了 Python 为初学者提供的附加资源列表，以帮助你完成培训。

*   了解如何创建 [Python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)。
*   [Python 列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)初学者指南。

只要有帮助和耐心，任何人都可以学习 Python 的基础知识。如果用 Python 连接字符串看起来令人畏惧，那么花点时间练习上面的例子。通过熟悉字符串变量和使用方法，您将很快挖掘出 Python 编程语言的无限潜力。