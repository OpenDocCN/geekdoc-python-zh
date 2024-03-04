# 字符串串联和格式化

> 原文：<https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting>

作为一名 Python 程序员，几乎可以保证你需要掌握[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)和格式化。

无论是准备一条最终将被用户阅读的消息，还是编写供内部使用的工具，知道如何构造完美的字符串对于几乎任何程序的成功都是至关重要的。

沟通往往是成功的关键。在一个由数据驱动的世界里尤其如此。

幸运的是，Python 为我们提供了一系列简化字符串格式化过程的工具。掌握这些工具将提供精确制作琴弦所需的技能。

发展您的 Python 技能将使您能够处理大量的数据，以及 Python 提供的所有令人兴奋的机会。

## 串联

在任何编程语言中，连接两个或多个字符串都是一项常见的任务。这样做被称为*串联*。Python 提供了几个连接字符串的工具。

当我们在 Python 中连接字符串时，两个或更多的字符串被组合成一个新的字符串。这有点像数学中的“加法”，但我们添加的不是数字，而是字符串。

在 Python 中连接字符串最简单的方法是使用'+'操作符。在 Python 中，'+'操作符将连接任意两个字符串。

下面是语法:

```py
final_string = string_one + string_two
```

当我们以这种方式“添加”字符串时，我们实际上是在连接它们。正如您所看到的，Python 几乎从字面上理解了添加字符串的概念。

让我们通过一个代码示例来看看'+'操作符的工作情况。

### 示例 Python 中的串联

```py
# joining strings with the '+' operator
# first create two strings that we’ll combine 
first_name = "Sherlock"
last_name = "Holmes"

# join the names, separated by a space character
full_name = first_name + " " + last_name

print("Hello, " + full_name + ".") 
```

### 输出

```py
Hello, Sherlock Holmes.
```

在本 Python 示例中，使用“+”运算符连接三个字符串，并将它们的值组合成一个新的字符串。

使用'+'操作符是 Python 中组合字符串的一种直接方法。

在创建了变量 *full_name* 之后，我们的程序使用 print()方法在命令提示符中显示字符串。使用'+'操作符，我们格式化字符串以包含正确的标点符号，并添加额外的空格。

使用“+”操作符连接字符串是创建更易读文本的一种很好的方式。创建人类可读的文本是软件设计的一个重要部分。当字符串格式正确且含义清晰时，软件使用起来就容易多了。

可以使用更长的 Python 语句连接多个字符串。

### 示例 2:组合多个字符串。

```py
act1 = "This is the beginning"
act2 = "this is the middle"
act3 = "this is the end"

story = act1 + ", " + act2 + ", and " + act3 + "."

print(story) 
```

### 输出

```py
This is the beginning, this is the middle, and this is the end.
```

从上面的例子可以看出，可以连接的字符串数量没有限制。通过使用'+'操作符，可以连接许多不同的字符串。

需要注意的是，Python 不能连接不同类型的对象。例如，不可能将一个字符串和一个数字连接起来。这样做会引起 Python 的抱怨。

```py
>>> str = "red"
>>> print(str + 3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: can only concatenate str (not "int") to str 
```

尝试连接字符串和整数会导致错误。从 Traceback 消息中可以看出，Python 只能将一个字符串与另一个字符串连接起来。

解决这个问题的一个简单方法是使用 **str()** 方法。使用这个方法会将一个数字转换成一个字符串。

```py
>>> print("red" + " " + str(13))
red 13 
```

**str()** 方法简化了将不同类型的数据转换成字符串对象的过程。这是一种特殊的 Python 方法，适用于许多不同类型的对象。使用它可以快速将数字、小数和其他对象转换为字符串。

### 示例 3:使用 str()方法将数字转换为字符串。

```py
# use the str() method to convert numbers to strings
num = 221
letter = "B"
street = "Baker Street"

print(str(num) + letter + " " + street) 
```

### 输出

```py
221B Baker Street
```

## Python 中的字符串格式

为了精确地格式化文本，有必要使用更高级的技术。有时我们想在字符串中插入一个变量而不使用连接。这可以通过一种叫做*插值*的方法来实现。

通过插值，Python 动态地将数据插入到字符串中。Python 提供了多种插值数据的方法。

这些方法非常灵活，并且根据您的需要，可以与其他工具结合使用。使用它们精确格式化字符串并提高 Python 程序的可读性。

## 使用%运算符的字符串格式

在 Python 中格式化字符串的一种方法是使用“%”运算符。使用“%”操作符将文本插入到占位符中，其方式类似于 **format()** 方法的工作方式。

当用“%”操作符格式化字符串时，我们必须指定将替换占位符的数据类型。这可以通过特殊字符来实现。

例如,“s”字符用于告诉 Python 我们想要用字符串数据替换占位符。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

### 示例 4:使用%运算符对数据进行插值。

```py
# %s for strings
# %d for numbers

print("%s never ends, %s." % ("Education","Watson")) 
```

### 输出

```py
Education never ends, Watson.
```

如果我们想用其他数据类型替换占位符，我们需要使用其他特殊字符。例如，您可以使用“%d”作为字符串中的占位符，将整数传递给字符串 buying。

```py
author = "Arthor Conan Doyle"
bday = 1859

print("%s was born on %d." % (author, bday)) 
```

## 用{ }运算符设置字符串格式

格式化复杂的字符串可能很繁琐。为了减轻负担，Python 提供了一种将数据插入字符串的特殊方法。

使用 **format()** 方法，占位符用于将数据插入到字符串中。 **format()** 方法将在字符串中查找这些占位符，并用新值替换它们，作为参数提供给该方法。

占位符是用花括号指定的。

我们来看看 **format()** 方法在起作用。

### 示例 5:使用。format()替换字符串中的文本。

```py
subject = "Dr. Watson"
clues = 3

message = "{}, I've found {} clues!".format(subject, clues)
print(message) 
```

### 输出

```py
Dr. Watson, I've found 3 clues!
```

## 在 Python 中使用 Join 方法

还可以使用命名索引来标识占位符。

**join()** 方法用于连接一个 Python iterable。该方法提供了一种将列表或字典转换为字符串数据的快速方法。

这个方法是一个强大而通用的工具，可用于处理字符串数据的列表或字典。

打印 Python 列表的内容是一项常见的任务。使用 **join()** 方法，可以将字符串列表转换成单个字符串。 **join()** 方法需要一个*连接符*，它是用来将列表“粘合”在一起的字符。

例如，如果使用逗号作为连接符，那么列表中的元素将在最后的字符串中用逗号分隔。任何字符串都可以作为连接符，但是通常使用空格这样的东西来分隔字符串。

### 示例 6:用 join()方法连接字符串。

```py
# create a list of string data
titles = ["A Study in Scarlet", "The Sign of Four", "A Scandal in Bohemia"]

# join the list using a comma and a space to separate the strings
print(', '.join(titles)) 
```

### 输出

```py
A Study in Scarlet, The Sign of Four, A Scandal in Bohemia
```

**join()** 方法也可以用来打印字典的内容。

### 示例 7:字典和 join()方法。

```py
# use join() to print the characters in the dictionary
print(' '.join(characters))

# populate a list of occupations
occupations = []
for character in characters:
    occupations.append(characters[character])

# use join to print the occupations
print(' '.join(occupations)) 
```

### 输出

```py
Sherlock Watson Moriarty 
```

在这个例子中，我们使用了 **join()** 方法来打印字典和列表。如您所见，如果使用 **join()** 来打印字典，那么只打印字典的键。

为了查看字典中的值，创建并填充一个新列表是很有用的。这个新列表将保存字典中分配给键的值。

创建并填充一个名为“职业”的新列表后，我们可以使用 **join()** 来打印该列表。

```py
Detective Doctor Criminal Mastermind
```

## 相关职位

学习正确格式化字符串是每个 Python 程序员都需要的一项重要技能。Python 提供了几个工具来完成这项任务。掌握这些工具将确保你的程序尽可能清晰和有用。

有几种方法可以提高你在这方面的技能。最简单的方法是练习你新发现的 Python 技能，并构建一些你自己的程序。例如，构建一个基于文本的游戏是练习字符串连接和格式化的绝佳方式。

学生可以使用其他资源。如果您对 Python 编程的更多信息感兴趣，请访问下面的链接。

*   [Python 字符串拆分](https://www.pythonforbeginners.com/dictionary/python-split)
*   [Python 列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)