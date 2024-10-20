# Python 中的列表理解

> 原文：<https://www.pythoncentral.io/list-comprehension-in-python/>

有时我们需要生成遵循一些自然逻辑的列表，比如迭代一个序列并在其中应用一些条件。我们可以使用 Python 的“列表理解”技术编写紧凑的代码来生成列表。我们可以循环遍历一个序列，并应用逻辑表达式。

首先，让我们来看一个特殊的函数***range***——顾名思义，它用于生成一个范围内的数字列表！尝试 Python IDLE 中的以下代码部分:

```py
>>range(10)

[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

>>range(5, 10)

[5, 6, 7, 8, 9]

>>range(0, 10, 2)

[0, 2, 4, 6, 8]

```

所以 range 实际上生成了一个列表。我们可以在 for 循环中使用它。看看下面的例子，我们生成了一个名为 *myList* 的列表:

```py
myList = []

for i in range(0, 5):

myList.append(i**2)

print(myList)
```

上述代码片段将有以下输出:

```py
[0, 1, 4, 9, 16]
```

我们还可以使用列表理解在一行代码中创建 *myList* :

```py
myList = [ i**2 for i in range(0, 5) ]
```

很酷吧。让我们看另一个例子。给定一个列表*输入列表*，让我们创建一个*输出列表*，它将只包含*输入列表*中具有正索引的元素(索引 0，2，4，...).首先，让我们尝试使用一个循环来实现这一点:

```py
inputList = ["bird", "mammal", "reptile", "fish", "insect"]

outputList = []

for i in range(0, len(inputList), 2):

outputList.append(inputList[i])

print(outputList)
```

现在让我们看看列表理解版本:

```py
outputList = [inputList[i] for i in range(0, len(inputList), 2)]

print(outputList)
```

这两种技术将有相同的输出:

```py
['bird', 'reptile', 'insect']
```

作为最后一个例子，给定一个整数列表，让我们找出奇数整数，并创建一个包含这些整数的新列表:

```py
input = [4, 7, 9, 3, 12, 25, 30]

output = []

for x in input:

if not x%2 == 0:

output.append(x)

print(output)
```

将打印:

```py
[7, 9, 3, 25]
```

使用列表理解方法完成的相同工作将是编写如下内容:

```py
output = [x for x in input if not x%2 == 0]

print(output)
```

就是这样！现在我们可以描述语法了。我们由第三个大括号——“***[***&*”——开始和结束，一切都要用大括号括起来。左大括号后面是一个 ***表达式*** ，后面是一个 ***for*** 子句，然后是零个或多个***for***&***if***子句。*

 *列表理解总是返回一个列表，评估我们放在左括号后的 ***表达式*** 。

## **评估“for”子句的顺序**

如果子句有一个以上的*，它们将按照在循环中被求值的顺序被求值。例如，注意下面的列表理解，*

```py
output = [x**y for x in range(1, 5) for y in range(1, 3)]
```

类似于:

```py
output = []

for x in range(1,5):

for y in range(1, 3):

output.append(x ** y)
```

两者都给出*【1，1，2，4，3，9，4，16】*作为输出。

现在我们已经学习了什么是列表理解，它的语法和一些显示它的用法的例子。问题是，为什么要用列表理解？在许多情况下，通过一起使用*映射*&*λ*函数，您可以在不使用列表理解的情况下获得相同的结果。然而，请注意，在大多数情况下，列表理解被认为比 map & lambda 函数放在一起更快。此外，列表理解使代码简洁易读——这是漂亮编码的必备条件。地图也帮助我们自然编码；例如，我们可以使用列表理解创建一个所有正数的列表，就像我们在数学课上思考的那样！*