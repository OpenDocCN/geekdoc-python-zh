# 在 Python 中使用列表理解

> 原文：<https://www.askpython.com/python/list/python-list-comprehension>

大家好！今天，我们将讨论如何在 Python 中使用列表理解。

列表理解通常是一种语法糖，使代码更容易阅读和编写。

通常，当我们处理涉及创建列表的代码时，反复编写嵌套循环是很麻烦的。

Python 通过引入这一特性使我们的工作变得更加容易。

现在让我们通过合适的例子来了解如何在我们的程序中使用它！

* * *

## 列表理解的基本结构

让我们考虑以下正常编写的代码:

```py
word = "Hello from AskPython"
letters = []

for letter in word:
    letters.append(letter)

print(letters)

```

输出

```py
['H', 'e', 'l', 'l', 'o', ' ', 'f', 'r', 'o', 'm', ' ', 'A', 's', 'k', 'P', 'y', 't', 'h', 'o', 'n']

```

上面的代码片段打印了单词中的字母列表。

我们可以使用列表理解来缩短这段代码，因为列表的元素有一个共同的属性:它们是字母，并且它们将被追加到列表中。

现在让我们使用列表理解来使它更简短和易读:

```py
word = "Hello from AskPython"

letters = [letter for letter in word]

print(letters)

```

**输出**

```py
['H', 'e', 'l', 'l', 'o', ' ', 'f', 'r', 'o', 'm', ' ', 'A', 's', 'k', 'P', 'y', 't', 'h', 'o', 'n']

```

看到有多简单了吗？代码的意图很清楚:我们选取单词的字母，并直接将其添加到我们的列表中！

现在，我们也可以用列表理解其他的可重复项了！

再举一个例子，我们可以生成 1 到 10 的数字的平方。

正常的方法如下:

```py
squares = []

for i in range(1, 11):
    squares.append(i * i)

print(squares)

```

**输出**

```py
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

```

虽然这已经足够小了，但是我们可以使用列表理解做得更好。我们可以把它减少到只有两行代码！

```py
squares = [i * i for i in range(1, 11)]
print(squares)

```

我们现在向您展示了理解列表的能力！现在让我们通过处理像`if`和`else`这样的条件句来给它添加一些趣味吧！

* * *

## 在列表理解中使用条件

在我们的列表理解中，我们可以使用`if`和`else` [条件句](https://www.askpython.com/python/python-if-else-elif-statement)。

让我们考虑第一种情况，这里我们只有一个`if`条件。

这种列表理解的一般结构如下:

```py
list = [item for item in iterable if condition]

```

所以在这里，`list`将只由`item`组成，其中`condition`成立。

让我们以之前构建正方形的例子为例，使用`if`将其限制为偶数。

```py
squares = [i * i for i in range(1, 11) if i % 2 == 0]
print(squares)

```

**输出**

```py
[4, 16, 36, 64, 100]

```

事实上，我们在这里只能得到偶数个元素方块，因为只有当 I 是偶数时。

现在让我们看看第二种情况，我们也有一个`else`条件。该结构现在将如下所示:

```py
list = [value1 if condition else value2 for item in iterable]

```

这里，如果`condition == True`，列表将包含`value1`的元素，如果`condition == False`，则包含`value2`的元素。

现在让我们举一个例子，我们一直打印整数方块，直到`i<=5`。如果我> 5，我们将打印 0。

我们的列表理解现在看起来像这样:

```py
my_list = [i * i if i <= 5 else 0 for i in range(10)]
print(my_list)

```

输出

```py
[0, 1, 4, 9, 16, 25, 0, 0, 0, 0]

```

如您所见，该列表仅包含所有< = 5 的数字的平方。其余元素设置为 0！

我们也可以使用其他条件，甚至 lambda 函数，如果我们愿意的话！

这里有一个稍微有点做作的例子，它使用一个`lambda`从 0 开始计算连续的对和。(0, 1 + 2, 2 + 3, 3 + 4..)

```py
pair_sums = [(lambda x, y: x + y)(i, j) if i > 1 else 0 for i, j in zip(range(1, 11), range(0, 10))]
print(pair_sums)

```

**输出**

```py
[0, 3, 5, 7, 9, 11, 13, 15, 17, 19]

```

如您所见，这段代码可读性不是很好，如果使用其他代码可能会更好！

所以，如果你想一次做太多事情，小心不要使用列表理解。当您想要运行简单的循环条件语句来构建列表时，最好坚持使用这种方法，而不是当您想要对每个元素单独执行数学计算时。

* * *

## 结论

在本文中，我们学习了使用 Python 的列表理解语义。这使得使用迭代循环来减少重复编写代码变得更加容易！

## 参考

*   Python 列表上的 [AskPython 文章](https://www.askpython.com/python/list/python-list)

* * *