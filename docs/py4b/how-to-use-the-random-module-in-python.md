# 如何使用 Python 中的随机模块

> 原文：<https://www.pythonforbeginners.com/random/how-to-use-the-random-module-in-python>

在这篇文章中，我将描述 Python 中随机模块的使用。随机模块提供对支持许多操作的功能的访问。也许最重要的是它允许你生成随机数。

## Python 中什么时候使用 Random 模块？

如果你想让计算机在一个给定的范围内挑选一个随机数，从一个 [python 列表](https://www.pythonforbeginners.com/lists/python-lists-cheat-sheet-2)中挑选一个随机元素，从一副牌中挑选一张随机牌，掷硬币等等，你可以使用 python 中的 random 模块。您还可以使用随机模块来创建随机字符串，同时选择密码以使您的密码数据库更加安全，或者为您网站的随机页面功能提供动力。

## Python 中的随机函数

随机模块包含一些非常有用的函数，即 randint()函数、Random()函数、choice()函数、randrange()函数和 shuffle()函数。让我们逐一讨论这些功能。

#### Python 中的 randint()函数

python random 模块中定义的 randint()函数可用于在一个范围内创建随机字符串。该函数将两个数字作为其输入参数。第一个输入参数是范围的开始，第二个输入参数是范围的结束。执行后，randint()函数返回给定范围内的随机整数。

如果我们想要一个随机整数，我们可以使用 randint()函数。例如，您可以生成 1 到 5 之间的随机整数，如下例所示。

```py
import random
print random.randint(0, 5) 
```

上述代码将输出 1、2、3、4 或 5。这里，您应该确保 randint()函数中的第一个输入参数应该小于第二个输入参数。否则，程序会出错。

#### random()函数

python random 模块中的 random()函数用于生成 0 到 1 之间的随机数。执行时，random()函数返回一个介于 0 和 1 之间的浮点数。

如果你想要一个更大的数字，你可以把它乘以一个更大的值。例如，要创建一个 0 到 100 之间的随机数，可以将 random()函数的输出乘以 100，如下所示。

```py
import random
random.random() * 100 
```

#### choice()函数

choice()函数用于从一个集合对象中选择一个随机元素，比如列表、集合、元组等。该函数将集合对象作为其输入参数，并返回一个随机元素。

例如，通过将颜色名称列表作为输入传递给 choice()函数，可以从颜色列表中选择一种随机颜色，如下例所示。

```py
random.choice( ['red', 'black', 'green'] ). 
```

choice 函数通常用于从列表中选择一个随机元素。

```py
import random
myList = [2, 109, False, 10, "Lorem", 482, "Ipsum"]
random.choice(myList) 
```

#### 函数的作用是

顾名思义，shuffle 函数将列表中的元素打乱顺序。shuffle()函数将一个列表作为输入参数。执行后，列表中的元素以随机的顺序被打乱，如下例所示。

```py
from random import shuffle
x = [[i] for i in range(10)]
shuffle(x) 
```

```py
Output:
# print x  gives  [[9], [2], [7], [0], [4], [5], [3], [1], [8], [6]]
# of course your results will vary 
```

#### randrange()函数

python random 模块中的 randrange()函数用于从给定的范围中选择一个随机元素。它采用三个数字作为输入参数，即开始、停止和步进。执行后，它从范围(开始、停止、步进)中生成一个随机选择的元素。要了解 range()函数的工作原理，可以阅读这篇关于 [python range](https://www.pythonforbeginners.com/modules-in-python/python-range-function) 的文章。

```py
random.randrange(start, stop[, step]) 
```

```py
import random
for i in range(3):
    print random.randrange(0, 101, 5) 
```

实际上，randrange()函数是 choice()函数和 range()函数的组合。

## Python 中随机模块的代码示例

下面的代码示例使用 choice()函数来计算 10000 次投掷中正面和反面的数量。为此，我们定义了一个名为 outcomes 的字典来存储正面和反面的数量。接下来，我们使用 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)的 keys()方法来获取列表["heads "，" tails"]。之后，我们使用 choice()函数从列表中随机选择一个值，并根据输出更新结果字典。

```py
import random
import itertools

outcomes = { 'heads':0,
             'tails':0,
             }
sides = outcomes.keys()

for i in range(10000):
    outcomes[ random.choice(sides) ] += 1

print 'Heads:', outcomes['heads']
print 'Tails:', outcomes['tails'] 
```

只允许两种结果，所以与其使用数字并转换它们，不如将单词“正面”和“反面”与 choice()一起使用。

使用结果名称作为关键字，将结果列表显示在字典中。

```py
$ python random_choice.py

Heads: 4984
Tails: 5016 
```

## 结论

在本文中，我们讨论了 Python 中 random 模块的不同函数。我们还讨论了一个使用 choice()函数模拟抛硬币的代码示例。

要了解更多关于 python 编程的知识，你可以阅读这篇关于 Python 中的[列表理解的文章。你可能也会喜欢这篇关于](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)[字符串操作](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)的文章。

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！