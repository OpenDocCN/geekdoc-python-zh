# 使用 Python 进行排列和组合

> 原文：<https://www.askpython.com/python/permutations-and-combinations-using-python>

在本文中，我们将学习如何使用 Python 寻找排列和组合。

Python 提供了一个名为 [itertools](https://www.askpython.com/python-modules/python-itertools-module) 的库，其中包含计算排列和组合的内置函数。让我们快速看一下这些函数的实现。

## 导入所需的库

在我们能够使用以下任何函数之前，我们需要导入`itertools`库。这是通过以下方式实现的:

```py
import itertools

```

上面的语句导入了库，并形成了使用其函数的路径。

* * *

## 寻找排列

排列在数学上是指*“某些数字或字母的排列”*。`itertools`库中的`permutations()`函数就是这样做的。

### 1.Python 字符串的置换

如果给我们一个 [Python 字符串](https://www.askpython.com/python/string/strings-in-python)，并要求我们找出其字母排列的所有方式，那么这个任务可以通过`permutations()`函数轻松完成。

```py
import itertools

st = "ABC"

per = itertools.permutations(st)

for val in per:
	print(*val)

```

**输出:**

```py
A B C
A C B
B A C
B C A
C A B
C B A

```

函数`permutations()`接受一个字符串参数，并提供一个`itertools`对象。如果我们试图直接打印变量`'per'`，我们将得到如下结果:

```py
<itertools.permutations object at 0x7fc9abdd8308>

```

因此，有必要运行一个循环来打印每个条目。

* * *

### 2.多个数字的排列

`permuatations()`函数采用可迭代的参数，因此为了找出数字的排列，我们需要将数字作为列表、集合或[元组](https://www.askpython.com/python/tuple/python-tuple)来传递。

```py
import itertools

values = [1, 2, 3]

per = itertools.permutations(values)

for val in per:
	print(*val)

```

**输出:**

```py
1 2 3
1 3 2
2 1 3
2 3 1
3 1 2
3 2 1

```

在上述计算排列的技术中，我们包括了所有的数字或字母。有可能限制排列中元素的数量。

* * *

### 3.具有一定数量元素的排列

类似于**‘nPr’**的概念，即*“从 n 个元素中排列 r 个元素”*，这可以通过在元素集后传递一个整数来实现。

```py
import itertools

values = [1, 2, 3, 4]

per = itertools.permutations(values, 2)

for val in per:
	print(*val)

```

**输出:**

```py
1 2
1 3
1 4
2 1
2 3
2 4
3 1
3 2
3 4
4 1
4 2
4 3

```

在上面的代码片段中，`permutations()`函数被要求一次只排列所提供的数字列表中的 2 个元素。

* * *

## 寻找组合

术语组合指的是从一组对象中提取元素的方法。`itertools`库为这个功能提供了一个方法`combinations()`。

这里需要注意的一点是，挑选一组对象并不涉及排列。`combinations()`函数有两个参数:

1.  一组值
2.  一个整数，表示要为组合选取的值的数量。

### 1.单词中字母的组合

给定一个单词，如果我们需要从这个单词中找出恰好包含两个字母的所有组合，那么`combinations()`就是定位函数。

```py
import itertools

st = "ABCDE"

com = itertools.combinations(st, 2)

for val in com:
	print(*val)

```

**输出:**

```py
A B
A C
A D
A E
B C
B D
B E
C D
C E
D E

```

* * *

### 2.数字集合的组合

类似于我们从一个单词中的字母得到的组合结果，它可以在一个列表中的数字上实现。

```py
import itertools

values = [1, 2, 3, 4]

com = itertools.combinations(values, 2)

for val in com:
	print(*val)

```

**输出:**

```py
1 2
1 3
1 4
2 3
2 4
3 4

```

> **注意:**生成的组合基于每个对象的索引值，而不是它们的实际值，因此，如果有任何值重复，则该函数会打印相似的组合，认为每个值都不同。

* * *

### 3.重复数字的组合

为了进一步解释上面的**注释**，让我们为它运行一个例子。

```py
import itertools

values = [1, 1, 2, 2]

com = itertools.combinations(values, 2)

for val in com:
	print(*val)

```

**输出:**

```py
1 1
1 2
1 2
1 2
1 2
2 2

```

结果不言自明，因为列表中的数字是重复的。

* * *

### 4.数字自身的组合

在`itertools`库中还有另一个与排列和组合相关的函数叫做`combinations_with_replacement()`。这个函数是`combinations()`函数的一个变体，略有不同的是它包括元素自身的组合。

```py
import itertools

values = [1, 2, 3, 4]

com = itertools.combinations_with_replacement(values, 2)

for val in com:
	print(*val)

```

**输出:**

```py
1 1
1 2
1 3
1 4
2 2
2 3
2 4
3 3
3 4
4 4

```

当传递给`combinations()`函数时，我们可以看到上述输出与具有相同参数的输出的明显区别。

数字有它们自己的组合，就像列表中有它们的多个实例。这基本上是因为，当我们从列表中选择一个元素时，上面的函数会再次放置相同的值，以获得与其自身的组合。

这总结了使用 Python 进行排列和组合的主题。

## 结论

排列和组合在数学领域的应用非常广泛。本教程中解释的方法在解决关于这种数学技术的问题时会派上用场。

我们希望这篇文章简单易懂。如有疑问和建议，欢迎在下面发表评论。