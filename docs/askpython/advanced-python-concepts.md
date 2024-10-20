# 高级 Python 概念

> 原文：<https://www.askpython.com/python/advanced-python-concepts>

今天让我们来看看一些更高级的 Python 概念。虽然我们已经在之前的教程中讨论了其中的一些概念，但本页将为您提供 Python 学习中常见高级概念的快速指南。

## 高级 Python 概念的简要列表

事不宜迟，让我们继续我们的第一个先进概念。

### 1.λ函数

在 Python 中， [lambda 函数](https://www.askpython.com/python/python-lambda-anonymous-function)是一个声明为匿名的单行函数，即声明时没有名字，它可能有许多参数，但只有一个表达式。

语法:

```py
lambda arguments: expression

```

*   如下面的语法所示，lambda 函数是通过使用关键字“lambda”来声明的。
*   然后我们写一个参数列表，lambda 函数可以带任意数量的参数，但不能是零。在冒号之后，我们编写一个表达式，将这些参数应用于任何实际操作。从语法上讲，lambda 函数仅限于单个表达式，即它只能包含一个表达式，不能超过一个。

**举例:**

```py
remainder = lambda number: number%2
print (remainder (25))

```

**说明:**

在上面的代码中，`lambda num: number%2`是 lambda 函数。数字是参数，而数字% 2 是被计算并返回结果的表达式。

该表达式推导出输入 2 的输入模数。我们给 25 作为参数，除以 2，我们得到剩下的 1。

您应该注意到，上面脚本中的 lambda 函数没有被赋予任何名称。它只是将给定的项返回给标识符的其余部分。

然而，即使它是未知的，我们也有可能称它为正常函数。

这是 lambda 函数的另一个例子:

```py
addition = lambda a, b: a+b
print (addition (19,55))

```

**输出:** **74**

* * *

### 2.Python 中的理解

Python 中理解力为我们提供了一种压缩但简洁的方式来创造新的序列(如列表、集合、字典等)。)

Python 支持 4 种理解类型

*   列表理解
*   词典理解
*   一组
*   发电机

* * *

#### 列表理解

列表是 Python 中基本的[数据类型之一。每当您遇到一个变量名后跟一个方括号[ ]，或列表生成器，它是一个可以包含多个项目的列表，使其成为一种集成的数据类型。同样，宣布一个新的列表，然后向其中添加一个或多个项目也是一个好主意。](https://www.askpython.com/python/python-data-types)

示例:

```py
even_numbers = [2, 4, 6, 8, 10]
print (even_numbers)

```

**输出:**

```py
[2,4,6,8,10]

```

**什么是列表理解？**

简单来说，[列表理解](https://www.askpython.com/python/list/python-list-comprehension)就是从现有列表中构建新列表的过程。或者，你可以说这是 Python 独特的方式，将循环的[添加到列表中。事实上，列表理解比传统列表有很多优势。](https://www.askpython.com/python/python-for-loop)

首先，代码不超过一行，易于声明和阅读。使用理解比使用 for 循环更便于理解列表。最后，这也是创建一个新的、更动态的列表的简单、快速和准确的方法。

**语法:**

```py
[expression for item in list]

```

运筹学

```py
[expression for item in list if conditional]

```

list comprehension 的语法与其他语法有点不同，因为表达式是在循环之前提到的，但这就是它的工作方式。

**举例:**

```py
n_letter = [letter for letter in 'encyclopedia']
print(n_letter)

```

**输出:**

['e '，' n '，' c '，' y '，' c '，' l '，' o '，' p '，' e '，' d '，' I '，' a']

* * *

#### 词典理解

[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)是一种称为关联数组的数据结构的 Python 实现。字典包含一组键值。每对密钥将密钥设置为其对应的值。您可以通过用大括号({})括起逗号分隔的键值对列表来定义字典。冒号(:)将每个键与其关联的值分隔开:

**举例:**

```py
thisdict = {"name": "Ford","age": 34, "last_name": "Mustang"}
print(thisdict)

```

**输出:**

```py
{'name': 'Ford', 'age': 34, 'last_name': 'Mustang'}

```

**什么是词典理解？**

[字典理解](https://www.askpython.com/python/dictionary/python-dictionary-comprehension)类似于列表理解，但需要定义一个关键字:

**语法:**

```py
output_dict = {key:value for (key, value) in iterable if (key, value satisfy this condition)}

```

**举例:**

在这个例子中，我们将使用一个常规函数执行与理解相同的功能。

```py
sq_dict = dict()
for number in range(1, 9):
    sq_dict[number] = number*number
print(sq_dict)

```

现在，让我们使用字典理解来尝试相同的功能

```py
square_dict = {num: num*num for num in range(1, 9)}
print(square_dict)

```

**输出:**

```py
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64}

```

* * *

#### 集合理解

[集合](https://www.askpython.com/python/set/python-set)用于在单个变量中存储多个项目。集合是 Python 内置的用于存储数据集合的 4 种数据类型之一。其他 3 个列表、元组和字典，都有不同的属性和用途。

**举例:**

```py
brand_set = {"Mustang", "Ferrari", "Ford","Aston Martin"}
print(brand_set)

```

输出:

```py
{'Aston Martin', 'Mustang', 'Ford', 'Ferrari'}

```

**什么是集合理解？**

集合理解类似于列表理解。它们之间唯一的区别是集合理解使用了花括号{}。我们看下面这个例子来理解集合理解。

**语法:**

```py
{expr for variable in iterable}

```

运筹学

```py
{expression for variable in iterable if condition}

```

**举例:**

```py
s = [1,2,3,4,5,4,6,6,7,8,8,]
using_comp = {var for var in s if var % 2 ==0}
print(using_comp)

```

**输出:**

```py
{8, 2, 4, 6}

```

* * *

#### 生成器理解

一个[生成器是一种特殊类型的迭代器](https://www.askpython.com/python/examples/generators-in-python)，它维护着关于如何分别产生它的单个组件的指令，以及它的当前复制状态。它只在迭代请求时产生每个成员，一次一个。

**语法:**

```py
(expression for var in iterable if condition)

```

**什么是生成器理解？**

生成器理解和列表理解非常相似。两者的一个区别是，生成器理解用圆括号，列表理解用方括号。

它们之间的主要区别是生成器不为整个列表设置内存。相反，它们单独产生每个值，这就是为什么它们在内存中工作得如此之好。

**举例:**

```py
input_list = [1, 2, 3, 4, 4, 5, 6, 7, 7] 
output_gen = (var for var in input_list if var % 2 == 0) 
print("Output values using generator comprehensions:", end = ' ') 
for var in output_gen: 
     print(var, end = ' ')

```

**输出:**

```py
Output values using generator comprehensions: 2 4 4 6

```

* * *

### 3.装饰函数

装饰器是强大而足智多谋的工具，允许程序员在不影响基本功能的情况下改变功能的性能。

你可以想到其他活动，比如普通甜甜圈；在甜甜圈上涂涂料的装饰工艺。不管你怎么装饰它们，它们还是甜甜圈。

换句话说，decorators 允许程序员包装另一个函数，以便在不改变其内部算法的情况下提高包装函数的性能。

**语法:**

```py
@dec2
@dec1
def func (arg1, arg2, ...):
    pass

```

* * *

### 4.哈希能力

Hashability 是 [Python objects](https://www.askpython.com/python/oops/python-classes-objects) 的一个特性，它告诉我们一个对象是否有哈希值。如果一个项目有一个散列值，它可以被用作一个字典键或一个预置项目。

如果一个对象在其整个生命周期中有一个固定的哈希值，那么它就是可哈希的。Python 有一个内置的 hash 方法(__hash __())，可以和其他对象进行比较。

比较需要 __eq __()或 __cmp __()方法，如果可散列项相等，则它们具有相同的散列值。

**举例:**

```py
s1 = (2,4,6,8,10)
s2 = (1,3,5,7,9)
#shows the id of the object
print(id(s1))
print(id(s2))

```

**输出:**

```py
1898434378944
1898436290656

```

在上面的例子中，两个项目是不同的，因为不可转换类型的哈希值依赖于存储的数据，而不是它们的 id。

使用散列的最大优点是从字典中获取条目的快速搜索时间(例如，O (1)复数时间)。类似地，检查某个东西是否是一个集合需要正常的时间。

换句话说，使用散列作为启动过程为各种标准操作提供了高性能，例如对象检测、对象安装和对象测试，使用了在引擎盖下具有散列表的头部。

* * *

## 结论

在本文中，我们回顾了 Python 中的五个高级概念。这里快速回顾一下最重要的信息。

*   **Lambda activities** :使用 Lambda 函数执行一个简单的任务，通常是在另一个函数调用中，比如 filter()或 max()。
*   理解:这是一种简单有效的方法，可以从系统中制作列表、字典和收藏。
*   生成器:延迟求值的迭代器，仅在被请求时才提供条目，因此，它们在内存中工作得很好。当按顺序处理大数据时，应该使用它们。
*   **decorator**:当你想寻找其他非算法变化和当前函数时，decorator 很有用。另外，装饰者可以重复使用。一旦定义好了，它们就可以随心所欲地修饰许多功能。
*   **Hashability** : Strength 是 Python 对象的必备组件，可以作为字典键或者 set 对象使用。他们提供了一种方法来恢复和安装一些有效的东西，以及成员测试。

* * *

这是关于 python 中一些高级主题的简单介绍。

希望这有所帮助！