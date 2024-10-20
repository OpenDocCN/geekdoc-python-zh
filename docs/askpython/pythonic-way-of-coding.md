# 编写代码的 Pythonic 方式

> 原文：<https://www.askpython.com/python/examples/pythonic-way-of-coding>

在大多数编程语言中，我们经常遇到声称它是最强大的语言的说法。嗯，这个说法好像比较主观，为什么会这样呢？这是因为用一种语言编写的程序可以用另一种语言编写。

不同语言的语法可能不同，但不存在一种语言可以实现而另一种语言不能实现的算法。在本文中，我们将看到 Python 在最佳实践、简单性和可读性方面提供的一些方法，可以表述为***Python 化的*** 。

## Pythonic 代码的要点

就像任何其他语言一样，Python 中有各种各样的方法或设计模式。Python 本身使用的语法读起来非常接近英语，我们可以找到大量的文章或在线评论，大部分来自其他语言的人，Python 可以在几周内学会等等。这对于任何语言来说都是不正确的，对于 Python 来说更是如此。

***Python 化*** *指的是 Python 特有的惯用编码做法，确实不仅仅是使用 Python 语法和使用标准库。*

Python 也被称为**习惯用法**，这意味着基于 Python 代码与标准库提供的特性保持一致的想法来编写 Python 代码。对于不同的程序员来说，这种说法可能是不同的，它当然不是任何人的绝对指南。

这是像一个有经验的程序员一样用最佳实践和技术编写 Python 的想法，就像任何其他语言一样，它通常被一个巨大的社区所遵循。

## 我们到底为什么需要 Pythonic 代码？

每当我们编写一个程序时，它都是为了被正确执行。编写程序有多种方式，但有些方式可以被视为**【更多】**正确的方式。这对 Python 来说尤其如此，因为它使用极简语法，可读性很强。所以，如果我们用错了**【少】**的正确方式，往往很难被忽视。

用“更”正确的方式编写代码有很多好处。其中一些可以表述为:

*   我们正在利用数千名非常聪明的开发人员，他们拥有多年编写 Python 的经验，并对开源语言做出了贡献。利用他们长期使用该语言所获得的专业知识，实现使用 Python 的最佳实践和惯用方法，这当然是一个好主意。
*   具体谈到 CPython，它期望某种类型的代码实现以提高效率，因为它已经针对许多操作进行了调整。例如，我们可以通过使用 ***[for 循环](https://www.askpython.com/course/python-course-for-loop)*** 从列表中创建*子列表，或者我们可以使用 ***[切片](https://www.askpython.com/python/array/array-slicing-in-python)*** 来实现相同的目的。切片稍微好一点，因为它的工作方式与我们预期的完全一样。*
*   一段代码应该是易于阅读和全面的。比方说，当我们看到常见的模式如 ***[列表理解](https://www.askpython.com/python/list/python-list-comprehension)*** 时，我们甚至可以在解析所有符号之前更快地理解代码。相反，以非 pythonic 的方式，我们必须分析整个代码才能理解它是如何工作的。因此，以惯用的方式编写一段代码通常更干净、更简单。

作为 Python 开发者，这些对我们来说是显而易见的优势。然而，还有一些不太明显或更广泛的好处，例如:

*   当我们运行开源项目时，如果使用最佳实践，人们更容易做出贡献。为我们的项目做出贡献的人看到代码以聪明的方式编写会很兴奋。错过这个想法的项目之一是传统的 T2 PyPI。当我们 **[pip 安装](https://www.askpython.com/python/pyinstaller-executable-files)** 任何包时，我们运行的就是 **PyPI** 。它已经存在很多年了，但是很难找到开源的贡献者。嗯， **PyPI** 源代码缺乏惯用的实践。它由一个文件中的数千行代码和一个相似的代码库组成。所以，当人们试图让它变得更好时，他们很难读懂这些代码。

*值得庆幸的是，它被重写并在 [PyPI 的官方网站](https://pypi.org/)上发布，使用了最佳实践和现代习语，当然这次有更多的贡献者。*

*   使用封闭的源代码工作，例如为一个组织或公司工作，只有最好的开发人员会被期望工作并被保留。因此，项目中的最佳编码实践和一致性为那些开发人员提供了在更习惯、更易理解和更可读的代码上工作的好处。

## Pythonic 方式的代码示例

让我们试着通过一些例子来对我们在本文前面讨论过的 Pythonic 代码有一个大致的了解。

### **1。** **命名约定**

也被称为*标识符名称*的变量作为一个基本部分在所有编程语言中使用。通常，命名变量没有固定的准则，但是遵循一个模式会有所帮助。

PEP8 提供了一些遵循 Pythonic 方式命名变量的最佳实践或风格指南。即使只是简单地看一眼，也能很好地了解正在阅读的代码类型。下面提供了一些例子。

| **命名惯例** | **摘自人教版 8 风格指南** |
| 包装 | 公用事业 |
| 模块 | db_utils, dbutils, db_connect |
| 班级 | 客户详细信息 |
| 功能 | 创建订单 |
| 变量 | 订单 id |
| 常数 | 运输 _ 成本 |

PEP 8 Style Guide

### **2。** **使用 f 字符串进行字符串格式化**

使用字符串输出值或返回包含表达式的插值字符串的语句是 Python 中的一种常见模式。早些时候，字符串被*连接*或者使用 *+操作符*连接，这导致了一长串代码，看起来非常混乱，经常令人困惑。这个问题很快得到解决，语言提供了一些非常简洁的解决方案。

在 **Python 3.6** 中引入了 [***f 弦***](https://www.askpython.com/python/string/string-formatting) ，为上述问题提供了更好的解决方案。*f 字符串可以被认为是对用 Python 编写字符串*的改进，因为用 f 字符串格式编写的代码不会向后兼容，但是实现它们符合 Python 的标准。让我们通过一些例子来获得一个概述。

```py
first_name, age, job = "John", 33, "programmer"

# Method 1: String Concatenation
print(first_name + " " + "is a" + " " + str(age) + " " + "year old" + " " + job)

# Method 2: % Conversion specifier
print("%s is a %d year old %s" % (first_name, age, job))

# Method 3: Using format: Until Python 3.6
print("{0} is a {1} year old {2}".format(first_name, age, job))

# Method 4: The Pythonic Way: From Python 3.6 and above
print(f"{first_name} is a {age} year old {job}")

"""
Output:

John is a 33 year old programmer
John is a 33 year old programmer
John is a 33 year old programmer
John is a 33 year old programmer

"""

```

### **3。使用 enumerate()而不是 len()或 range()函数与 *for-loops***

通常，我们必须使用 *for-loops* 来迭代 Python 中的可迭代对象，例如列表、字符串或元组，并且还需要通过它们进行*索引*。有很多方法可以实现这一点。一些常见的模式使用 **`len()`** 或 **`range()`** 函数来访问那些可迭代对象的索引。Python 有一个内置的[**`enumerate()`**](https://www.askpython.com/python/built-in-methods/python-enumerate-method)函数来执行这个操作，这可以被认为是 python 提供了对索引的直接访问，而不是使用多个函数对其进行编码。

```py
# ------------------------------
# Looping and Indexing through
# -----------------------------

# Method 1
greet = "hello"
i = 0
for eachChar in greet:
    print(i, eachChar)
    i += 1

"""
0 h
1 e
2 l
3 l
4 o

"""

# Method 2: A better way
greet = "there"
for idx in range(len(greet)):
    print(idx, greet[idx])

"""
0 t
1 h
2 e
3 r
4 e

"""

# Method 3: The Pythonic way: enumerate()
greet = "hello"
for idx, eachChar in enumerate(greet):
    print(idx, eachChar)

"""
0 h
1 e
2 l
3 l
4 o

"""

```

### **4。使用列表理解代替 *for 循环***

*列表理解*是 Python 特有的。这是我们用 Python 快速创建列表的一种方式，而不是循环和使用 append。它提供了一个更短的语法来基于原始列表中包含的条目创建一个全新的列表，或者更具体地说是基于 iterable 对象。下面是使用 append 函数和更多的*python 式*列表理解的例子。请注意，它不能被过度使用，因为我们不希望在一行代码中有太多的逻辑，但它在各种情况下都非常有用。它使我们的代码简洁易读。

```py
# Using loops
my_list = []
for char in "hello":
    my_list.append(char)
print(my_list)  # ['h', 'e', 'l', 'l', 'o']

# The Pythonic Way: Using List Comprehensions
my_list1 = [char for char in "hello"]
print(my_list1)  #  ['h', 'e', 'l', 'l', 'o']

```

### **5。合并字典**

Python 中到处都在使用字典。它们是最重要的数据结构之一，有多种用途。字典中的键|值对是构建块，在 Python 中对它们执行多重操作是很常见的。对于这个例子，我们将按照惯例或程序以及一些 Pythonic 方式来合并字典。

```py
# Declaring two variables for dictionaries
customer_details = {"name": "john", "age": 34, "email": "[email protected]"}
order_items = {"item1": "apple", "item2": "orange"}

# Method 1: Not So Pythonic: Using Loops
order_details = {}
for i in customer_details:
    order_details[i] = customer_details[i]
for i in order_items:
    order_details[i] = order_items[i]

print(f"Result_one: {order_details}")

# The Pythonic way: Spreading out and merging using **kwargs
order_details = {**customer_details, **order_items}
print(f"Result_two: {order_details}")

# Python 3.10 Updated:  Union Operator (with | pipe character)
# The Pythonic Way
order_details = customer_details | order_items
print(f"Result_three: {order_details}")

"""
Output:

Result_one: {'name': 'john', 'age': 34, 'email': '[email protected]', 'item1': 'apple', 'item2': 'orange'}  
Result_two: {'name': 'john', 'age': 34, 'email': '[email protected]', 'item1': 'apple', 'item2': 'orange'}  
Result_three: {'name': 'john', 'age': 34, 'email': '[email protected]', 'item1': 'apple', 'item2': 'orange'}
"""

```

## 结论

在所有编程语言中，都有特定于它们的最佳实践。作为最常用的语言之一，Python 也不例外。它有其久经考验的技术和写作风格。编写 Pythonic 代码不仅使我们的代码更简单易读，而且易于维护。

惯用的 Python 提供了一种易于识别的设计模式，这种模式使得编程成为一种无缝的体验，混淆最小。在本文中，我们浏览了其中的一些方法，以获得 Pythonic 方式的概述。

还有很多其他的例子，而且随着时间的推移，这种例子只会越来越多。Python 确实在 Python 开发人员社区中占有重要地位。