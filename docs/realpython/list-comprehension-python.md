# Python 中何时使用列表理解

> 原文：<https://realpython.com/list-comprehension-python/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**理解 Python 列表理解**](/courses/understand-list-comprehensions/)

Python 以允许您编写优雅、易于编写、几乎和普通英语一样易读的代码而闻名。该语言最与众不同的特性之一是**列表理解**，您可以使用它在一行代码中创建强大的功能。然而，许多开发人员努力充分利用 Python 中列表理解的更高级特性。一些程序员甚至过度使用它们，这会导致代码效率更低，更难阅读。

本教程结束时，您将了解 Python list comprehensions 的全部功能，以及如何舒适地使用它们的特性。您还将了解使用这些方法的利弊，这样您就可以决定何时使用其他方法更好。

**在本教程中，您将学习如何:**

*   用 Python 重写循环和`map()`调用作为**列表理解**
*   **在理解、循环和`map()`调用之间选择**
*   用**条件逻辑**增强你的理解力
*   **用理解**代替`filter()`
*   **剖析**您的代码以解决性能问题

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 如何在 Python 中创建列表

在 Python 中有几种不同的方法可以创建[列表](https://realpython.com/python-lists-tuples/)。为了更好地理解在 Python 中使用列表理解的利弊，让我们首先看看如何用这些方法创建列表。

[*Remove ads*](/account/join/)

### 使用`for`循环

最常见的循环类型是 [`for`](https://realpython.com/courses/python-for-loop/) 循环。您可以使用一个`for`循环分三步创建一个元素列表:

1.  实例化一个空列表。
2.  循环遍历可迭代或[范围](https://realpython.com/python-range/)的元素。
3.  将每个元素追加到列表的末尾。

如果您想创建一个包含前十个完美方块的列表，那么您可以用三行代码完成这些步骤:

>>>

```py
>>> squares = []
>>> for i in range(10):
...     squares.append(i * i)
>>> squares
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

在这里，您实例化了一个空列表`squares`。然后，使用一个`for`循环来迭代`range(10)`。最后，将每个[数字](https://realpython.com/python-numbers/)乘以自身，并将结果附加到列表的末尾。

### 使用`map()`对象

[`map()`](https://realpython.com/python-map-function/) 提供了另一种基于[函数式编程](https://realpython.com/python-functional-programming/)的方法。你传入一个函数和一个 iterable，然后`map()`会创建一个对象。这个对象包含通过提供的函数运行每个 iterable 元素得到的输出。

例如，考虑这样一种情况，您需要计算一系列交易的税后价格:

>>>

```py
>>> txns = [1.09, 23.56, 57.84, 4.56, 6.78]
>>> TAX_RATE = .08
>>> def get_price_with_tax(txn):
...     return txn * (1 + TAX_RATE)
>>> final_prices = map(get_price_with_tax, txns)
>>> list(final_prices)
[1.1772000000000002, 25.4448, 62.467200000000005, 4.9248, 7.322400000000001]
```

这里，你有一个可迭代的`txns`和一个函数`get_price_with_tax()`。您将这两个参数传递给`map()`，并将结果对象存储在`final_prices`中。您可以使用`list()`轻松地将这个地图对象转换成一个列表。

### 使用列表理解

列表理解是列表的第三种方式。使用这种优雅的方法，您可以用一行代码重写第一个示例中的`for`循环:

>>>

```py
>>> squares = [i * i for i in range(10)]
>>> squares
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

您只需按照以下格式同时定义列表及其内容，而不是创建一个空列表并将每个元素添加到末尾:

>>>

```py
new_list = [expression for member in iterable]
```

Python 中的每个列表理解都包括三个元素:

1.  **`expression`** 是成员本身、对方法的调用或者任何其他返回值的有效表达式。在上面的例子中，表达式`i * i`是成员值的平方。
2.  **`member`** 是列表中的对象或值或 iterable。在上面的例子中，成员值是`i`。
3.  **`iterable`** 是一个列表、[集合](https://realpython.com/python-sets/)、序列、[生成器](https://realpython.com/introduction-to-python-generators/)，或者任何其他可以一次返回一个元素的对象。在上面的例子中，iterable 是`range(10)`。

因为**表达式**的需求非常灵活，所以 Python 中的列表理解在很多你会用到`map()`的地方都工作得很好。您可以用自己的列表理解重写定价示例:

>>>

```py
>>> txns = [1.09, 23.56, 57.84, 4.56, 6.78]
>>> TAX_RATE = .08
>>> def get_price_with_tax(txn):
...     return txn * (1 + TAX_RATE)
>>> final_prices = [get_price_with_tax(i) for i in txns] >>> final_prices
[1.1772000000000002, 25.4448, 62.467200000000005, 4.9248, 7.322400000000001]
```

这个实现和`map()`的唯一区别是 Python 中的 list comprehension 返回一个列表，而不是一个 map 对象。

[*Remove ads*](/account/join/)

### 使用列表理解的好处

列表理解通常被描述为比循环或 T0 更具 T2 风格。但是，与其盲目接受这种评价，不如理解在 Python 中使用列表理解与其他选择相比的好处。稍后，您将了解替代方案是更好选择的一些场景。

在 Python 中使用 list comprehension 的一个主要好处是，它是一个可以在许多不同情况下使用的单一工具。除了标准的[列表创建](https://realpython.com/python-lists-tuples/)之外，列表理解还可以用于映射和过滤。您不必为每个场景使用不同的方法。

这就是为什么列表理解被认为是**Python**的主要原因，因为 Python 包含了简单而强大的工具，可以在各种各样的情况下使用。一个额外的好处是，无论何时在 Python 中使用 list comprehension，您都不需要像调用`map()`时那样记住参数的正确顺序。

列表理解也比循环更具有声明性，这意味着它们更容易阅读和理解。循环要求你关注列表是如何创建的。您必须手动创建一个空列表，循环遍历元素，并将每个元素添加到列表的末尾。有了对 Python 中列表的理解，你可以专注于*你想在列表中放什么*，并且相信 Python 会处理*如何*列表的构造。

## 如何增强你的理解力

为了理解[列表理解](https://realpython.com/courses/using-list-comprehensions-effectively/)能够提供的全部价值，理解它们可能的功能范围是有帮助的。你也会想了解在 [Python 3.8](https://realpython.com/python38-new-features/) 中列表理解的变化。

### 使用条件逻辑

前面，您看到了如何创建列表理解的公式:

>>>

```py
new_list = [expression for member in iterable]
```

虽然这个公式是准确的，但它也有点不完整。对理解公式更完整的描述增加了对可选的**条件句**的支持。将[条件逻辑](https://realpython.com/python-conditional-statements/)添加到列表理解中最常见的方法是在表达式的末尾添加一个条件:

>>>

```py
new_list = [expression for member in iterable (if conditional)]
```

这里，您的条件语句就在右括号之前。

条件非常重要，因为它们允许列表理解过滤掉不需要的值，这通常需要调用 [`filter()`](https://realpython.com/python-filter-function/) :

>>>

```py
>>> sentence = 'the rocket came back from mars'
>>> vowels = [i for i in sentence if i in 'aeiou']
>>> vowels
['e', 'o', 'e', 'a', 'e', 'a', 'o', 'a']
```

在这个代码块中，条件语句过滤掉`sentence`中不是元音的任何字符。

条件可以测试任何有效的表达式。如果您需要一个更复杂的过滤器，那么您甚至可以将条件逻辑移到一个单独的函数中:

>>>

```py
>>> sentence = 'The rocket, who was named Ted, came back \
... from Mars because he missed his friends.'
>>> def is_consonant(letter):
...     vowels = 'aeiou'
...     return letter.isalpha() and letter.lower() not in vowels
>>> consonants = [i for i in sentence if is_consonant(i)]
['T', 'h', 'r', 'c', 'k', 't', 'w', 'h', 'w', 's', 'n', 'm', 'd', \
'T', 'd', 'c', 'm', 'b', 'c', 'k', 'f', 'r', 'm', 'M', 'r', 's', 'b', \
'c', 's', 'h', 'm', 's', 's', 'd', 'h', 's', 'f', 'r', 'n', 'd', 's']
```

在这里，您创建了一个复杂的过滤器`is_consonant()`，并将这个函数作为条件语句传递给您的列表理解。注意，成员值`i`也作为参数传递给函数。

您可以将条件放在语句的末尾进行简单的过滤，但是如果您想要*更改*一个成员值而不是将其过滤掉，该怎么办呢？在这种情况下，将条件放在表达式的*开头*附近很有用:

>>>

```py
new_list = [expression (if conditional) for member in iterable]
```

使用此公式，您可以使用条件逻辑从多个可能的输出选项中进行选择。例如，如果您有一个价格列表，那么您可能希望用`0`替换负价格，而保持正值不变:

>>>

```py
>>> original_prices = [1.25, -9.45, 10.22, 3.78, -5.92, 1.16]
>>> prices = [i if i > 0 else 0 for i in original_prices]
>>> prices
[1.25, 0, 10.22, 3.78, 0, 1.16]
```

这里，您的表达式`i`包含一个条件语句`if i > 0 else 0`。这告诉 Python 如果数字是正数就输出`i`的值，但是如果数字是负数就把`i`改为`0`。如果这看起来让人不知所措，那么将条件逻辑视为其自身的功能可能会有所帮助:

>>>

```py
>>> def get_price(price): ...     return price if price > 0 else 0 >>> prices = [get_price(i) for i in original_prices]
>>> prices
[1.25, 0, 10.22, 3.78, 0, 1.16]
```

现在，您的条件语句包含在`get_price()`中，您可以将它用作列表理解表达式的一部分。

[*Remove ads*](/account/join/)

### 使用集合和字典理解

虽然 Python 中的 list comprehension 是一个常用工具，但是您也可以创建 set 和[dictionary](https://realpython.com/python-dicts/)comprehension。一个**集合理解**和 Python 中的列表理解几乎完全一样。区别在于集合理解确保输出不包含重复项。您可以通过使用花括号而不是括号来创建一个集合理解:

>>>

```py
>>> quote = "life, uh, finds a way"
>>> unique_vowels = {i for i in quote if i in 'aeiou'}
>>> unique_vowels
{'a', 'e', 'u', 'i'}
```

您的 set comprehension 会输出在`quote`中找到的所有独特元音。与列表不同，集合不保证项目将以任何特定的顺序保存。这就是为什么集合中的第一个成员是`a`，尽管`quote`中的第一个元音是`i`。

**字典理解**是相似的，额外的要求是定义一个键:

>>>

```py
>>> squares = {i: i * i for i in range(10)}
>>> squares
{0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81}
```

为了创建`squares`字典，在表达式中使用花括号(`{}`)和键值对(`i: i * i`)。

### 使用 Walrus 运算符

Python 3.8 将引入[赋值表达式](https://www.python.org/dev/peps/pep-0572/)，也被称为**海象运算符**。为了理解如何使用它，考虑下面的例子。

假设您需要向一个将返回温度数据的 API 发出十个请求。您只想返回大于 100 华氏度的结果。假设每个请求将返回不同的数据。在这种情况下，无法使用 Python 中的列表理解来解决这个问题。公式`expression for member in iterable (if conditional)`没有为条件提供将数据赋给表达式可以访问的[变量](https://realpython.com/python-variables/)的方法。

海象运营商 T2 解决了这个问题。它允许您在将输出值赋给变量的同时运行表达式。下面的例子展示了这是如何实现的，使用`get_weather_data()`来生成假的天气数据:

>>>

```py
>>> import random
>>> def get_weather_data():
...     return random.randrange(90, 110)
>>> hot_temps = [temp for _ in range(20) if (temp := get_weather_data()) >= 100]
>>> hot_temps
[107, 102, 109, 104, 107, 109, 108, 101, 104]
```

在 Python 的 list comprehension 中，您不经常需要使用赋值表达式，但在必要时，它是一个非常有用的工具。

## Python 中何时不使用列表理解

列表理解非常有用，可以帮助您编写易于阅读和调试的优雅代码，但是它们并不是所有情况下的正确选择。它们可能会使您的代码运行得更慢或使用更多的内存。如果您的代码性能较差或难以理解，那么最好选择一种替代方案。

### 小心嵌套的理解

理解可以**嵌套**以在集合中创建列表、字典和集合的组合。例如，假设一个气候实验室正在跟踪六月第一周五个不同城市的高温。存储这些数据的完美数据结构可以是嵌套在字典理解中的 Python 列表理解:

>>>

```py
>>> cities = ['Austin', 'Tacoma', 'Topeka', 'Sacramento', 'Charlotte']
>>> temps = {city: [0 for _ in range(7)] for city in cities}
>>> temps
{
 'Austin': [0, 0, 0, 0, 0, 0, 0],
 'Tacoma': [0, 0, 0, 0, 0, 0, 0],
 'Topeka': [0, 0, 0, 0, 0, 0, 0],
 'Sacramento': [0, 0, 0, 0, 0, 0, 0],
 'Charlotte': [0, 0, 0, 0, 0, 0, 0]
}
```

您用字典理解创建外部集合`temps`。这个表达式是一个键值对，它包含了另一种理解。这段代码将快速生成`cities`中每个城市的数据列表。

嵌套列表是创建**矩阵**的常用方法，通常用于数学目的。看看下面的代码块:

>>>

```py
>>> matrix = [[i for i in range(5)] for _ in range(6)]
>>> matrix
[
 [0, 1, 2, 3, 4],
 [0, 1, 2, 3, 4],
 [0, 1, 2, 3, 4],
 [0, 1, 2, 3, 4],
 [0, 1, 2, 3, 4],
 [0, 1, 2, 3, 4]
]
```

外部列表理解`[... for _ in range(6)]`创建六行，而内部列表理解`[i for i in range(5)]`用值填充每一行。

到目前为止，每个嵌套理解的目的都非常直观。然而，还有其他情况，比如**展平**嵌套列表，其中的逻辑可能会使您的代码更加混乱。以这个例子为例，它使用嵌套列表理解来展平矩阵:

>>>

```py
matrix = [
...     [0, 0, 0],
...     [1, 1, 1],
...     [2, 2, 2],
... ]
>>> flat = [num for row in matrix for num in row]
>>> flat
[0, 0, 0, 1, 1, 1, 2, 2, 2]
```

展平矩阵的代码很简洁，但理解它的工作原理可能不那么直观。另一方面，如果您使用`for`循环来展平同一个矩阵，那么您的代码会简单得多:

>>>

```py
>>> matrix = [
...     [0, 0, 0],
...     [1, 1, 1],
...     [2, 2, 2],
... ]
>>> flat = []
>>> for row in matrix:
...     for num in row:
...         flat.append(num)
...
>>> flat
[0, 0, 0, 1, 1, 1, 2, 2, 2]
```

现在您可以看到代码一次遍历矩阵的一行，在移动到下一行之前取出该行中的所有元素。

虽然单行嵌套列表理解可能看起来更 Pythonic 化，但最重要的是编写您的团队可以容易理解和修改的代码。当你选择你的方法时，你必须根据你认为理解有助于还是有损于可读性来做出判断。

[*Remove ads*](/account/join/)

### 为大型数据集选择生成器

Python 中的列表理解通过将整个输出列表加载到内存中来实现。对于小型甚至中型的列表，这通常是好的。如果你想计算前 1000 个整数的平方和，那么列表理解可以很好地解决这个问题:

>>>

```py
>>> sum([i * i for i in range(1000)])
332833500
```

但是如果你想计算前十亿个整数的平方和呢？如果您在您的机器上尝试了，那么您可能会注意到您的计算机变得没有响应。这是因为 Python 试图创建一个包含 10 亿个整数的列表，这会消耗比你的计算机所希望的更多的内存。您的计算机可能没有生成庞大列表并将其存储在内存中所需的资源。如果你试图这样做，那么你的机器可能会变慢甚至崩溃。

当列表的大小有问题时，使用一个[生成器](https://realpython.com/courses/python-generators/)来代替 Python 中的列表理解通常是有帮助的。一个**生成器**不会在内存中创建一个单一的大型数据结构，而是返回一个 iterable。您的代码可以根据需要多次从 iterable 中请求下一个值，或者直到到达序列的末尾，同时一次只存储一个值。

如果你要用一个生成器对前十亿个平方求和，那么你的程序可能会运行一段时间，但不会导致你的计算机死机。以下示例使用了一个生成器:

>>>

```py
>>> sum(i * i for i in range(1000000000))
333333332833333333500000000
```

您可以看出这是一个生成器，因为表达式没有用括号或花括号括起来。或者，生成器可以用括号括起来。

上面的例子仍然需要大量的工作，但是它缓慢地执行了操作**。由于惰性求值，只有在显式请求时才会计算值。在生成器生成一个值(例如，`567 * 567`)后，它可以将该值添加到运行总和中，然后丢弃该值并生成下一个值(`568 * 568`)。当 sum 函数请求下一个值时，循环重新开始。这个过程保持了较小的内存占用。*

*`map()`也运行缓慢，这意味着如果您选择在这种情况下使用内存，它将不是问题:

>>>

```py
>>> sum(map(lambda i: i*i, range(1000000000)))
333333332833333333500000000
```

你更喜欢生成器表达式还是`map()`由你决定。

### 优化性能的配置文件

那么，哪种方法更快呢？你应该使用列表理解还是它们的替代品？与其坚持一个在所有情况下都适用的规则，不如问问自己在你的具体情况下，表现是否重要。如果不是，那么通常最好选择能产生最干净代码的方法！

如果你处在一个性能很重要的场景中，那么通常最好的方法是**描述**不同的方法并倾听数据。 [`timeit`](https://docs.python.org/3/library/timeit.html) 是一个有用的库，用于计时大块代码运行的时间。您可以使用`timeit`来比较`map()`、`for`循环的运行时间，并列出理解:

>>>

```py
>>> import random
>>> import timeit
>>> TAX_RATE = .08
>>> txns = [random.randrange(100) for _ in range(100000)]
>>> def get_price(txn):
...     return txn * (1 + TAX_RATE)
...
>>> def get_prices_with_map():
...     return list(map(get_price, txns))
...
>>> def get_prices_with_comprehension():
...     return [get_price(txn) for txn in txns]
...
>>> def get_prices_with_loop():
...     prices = []
...     for txn in txns:
...         prices.append(get_price(txn))
...     return prices
...
>>> timeit.timeit(get_prices_with_map, number=100)
2.0554370979998566
>>> timeit.timeit(get_prices_with_comprehension, number=100)
2.3982384680002724
>>> timeit.timeit(get_prices_with_loop, number=100)
3.0531821520007725
```

这里，您定义了三种方法，每种方法使用不同的方法来创建列表。然后，你告诉`timeit`每个函数运行 100 次。`timeit`返回运行这 100 次执行所花费的总时间。

正如代码所展示的，基于循环的方法和`map()`之间最大的区别是，循环的执行时间延长了 50%。这是否重要取决于您的应用程序的需求。

## 结论

在本教程中，您学习了如何使用 Python 中的**列表理解**来完成复杂的任务，而不会使您的代码过于复杂。

现在您可以:

*   用声明性**列表理解**简化循环和`map()`调用
*   用**条件逻辑**增强你的理解力
*   创建**集合**和**字典**释义
*   确定何时代码清晰性或性能决定了**替代方法**

每当您必须选择一种列表创建方法时，请尝试多种实现，并考虑在您的特定场景中最容易阅读和理解的是什么。如果性能很重要，那么您可以使用分析工具为您提供可操作的数据，而不是依靠直觉或猜测来判断什么是最好的。

请记住，虽然 Python 列表理解得到了很多关注，但您的直觉和使用数据的能力将帮助您编写干净的代码来完成手头的任务。最终，这是使您的代码 Pythonic 化的关键！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**理解 Python 列表理解**](/courses/understand-list-comprehensions/)********