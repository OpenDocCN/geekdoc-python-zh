# Python 的 all():检查你的 Iterables 的真实性

> 原文：<https://realpython.com/python-all/>

在编程时，你经常需要检查一个 iterable 中的所有项是否都是真实的。重复编写这种功能的代码可能会很烦人，而且效率很低。幸运的是，Python 提供了内置的`all()`函数来解决这个问题。这个函数接受一个 iterable 并检查它的所有项的真值，这对于发现这些项是否具有给定的属性或满足特定的条件很方便。

Python 的`all()`是一个强大的工具，可以帮助你用 Python 编写干净、可读、高效的代码。

**在本教程中，您将学习如何:**

*   使用 **`all()`** 检查一个 iterable 中的所有项是否为真
*   将`all()`与不同的**可迭代类型**一起使用
*   结合`all()`和**的理解**和**生成器表达式**
*   区分`all()`和布尔**运算符`and`和**

为了补充这些知识，您将编写几个例子来展示令人兴奋的`all()`用例，并强调在 Python 编程中使用该函数的许多方式。

为了理解本教程中的主题，您应该对几个 Python 概念有基本的了解，比如可迭代的[数据结构](https://realpython.com/python-data-structures/)、[布尔类型](https://realpython.com/python-boolean/)、[表达式](https://realpython.com/python-operators-expressions/)、[操作符](https://realpython.com/python-boolean/#boolean-operators)、[列表理解](https://realpython.com/list-comprehension-python/)，以及[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)。

**免费 PDF 下载:** [Python 3 备忘单](https://realpython.com/bonus/python-cheat-sheet-short/)

## 评估项目的真值

编程中一个很常见的问题是确定一个列表或数组中的所有元素是否都是真的。例如，您可能有以下条件列表:

*   `5 > 2`
*   `1 == 1`
*   `42 < 50`

为了确定这些条件是否为真，您需要迭代它们并测试每个条件的真实性。在这个例子中，你知道`5 > 2`为真，`1 == 1`为真，`42 < 50`也为真。因此，你可以说*所有*这些条件都是真实的。如果至少有一个条件为假，那么你会说*不是所有的*条件都为真。

注意，一旦你发现一个假条件，你可以停止评估条件，因为在这种情况下，你已经知道最终结果:*不是所有的*都是真的。

要通过编写定制的 Python 代码来解决这个问题，您可以使用一个 [`for`循环](https://realpython.com/python-for-loop/)来迭代每个条件并评估其真实性。您的循环将进行迭代，直到找到一个错误项，这时它将停止，因为您已经有了一个结果:

>>>

```py
>>> def all_true(iterable):
...     for item in iterable:
...         if not item:
...             return False
...     return True
...
```

这个[函数](https://realpython.com/defining-your-own-python-function/)以一个**可迭代**作为参数。循环迭代输入参数，同时[条件`if`语句](https://realpython.com/python-conditional-statements/)使用 [`not`](https://realpython.com/python-not-operator/) 运算符检查是否有任何项目为 falsy。如果一个项目是假的，那么函数立即[返回](https://realpython.com/python-return-statement/) `False`，表明不是所有的项目都是真的。否则返回`True`。

这个函数非常通用。它需要一个 iterable，这意味着你可以传入一个[列表，元组](https://realpython.com/python-lists-tuples/)，[字符串](https://realpython.com/python-strings/)，[字典](https://realpython.com/python-dicts/)，或者任何其他的 iterable [数据结构](https://realpython.com/python-data-structures/)。为了检查当前项目是真还是假，`all_true()`使用`not`运算符来反转其操作数的**真值**。换句话说，如果它的操作数计算结果为假，它将返回`True`，反之亦然。

Python 的[布尔](https://realpython.com/python-boolean/)操作符可以计算表达式和对象的真值，这保证了你的函数可以接受包含对象、表达式或两者的可迭代对象。例如，如果您传入一个布尔表达式的 iterable，那么`not`只对表达式求值并对结果求反。

下面是`all_true()`的行动:

>>>

```py
>>> bool_exps = [
...     5 > 2,
...     1 == 1,
...     42 < 50,
... ]
>>> all_true(bool_exps)
True
```

因为输入 iterable 中的所有表达式都为真，`not`对结果求反，`if`代码块永远不会运行。在这种情况下，`all_true()`返回`True`。

当输入 iterable 包含 Python 对象和非布尔表达式时，也会发生类似的情况:

>>>

```py
>>> objects = ["Hello!", 42, {}]
>>> all_true(objects)
False

>>> general_expressions = [
...     5 ** 2,
...     42 - 3,
...     int("42")
... ]
>>> all_true(general_expressions)
True

>>> empty = []
>>> all_true(empty)
True
```

在第一个例子中，输入列表包含常规的 Python 对象，包括一个字符串、一个[号](https://realpython.com/python-numbers/)和一个字典。在这种情况下，`all_true()`返回`False`，因为字典是空的，在 Python 中计算结果为 false。

为了对对象执行[真值测试](https://docs.python.org/3/library/stdtypes.html#truth-value-testing)，Python 为评估为假的对象提供了一组内部规则:

*   天生的负常数，像 [`None`](https://realpython.com/null-in-python/) 和`False`
*   带有零值的数字类型，如`0`、`0.0`、[、`0j`、](https://realpython.com/python-complex-numbers/)[、`Decimal("0")`、](https://docs.python.org/3/library/decimal.html#decimal.Decimal)、[、](https://realpython.com/python-fractions/)
*   空序列和集合，如`""`、`()`、`[]`、`{}`、[、](https://realpython.com/python-sets/)、[、`range(0)`、](https://realpython.com/python-range/)
*   实现返回值为`False`的 [`.__bool__()`](https://docs.python.org/3/reference/datamodel.html#object.__bool__) 或返回值为`0`的 [`.__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__) 的对象

当您在 Python 中测试其他任何对象的真值时，它的值都为 true。

在第二个示例中，输入列表包含一般的 Python 表达式，如数学表达式和函数调用。在这种情况下，Python 首先计算表达式以获得其结果值，然后检查该值的真实性。

第三个例子突出了`all_true()`的一个重要细节。当输入 iterable 为空时，`for`循环不运行，函数立即返回`True`。这种行为乍一看似乎很奇怪。然而，其背后的逻辑是，如果输入 iterable 中没有条目，那么就没有办法判断任何条目是否为 falsy。因此，该函数返回空的 iterables。

尽管用 Python 编写`all_true()`代码非常简单，但每次需要它的功能时都要编写这个自定义函数，这可能很烦人。确定 iterable 中的所有项是否都为真是编程中的一项常见任务，Python 为此提供了内置的 [`all()`](https://docs.python.org/3/library/functions.html#all) 函数。

[*Remove ads*](/account/join/)

## Python 的`all()` 入门

如果您查看 Python 的`all()`的[文档](https://docs.python.org/3/library/functions.html#all)，那么您会注意到该函数与您在上一节中编写的函数是等效的。然而，像所有内置函数一样，`all()`是一个 [C](https://realpython.com/c-for-python-programmers/) 函数，并针对性能进行了优化。

[提出了](https://www.artima.com/weblogs/viewpost.jsp?thread=98196)`all()`[`any()`](https://realpython.com/any-python/)函数，力图从 Python 中去掉 [`functools.reduce()`](https://realpython.com/python-reduce-function/) 等功能工具，如 [`filter()`](https://realpython.com/python-filter-function/) 和 [`map()`](https://realpython.com/python-map-function/) 。然而，Python 社区对移除这些工具并不满意。即便如此，`all()`和`any()`还是作为内置函数被添加到 [Python 2.5](https://docs.python.org/3/whatsnew/2.5.html#other-language-changes) 中，由 [Raymond Hettinger](https://twitter.com/raymondh) 实现。

可以说 Python 的`all()`执行了一个归约或[折叠](https://en.wikipedia.org/wiki/Fold_(higher-order_function))操作，因为它将一个 iterable items 项归约为一个单独的对象。然而，它不是一个[高阶函数](https://en.wikipedia.org/wiki/Higher-order_function)，因为它不将其他函数作为参数来执行其计算。因此，您可以将`all()`视为常规的**谓词**或[布尔值函数](https://en.wikipedia.org/wiki/Boolean-valued_function)。

您可以使用`all()`来检查输入 iterable 中的所有项是否都为真。因为是内置函数，所以不需要导入。它是这样工作的:

>>>

```py
>>> bool_exps = [
...     5 > 2,
...     1 == 1,
...     42 < 50,
... ]
>>> all(bool_exps)
True

>>> objects = ["Hello!", 42, {}]
>>> all(objects)
False

>>> general_exps = [
...     5 ** 2,
...     42 - 3,
...     int("42")
... ]
>>> all(general_exps)
True

>>> empty = []
>>> all(empty)
True
```

这些例子显示了`all()`与您的自定义函数`all_true()`的工作原理相同。在内部，`all()`循环遍历输入 iterable 中的条目，检查它们的真值。如果它发现一个错误的条目，那么它返回`False`。否则，它返回`True`。

如果你用一个空的 iterable 调用`all()`，就像你在上面最后一个例子中所做的那样，那么你会得到`True`,因为在一个空的 iterable 中没有 falsy 项。注意，`all()`评估的是输入 iterable 中的项，而不是 iterable 本身。参见[如果 Iterable 为空，为什么`all()`返回`True`？](https://blog.carlmjohnson.net/post/2020/python-square-of-opposition/)关于这一点的更多哲学讨论。

为了总结`all()`的行为，下面是它的真值表:

| 情况 | 结果 |
| --- | --- |
| 所有项目评估为真。 | `True` |
| 所有项目评估为假。 | `False` |
| 一个或多个项目评估为假。 | `False` |
| 输入 iterable 为空。 | `True` |

您可以运行以下对`all()`的调用来确认该表中的信息:

>>>

```py
>>> all([True, True, True])
True

>>> all([False, False, False])
False

>>> all([False, True, True])
False

>>> all([])
True
```

这些例子表明，当输入 iterable 中的所有项都为真或者 iterable 为空时，`all()`返回`True`。否则，函数返回`False`。

就像你的`all_true()`函数一样，`all()`也实现了所谓的[短路评估](https://en.wikipedia.org/wiki/Short-circuit_evaluation)。这种评估意味着`all()`一旦确定了行动的最终结果，就会立刻返回。

当函数在 iterable 中找到一个错误的项时，就会发生短路。在这种情况下，没有必要评估其余的项目，因为函数已经知道最终结果。请注意，这种类型的实现意味着当您测试具有副作用的条件时，您可以获得不同的行为。考虑下面的例子:

>>>

```py
>>> def is_true(value):
...     print("Side effect!")
...     return bool(value)
...

>>> values = [0, 1]

>>> conditions = (is_true(n) for n in values)
>>> all(conditions)
Side effect!
False

>>> conditions = (is_true(n) for n in reversed(values))
>>> all(conditions)
Side effect!
Side effect!
False
```

`is_true()`函数将一个对象作为参数，并返回其真值。在函数的执行过程中，一个副作用发生了:函数[将](https://realpython.com/python-print/)的东西打印到屏幕上。

`conditions`的第一个实例保存了一个[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)，它在[对来自输入 iterable(在本例中为`values`)的每一项进行惰性求值](https://en.wikipedia.org/wiki/Lazy_evaluation)之后产生真值。这次，`all()`只对函数求值一次，因为`is_true(0)`返回`False`。副作用只会出现一次。

现在来看看`conditions`的第二个实例。如果你[反转](https://realpython.com/python-reverse-list/)输入的 iterable，那么`all()`会评估两个条目，因为用`1`作为参数对`is_true()`的调用会返回`True`。副作用会持续两次。

这种行为可能是微妙问题的来源，因此您应该避免在代码中评估具有副作用的条件。

最后，当谈到使用`all()`函数时，可以说它至少有两个通用用例。您可以使用`all()`来检查 iterable 中的所有项目:

1.  评估为真
2.  具有给定的属性或者满足一定的条件

在下一节中，您将学习如何在 Python 中对不同的可迭代类型使用`all()`。之后，您将学习如何使用`all()`和[列表理解](https://realpython.com/list-comprehension-python/)以及生成器表达式来解决上面列出的第二个用例。

[*Remove ads*](/account/join/)

## 将`all()`用于不同的可迭代类型

内置的`all()`函数包含了 Python 的[鸭子类型](https://en.wikipedia.org/wiki/Duck_typing)风格，并且接受不同的参数类型，只要它们是可迭代的。您可以将`all()`用于列表、元组、字符串、字典、[集](https://realpython.com/python-sets/)等。

在所有情况下，`all()`都按预期工作，如果所有项目都正确，则返回`True`,否则返回`False`。在本节中，您将使用不同的可迭代类型编写使用`all()`的示例。

### 序列

至此，您已经了解了`all()`如何使用 Python 列表。在本节中，您将了解到列表和其他序列数据类型之间没有真正的区别，例如元组和 [`range`](https://realpython.com/python-range/) 对象。函数所需要的就是输入对象是可迭代的。

下面是一些将`all()`用于元组和`range`对象的例子:

>>>

```py
>>> # With tuples
>>> all((1, 2, 3))
True
>>> all((0, 1, 2, 3))
False
>>> all(())
True
>>> all(tuple())
True

>>> # With range objects
>>> all(range(10))
False
>>> all(range(1, 11))
True
>>> all(range(0))
True
```

通常，如果输入 iterable 中的所有项都是真的，那么您将得到`True`。否则，你得到`False`。空元组和范围产生一个`True`结果。在最后一个例子中，用`0`作为参数调用`range()`会返回一个空的`range`对象，因此`all()`会给出结果`True`。

还可以将包含表达式、布尔表达式或任何类型的 Python 对象的元组传递给`all()`。来吧，试一试！

### 字典

字典是键值对的集合。如果你[直接遍历字典](https://realpython.com/iterate-through-dictionary-python/)，那么你会自动遍历它的键。此外，您可以使用方便的方法显式迭代字典的键、值和项。

**注:**用字典的`.items()`方法使用`all()`没有多大意义。该方法以两项元组的形式返回键-值对，在 Python 中这些元组的值总是为 true。

如果您将字典直接传递给`all()`，那么该函数将自动检查字典的键:

>>>

```py
>>> all({"gold": 1, "silver": 2, "bronze": 3})
True

>>> all({0: "zero", 1: "one", 2: "two"})
False
```

因为第一个字典中的所有键都是真的，所以结果是得到`True`。在第二个字典中，第一个键是`0`，其值为 false。在这种情况下，您从`all()`处取回`False`。

如果您想获得与上面示例相同的结果，但是代码更可读、更显式，那么您可以使用 [`.keys()`](https://docs.python.org/3/library/stdtypes.html#dict.keys) 方法，该方法从底层字典返回所有键:

>>>

```py
>>> medals = {"gold": 1, "silver": 2, "bronze": 3}
>>> all(medals.keys())
True

>>> numbers = {0: "zero", 1: "one", 2: "two"}
>>> all(numbers.keys())
False
```

使用`.keys()`，您可以明确您的代码调用`all()`来确定输入字典中的所有当前键是否都是真的。

另一个常见的需求是，您需要检查给定字典中的所有值是否都评估为 true。在这种情况下，可以使用 [`.values()`](https://docs.python.org/3/library/stdtypes.html#dict.values) :

>>>

```py
>>> monday_inventory = {"book": 2, "pencil": 5, "eraser": 1}
>>> all(monday_inventory.values())
True

>>> tuesday_inventory = {"book": 2, "pencil": 3, "eraser": 0}
>>> all(tuesday_inventory.values())
False
```

在这些例子中，你首先检查你当前的学习用品库存中是否至少有一件物品。星期一，你所有的项目至少有一个单位，所以`all()`返回`True`。然而，在星期二，对`all()`的调用返回`False`，因为您已经用完了至少一种供应品中的单位，在本例中是`eraser`。

[*Remove ads*](/account/join/)

## 将`all()`用于理解和生成器表达式

正如您之前了解到的，Python 的`all()`的第二个用例是检查 iterable 中的所有项是否都有给定的属性或满足特定的条件。为了进行这种检查，您可以使用带有列表理解或生成器表达式的`all()`作为参数，这取决于您的需要。

通过将`all()`与列表理解和生成器表达式相结合，您获得的协同效应释放了这个函数的全部能力，并使它在您的日常编码中非常有价值。

利用`all()`这种超级能力的一种方法是使用谓词函数来测试所需的属性。这个谓词函数将是 list comprehension 中的表达式，您将把它作为参数传递给`all()`。下面是所需的语法:

```py
all([predicate(item) for item in iterable])
```

这个列表理解使用`predicate()`来测试给定属性的`iterable`中的每个`item`。然后对`all()`的调用将结果列表缩减为一个单独的`True`或`False`值，这将告诉您是否所有的条目都具有`predicate()`定义和测试的属性。

例如，下面的代码检查序列中的所有值是否都是质数:

>>>

```py
>>> import math

>>> def is_prime(n):
...     if n <= 1:
...         return False
...     for i in range(2, math.isqrt(n) + 1):
...         if n % i == 0:
...             return False
...     return True
...

>>> numbers = [2, 3, 5, 7, 11]
>>> all([is_prime(x) for x in numbers])
True

>>> numbers = [2, 4, 6, 8, 10]
>>> all([is_prime(x) for x in numbers])
False
```

在这个例子中，您将`all()`与列表理解结合起来。理解使用`is_prime()`谓词函数来测试`numbers`中的每个值的素性。结果列表将包含每次检查结果的布尔值(`True`或`False`)。然后`all()`获取这个列表作为参数，并处理它以确定所有的数字是否都是质数。

**注意:**`is_prime()`谓词基于维基百科关于[素性测试](https://en.wikipedia.org/wiki/Primality_test#Simple_methods)的文章中的算法。

这个神奇组合的第二个用例，`all()`加上一个列表理解，是检查 iterable 中的所有条目是否满足给定的条件。下面是所需的语法:

```py
all([condition for item in iterable])
```

这个对`all()`的调用使用一个列表理解来检查`iterable`中的所有项目是否满足所需的`condition`，这通常是根据单个`item`来定义的。按照这个想法，下面有几个例子来检查列表中的所有数字是否都大于`0`:

>>>

```py
>>> numbers = [1, 2, 3]
>>> all([number > 0 for number in numbers])
True

>>> numbers = [-2, -1, 0, 1, 2]
>>> all([number > 0 for number in numbers])
False
```

在第一个例子中，`all()`返回`True`，因为输入列表中的所有数字都满足大于`0`的条件。在第二个例子中，结果是`False`，因为输入 iterable 包含`0`和负数。

正如您已经知道的，`all()`返回带有空 iterable 作为参数的`True`。这种行为可能看起来很奇怪，并可能导致错误的结论:

>>>

```py
>>> numbers = []

>>> all([number < 0 for number in numbers])
True

>>> all([number == 0 for number in numbers])
True

>>> all([number > 0 for number in numbers])
True
```

这段代码显示`numbers`中的所有值都小于`0`，但是它们也等于并且大于`0`，这是不可能的。这种不合逻辑的结果的根本原因是所有这些对`all()`的调用都计算空的 iterables，这使得`all()`返回`True`。

要解决这个问题，您可以使用内置的 [`len()`函数](https://realpython.com/len-python-function/)来获取输入 iterable 中的项数。如果`len()`返回`0`，那么你可以跳过调用`all()`来处理空的输入 iterable。这个策略将使你的代码不容易出错。

您在本节中编写的所有示例都使用列表理解作为`all()`的参数。列表理解在内存中创建一个完整的列表，这可能是一个浪费的操作。如果您的代码中不再需要结果列表，这种行为尤其成立，这是典型的`all()`情况。

在这种情况下，使用带有**生成器表达式**的`all()`总是更有效，尤其是当你处理一个长输入列表时。生成器表达式不是在内存中构建一个全新的列表，而是根据需要生成条目，从而使您的代码更加高效。

构建生成器表达式的语法几乎与理解列表所用的语法相同:

```py
# With a predicate
all(predicate(item) for item in iterable)

# With a condition
all(condition for item in iterable)
```

唯一的区别是生成器表达式使用括号(`()`)而不是方括号(`[]`)。因为函数调用已经需要圆括号，所以只需要去掉方括号。

与列表理解不同，生成器表达式按需生成条目，这使得它们在内存使用方面非常有效。此外，你不会创建一个新的列表，然后在`all()`返回后扔掉它。

[*Remove ads*](/account/join/)

## 将`all()`与`and`布尔运算符进行比较

你可以大致把`all()`想象成通过布尔 [`and`](https://realpython.com/python-and-operator/) 运算符连接起来的一系列项目。例如，函数调用`all([item1, item2, ..., itemN])`在语义上等同于表达式`item1 and item2 ... and itemN`。然而，它们之间有一些微小的差异。

在本节中，您将了解这些差异。第一个与语法有关，第二个与返回值有关。此外，您将了解到`all()`和`and`操作符都实现短路评估。

### 理解语法差异

对`all()`的调用使用与 Python 中任何函数调用相同的语法。你需要用一对括号来调用这个函数。在`all()`的特定情况下，您必须传入一个值的 iterable 作为参数:

>>>

```py
>>> all([True, False])
False
```

输入 iterable 中的项可以是通用表达式、布尔表达式或任何类型的 Python 对象。此外，input iterable 中的项数只取决于系统中可用的内存量。

另一方面，`and`运算符是一个**二元运算符**，它连接表达式中的两个操作数:

>>>

```py
>>> True and False
False
```

逻辑运算符`and`采用左操作数和右操作数来构建复合表达式。就像使用`all()`一样，`and`表达式中的操作数可以是通用表达式、布尔表达式或 Python 对象。最后，您可以使用多个`and`操作符来连接任意数量的操作数。

### 返回布尔值 vs 操作数

`all()`和`and`操作符之间的第二个甚至更重要的区别是它们各自的返回值。当`all()`总是返回`True`或`False`时，`and`操作符总是返回它的一个操作数。如果返回的操作数显式地评估为任一值，则它仅返回`True`或`False`:

>>>

```py
>>> all(["Hello!", 42, {}])
False
>>> "Hello!" and 42 and {}
{}

>>> all([1, 2, 3])
True
>>> 1 and 2 and 3
3

>>> all([0, 1, 2, 3])
False
>>> 0 and 1 and 2 and 3
0

>>> all([5 > 2, 1 == 1])
True
>>> 5 > 2 and 1 == 1
True
```

这些例子展示了`all()`如何总是返回`True`或`False`，这与谓词函数的状态一致。另一方面，`and`返回最后计算的操作数。如果它恰好是一个表达式中的最后一个操作数，那么前面的所有操作数一定都是真的。否则，`and`将返回第一个 falsy 操作数，指示求值停止的位置。

注意，在最后一个例子中，`and`操作符返回`True`，因为隐含的操作数是[比较](https://realpython.com/python-boolean/#comparison-operators)表达式，它们总是显式返回`True`或`False`。

这是`all()`函数和`and`操作符之间的一个重要区别。因此，您应该考虑到这一点，以防止代码中出现微妙的错误。然而，在[布尔上下文](https://realpython.com/python-and-operator/#using-pythons-and-operator-in-boolean-contexts)中，比如`if`语句和 [`while`循环](https://realpython.com/python-while-loop/)，这种差异根本不相关。

### 短路评估

正如您已经了解到的，`all()`在决定最终结果时，会缩短对输入 iterable 中各项的评估。`and`操作员还执行[短路评估](https://realpython.com/python-and-operator/#short-circuiting-the-evaluation)。

此功能的优点是，一旦出现错误的项目，就跳过剩余的检查，从而提高操作效率。

要尝试短路评估，您可以使用[发生器函数](https://realpython.com/introduction-to-python-generators/#using-generators)，如下例所示:

>>>

```py
>>> def generate_items(iterable):
...     for i, item in enumerate(iterable):
...         print(f"Checking item: {i}")
...         yield item
...
```

`generate_items()`中的循环遍历`iterable`中的条目，使用内置的 [`enumerate()`](https://realpython.com/python-enumerate/) 函数获取每个选中条目的索引。然后，该循环打印一条标识选中物品的消息，并生成手边的物品。

有了`generate_items()`,您可以运行以下代码来测试`all()`的短路评估:

>>>

```py
>>> # Check both items to get the result
>>> items = generate_items([True, True])
>>> all(items)
Checking item: 0
Checking item: 1
True

>>> # Check the first item to get the result
>>> items = generate_items([False, True])
>>> all(items)
Checking item: 0
False

>>> # Still have a remaining item
>>> next(items)
Checking item: 1
True
```

对`all()`的第一次调用展示了该函数如何检查这两项以确定最终结果。第二次调用确认`all()`只检查第一项。由于这一项为 false sy，所以该函数不检查第二项就立即返回。这就是为什么当你调用 [`next()`](https://docs.python.org/3/library/functions.html#next) 的时候，生成器还是会产生第二个项目。

现在您可以使用`and`操作符运行一个类似的测试:

>>>

```py
>>> # Check both items to get the result
>>> items = generate_items([True, True])
>>> next(items) and next(items)
Checking item: 0
Checking item: 1
True

>>> # Check the first item to get the result
>>> items = generate_items([False, True])
>>> next(items) and next(items)
Checking item: 0
False

>>> # Still have a remaining item
>>> next(items)
Checking item: 1
True
```

第一个`and`表达式评估两个操作数以获得最终结果。第二个`and`表达式只计算第一个操作数来决定结果。用`items`作为参数调用`next()`,显示生成器函数仍然产生一个剩余项。

[*Remove ads*](/account/join/)

## 将`all()`付诸行动:实例

到目前为止，您已经学习了 Python 的`all()`的基础知识。你已经学会了在序列、字典、列表理解和生成器表达式中使用它。此外，您已经了解了这个内置函数和逻辑操作符`and`之间的区别和相似之处。

在这一节中，您将编写一系列实际例子，帮助您评估在使用 Python 编程时`all()`有多有用。所以，请继续关注并享受您的编码吧！

### 提高长复合条件的可读性

`all()`的一个有趣的特性是，当您处理基于`and`操作符的长复合布尔表达式时，这个函数如何提高代码的可读性。

例如，假设您需要在一段代码中验证用户的输入。为了使输入有效，它应该是一个介于`0`和`100`之间的整数，也是一个偶数。要检查所有这些条件，可以使用下面的`if`语句:

>>>

```py
>>> x = 42

>>> if isinstance(x, int) and 0 <= x <= 100 and x % 2 == 0:
...     print("Valid input")
... else:
...     print("Invalid input")
...
Valid input
```

`if`条件包括对 [`isinstance()`](https://docs.python.org/3/library/functions.html#isinstance) 的调用，用于检查输入是否为整数；一个[链式比较](https://realpython.com/python-operators-expressions/#chained-comparisons)表达式，用于检查数字是否在`0`和`100`之间；以及一个表达式，用于检查输入值是否为偶数。

尽管这段代码可以工作，但是条件相当长，这使得解析和理解起来很困难。此外，如果您需要在未来的更新中添加更多的验证检查，那么条件将变得更长、更复杂。它还需要一些代码格式化。

为了提高这个条件的可读性，你可以使用`all()`，就像下面的代码:

>>>

```py
>>> x = 42

>>> validation_conditions = (
...     isinstance(x, int),
...     0 <= x <= 100,
...     x % 2 == 0,
... )

>>> if all(validation_conditions):
...     print("Valid input")
... else:
...     print("Invalid input")
...
Valid input
```

在这个例子中，所有的验证条件都存在于一个具有描述性名称的元组中。使用这种技术还有一个额外的好处:如果您需要添加一个新的验证条件，那么您只需要向您的`validation_conditions`元组添加一个新行。请注意，现在您的`if`语句拥有了一个基于`all()`的非常易读、明确和简洁的表达式。

在现实生活中，验证策略通常允许您重用验证代码。例如，您可以编写可重用的验证函数，而不是指定只计算一次的普通条件:

>>>

```py
>>> def is_integer(x):
...     return isinstance(x, int)
...

>>> def is_between(a=0, b=100):
...     return lambda x: a <= x <= b
...

>>> def is_even(x):
...     return x % 2 == 0
...

>>> validation_conditions = (
...     is_integer,
...     is_between(0, 100),
...     is_even,
... )

>>> for x in (4.2, -42, 142, 43, 42):
...     print(f"Is {x} valid?", end=" ")
...     print(all(condition(x) for condition in validation_conditions))
...
Is 4.2 valid? False
Is -42 valid? False
Is 142 valid? False
Is 43 valid? False
Is 42 valid? True
```

在这个例子中，有三个函数以可重用的方式检查三个初始条件。然后，使用刚刚编写的函数重新定义验证条件元组。最后的`for`循环展示了如何使用`all()`重用这些函数来验证几个输入对象。

### 验证数值的可重复项

`all()`的另一个有趣的用例是检查一个 iterable 中的所有数值是否都在给定的区间内。下面是几个示例，说明如何在不同的条件下，借助生成器表达式来实现这一点:

>>>

```py
>>> numbers = [10, 5, 6, 4, 7, 8, 20]

>>> # From 0 to 20 (Both included)
>>> all(0 <= x <= 20 for x in numbers)
True

>>> # From 0 to 20 (Both excluded)
>>> all(0 < x < 20 for x in numbers)
False

>>> # From 0 to 20 (integers only)
>>> all(x in range(21) for x in numbers)
True

>>> # All greater than 0
>>> all(x > 0 for x in numbers)
True
```

这些例子展示了如何构建生成器表达式来检查一个可迭代数字中的所有值是否都在给定的区间内。

上面例子中的技术允许很大的灵活性。您可以调整条件并使用`all()`在目标 iterable 上运行各种检查。

[*Remove ads*](/account/join/)

### 验证字符串和字符串的可重复项

内置的 [`str`](https://docs.python.org/3/library/stdtypes.html#textseq) 类型实现了几个谓词[字符串方法](https://docs.python.org/3/library/stdtypes.html#string-methods)，当您需要验证字符串的可重复项和给定字符串中的单个字符时，这些方法会很有用。

例如，使用这些方法，您可以检查一个字符串是否是有效的十进制数，是否是字母数字字符，或者是否是有效的 ASCII 字符。

下面是一些在代码中使用字符串方法的示例:

>>>

```py
>>> numbers = ["1", "2", "3.0"]

>>> all(number.isdecimal() for number in numbers)
True

>>> chars = "abcxyz123"

>>> all(char.isalnum() for char in chars)
True

>>> all(char.isalpha() for char in chars)
False

>>> all(char.isascii() for char in chars)
True

>>> all(char.islower() for char in chars)
False

>>> all(char.isnumeric() for char in chars)
False

>>> all(char.isprintable() for char in chars)
True
```

这些`.is*()`方法中的每一个都检查底层字符串的特定属性。您可以利用这些和其他几个 string 方法来验证可重复字符串中的项以及给定字符串中的单个字符。

### 从表格数据中删除带有空字段的行

当您处理表格数据时，可能会遇到空字段的问题。您可能需要清理包含空字段的行。如果是这种情况，那么您可以使用`all()`和 [`filter()`](https://realpython.com/python-filter-function/) 来提取所有字段中都有数据的行。

内置的`filter()`函数以一个函数对象和一个 iterable 作为参数。通常，您将使用谓词函数作为`filter()`的第一个参数。对`filter()`的调用将谓词应用于 iterable 中的每一项，并返回一个迭代器，其中包含使谓词返回`True`的项。

您可以在`filter()`调用中使用`all()`作为谓词。这样，您可以处理列表的列表，这在您处理表格数据时会很有用。

举一个具体的例子，假设您有一个 [CSV 文件](https://realpython.com/python-csv/)，其中包含关于您公司员工的数据:

```py
name,job,email
"Linda","Technical Lead","" "Joe","Senior Web Developer","joe@example.com"
"Lara","Project Manager","lara@example.com"
"David","","david@example.com" "Jane","Senior Python Developer","jane@example.com"
```

快速浏览一下这个文件，您会注意到有些行包含空字段。例如，第一行没有电子邮件，第四行没有提供职位或角色。您需要通过删除包含空字段的行来清理数据。

下面是如何通过在一个`filter()`调用中使用`all()`作为谓词来满足这个需求:

>>>

```py
>>> import csv
>>> from pprint import pprint

>>> with open("employees.csv", "r") as csv_file:
...     raw_data = list(csv.reader(csv_file))
...

>>> # Before cleaning
>>> pprint(raw_data)
[['name', 'job', 'email'],
 ['Linda', 'Technical Lead', ''], ['Joe', 'Senior Web Developer', 'joe@example.com'],
 ['Lara', 'Project Manager', 'lara@example.com'],
 ['David', '', 'david@example.com'], ['Jane', 'Senior Python Developer', 'jane@example.com']]

>>> clean_data = list(filter(all, raw_data)) 
>>> # After cleaning
>>> pprint(clean_data)
[['name', 'job', 'email'],
 ['Joe', 'Senior Web Developer', 'joe@example.com'],
 ['Lara', 'Project Manager', 'lara@example.com'],
 ['Jane', 'Senior Python Developer', 'jane@example.com']]
```

在这个例子中，首先使用 Python [标准库](https://docs.python.org/3/library/index.html)中的 [`csv`](https://docs.python.org/3/library/csv.html) 模块将目标 CSV 文件的内容加载到`raw_data`中。对 [`pprint()`](https://realpython.com/python-pretty-print/) 函数的调用显示，数据包含带有空字段的行。然后你用`filter()`和`all()`清理数据。

**注意:**如果你觉得用`filter()`不舒服，那么你可以用列表理解来代替。

继续运行下面的代码行:

>>>

```py
>>> clean_data = [row for row in raw_data if all(row)]
```

一旦有了干净数据的列表，就可以再次运行`for`循环来检查是否一切正常。

`filter()`和`all()`函数如何协同执行任务？嗯，如果`all()`在一行中找到一个空字段，那么它返回`False`。因此，`filter()`不会将该行包含在最终数据中。为了确保这种技术有效，您可以使用干净的数据作为参数来调用`pprint()`。

### 比较自定义数据结构

作为如何使用`all()`的另一个例子，假设您需要创建一个定制的类似列表的类，它允许您检查它的所有值是否都大于一个特定值。

要创建这个自定义类，您可以从 [`collections`](https://realpython.com/python-collections-module/) 模块中子类化 [`UserList`](https://docs.python.org/3/library/collections.html#collections.UserList) ，然后覆盖被称为 [`.__gt__()`](https://docs.python.org/3/reference/datamodel.html#object.__gt__) 的[特殊方法](https://docs.python.org/3/glossary.html#term-special-method)。覆盖这个方法允许您重载大于(`>`)操作符，为它提供一个自定义行为:

>>>

```py
>>> from collections import UserList

>>> class ComparableList(UserList):
...     def __gt__(self, threshold):
...         return all(x > threshold for x in self)
...

>>> numbers = ComparableList([1, 2, 3])

>>> numbers > 0
True

>>> numbers > 5
False
```

在`.__gt__()`中，您使用`all()`来检查当前列表中的所有数字是否都大于应该来自用户的特定`threshold`值。

这段代码末尾的比较表达式展示了如何使用您的自定义列表，以及它如何与大于号(`>`)操作符一起工作。在第一个表达式中，列表中的所有值都大于`0`，所以结果是`True`。在第二个表达式中，所有的数字都小于`5`，这导致了一个`False`结果。

[*Remove ads*](/account/join/)

### 部分模拟 Python 的`zip()`函数

Python 内置的 [`zip()`](https://realpython.com/python-zip-function/) 函数对于并行循环多个可迭代对象很有用。该函数将给定数量的可重复项( *N* )作为参数，并将每个可重复项中的元素聚合到 *N 项*元组中。在这个例子中，您将学习如何使用`all()`来部分模拟这个功能。

为了更好地理解这一挑战，请查看`zip()`的基本功能:

>>>

```py
>>> numbers = zip(["one", "two"], [1, 2])

>>> list(numbers)
[('one', 1), ('two', 2)]
```

在这个例子中，您将两个列表作为参数传递给`zip()`。该函数返回一个迭代器，每个迭代器产生两个条目的元组，您可以通过调用`list()`将结果迭代器作为参数来确认。

这里有一个模拟这种功能的函数:

>>>

```py
>>> def emulated_zip(*iterables):
...     lists = [list(iterable) for iterable in iterables]
...     while all(lists):
...         yield tuple(current_list.pop(0) for current_list in lists)
...

>>> numbers = emulated_zip(["one", "two"], [1, 2])

>>> list(numbers)
[('one', 1), ('two', 2)]
```

您的`emulated_zip()`函数可以接受由可迭代对象组成的[可变数量的参数](https://realpython.com/python-kwargs-and-args/#using-the-python-args-variable-in-function-definitions)。函数中的第一行使用 list comprehension 将每个输入 iterable 转换成 Python 列表，以便您稍后可以使用它的`.pop()`方法。循环条件依赖于`all()`来检查所有的输入列表是否至少包含一个条目。

在每次迭代中， [`yield`](https://realpython.com/introduction-to-python-generators/#understanding-the-python-yield-statement) 语句从每个输入列表中返回一个包含一个条目的元组。以`0`为参数调用`.pop()`从每个列表中检索并移除第一个项目。

一旦循环迭代的次数足够多，以至于`.pop()`从列表中删除了所有的条目，那么条件就变为假，函数就终止了。当最短的 iterable 用尽时，循环结束，截断较长的 iterable。该行为与`zip()`的默认行为一致。

请注意，您的函数只是部分模拟了内置的`zip()`函数，因为您的函数没有采用`strict`参数。这个参数是在 [Python 3.10](https://realpython.com/python310-new-features/) 中添加的，作为处理[不相等长度](https://realpython.com/python-zip-function/#passing-arguments-of-unequal-length)的可重复项的一种安全方式。

## 结论

现在您知道了如何使用 Python 的内置函数`all()`来检查现有 iterable 中的所有项是否都是真的。您还知道如何使用这个函数来确定 iterable 中的项是否满足给定的条件或者是否具有特定的属性。

有了这些知识，您现在就能够编写可读性更强、效率更高的 Python 代码了。

**在本教程中，您学习了:**

*   如何使用 Python 的 **`all()`** 检查一个 iterable 中的所有项是否为真
*   `all()`如何处理不同的**可迭代类型**
*   如何将`all()`和**结合起来理解**和**生成器表达式**
*   是什么使得`all()`与 **`and`运算符**有所不同和相似

此外，您编写了几个实际例子，帮助您理解`all()`有多强大，以及它在 Python 编程中最常见的一些用例是什么。

**免费 PDF 下载:** [Python 3 备忘单](https://realpython.com/bonus/python-cheat-sheet-short/)*******