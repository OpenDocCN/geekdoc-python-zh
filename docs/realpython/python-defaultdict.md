# 使用 Python defaultdict 类型处理丢失的键

> 原文：<https://realpython.com/python-defaultdict/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 Python defaultdict Type**](/courses/python-defaultdict-type/) 处理丢失的键

使用 Python[dictionary](https://realpython.com/python-dicts/)时可能会遇到的一个常见问题是试图访问或修改字典中不存在的键。这将引发一个 [`KeyError`](https://realpython.com/python-keyerror/) 并中断您的代码执行。为了处理这些情况，[标准库](https://docs.python.org/3/library/index.html)提供了 Python **`defaultdict`** 类型，这是一个类似字典的类，可以在 [`collections`](https://docs.python.org/3/library/collections.html#module-collections) 中找到。

Python [`defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict) 类型的行为几乎与常规 Python 字典完全一样，但是如果您试图访问或修改一个丢失的键，那么`defaultdict`将自动创建该键并为其生成一个默认值。这使得`defaultdict`成为处理字典中丢失键的一个有价值的选择。

在本教程中，您将学习:

*   如何使用 Python `defaultdict`类型为**处理字典中丢失的键**
*   何时以及为什么要使用 Python `defaultdict`而不是普通的 [`dict`](https://realpython.com/python-dicts/)
*   如何使用一个`defaultdict`进行**分组**、**计数**、**累加**操作

有了这些知识，您将能够更好地在日常编程挑战中有效地使用 Python `defaultdict`类型。

为了从本教程中获得最大收益，您应该对什么是 Python [字典](https://realpython.com/courses/dictionaries-python/)以及如何使用它们有所了解。如果你需要梳洗一下，那么看看下面的资源:

*   [Python 中的字典](https://realpython.com/python-dicts/)(教程)
*   [Python 中的字典](https://realpython.com/courses/dictionaries-python/)(课程)
*   [如何在 Python 中迭代字典](https://realpython.com/iterate-through-dictionary-python/)

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 处理字典中丢失的键

使用 Python 字典时，你可能会面临的一个常见问题是如何处理丢失的键。如果你的代码大量基于字典，或者如果你一直在动态地创建字典，那么你很快就会注意到处理频繁的[`KeyError`](https://realpython.com/python-keyerror/)异常会很烦人，会给你的代码增加额外的复杂性。使用 Python 字典，至少有四种方法可以处理丢失的键:

1.  使用`.setdefault()`
2.  使用`.get()`
3.  使用`key in dict`习语
4.  使用 [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) 和`except`块

[Python 文档](https://docs.python.org/)对`.setdefault()`和`.get()`解释如下:

> **T2`setdefault(key[, default])`**
> 
> 如果`key`在字典中，返回它的值。如果不是，插入值为`default`的`key`并返回`default`。`default`默认为`None`。
> 
> **T2`get(key[, default])`**
> 
> 如果`key`在字典中，返回`key`的值，否则返回`default`。如果没有给出`default`，则默认为`None`，这样这个方法就永远不会引发一个 [`KeyError`](https://docs.python.org/3/library/exceptions.html#KeyError) 。
> 
> ([来源](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict))

这里有一个如何使用`.setdefault()`来处理字典中丢失的键的例子:

>>>

```py
>>> a_dict = {}
>>> a_dict['missing_key']
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    a_dict['missing_key']
KeyError: 'missing_key'
>>> a_dict.setdefault('missing_key', 'default value')
'default value'
>>> a_dict['missing_key']
'default value'
>>> a_dict.setdefault('missing_key', 'another default value')
'default value'
>>> a_dict
{'missing_key': 'default value'}
```

在上面的代码中，您使用`.setdefault()`为`missing_key`生成一个默认值。请注意，您的字典`a_dict`现在有了一个名为`missing_key`的新键，其值为`'default value'`。这个键在你调用`.setdefault()`之前是不存在的。最后，如果您在一个现有的键上调用`.setdefault()`，那么这个调用不会对字典产生任何影响。您的密钥将保存原始值，而不是新的默认值。

**注意:**在上面的代码示例中，您得到了一个异常，Python 向您显示了一条**回溯**消息，告诉您正在尝试访问`a_dict`中丢失的键。如果你想更深入地了解如何破译和理解 Python 回溯，那么请查看[了解 Python 回溯](https://realpython.com/python-traceback/)和[充分利用 Python 回溯](https://realpython.com/courses/python-traceback/)。

另一方面，如果您使用`.get()`，那么您可以编写如下代码:

>>>

```py
>>> a_dict = {}
>>> a_dict.get('missing_key', 'default value')
'default value'
>>> a_dict
{}
```

这里，您使用`.get()`为`missing_key`生成一个默认值，但是这一次，您的字典保持为空。这是因为`.get()`返回默认值，但是这个值没有被添加到底层字典中。例如，如果你有一本名为`D`的字典，那么你可以假设`.get()`是这样工作的:

```py
D.get(key, default) -> D[key] if key in D, else default
```

有了这段伪代码，你就能明白`.get()`内部是怎么工作的了。如果键存在，那么`.get()`返回映射到那个键的值。否则，将返回默认值。您的代码从来不会为`key`创建或赋值。本例中，`default`默认为 [`None`](https://realpython.com/null-in-python/) 。

您还可以使用[条件语句](https://realpython.com/courses/python-conditional-statements/)来处理字典中丢失的键。看看下面的例子，它使用了`key in dict`习语:

>>>

```py
>>> a_dict = {}
>>> if 'key' in a_dict:
...     # Do something with 'key'...
...     a_dict['key']
... else:
...     a_dict['key'] = 'default value'
...
>>> a_dict
{'key': 'default value'}
```

在这段代码中，您使用一个`if`语句和 [`in`操作符](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations)来检查`a_dict`中是否存在`key`。如果是这样，那么你可以用`key`或者它的值来执行任何操作。否则，您将创建新的键`key`，并将其指定为`'default value'`。注意，上面的代码类似于`.setdefault()`，但是需要四行代码，而`.setdefault()`只需要一行代码(除了可读性更好之外)。

您还可以通过使用`try`和`except`块来处理异常，从而绕过`KeyError`。考虑下面这段代码:

>>>

```py
>>> a_dict = {}
>>> try:
...     # Do something with 'key'...
...     a_dict['key']
... except KeyError:
...     a_dict['key'] = 'default value'
...
>>> a_dict
{'key': 'default value'}
```

上例中的`try`和`except`块在您试图访问一个丢失的键时捕获`KeyError`。在`except`子句中，您创建了`key`，并给它分配了一个`'default value'`。

**注意:**如果缺少键在你的代码中不常见，那么你可能更喜欢使用`try`和`except`块( [EAFP 编码风格](https://docs.python.org/3/glossary.html#term-eafp))来捕捉`KeyError`异常。这是因为代码不会检查每个键的存在，如果有的话，只会处理少数异常。

另一方面，如果缺少键在您的代码中很常见，那么条件语句( [LBYL 编码风格](https://docs.python.org/3/glossary.html#term-lbyl))可能是更好的选择，因为检查键比处理频繁的异常成本更低。

到目前为止，您已经学会了如何使用`dict`和 Python 提供的工具来处理丢失的键。然而，您在这里看到的例子非常冗长，难以阅读。它们可能不像你想的那样简单。这就是为什么 [Python 标准库](https://docs.python.org/3/library/index.html)提供了一个更加优雅、[Python](https://realpython.com/learning-paths/writing-pythonic-code/)和高效的解决方案。这个解决方案就是 [`collections.defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict) ，这就是你从现在开始要覆盖的内容。

[*Remove ads*](/account/join/)

## 了解 Python `defaultdict`类型

Python 标准库提供了 [`collections`](https://docs.python.org/3/library/collections.html#module-collections) ，这是一个实现专门容器类型的模块。其中之一是 Python `defaultdict`类型，它是`dict`的替代，专门设计来帮助你解决丢失的密钥。`defaultdict`是继承自`dict`的 Python 类型:

>>>

```py
>>> from collections import defaultdict
>>> issubclass(defaultdict, dict)
True
```

上面的代码显示 Python `defaultdict`类型是`dict`的**子类**。这意味着`defaultdict`继承了`dict`的大部分行为。所以，你可以说`defaultdict`很像一本普通的字典。

`defaultdict`和`dict`的主要区别在于，当你试图访问或修改一个不在字典中的`key`时，默认的`value`会自动赋予那个`key`。为了提供这个功能，Python `defaultdict`类型做了两件事:

1.  它覆盖了 [`.__missing__()`](https://docs.python.org/3/library/collections.html#collections.defaultdict.__missing__) 。
2.  它增加了`.default_factory`，一个需要在实例化时提供的可写实例变量。

实例变量`.default_factory`将保存传入 [`defaultdict.__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__) 的第一个参数。该参数可以采用有效的 Python callable 或`None`。如果提供了一个 callable，那么每当您试图访问或修改与一个丢失的键相关联的值时，它将被`defaultdict`自动调用。

**注意:**类初始化器的所有剩余参数都被视为传递给了正则`dict`的初始化器，包括关键字参数。

看看如何创建并正确初始化一个`defaultdict`:

>>>

```py
>>> # Correct instantiation
>>> def_dict = defaultdict(list)  # Pass list to .default_factory
>>> def_dict['one'] = 1  # Add a key-value pair
>>> def_dict['missing']  # Access a missing key returns an empty list
[]
>>> def_dict['another_missing'].append(4)  # Modify a missing key
>>> def_dict
defaultdict(<class 'list'>, {'one': 1, 'missing': [], 'another_missing': [4]})
```

在这里，当您创建字典时，您将 [`list`](https://realpython.com/python-lists-tuples/#python-lists) 传递给`.default_factory`。然后，你可以像使用普通字典一样使用`def_dict`。注意，当您试图访问或修改映射到一个不存在的键的值时，字典会给它分配调用`list()`得到的默认值。

请记住，您必须将一个有效的 Python 可调用对象传递给`.default_factory`，所以记住不要在初始化时使用括号来调用它。当您开始使用 Python `defaultdict`类型时，这可能是一个常见的问题。看一下下面的代码:

>>>

```py
>>> # Wrong instantiation
>>> def_dict = defaultdict(list())
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    def_dict = defaultdict(list())
TypeError: first argument must be callable or None
```

在这里，您试图通过将`list()`传递给`.default_factory`来创建一个`defaultdict`。对`list()`的调用引发了一个`TypeError`，它告诉你第一个参数必须是可调用的或者是`None`。

通过对 Python `defaultdict`类型的介绍，您可以通过实际例子开始编码。接下来的几节将带您浏览一些常见的用例，在这些用例中，您可以依靠一个`defaultdict`来提供一个优雅、高效的 Pythonic 解决方案。

## 使用 Python `defaultdict`类型

有时，您将使用可变的内置集合(一个`list`、`dict`或 [`set`](https://realpython.com/python-sets/) )作为 Python 字典中的值。在这些情况下，你需要在第一次使用前初始化**键**，否则你会得到一个`KeyError`。您可以手动完成这个过程，也可以使用 Python `defaultdict`自动完成。在本节中，您将学习如何使用 Python `defaultdict`类型来解决一些常见的编程问题:

*   **分组**集合中的项目
*   清点收藏中的物品
*   **累加**集合中的值

您将会看到一些使用`list`、`set`、`int`和`float`的例子，以一种用户友好和有效的方式执行分组、计数和累加操作。

### 分组项目

Python `defaultdict`类型的典型用法是将`.default_factory`设置为`list`，然后构建一个将键映射到值列表的字典。使用这个`defaultdict`，如果您试图访问任何丢失的键，那么字典将运行以下步骤:

1.  **调用** `list()`创建一个新的空`list`
2.  **将**空的`list`插入字典，使用丢失的键作为`key`
3.  **返回**对那个`list`的引用

这允许您编写如下代码:

>>>

```py
>>> from collections import defaultdict
>>> dd = defaultdict(list)
>>> dd['key'].append(1)
>>> dd
defaultdict(<class 'list'>, {'key': [1]})
>>> dd['key'].append(2)
>>> dd
defaultdict(<class 'list'>, {'key': [1, 2]})
>>> dd['key'].append(3)
>>> dd
defaultdict(<class 'list'>, {'key': [1, 2, 3]})
```

在这里，您创建了一个名为`dd`的 Python `defaultdict`，并将`list`传递给`.default_factory`。注意，即使没有定义`key`，您也可以[将](https://realpython.com/python-append/)值附加到它上面，而不会得到`KeyError`。那是因为`dd`自动调用`.default_factory`为缺失的`key`生成默认值。

您可以将`defaultdict`与`list`一起使用，对序列或集合中的项目进行分组。假设您已经从贵公司的数据库中检索到以下数据:

| 部门 | 员工姓名 |
| --- | --- |
| 销售 | 无名氏 |
| 销售 | 马丁·史密斯 |
| 会计 | 简·多伊 |
| 营销 | 伊丽莎白·史密斯 |
| 营销 | 亚当·多伊 |
| … | … |

有了这些数据，你创建一个初始的 [`tuple`](https://realpython.com/python-lists-tuples/#python-tuples) 对象的`list`，如下:

```py
dep = [('Sales', 'John Doe'),
       ('Sales', 'Martin Smith'),
       ('Accounting', 'Jane Doe'),
       ('Marketing', 'Elizabeth Smith'),
       ('Marketing', 'Adam Doe')]
```

现在，您需要创建一个字典，按部门对雇员进行分组。为此，您可以使用如下的`defaultdict`:

```py
from collections import defaultdict

dep_dd = defaultdict(list)
for department, employee in dep:
    dep_dd[department].append(employee)
```

在这里，您创建一个名为`dep_dd`的`defaultdict`，并使用一个 [`for`循环](https://realpython.com/python-for-loop/)来遍历您的`dep`列表。语句`dep_dd[department].append(employee)`为部门创建键，将它们初始化为一个空列表，然后将雇员添加到每个部门。一旦您运行了这段代码，您的`dep_dd`将看起来像这样:

>>>

```py
defaultdict(<class 'list'>, {'Sales': ['John Doe', 'Martin Smith'],
 'Accounting' : ['Jane Doe'],
 'Marketing': ['Elizabeth Smith', 'Adam Doe']})
```

在这个例子中，您使用一个`defaultdict`将员工按部门分组，其中`.default_factory`设置为`list`。要使用常规字典来实现这一点，您可以如下使用`dict.setdefault()`:

```py
dep_d = dict()
for department, employee in dep:
    dep_d.setdefault(department, []).append(employee)
```

这段代码很简单，作为一名 Python 程序员，您会经常在工作中发现类似的代码。然而，`defaultdict`版本可以说更具可读性，对于大型数据集，它也可以快很多[并且更高效](#defaultdict-vs-dictsetdefault)。所以，如果速度是你关心的问题，那么你应该考虑使用`defaultdict`而不是标准的`dict`。

[*Remove ads*](/account/join/)

### 对唯一项目进行分组

继续使用上一节中的部门和员工数据。经过一些处理后，您意识到一些员工被错误地在数据库中复制成了**。您需要清理数据，并从您的`dep_dd`字典中删除重复的雇员。为此，您可以使用一个`set`作为`.default_factory`，并如下重写您的代码:**

```py
dep = [('Sales', 'John Doe'),
       ('Sales', 'Martin Smith'),
       ('Accounting', 'Jane Doe'),
       ('Marketing', 'Elizabeth Smith'),
       ('Marketing', 'Elizabeth Smith'),
       ('Marketing', 'Adam Doe'),
       ('Marketing', 'Adam Doe'),
       ('Marketing', 'Adam Doe')]

dep_dd = defaultdict(set)
for department, employee in dep:
    dep_dd[department].add(employee)
```

在本例中，您将`.default_factory`设置为`set`。**集合**是唯一对象的**集合，这意味着你不能创建一个有重复项目的`set`。这是 set 的一个非常有趣的特性，它保证了在最终的字典中不会有重复的条目。**

### 清点物品

如果您将`.default_factory`设置为 [`int`](https://docs.python.org/3/library/functions.html#int) ，那么您的`defaultdict`将对**计数序列或集合中的项目**有用。当您不带参数调用`int()`时，该函数返回`0`，这是您用来初始化计数器的典型值。

继续以公司数据库为例，假设您想要构建一个字典来计算每个部门的雇员人数。在这种情况下，您可以编写如下代码:

>>>

```py
>>> from collections import defaultdict
>>> dep = [('Sales', 'John Doe'),
...        ('Sales', 'Martin Smith'),
...        ('Accounting', 'Jane Doe'),
...        ('Marketing', 'Elizabeth Smith'),
...        ('Marketing', 'Adam Doe')]
>>> dd = defaultdict(int)
>>> for department, _ in dep:
...     dd[department] += 1
>>> dd
defaultdict(<class 'int'>, {'Sales': 2, 'Accounting': 1, 'Marketing': 2})
```

在这里，您将`.default_factory`设置为`int`。当你不带参数调用`int()`时，返回值是`0`。您可以使用这个默认值开始计算在每个部门工作的员工。为了让这段代码正确运行，您需要一个干净的数据集。不得有重复数据。否则，您需要过滤掉重复的员工。

另一个计算项目的例子是`mississippi`的例子，你计算一个单词中每个字母重复的次数。看一下下面的代码:

>>>

```py
>>> from collections import defaultdict
>>> s = 'mississippi'
>>> dd = defaultdict(int)
>>> for letter in s:
...     dd[letter] += 1
...
>>> dd
defaultdict(<class 'int'>, {'m': 1, 'i': 4, 's': 4, 'p': 2})
```

在上面的代码中，您创建了一个将`.default_factory`设置为`int`的`defaultdict`。这将任何给定键的默认值设置为`0`。然后，使用一个`for`循环遍历[字符串](https://realpython.com/python-strings/) `s`，并使用一个[增强赋值操作](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements)在每次迭代中将`1`添加到计数器中。`dd`的按键将是`mississippi`中的字母。

**注意:** Python 的[增强赋值操作符](https://docs.python.org/3/whatsnew/2.0.html#augmented-assignment)是常见操作的便捷快捷方式。

看看下面的例子:

*   `var += 1`相当于`var = var + 1`
*   `var -= 1`相当于`var = var - 1`
*   `var *= 1`相当于`var = var * 1`

这只是增强赋值操作符如何工作的一个例子。你可以看一下[官方文档](https://docs.python.org/3/whatsnew/2.0.html#augmented-assignment)来了解这个特性的更多信息。

由于计数在编程中是一个相对常见的任务，类似 Python 字典的类 [`collections.Counter`](https://realpython.com/python-counter/) 是专门为计数序列中的项目而设计的。使用`Counter`，您可以编写如下的`mississippi`示例:

>>>

```py
>>> from collections import Counter
>>> counter = Counter('mississippi')
>>> counter
Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})
```

在这种情况下，`Counter`会为您完成所有工作！您只需要传入一个序列，字典将对其条目进行计数，将它们存储为键，将计数存储为值。注意，这个例子是可行的，因为 Python 字符串也是一种序列类型。

### 累计值

有时你需要计算一个序列或集合中的值的总和。假设您有以下 [Excel 表格](https://realpython.com/openpyxl-excel-spreadsheets-python/)，其中包含您的 Python 网站的销售数据:

| 制品 | 七月 | 八月 | 九月 |
| --- | --- | --- | --- |
| 书 | One thousand two hundred and fifty | One thousand three hundred | One thousand four hundred and twenty |
| 教程 | Five hundred and sixty | Six hundred and thirty | Seven hundred and fifty |
| 课程 | Two thousand five hundred | Two thousand four hundred and thirty | Two thousand seven hundred and fifty |

接下来，您使用 Python 处理数据并获得以下`tuple`个对象中的`list`:

```py
incomes = [('Books', 1250.00),
           ('Books', 1300.00),
           ('Books', 1420.00),
           ('Tutorials', 560.00),
           ('Tutorials', 630.00),
           ('Tutorials', 750.00),
           ('Courses', 2500.00),
           ('Courses', 2430.00),
           ('Courses', 2750.00),]
```

有了这些数据，你想计算每件产品的总收入。要做到这一点，您可以使用一个带`float`的 Python `defaultdict`作为`.default_factory`，然后编写如下代码:

```py
 1from collections import defaultdict
 2
 3dd = defaultdict(float)
 4for product, income in incomes:
 5    dd[product] += income
 6
 7for product, income in dd.items():
 8    print(f'Total income for {product}: ${income:,.2f}')
```

下面是这段代码的作用:

*   **在第 1 行**中，您导入了 Python `defaultdict`类型。
*   **在第 3 行**中，你创建了一个`defaultdict`对象，并将`.default_factory`设置为`float`。
*   **在第 4 行**中，您定义了一个`for`循环来遍历`incomes`的条目。
*   **在第 5 行**中，您使用一个增强的赋值操作(`+=`)来累加字典中每个产品的收入。

第二个循环遍历`dd`的条目，并将收入打印到屏幕上。

**注意:**如果你想更深入地研究字典迭代，请查看[如何在 Python 中迭代字典](https://realpython.com/iterate-through-dictionary-python/)。

如果您将所有这些代码放入一个名为`incomes.py`的文件中，并从命令行运行它，那么您将得到以下输出:

```py
$ python3 incomes.py
Total income for Books: $3,970.00
Total income for Tutorials: $1,940.00
Total income for Courses: $7,680.00
```

你现在有了每件产品的收入汇总，所以你可以决定采取什么策略来增加你网站的总收入。

[*Remove ads*](/account/join/)

## 深入到`defaultdict`

到目前为止，通过编写一些实际例子，您已经学会了如何使用 Python `defaultdict`类型。此时，您可以更深入地了解**类型实现**和其他工作细节。这是您将在接下来的几节中涉及的内容。

### `defaultdict`vs`dict`T2】

为了更好地理解 Python `defaultdict`类型，一个很好的练习是将其与其超类`dict`进行比较。如果您想知道特定于 Python `defaultdict`类型的方法和属性，那么您可以运行下面一行代码:

>>>

```py
>>> set(dir(defaultdict)) - set(dir(dict))
{'__copy__', 'default_factory', '__missing__'}
```

在上面的代码中，您使用 [`dir()`](https://docs.python.org/3/library/functions.html#dir) 来获取`dict`和`defaultdict`的有效属性列表。然后，你用一个 [`set`](https://realpython.com/python-sets/) 的区别来得到你只能在`defaultdict`中找到的方法和属性的集合。正如您所看到的，这两个类之间的区别是。您有两个方法和一个实例属性。下表显示了这些方法和属性的用途:

| 方法或属性 | 描述 |
| --- | --- |
| `.__copy__()` | 为`copy.copy()`提供支持 |
| `.default_factory` | 保存由`.__missing__()`调用的 callable，以自动为丢失的键提供默认值 |
| `.__missing__(key)` | 当`.__getitem__()`找不到`key`时被调用 |

在上表中，您可以看到使`defaultdict`不同于常规`dict`的方法和属性。这两个类中的其余方法是相同的。

**注意:**如果你使用一个有效的可调用函数初始化一个`defaultdict`，那么当你试图访问一个丢失的键时，你不会得到一个`KeyError`。任何不存在的键都会得到由`.default_factory`返回的值。

此外，您可能会注意到一个`defaultdict`等于一个`dict`，具有相同的项目:

>>>

```py
>>> std_dict = dict(numbers=[1, 2, 3], letters=['a', 'b', 'c'])
>>> std_dict
{'numbers': [1, 2, 3], 'letters': ['a', 'b', 'c']}
>>> def_dict = defaultdict(list, numbers=[1, 2, 3], letters=['a', 'b', 'c'])
>>> def_dict
defaultdict(<class 'list'>, {'numbers': [1, 2, 3], 'letters': ['a', 'b', 'c']})
>>> std_dict == def_dict
True
```

在这里，您创建了一个包含一些任意条目的常规字典`std_dict`。然后，用相同的条目创建一个`defaultdict`。如果您测试两个字典的内容是否相等，那么您会发现它们是相等的。

### `defaultdict.default_factory`

Python `defaultdict`类型的第一个参数必须是一个**可调用的**，它不接受任何参数并返回值。该参数被分配给实例属性`.default_factory`。为此，您可以使用任何可调用的对象，包括函数、方法、类、类型对象或任何其他有效的可调用对象。`.default_factory`的默认值为`None`。

如果您实例化了`defaultdict`而没有传递一个值给`.default_factory`，那么字典将像常规的`dict`一样运行，并且通常的`KeyError`将会因缺少键查找或修改尝试而被引发:

>>>

```py
>>> from collections import defaultdict
>>> dd = defaultdict()
>>> dd['missing_key']
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    dd['missing_key']
KeyError: 'missing_key'
```

这里，您实例化了不带参数的 Python `defaultdict`类型。在这种情况下，实例的行为就像一个标准字典。因此，如果您试图访问或修改一个丢失的密钥，那么您将得到通常的`KeyError`。从这一点开始，您可以将`dd`作为普通的 Python 字典来使用，除非您为`.default_factory`分配一个新的 callable，否则您将无法使用`defaultdict`的能力来自动处理丢失的键。

如果您将`None`传递给`defaultdict`的第一个参数，那么实例的行为将与您在上面的例子中看到的一样。那是因为`.default_factory`默认为`None`，所以两种初始化是等价的。另一方面，如果你传递一个有效的可调用对象给`.default_factory`，那么你可以用它以一种用户友好的方式处理丢失的键。下面是一个将`list`传递给`.default_factory`的例子:

>>>

```py
>>> dd = defaultdict(list, letters=['a', 'b', 'c'])
>>> dd.default_factory
<class 'list'>
>>> dd
defaultdict(<class 'list'>, {'letters': ['a', 'b', 'c']})
>>> dd['numbers']
[]
>>> dd
defaultdict(<class 'list'>, {'letters': ['a', 'b', 'c'], 'numbers': []})
>>> dd['numbers'].append(1)
>>> dd
defaultdict(<class 'list'>, {'letters': ['a', 'b', 'c'], 'numbers': [1]})
>>> dd['numbers'] += [2, 3]
>>> dd
defaultdict(<class 'list'>, {'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]})
```

在这个例子中，您创建了一个名为`dd`的 Python `defaultdict`，然后使用`list`作为它的第一个参数。第二个参数称为`letters`，它保存一个字母列表。您会看到`.default_factory`现在持有一个`list`对象，当您需要为任何丢失的键提供一个默认的`value`时，该对象将被调用。

注意，当你试图访问`numbers`，`dd`测试`numbers`是否在字典中。如果不是，那么它调用`.default_factory()`。由于`.default_factory`持有一个`list`对象，返回的`value`是一个空列表(`[]`)。

现在`dd['numbers']`已经用空的`list`初始化，您可以使用`.append()`向`list`添加元素。您还可以使用一个增强的赋值操作符(`+=`)来连接列表`[1]`和`[2, 3]`。这样，您可以用一种更 Pythonic 化、更高效的方式来处理丢失的键。

另一方面，如果您将一个**不可调用的**对象传递给 Python `defaultdict`类型的初始化器，那么您将得到一个`TypeError`，如下面的代码所示:

>>>

```py
>>> defaultdict(0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    defaultdict(0)
TypeError: first argument must be callable or None
```

在这里，您将`0`传递给`.default_factory`。因为`0`不是一个可调用的对象，你得到一个`TypeError`告诉你第一个参数必须是可调用的或者是`None`。否则，`defaultdict`不起作用。

记住`.default_factory`只是从 [`.__getitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__getitem__) 中调用，而不是从其他方法中调用。这意味着如果`dd`是一个`defaultdict`并且`key`是一个丢失的键，那么`dd[key]`将调用`.default_factory`来提供一个默认的`value`，但是`dd.get(key)`仍然返回`None`而不是`.default_factory`将提供的值。那是因为`.get()`没有调用`.__getitem__()`来检索`key`。

看一下下面的代码:

>>>

```py
>>> dd = defaultdict(list)
>>> # Calls dd.__getitem__('missing')
>>> dd['missing']
[]
>>> # Don't call dd.__getitem__('another_missing')
>>> print(dd.get('another_missing'))
None
>>> dd
defaultdict(<class 'list'>, {'missing': []})
```

在这段代码中，你可以看到`dd.get()`返回`None`，而不是`.default_factory`提供的默认值。那是因为`.default_factory`只从`.__missing__()`调用，`.get()`不调用。

请注意，您还可以向 Python `defaultdict`添加**任意值**。这意味着您不局限于与由`.default_factory`生成的值类型相同的值。这里有一个例子:

>>>

```py
>>> dd = defaultdict(list)
>>> dd
defaultdict(<class 'list'>, {})
>>> dd['string'] = 'some string'
>>> dd
defaultdict(<class 'list'>, {'string': 'some string'})
>>> dd['list']
[]
>>> dd
defaultdict(<class 'list'>, {'string': 'some string', 'list': []})
```

在这里，您创建了一个`defaultdict`并向`.default_factory`传递了一个`list`对象。这会将您的默认值设置为空列表。但是，您可以自由添加保存不同类型值的新键。键`string`就是这种情况，它持有一个`str`对象，而不是一个`list`对象。

最后，您总是可以用与处理任何实例属性相同的方式,**更改或更新最初分配给`.default_factory`的可调用的**:

>>>

```py
>>> dd.default_factory = str
>>> dd['missing_key']
''
```

在上面的代码中，您将`.default_factory`从`list`更改为`str`。现在，每当您试图访问一个丢失的键时，您的缺省值将是一个空字符串(`''`)。

根据 Python `defaultdict`类型的用例，一旦完成创建，您可能需要冻结字典，并使其成为只读的。为此，您可以在完成字典填充后将`.default_factory`设置为`None`。这样，您的字典将表现得像一个标准的`dict`，这意味着您不会有更多自动生成的默认值。

[*Remove ads*](/account/join/)

### `defaultdict`vs`dict.setdefault()`T2】

正如您之前看到的，`dict`提供了`.setdefault()`，它将允许您动态地为丢失的键赋值。相比之下，使用`defaultdict`可以在初始化容器时预先指定默认值。您可以使用`.setdefault()`分配默认值，如下所示:

>>>

```py
>>> d = dict()
>>> d.setdefault('missing_key', [])
[]
>>> d
{'missing_key': []}
```

在这段代码中，您创建一个常规字典，然后使用`.setdefault()`为键`missing_key`赋值(`[]`)，这个键还没有定义。

**注意:**你可以使用`.setdefault()`分配任何类型的 Python 对象。如果你考虑到`defaultdict`只接受可调用或`None`，这是与`defaultdict`的一个重要区别。

另一方面，如果您使用一个`defaultdict`来完成相同的任务，那么每当您试图访问或修改一个丢失的键时，就会根据需要生成缺省值。注意，使用`defaultdict`，默认值是由您传递给类的初始化器的可调用函数生成的。它是这样工作的:

>>>

```py
>>> from collections import defaultdict
>>> dd = defaultdict(list)
>>> dd['missing_key']
[]
>>> dd
defaultdict(<class 'list'>, {'missing_key': []})
```

这里，首先从`collections`导入 Python `defaultdict`类型。然后，您创建一个`defaultdict`，并将`list`传递给`.default_factory`。当您试图访问一个丢失的键时，`defaultdict`在内部调用`.default_factory()`，它保存了对`list`的引用，并将结果值(一个空的`list`)赋给`missing_key`。

上面两个例子中的代码做了同样的工作，但是`defaultdict`版本更具可读性、用户友好性、Pythonic 化和简单明了。

**注意:**对内置类型如`list`、`set`、`dict`、`str`、`int`或`float`的调用将返回一个空对象，对于数值类型则返回零。

看一下下面的代码示例:

>>>

```py
>>> list()
[]
>>> set()
set([])
>>> dict()
{}
>>> str()
''
>>> float()
0.0
>>> int()
0
```

在这段代码中，您调用了一些没有参数的内置类型，并获得了一个空对象或零数值类型。

最后，使用`defaultdict`处理丢失的键比使用`dict.setdefault()`更快。看看下面的例子:

```py
# Filename: exec_time.py

from collections import defaultdict
from timeit import timeit

animals = [('cat', 1), ('rabbit', 2), ('cat', 3), ('dog', 4), ('dog', 1)]
std_dict = dict()
def_dict = defaultdict(list)

def group_with_dict():
    for animal, count in animals:
        std_dict.setdefault(animal, []).append(count)
    return std_dict

def group_with_defaultdict():
    for animal, count in animals:
        def_dict[animal].append(count)
    return def_dict

print(f'dict.setdefault() takes {timeit(group_with_dict)} seconds.')
print(f'defaultdict takes {timeit(group_with_defaultdict)} seconds.')
```

如果您从系统的命令行运行脚本,那么您将得到如下结果:

```py
$ python3 exec_time.py
dict.setdefault() takes 1.0281260240008123 seconds.
defaultdict takes 0.6704721650003194 seconds.
```

这里用 [`timeit.timeit()`](https://docs.python.org/3/library/timeit.html#python-interface) 来衡量`group_with_dict()`和`group_with_defaultdict()`的执行时间。这些函数执行相同的动作，但是第一个使用`dict.setdefault()`，第二个使用`defaultdict`。时间测量将取决于您当前的硬件，但是您可以在这里看到`defaultdict`比`dict.setdefault()`快。随着数据集变大，这种差异会变得更加重要。

此外，您需要考虑创建一个常规的`dict`可能比创建一个`defaultdict`更快。看一下这段代码:

>>>

```py
>>> from timeit import timeit
>>> from collections import defaultdict
>>> print(f'dict() takes {timeit(dict)} seconds.')
dict() takes 0.08921320698573254 seconds.
>>> print(f'defaultdict() takes {timeit(defaultdict)} seconds.')
defaultdict() takes 0.14101867799763568 seconds.
```

这一次，您使用`timeit.timeit()`来测量`dict`和`defaultdict`实例化的执行时间。注意，创建一个`dict`花费的时间几乎是创建一个`defaultdict`的一半。如果您考虑到在真实世界的代码中，您通常只实例化`defaultdict`一次，这可能不是问题。

还要注意，默认情况下，`timeit.timeit()`将运行您的代码一百万次。这就是将`std_dict`和`def_dict`定义在`group_with_dict()`和`exec_time.py`范围之外的原因。否则，时间度量将受到`dict`和`defaultdict`的实例化时间的影响。

在这一点上，你可能已经知道什么时候使用一个`defaultdict`而不是一个常规的`dict`。这里有三点需要考虑:

1.  **如果你的代码在很大程度上基于字典**，并且你一直在处理丢失的键，那么你应该考虑使用`defaultdict`而不是常规的`dict`。

2.  **如果你的字典条目需要用一个常量默认值初始化**，那么你应该考虑用`defaultdict`代替`dict`。

3.  **如果你的代码依赖字典**来聚合、累加、计数或分组值，并且性能是个问题，那么你应该考虑使用`defaultdict`。

在决定是使用`dict`还是`defaultdict`时，可以考虑上面的指导方针。

[*Remove ads*](/account/join/)

### `defaultdict.__missing__()`

在幕后，Python `defaultdict`类型通过调用`.default_factory`为丢失的键提供默认值。使这成为可能的机制是`.__missing__()`，一种所有标准映射类型都支持的特殊方法，包括`dict`和`defaultdict`。

**注意:**注意，`.__missing__()`被`.__getitem__()`自动调用来处理丢失的密钥，`.__getitem__()`被 Python 同时自动调用来进行[订阅操作](https://docs.python.org/3/reference/expressions.html#subscriptions)像`d[key]`。

那么，`.__missing__()`是如何工作的呢？如果您将`.default_factory`设置为`None`，那么`.__missing__()`将引发一个`KeyError`，并将`key`作为参数。否则，不带参数调用`.default_factory`，为给定的`key`提供默认的`value`。这个`value`插入字典，最后返回。如果调用`.default_factory`引发了一个异常，那么这个异常会被原封不动地传播。

下面的代码展示了一个可行的 Python 实现用于`.__missing__()`:

```py
 1def __missing__(self, key):
 2    if self.default_factory is None:
 3        raise KeyError(key)
 4    if key not in self:
 5        self[key] = self.default_factory()
 6    return self[key]
```

下面是这段代码的作用:

*   **在第 1 行**中，您定义了方法及其签名。
*   **在第 2 行和第 3 行**，你测试一下`.default_factory`是不是`None`。如果是这样，那么你用`key`作为参数来引发一个`KeyError`。
*   **在第 4 行和第 5 行**，你检查`key`是否不在字典中。如果不是，那么您调用`.default_factory`并将它的返回值赋给`key`。
*   **在第 6 行**中，你如期返回了`key`。

请记住,`.__missing__()`在映射中的出现对其他查找键的方法的行为没有影响，比如实现了`in`操作符的`.get()`或`.__contains__()`。那是因为只有当被请求的`key`在字典中找不到的时候`.__missing__()`才会被`.__getitem__()`调用。无论`.__missing__()`返回或引发什么，都会被`.__getitem__()`返回或引发。

既然您已经介绍了`.__missing__()`的另一种 Python 实现，那么尝试用一些 Python 代码来模拟`defaultdict`将是一个很好的练习。这就是您在下一部分要做的事情。

## 模仿 Python `defaultdict`类型

在本节中，您将编写一个行为类似于`defaultdict`的 Python 类。为此，您将子类化 [`collections.UserDict`](https://docs.python.org/3/library/collections.html#collections.UserDict) ，然后添加`.__missing__()`。此外，您需要添加一个名为`.default_factory`的实例属性，它将保存用于按需生成默认值的可调用。这里有一段代码模拟了 Python `defaultdict`类型的大部分行为:

```py
 1import collections
 2
 3class my_defaultdict(collections.UserDict):
 4    def __init__(self, default_factory=None, *args, **kwargs):
 5        super().__init__(*args, **kwargs)
 6        if not callable(default_factory) and default_factory is not None:
 7            raise TypeError('first argument must be callable or None')
 8        self.default_factory = default_factory
 9
10    def __missing__(self, key):
11        if self.default_factory is None:
12            raise KeyError(key)
13        if key not in self:
14            self[key] = self.default_factory()
15        return self[key]
```

下面是这段代码的工作原理:

*   **在第 1 行**，你导入`collections`来访问`UserDict`。

*   在第 3 行中，你创建了一个子类`UserDict`。

*   在第 4 行中，你定义了类初始化器`.__init__()`。这个方法使用一个名为`default_factory`的参数来保存您将用来生成默认值的可调用对象。注意`default_factory`默认为`None`，就像在`defaultdict`中一样。您还需要 [`*args`和`**kwargs`](https://realpython.com/python-kwargs-and-args/) 来模拟常规`dict`的正常行为。

*   **在第 5 行**，你调用超类`.__init__()`。这意味着你正在调用`UserDict.__init__()`并将`*args`和`**kwargs`传递给它。

*   **在第 6 行**中，首先检查`default_factory`是否是有效的可调用对象。在这种情况下，您使用 [`callable(object)`](https://docs.python.org/3/library/functions.html#callable) ，这是一个内置函数，如果`object`是可调用的，则返回`True`，否则返回`False`。如果您需要为任何缺失的`key`生成默认的`value`，这个检查确保您可以调用`.default_factory()`。然后，你检查一下`.default_factory`是不是`None`。

*   **在第 7 行**，如果`default_factory`是`None`，你就像普通的`dict`一样养一只`TypeError`。

*   **在第 8 行**，你初始化`.default_factory`。

*   **在第 10 行**中，你定义了`.__missing__()`，它的实现和你之前看到的一样。回想一下，当给定的`key`不在字典中时，`.__getitem__()`会自动调用`.__missing__()`。

如果你有心情阅读一些 [C](https://realpython.com/build-python-c-extension-module/) 代码，那么你可以看看 [CPython 源代码](https://realpython.com/cpython-source-code-guide/)中 Python `defaultdict`类型的[完整代码](https://github.com/python/cpython/blob/master/Modules/_collectionsmodule.c)。

现在您已经完成了这个类的编码，您可以通过将代码放入名为`my_dd.py`的 Python 脚本中并从交互式会话中导入它来测试它。这里有一个例子:

>>>

```py
>>> from my_dd import my_defaultdict
>>> dd_one = my_defaultdict(list)
>>> dd_one
{}
>>> dd_one['missing']
[]
>>> dd_one
{'missing': []}
>>> dd_one.default_factory = int
>>> dd_one['another_missing']
0
>>> dd_one
{'missing': [], 'another_missing': 0}
>>> dd_two = my_defaultdict(None)
>>> dd_two['missing']
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    dd_two['missing']
 File "/home/user/my_dd.py", line 10,
 in __missing__
 raise KeyError(key)
KeyError: 'missing'
```

这里首先从`my_dd`导入`my_defaultdict`。然后，创建一个`my_defaultdict`的实例，并将`list`传递给`.default_factory`。如果你试图通过订阅操作访问一个键，比如`dd_one['missing']`，那么`.__getitem__()`会被 Python 自动调用。如果这个键不在字典中，那么调用`.__missing__()`，它通过调用`.default_factory()`生成一个默认值。

你也可以像在`dd_one.default_factory = int`中一样使用普通的赋值操作来改变分配给`.default_factory`的可调用函数。最后，如果你将`None`传给`.default_factory`，那么当你试图找回丢失的钥匙时，你将得到一个`KeyError`。

**注意:**一个`defaultdict`的行为本质上和这个 Python 等价物是一样的。然而，您很快就会注意到，您的 Python 实现不是作为真正的`defaultdict`打印的，而是作为标准的`dict`打印的。您可以通过覆盖`.__str__()`和`.__repr__()`来修改这个细节。

您可能想知道为什么在这个例子中您子类化了`collections.UserDict`而不是常规的`dict`。其主要原因是，对内置类型进行子类化可能容易出错，因为内置类型的 C 代码似乎不会始终如一地调用由用户覆盖的特殊方法。

这里有一个例子，展示了在子类化`dict`时可能会遇到的一些问题:

>>>

```py
>>> class MyDict(dict):
...     def __setitem__(self, key, value):
...         super().__setitem__(key, None)
...
>>> my_dict = MyDict(first=1)
>>> my_dict
{'first': 1}
>>> my_dict['second'] = 2
>>> my_dict
{'first': 1, 'second': None}
>>> my_dict.setdefault('third', 3)
3
>>> my_dict
{'first': 1, 'second': None, 'third': 3}
```

在这个例子中，您创建了`MyDict`，它是`dict`的子类。您的 [`.__setitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__setitem__) 实现总是将值设置为`None`。如果你创建了一个`MyDict`的实例并传递了一个关键字参数给它的初始化器，那么你会注意到这个类并没有调用你的`.__setitem__()`来处理这个赋值。你知道那是因为键`first`没有被赋值`None`。

相比之下，如果您运行类似于`my_dict['second'] = 2`的订阅操作，那么您会注意到`second`被设置为`None`而不是`2`。所以，这个时候你可以说订阅操作调用你的客户`.__setitem__()`。最后，注意`.setdefault()`也不调用`.__setitem__()`，因为您的`third`键以值`3`结束。

**`UserDict`** 并没有继承`dict`而是模拟了标准字典的行为。该类有一个名为`.data`的内部`dict`实例，用于存储字典的内容。在创建**自定义映射**时，`UserDict`是一个更可靠的类。如果你使用`UserDict`，那么你将会避免你之前看到的问题。为了证明这一点，回到`my_defaultdict`的代码并添加以下方法:

```py
 1class my_defaultdict(collections.UserDict):
 2    # Snip
 3    def __setitem__(self, key, value): 4        print('__setitem__() gets called')
 5        super().__setitem__(key, None)
```

这里，您添加了一个调用超类`.__setitem__()`的自定义`.__setitem__()`，它总是将值设置为`None`。更新脚本`my_dd.py`中的代码，并从交互式会话中导入，如下所示:

>>>

```py
>>> from my_dd import my_defaultdict
>>> my_dict = my_defaultdict(list, first=1)
__setitem__() gets called
>>> my_dict
{'first': None}
>>> my_dict['second'] = 2
__setitem__() gets called
>>> my_dict
{'first': None, 'second': None}
```

在这种情况下，当您实例化`my_defaultdict`并将`first`传递给类初始化器时，您的自定义`__setitem__()`会被调用。同样，当你给键`second`赋值时，`__setitem__()`也会被调用。您现在有了一个`my_defaultdict`，它始终如一地调用您的定制特殊方法。注意，现在字典中的所有值都等于`None`。

[*Remove ads*](/account/join/)

## 将参数传递给`.default_factory`

正如您前面看到的，`.default_factory`必须被设置为一个不接受参数并返回值的可调用对象。该值将用于为字典中任何缺失的键提供默认值。即使当`.default_factory`不应该接受参数时，Python 也提供了一些技巧，如果您需要向它提供参数，您可以使用这些技巧。在本节中，您将介绍两个可以实现这一目的的 Python 工具:

1.  [T2`lambda`](https://realpython.com/python-lambda/)
2.  [T2`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial)

有了这两个工具，您可以为 Python `defaultdict`类型增加额外的灵活性。例如，您可以用一个带参数的 callable 来初始化一个`defaultdict`,经过一些处理后，您可以用一个新的参数来更新这个 callable，以改变您将从现在开始创建的键的默认值。

### 使用`lambda`

将参数传递给`.default_factory`的一种灵活方式是使用 **`lambda`** 。假设您想创建一个函数来在`defaultdict`中生成默认值。该函数执行一些处理并返回一个值，但是您需要传递一个参数以使该函数正确工作。这里有一个例子:

>>>

```py
>>> def factory(arg):
...     # Do some processing here...
...     result = arg.upper()
...     return result
...
>>> def_dict = defaultdict(lambda: factory('default value'))
>>> def_dict['missing']
'DEFAULT VALUE'
```

在上面的代码中，您创建了一个名为`factory()`的函数。该函数接受一个参数，进行一些处理，然后返回最终结果。然后，创建一个`defaultdict`并使用`lambda`将字符串`'default value'`传递给`factory()`。当您尝试访问丢失的密钥时，将运行以下步骤:

1.  **字典`def_dict`** 调用它的`.default_factory`，它保存了对一个 [`lambda`函数](https://realpython.com/python-lambda/)的引用。
2.  **`lambda`函数**被调用，并返回以`'default value'`作为参数调用`factory()`的结果值。

如果您正在使用`def_dict`并突然需要将参数更改为`factory()`，那么您可以这样做:

>>>

```py
>>> def_dict.default_factory = lambda: factory('another default value')
>>> def_dict['another_missing']
'ANOTHER DEFAULT VALUE'
```

这一次，`factory()`接受一个新的字符串参数(`'another default value'`)。从现在开始，如果您试图访问或修改一个丢失的键，那么您将获得一个新的默认值，这就是字符串`'ANOTHER DEFAULT VALUE'`。

最后，您可能会面临这样一种情况，您需要一个不同于`0`或`[]`的默认值。在这种情况下，你也可以使用`lambda`来**生成一个不同的默认值**。比如，假设你有一个`list`的整数，你需要计算每个数的累积积。然后，您可以使用`defaultdict`和`lambda`，如下所示:

>>>

```py
>>> from collections import defaultdict
>>> lst = [1, 1, 2, 1, 2, 2, 3, 4, 3, 3, 4, 4]
>>> def_dict = defaultdict(lambda: 1)
>>> for number in lst:
...     def_dict[number] *= number
...
>>> def_dict
defaultdict(<function <lambda> at 0x...70>, {1: 1, 2: 8, 3: 27, 4: 64})
```

这里，您使用`lambda`来提供默认值`1`。有了这个初始值，你就可以计算出`lst`中每个数字的累积积。注意，使用`int`不能得到相同的结果，因为`int`返回的默认值总是`0`，这对于这里需要执行的乘法运算来说不是一个好的初始值。

### 使用`functools.partial()`

[`functools.partial(func, *args, **keywords)`](https://docs.python.org/3/library/functools.html#functools.partial) 是返回一个`partial`对象的函数。当您用位置参数(`args`)和关键字参数(`keywords`)调用这个对象时，它的行为类似于您调用`func(*args, **keywords)`时的行为。您可以利用`partial()`的这种行为，并使用它在 Python `defaultdict`中将参数传递给`.default_factory`。这里有一个例子:

>>>

```py
>>> def factory(arg):
...     # Do some processing here...
...     result = arg.upper()
...     return result
...
>>> from functools import partial
>>> def_dict = defaultdict(partial(factory, 'default value'))
>>> def_dict['missing']
'DEFAULT VALUE'
>>> def_dict.default_factory = partial(factory, 'another default value')
>>> def_dict['another_missing']
'ANOTHER DEFAULT VALUE'
```

在这里，您创建一个 Python `defaultdict`并使用`partial()`向`.default_factory`提供一个参数。注意，您也可以更新`.default_factory`来为可调用的`factory()`使用另一个参数。这种行为可以给你的`defaultdict`对象增加很多灵活性。

## 结论

Python `defaultdict`类型是一种类似字典的数据结构，由 Python 标准库在一个名为`collections`的模块中提供。该类继承自`dict`，它主要增加的功能是为丢失的键提供默认值。在本教程中，您已经学习了如何使用 Python `defaultdict`类型来处理字典中丢失的键。

**您现在能够:**

*   **创建并使用**一个 Python `defaultdict`来处理丢失的键
*   **解决**现实世界中与分组、计数和累加操作相关的问题
*   **知道**`defaultdict`和`dict`的实现差异
*   **决定**何时以及为什么使用 Python `defaultdict`而不是标准的`dict`

Python `defaultdict`类型是一种方便而有效的数据结构，旨在帮助您处理字典中丢失的键。试一试，让你的代码更快、更可读、更 Pythonic 化！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 Python defaultdict Type**](/courses/python-defaultdict-type/) 处理丢失的键********