# Python 的 ChainMap:有效管理多个上下文

> 原文：<https://realpython.com/python-chainmap/>

有时，当您使用几个不同的词典时，您需要将它们作为一个单独的词典进行分组和管理。在其他情况下，您可以拥有多个字典来表示不同的**范围**或**上下文**，并且需要将它们作为单个字典来处理，以允许您按照给定的顺序或优先级来访问底层数据。在那些情况下，你可以从 [`collections`模块](https://docs.python.org/3/library/collections.html#module-collections)中利用 Python 的 [`ChainMap`](https://docs.python.org/3/library/collections.html?highlight=collections#collections.ChainMap) 。

`ChainMap`以类似字典的行为将多个字典和映射组合在一个可更新的视图中。此外，`ChainMap`还提供了允许您有效管理各种字典、定义关键字查找优先级等功能。

**在本教程中，您将学习如何:**

*   在你的 Python 程序中创建 **`ChainMap`实例**
*   探索`ChainMap`和`dict`之间的**差异**
*   使用`ChainMap`将**几本字典合二为一**
*   使用`ChainMap`管理**键查找优先级**

为了从本教程中获得最大收益，你应该知道在 Python 中使用[字典](https://realpython.com/python-dicts/)和[列表](https://realpython.com/python-lists-tuples/)的基本知识。

在旅程结束时，您会发现一些实际的例子，它们将帮助您更好地理解`ChainMap`最相关的特性和用例。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## Python 的`ChainMap` 入门

Python 的 [`ChainMap`](https://docs.python.org/3/library/collections.html#collections.ChainMap) 在 [Python 3.3](https://docs.python.org/3/whatsnew/3.3.html#collections) 中被添加到`collections`中，作为管理多个**范围**和**上下文**的便捷工具。这个类允许你将几个字典和其他[映射](https://docs.python.org/3/glossary.html#term-mapping)组合在一起，使它们在逻辑上表现得像一个整体。它创建了一个单独的**可更新视图**，其工作方式类似于常规字典，但有一些内部差异。

`ChainMap`不会将其映射合并在一起。相反，它将它们保存在内部映射列表中。然后`ChainMap`在列表顶部重新实现常用的字典操作。由于内部列表保存了对原始输入映射的引用，这些映射中的任何变化都会影响整个`ChainMap`对象。

将输入映射存储在一个列表中允许在给定的链表中有重复的键。如果您执行**键查找**，那么`ChainMap`将搜索映射列表，直到找到目标键的第一个匹配项。如果密钥丢失，那么你照常得到一个 [`KeyError`](https://realpython.com/python-keyerror/) 。

当您需要管理嵌套的[作用域](https://realpython.com/python-namespaces-scope/)时，将映射存储在一个列表中确实非常有用，其中每个映射代表一个特定的作用域或上下文。

为了更好地理解什么是作用域和上下文，考虑一下 Python 是如何解析名称的。Python 在查找名称时，会在 [`locals()`](https://realpython.com/python-scope-legb-rule/#locals) 、 [`globals()`](https://realpython.com/python-scope-legb-rule/#globals) 中搜索，最后在 [`builtins`](https://realpython.com/python-scope-legb-rule/#builtins-the-built-in-scope) 中搜索，直到找到目标名称的第一次出现。如果名字不存在，那么你得到一个`NameError`。处理范围和上下文是你可以用`ChainMap`解决的最常见的问题。

当你使用`ChainMap`时，你可以用[不相交](https://en.wikipedia.org/wiki/Disjoint_sets)或者[相交](https://en.wikipedia.org/wiki/Intersection_(set_theory))的键来链接几个字典。

在第一种情况下，`ChainMap`允许您将所有的字典视为一个字典。因此，您可以像使用单个字典一样访问键值对。在第二种情况下，除了将您的字典作为一个字典来管理之外，您还可以利用内部映射列表来为字典中的重复键定义某种类型的**访问优先级**。这就是为什么`ChainMap`对象非常适合处理多种上下文。

`ChainMap`的一个奇怪行为是**突变**，比如更新、添加、删除、清除和弹出键，只作用于内部映射列表中的第一个映射。下面总结一下`ChainMap`的主要特点:

*   从几个输入映射构建一个可更新的视图
*   提供了与字典几乎相同的**接口，但是增加了一些额外的特性**
*   不合并输入映射，而是将它们保存在一个**内部公共列表**中
*   在输入映射中看到**外部变化**
*   可以包含具有不同值的**重复键**
*   **在内部映射列表中顺序搜索关键字**
*   在搜索了整个映射列表后，当缺少一个键时抛出一个 **`KeyError`**
*   仅对内部列表中的第一个映射执行**突变**

在本教程中，你会学到更多关于`ChainMap`所有这些很酷的特性。下一节将指导您如何在代码中创建新的`ChainMap`实例。

[*Remove ads*](/account/join/)

### 实例化`ChainMap`

要在 Python 代码中创建`ChainMap`，首先需要从`collections`导入类，然后像往常一样调用它。类初始化器可以接受零个或多个映射作为参数。在没有参数的情况下，它用一个空字典初始化一个链图:

>>>

```py
>>> from collections import ChainMap
>>> from collections import OrderedDict, defaultdict

>>> # Use no arguments
>>> ChainMap()
ChainMap({})

>>> # Use regular dictionaries
>>> numbers = {"one": 1, "two": 2}
>>> letters = {"a": "A", "b": "B"}

>>> ChainMap(numbers, letters)
ChainMap({'one': 1, 'two': 2}, {'a': 'A', 'b': 'B'})

>>> ChainMap(numbers, {"a": "A", "b": "B"})
ChainMap({'one': 1, 'two': 2}, {'a': 'A', 'b': 'B'})

>>> # Use other mappings
>>> numbers = OrderedDict(one=1, two=2)
>>> letters = defaultdict(str, {"a": "A", "b": "B"})
>>> ChainMap(numbers, letters)
ChainMap(
 OrderedDict([('one', 1), ('two', 2)]),
 defaultdict(<class 'str'>, {'a': 'A', 'b': 'B'})
)
```

这里，您使用不同的映射组合创建了几个`ChainMap`对象。在每种情况下，`ChainMap`返回所有输入映射的一个类似字典的视图。注意，你可以使用任何类型的映射，比如 [`OrderedDict`](https://realpython.com/python-ordereddict/) 和 [`defaultdict`](https://realpython.com/python-defaultdict/) 。

你也可以使用[类方法](https://realpython.com/instance-class-and-static-methods-demystified/) `.fromkeys()`创建`ChainMap`对象。这个方法可以接受一个可迭代的键和一个可选的所有键的默认值:

>>>

```py
>>> from collections import ChainMap

>>> ChainMap.fromkeys(["one", "two","three"])
ChainMap({'one': None, 'two': None, 'three': None})

>>> ChainMap.fromkeys(["one", "two","three"], 0)
ChainMap({'one': 0, 'two': 0, 'three': 0})
```

如果您调用`ChainMap`上的`.fromkeys()`并把一个可迭代的键作为参数，那么您将得到一个只有一个字典的链表。密钥来自输入 iterable，值默认为 [`None`](https://realpython.com/null-in-python/) 。可选地，您可以向`.fromkeys()`传递第二个参数，为每个键提供一个合理的默认值。

### 运行类似字典的操作

`ChainMap`支持与常规字典相同的 API 来访问现有的键。一旦有了`ChainMap`对象，就可以用字典式的键查找来检索现有的键，或者可以使用 [`.get()`](https://docs.python.org/3/library/stdtypes.html#dict.get) :

>>>

```py
>>> from collections import ChainMap

>>> numbers = {"one": 1, "two": 2}
>>> letters = {"a": "A", "b": "B"}

>>> alpha_num = ChainMap(numbers, letters)
>>> alpha_num["two"]
2

>>> alpha_num.get("a")
'A'

>>> alpha_num["three"]
Traceback (most recent call last):
    ...
KeyError: 'three'
```

键查找搜索目标链映射中的所有映射，直到找到所需的键。如果键不存在，那么你得到通常的`KeyError`。现在，当您有重复的键时，查找操作会如何表现呢？在这种情况下，您将获得目标键的第一个匹配项:

>>>

```py
>>> from collections import ChainMap

>>> for_adoption = {"dogs": 10, "cats": 7, "pythons": 3}
>>> vet_treatment = {"dogs": 4, "cats": 3, "turtles": 1}
>>> pets = ChainMap(for_adoption, vet_treatment)

>>> pets["dogs"]
10
>>> pets.get("cats")
7
>>> pets["turtles"]
1
```

当你访问一个重复的键时，比如`"dogs"`和`"cats"`，链表只返回那个键的第一次出现。在内部，查找操作按照输入映射在内部映射列表中出现的顺序搜索输入映射，这也是您将它们传递到类的初始化器中的确切顺序。

这个一般行为也适用于迭代:

>>>

```py
>>> from collections import ChainMap

>>> for_adoption = {"dogs": 10, "cats": 7, "pythons": 3}
>>> vet_treatment = {"dogs": 4, "cats": 3, "turtles": 1}
>>> pets = ChainMap(for_adoption, vet_treatment)

>>> for key, value in pets.items():
...     print(key, "->", value)
...
dogs -> 10
cats -> 7
turtles -> 1
pythons -> 3
```

[`for`循环](https://realpython.com/python-for-loop/)遍历`pets`中的字典，[打印](https://realpython.com/python-print/)每个键值对的第一次出现。您还可以直接[遍历字典](https://realpython.com/iterate-through-dictionary-python/)，或者像对任何字典一样使用 [`.keys()`](https://docs.python.org/3/library/stdtypes.html#dict.keys) 和 [`.values()`](https://docs.python.org/3/library/stdtypes.html#dict.values) 遍历字典:

>>>

```py
>>> from collections import ChainMap

>>> for_adoption = {"dogs": 10, "cats": 7, "pythons": 3}
>>> vet_treatment = {"dogs": 4, "cats": 3, "turtles": 1}
>>> pets = ChainMap(for_adoption, vet_treatment)

>>> for key in pets:
...     print(key, "->", pets[key])
...
dogs -> 10
cats -> 7
turtles -> 1
pythons -> 3

>>> for key in pets.keys():
...     print(key, "->", pets[key])
...
dogs -> 10
cats -> 7
turtles -> 1
pythons -> 3

>>> for value in pets.values():
...     print(value)
...
10
7
1
3
```

同样，行为是相同的。每次迭代都会遍历底层链表中每个键、项和值的第一次出现。

`ChainMap`也支持**突变**。换句话说，它允许您更新、添加、删除和弹出键值对。这种情况下的不同之处在于，这些操作仅作用于第一个映射:

>>>

```py
>>> from collections import ChainMap

>>> numbers = {"one": 1, "two": 2}
>>> letters = {"a": "A", "b": "B"}

>>> alpha_num = ChainMap(numbers, letters)
>>> alpha_num
ChainMap({'one': 1, 'two': 2}, {'a': 'A', 'b': 'B'})

>>> # Add a new key-value pair
>>> alpha_num["c"] = "C"
>>> alpha_num
ChainMap({'one': 1, 'two': 2, 'c': 'C'}, {'a': 'A', 'b': 'B'})

>>> # Update an existing key
>>> alpha_num["b"] = "b"
>>> alpha_num
ChainMap({'one': 1, 'two': 2, 'c': 'C', 'b': 'b'}, {'a': 'A', 'b': 'B'})

>>> # Pop keys
>>> alpha_num.pop("two")
2
>>> alpha_num.pop("a")
Traceback (most recent call last):
    ...
KeyError: "Key not found in the first mapping: 'a'"

>>> # Delete keys
>>> del alpha_num["c"]
>>> alpha_num
ChainMap({'one': 1, 'b': 'b'}, {'a': 'A', 'b': 'B'})
>>> del alpha_num["a"]
Traceback (most recent call last):
    ...
KeyError: "Key not found in the first mapping: 'a'"

>>> # Clear the dictionary
>>> alpha_num.clear()
>>> alpha_num
ChainMap({}, {'a': 'A', 'b': 'B'})
```

改变给定链映射内容的操作只影响第一个映射，即使您试图改变的键存在于列表中的其他映射中。例如，当您试图在第二个映射中更新`"b"`时，实际发生的是您向第一个字典添加了一个新的键。

您可以利用这种行为来创建不修改原始输入字典的可更新的链映射。在这种情况下，您可以使用一个空字典作为`ChainMap`的第一个参数:

>>>

```py
>>> from collections import ChainMap

>>> numbers = {"one": 1, "two": 2}
>>> letters = {"a": "A", "b": "B"}

>>> alpha_num = ChainMap({}, numbers, letters)
>>> alpha_num
ChainMap({}, {'one': 1, 'two': 2}, {'a': 'A', 'b': 'B'})

>>> alpha_num["comma"] = ","
>>> alpha_num["period"] = "."

>>> alpha_num
ChainMap(
 {'comma': ',', 'period': '.'},
 {'one': 1, 'two': 2},
 {'a': 'A', 'b': 'B'}
)
```

这里，您使用一个空字典(`{}`)来创建`alpha_num`。这确保了您在`alpha_num`上执行的更改永远不会影响您的两个原始输入词典`numbers`和`letters`，并且只会影响列表开头的空词典。

[*Remove ads*](/account/join/)

### 合并与链接字典

作为用`ChainMap`链接多个字典的替代方法，您可以考虑用 [`dict.update()`](https://docs.python.org/3/library/stdtypes.html#dict.update) 将它们合并在一起:

>>>

```py
>>> from collections import ChainMap

>>> # Chain dictionaries with ChainMap
>>> for_adoption = {"dogs": 10, "cats": 7, "pythons": 3}
>>> vet_treatment = {"hamsters": 2, "turtles": 1}

>>> ChainMap(for_adoption, vet_treatment)
ChainMap(
 {'dogs': 10, 'cats': 7, 'pythons': 3},
 {'hamsters': 2, 'turtles': 1}
)

>>> # Merge dictionaries with .update()
>>> pets = {}
>>> pets.update(for_adoption)
>>> pets.update(vet_treatment)
>>> pets
{'dogs': 10, 'cats': 7, 'pythons': 3, 'hamsters': 2, 'turtles': 1}
```

在这个具体的例子中，当您从两个具有惟一键的现有字典构建一个链映射和一个等价字典时，您会得到类似的结果。

用`.update()`合并字典和用`ChainMap`链接字典相比，有利有弊。第一个也是最重要的缺点是，您放弃了使用多个作用域或上下文来管理和优先访问重复键的能力。使用`.update()`，您为给定键提供的最后一个值将始终有效:

>>>

```py
>>> for_adoption = {"dogs": 10, "cats": 7, "pythons": 3}
>>> vet_treatment = {"cats": 2, "dogs": 1}

>>> # Merge dictionaries with .update()
>>> pets = {}
>>> pets.update(for_adoption)
>>> pets.update(vet_treatment)
>>> pets
{'dogs': 1, 'cats': 2, 'pythons': 3}
```

常规词典不能存储重复的键。每当您使用现有键的值调用`.update()`时，该键就会用新值更新。在这种情况下，您将无法使用不同的作用域来区分访问重复键的优先级。

**注意:**从 [Python 3.5](https://docs.python.org/3/whatsnew/3.5.html#pep-448-additional-unpacking-generalizations) 开始，还可以使用[字典解包操作符(`**` )](https://realpython.com/iterate-through-dictionary-python/#using-the-dictionary-unpacking-operator) 将字典合并在一起。此外，如果您使用的是 [Python 3.9](https://realpython.com/python39-new-features/) ，那么您可以使用字典联合操作符(`|`)将两个字典合并成一个新字典。

现在假设您有 *n* 个不同的映射，每个映射最多有 *m* 个键。从它们创建一个`ChainMap`对象将花费 [*O* ( *n* )](https://en.wikipedia.org/wiki/Big_O_notation) 的执行时间，而在最坏的情况下，检索一个键将花费 *O* ( *n* )，其中目标键在内部映射列表的最后一个字典中。

或者，在一个循环中使用`.update()`创建一个常规字典将花费 *O* ( *nm* )，而从最终字典中检索一个键将花费 *O* (1)。

结论是，如果您经常创建字典链，并且每次只执行一些键查找，那么您应该使用`ChainMap`。如果反过来，那么使用常规字典，除非你需要重复的键或多个作用域。

合并和链接字典之间的另一个区别是，当您使用`ChainMap`时，输入字典中的外部变化会影响底层的链，而合并字典则不是这种情况。

## 探索`ChainMap`的附加特性

提供了与普通 Python 字典基本相同的 API 和特性，还有一些你已经知道的细微差别。`ChainMap`还支持一些特定于其设计和目标的附加特性。

在本节中，您将了解所有这些附加功能。您将了解当您访问字典中的键值对时，它们如何帮助您管理不同的范围和上下文。

### 使用`.maps` 管理映射列表

`ChainMap`将所有输入映射存储在一个内部列表中。这个列表可以通过一个名为 [`.maps`](https://docs.python.org/3/library/collections.html#collections.ChainMap.maps) 的公共[实例属性](https://realpython.com/python3-object-oriented-programming/#class-and-instance-attributes)来访问，并且是用户可更新的。`.maps`中映射的顺序与您将它们传递给`ChainMap`的顺序相匹配。当您执行键查找操作时，该顺序定义了**搜索顺序**。

以下是如何访问`.maps`的示例:

>>>

```py
>>> from collections import ChainMap

>>> for_adoption = {"dogs": 10, "cats": 7, "pythons": 3}
>>> vet_treatment = {"dogs": 4, "turtles": 1}

>>> pets = ChainMap(for_adoption, vet_treatment)
>>> pets.maps
[{'dogs': 10, 'cats': 7, 'pythons': 3}, {'dogs': 4, 'turtles': 1}]
```

这里，您使用`.maps`来访问`pets`保存的内部映射列表。该列表是一个常规的 Python 列表，因此您可以手动添加和移除映射、遍历列表、更改映射的顺序等等:

>>>

```py
>>> from collections import ChainMap

>>> for_adoption = {"dogs": 10, "cats": 7, "pythons": 3}
>>> vet_treatment = {"cats": 1}
>>> pets = ChainMap(for_adoption, vet_treatment)

>>> pets.maps.append({"hamsters": 2})
>>> pets.maps
[{'dogs': 10, 'cats': 7, 'pythons': 3}, {"cats": 1}, {'hamsters': 2}]

>>> del pets.maps[1]
>>> pets.maps
[{'dogs': 10, 'cats': 7, 'pythons': 3}, {'hamsters': 2}]

>>> for mapping in pets.maps:
...     print(mapping)
...
{'dogs': 10, 'cats': 7, 'pythons': 3}
{'hamsters': 2}
```

在这些例子中，首先使用 [`.append()`](https://realpython.com/python-append/) 向`.maps`添加一个新的字典。然后使用`del` [关键字](https://realpython.com/python-keywords/)删除位置`1`处的词典。您可以像管理任何常规 Python 列表一样管理`.maps`。

**注意:**内部映射列表`.maps`，将总是包含至少一个映射。例如，如果您使用没有参数的`ChainMap()`创建一个空的链表，那么这个列表将存储一个空的字典。

在对所有映射执行操作时，您可以使用`.maps`来迭代这些映射。遍历映射列表的可能性允许您对每个映射执行不同的操作。使用此选项，您可以解决仅变更列表中第一个映射的默认行为。

一个有趣的例子是，您可以使用 [`.reverse()`](https://realpython.com/python-reverse-list/) 反转当前映射列表的顺序:

>>>

```py
>>> from collections import ChainMap

>>> for_adoption = {"dogs": 10, "cats": 7, "pythons": 3}
>>> vet_treatment = {"cats": 1}
>>> pets = ChainMap(for_adoption, vet_treatment)
>>> pets
ChainMap({'dogs': 10, 'cats': 7, 'pythons': 3}, {'cats': 1})

>>> pets.maps.reverse()
>>> pets
ChainMap({'cats': 1}, {'dogs': 10, 'cats': 7, 'pythons': 3})
```

反转内部映射列表允许您在链表中查找给定的键时反转搜索顺序。现在，当你寻找`"cats"`时，你得到的是接受兽医治疗的猫的数量，而不是准备被收养的猫的数量。

[*Remove ads*](/account/join/)

### 用`.new_child()` 添加新的子上下文

`ChainMap`亦作 [`.new_child()`](https://docs.python.org/3/library/collections.html#collections.ChainMap.new_child) 。该方法可选地将一个映射作为参数，并返回一个新的`ChainMap`实例，该实例包含输入映射，后跟底层链映射中的所有当前映射:

>>>

```py
>>> from collections import ChainMap

>>> mom = {"name": "Jane", "age": 31}
>>> dad = {"name": "John", "age": 35}

>>> family = ChainMap(mom, dad)
>>> family
ChainMap({'name': 'Jane', 'age': 31}, {'name': 'John', 'age': 35})

>>> son = {"name": "Mike", "age": 0}
>>> family = family.new_child(son)

>>> for person in family.maps:
...     print(person)
...
{'name': 'Mike', 'age': 0}
{'name': 'Jane', 'age': 31}
{'name': 'John', 'age': 35}
```

这里，`.new_child()`返回一个新的`ChainMap`对象，包含一个新的映射`son`，后面跟着旧的映射`mom`和`dad`。请注意，新映射现在位于内部映射列表的第一个位置`.maps`。

使用`.new_child()`，您可以创建一个子上下文，您可以在不改变任何现有映射的情况下更新它。例如，如果您不带参数调用`.new_child()`，那么它将使用一个空字典，并将它放在`.maps`的开头。在这之后，您可以对新的空映射执行任何变化，保持映射的其余部分为只读。

### 跳过带有`.parents`和的子上下文

`ChainMap`的另一个有趣的特点是 [`.parents`](https://docs.python.org/3/library/collections.html#collections.ChainMap.parents) 。这个[属性](https://realpython.com/python-descriptors/#python-descriptors-in-properties)返回一个新的`ChainMap`实例，其中包含底层链表中除第一个映射之外的所有映射。当您在给定的链映射中搜索关键点时，此功能对于跳过第一个映射很有用:

>>>

```py
>>> from collections import ChainMap

>>> mom = {"name": "Jane", "age": 31}
>>> dad = {"name": "John", "age": 35}
>>> son = {"name": "Mike", "age":  0}

>>> family = ChainMap(son, mom, dad)
>>> family
ChainMap(
 {'name': 'Mike', 'age': 0},
 {'name': 'Jane', 'age': 31},
 {'name': 'John', 'age': 35}
)

>>> family.parents
ChainMap({'name': 'Jane', 'age': 31}, {'name': 'John', 'age': 35})
```

在这个例子中，您使用`.parents`跳过包含儿子数据的第一个字典。在某种程度上，`.parents`与`.new_child()`正好相反。前者删除一个字典，而后者向列表的开头添加一个新字典。在这两种情况下，您都会得到一个新的链图。

## 用`ChainMap` 管理范围和上下文

可以说，`ChainMap`的主要用例是提供一种有效的方式来管理多个**作用域**或**上下文**，并处理复制键的**访问优先级**。当您有几个存储重复键的字典，并且希望定义代码访问它们的顺序时，此功能非常有用。

在 [`ChainMap`文档](https://docs.python.org/3/library/collections.html#chainmap-examples-and-recipes)中，你会发现一个经典的例子，它模拟了 Python 如何解析不同名称空间中的变量名。

当 Python 寻找一个名称时，它会依次搜索本地、全局和内置范围，遵循相同的顺序，直到找到目标名称。Python 范围是将名称映射到对象的字典。

要模拟 Python 的内部查找链，可以使用链图:

>>>

```py
>>> import builtins

>>> # Shadow input with a global name
>>> input = 42

>>> pylookup = ChainMap(locals(), globals(), vars(builtins))

>>> # Retrieve input from the global namespace
>>> pylookup["input"]
42

>>> # Remove input from the global namespace
>>> del globals()["input"]

>>> # Retrieve input from the builtins namespace
>>> pylookup["input"]
<built-in function input>
```

在这个例子中，首先创建一个名为`input`的全局变量，它隐藏了 [`builtins`](https://realpython.com/python-scope-legb-rule/#builtins-the-built-in-scope) 范围内的内置 [`input()`](https://realpython.com/python-input-output/#reading-input-from-the-keyboard) 函数。然后创建`pylookup`作为包含三个字典的链式映射，这三个字典包含每个 Python 范围。

当您从`pylookup`中检索`input`时，您从全局范围中获得值`42`。如果您从`globals()`字典中移除`input`键并再次访问它，那么您将从`builtins`范围中获得内置的`input()`函数，它在 Python 的查找链中具有最低的优先级。

类似地，您可以使用`ChainMap`来定义和管理重复键的键查找顺序。这使您可以优先访问所需的复制键实例。

## 标准库中的跟踪`ChainMap`

`ChainMap`的由来与 [`ConfigParser`](https://docs.python.org/3/library/configparser.html#configparser.ConfigParser) 中的一个性能[问题](https://bugs.python.org/issue11089)密切相关，它存在于标准库中的 [`configparser`](https://docs.python.org/3/library/configparser.html#module-configparser) 模块中。有了`ChainMap`，核心 Python 开发者通过优化 [`ConfigParser.get()`](https://github.com/python/cpython/blob/142e5c5445c019542246d93fe2f9e195d3131686/Lib/configparser.py#L766) 的实现，大幅提升了这个模块整体的性能。

你也可以在 [`string`](https://docs.python.org/3/library/string.html#module-string) 模块中找到`ChainMap`作为 [`Template`](https://realpython.com/python-string-formatting/#4-template-strings-standard-library) 的一部分。这个类将一个[字符串模板](https://docs.python.org/3/library/string.html?highlight=template#template-strings)作为参数，并允许您执行字符串替换，如 [PEP 292](https://www.python.org/dev/peps/pep-0292) 中所述。输入[字符串](https://realpython.com/python-strings/)模板包含嵌入的标识符，稍后您可以用实际值替换这些标识符:

>>>

```py
>>> import string

>>> greeting = "Hey $name, welcome to $place!"
>>> template = string.Template(greeting)

>>> template.substitute({"name": "Jane", "place": "the World"})
'Hey Jane, welcome to the World!'
```

当您通过字典为`name`和`place`提供值时， [`.substitute()`](https://docs.python.org/3/library/string.html#string.Template.substitute) 会在模板字符串中替换它们。此外，`.substitute()`可以将值作为关键字参数(`**kwargs`，这在某些情况下会导致名称冲突:

>>>

```py
>>> import string

>>> greeting = "Hey $name, welcome to $place!"
>>> template = string.Template(greeting)

>>> template.substitute(
...     {"name": "Jane", "place": "the World"},
...     place="Real Python"
... )
'Hey Jane, welcome to Real Python!'
```

在本例中，`.substitute()`用您作为关键字参数提供的值替换`place`,而不是输入字典中的值。如果您稍微深入研究一下这个方法的[代码](https://github.com/python/cpython/blob/92ceb1c8402422412fcbb98ca19448677c667c3c/Lib/string.py#L104)，那么您会看到当名称冲突发生时，它使用`ChainMap`来有效地管理输入值的优先级。

下面是来自`.substitute()`的源代码片段:

```py
# string.py
# Snip...
from collections import ChainMap as _ChainMap

_sentinel_dict = {}

class Template:
    """A string class for supporting $-substitutions."""
    # Snip...

    def substitute(self, mapping=_sentinel_dict, /, **kws):
        if mapping is _sentinel_dict:
            mapping = kws
        elif kws:
 mapping = _ChainMap(kws, mapping)        # Snip...
```

在这里，突出显示的行具有魔力。它使用一个链式映射，该映射将两个字典`kws`和`mapping`作为参数。通过将`kws`作为第一个参数，该方法为输入数据中的重复标识符设置优先级。

[*Remove ads*](/account/join/)

## 将 Python 的`ChainMap`付诸行动

到目前为止，您已经学会了如何使用`ChainMap`将多个词典合二为一。您还了解了`ChainMap`的特性，以及这个类与普通字典的不同之处。`ChainMap`的用例相当具体。它们包括:

*   在一个视图中有效地将多个字典分组
*   以某个**优先级**搜索多个字典
*   提供一系列**默认值**并管理它们的优先级
*   提高频繁计算字典的**子集的代码的性能**

在这一节中，您将编写一些实际的例子，帮助您更好地理解如何使用`ChainMap`来解决现实世界中的问题。

### 一次访问多个存货

您将编写的第一个示例使用`ChainMap`在单个视图中高效地搜索多个字典。在这种情况下，您会假设有一堆独立的字典，它们之间有唯一的键。

假设你正在经营一家出售水果和蔬菜的商店。您已经编写了一个 Python 应用程序来管理您的库存。该应用程序从数据库中读取数据，并返回两本分别包含水果和蔬菜价格数据的字典。您需要一种有效的方法在单个字典中对这些数据进行分组和管理。

经过一些研究后，您最终使用了`ChainMap`:

>>>

```py
>>> from collections import ChainMap

>>> fruits_prices = {"apple": 0.80, "grape": 0.40, "orange": 0.50}
>>> veggies_prices = {"tomato": 1.20, "pepper": 1.30, "onion": 1.25}
>>> prices = ChainMap(fruits_prices, veggies_prices)

>>> order = {"apple": 4, "tomato": 8, "orange": 4}

>>> for product, units in order.items():
...     price = prices[product]
...     subtotal = units * price
...     print(f"{product:6}: ${price:.2f} × {units} = ${subtotal:.2f}")
...
apple : $0.80 × 4 = $3.20
tomato: $1.20 × 8 = $9.60
orange: $0.50 × 4 = $2.00
```

在这个例子中，您使用一个`ChainMap`来创建一个类似字典的对象，该对象将来自`fruits_prices`和`veggies_prices`的数据进行分组。这允许您访问底层数据，就像您实际上有一个字典一样。`for`循环遍历给定`order`中的产品。然后，它计算每种产品的支付小计，并将其打印在屏幕上。

您可能会想到在新的字典中对数据进行分组，在循环中使用`.update()`。当你的产品种类有限，库存很少的时候，这种方式就很好了。然而，如果你管理许多不同类型的产品，那么与`ChainMap`相比，使用`.update()`来构建一个新字典可能是低效的。

使用`ChainMap`解决这类问题还可以帮助你定义不同批次产品的优先级，让你以先进先出( [FIFO](https://en.wikipedia.org/wiki/FIFO_and_LIFO_accounting) )的方式管理你的库存。

### 排列命令行应用程序设置的优先级

`ChainMap`特别有助于管理应用程序中的默认配置值。正如您已经知道的，`ChainMap`的一个主要特性是它允许您为键查找操作设置优先级。这听起来像是解决应用程序中管理配置问题的正确工具。

例如，假设您正在开发一个[命令行界面(CLI)](https://en.wikipedia.org/wiki/Command-line_interface) 应用程序。该应用程序允许用户指定连接到互联网的代理服务。设置优先级包括:

1.  命令行选项(`--proxy`、`-p`)
2.  用户主目录中的本地配置文件
3.  系统范围的代理配置

如果用户在命令行提供代理，那么应用程序必须使用该代理。否则，应用程序应该使用下一个配置对象中提供的代理，依此类推。这是`ChainMap`最常见的用例之一。在这种情况下，您可以执行以下操作:

>>>

```py
>>> from collections import ChainMap

>>> cmd_proxy = {}  # The user doesn't provide a proxy
>>> local_proxy = {"proxy": "proxy.local.com"}
>>> system_proxy = {"proxy": "proxy.global.com"}

>>> config = ChainMap(cmd_proxy, local_proxy, system_proxy)
>>> config["proxy"]
'proxy.local.com'
```

`ChainMap`允许您为应用程序的代理配置定义适当的优先级。一个键查找搜索`cmd_proxy`，然后是`local_proxy`，最后是`system_proxy`，返回当前键的第一个实例。在这个例子中，用户没有在命令行提供代理，所以应用程序从列表中下一个设置提供者`local_proxy`那里获取代理。

### 管理默认参数值

`ChainMap`的另一个用例是管理方法和函数中的默认参数值。假设您正在编写一个应用程序来管理公司员工的数据。您有以下代表一般用户的类:

```py
class User:
    def __init__(self, name, user_id, role):
        self.name = name
        self.user_id = user_id
        self.role = role

    # Snip...
```

在某些时候，你需要添加一个特性，允许员工访问一个 [CRM](https://en.wikipedia.org/wiki/Customer_relationship_management) 系统的不同组件。您首先想到的是修改`User`来添加新功能。然而，这可能会使类过于复杂，所以您决定创建一个子类`CRMUser`，来提供所需的功能。

该类将用户`name`和 CRM `component`作为参数。这也需要一些 [`**kwargs`](https://realpython.com/python-kwargs-and-args/) 。你想以一种方式实现`CRMUser`，允许你为基类的初始化器提供合理的默认值，同时又不失去`**kwargs`的灵活性。

以下是如何使用`ChainMap`解决问题:

```py
from collections import ChainMap

class CRMUser(User):
    def __init__(self, name, component, **kwargs):
        defaults = {"user_id": next(component.user_id), "role": "read"}
 super().__init__(name, **ChainMap(kwargs, defaults))
```

在这个代码示例中，您创建了一个`User`的子类。在类初始化器中，你把`name`、`component`和`**kwargs`作为参数。然后您创建一个本地字典，其中包含`user_id`和`role`的默认值。然后你用 [`super()`](https://realpython.com/python-super/) 调用父类的 [`.__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__) 方法。在这个调用中，您将`name`直接传递给父类的初始化器，并使用一个链表为其余的参数提供默认值。

注意，`ChainMap`对象接受`kwargs`和`defaults`作为参数。这个顺序保证了在实例化类时，手动提供的参数(`kwargs`)优先于`defaults`值。

[*Remove ads*](/account/join/)

## 结论

Python 的`collections`模块中的`ChainMap`提供了一个有效的工具，可以将几个字典作为一个字典来管理。当您有多个字典表示不同的**范围**或**上下文**并且需要设置底层数据的访问优先级时，这个类非常方便。

在一个可更新的视图中对多个字典和映射进行分组，其工作方式非常类似于字典。您可以使用`ChainMap`对象有效地使用几个字典，定义键查找优先级，并管理 Python 中的多个上下文。

**在本教程中，您学习了如何:**

*   在你的 Python 程序中创建 **`ChainMap`实例**
*   探索`ChainMap`和`dict`之间的**差异**
*   使用`ChainMap`将**几个字典作为一个**进行管理
*   用`ChainMap`设置键查找操作的**优先级**

在本教程中，您还编写了一些实际例子，帮助您更好地理解何时以及如何在 Python 代码中使用`ChainMap`。*****