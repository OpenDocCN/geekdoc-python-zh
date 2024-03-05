# Python 参数和 kwargs:去神秘化

> 原文：<https://realpython.com/python-kwargs-and-args/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python args 和 kwargs:揭秘**](/courses/python-kwargs-and-args/)

有时候，当你查看 Python 中的函数定义时，你可能会发现它带有两个奇怪的参数: **`*args`** 和 **`**kwargs`** 。如果你想知道这些特殊的[变量](https://realpython.com/python-variables/)是什么，或者为什么你的 IDE 在 [`main()`](https://realpython.com/python-main-function/) 中定义它们，那么这篇文章就是为你准备的。您将学习如何在 Python 中使用 args 和 kwargs 来增加函数的灵活性。

**到文章结束，你就知道:**

*   `*args`和`**kwargs`到底是什么意思
*   如何在函数定义中使用`*args`和`**kwargs`
*   如何使用单个星号(`*`)解包可重复项
*   如何使用两个星号(`**`)解包字典

本文假设你已经知道如何定义 Python 函数和使用 T2 列表和字典。

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 向函数传递多个参数

**`*args`** 和 **`**kwargs`** 允许您向一个函数传递多个参数或关键字参数。考虑下面的例子。这是一个简单的函数，它接受两个参数并返回它们的和:

```py
def my_sum(a, b):
    return a + b
```

这个函数工作正常，但是它仅限于两个参数。如果您需要对不同数量的参数求和，而传递的参数的具体数量只能在运行时确定，该怎么办？创建一个函数，不管传递给它的整数有多少，它都可以对所有整数求和，这不是很好吗？

[*Remove ads*](/account/join/)

## 在函数定义中使用 Python args 变量

有几种方法可以将不同数量的参数传递给函数。对于有收藏经验的人来说，第一种方式通常是最直观的。你只需将所有参数的列表或集合传递给你的函数。所以对于`my_sum()`，你可以传递一个需要相加的所有整数的列表:

```py
# sum_integers_list.py
def my_sum(my_integers):
    result = 0
    for x in my_integers:
        result += x
    return result

list_of_integers = [1, 2, 3]
print(my_sum(list_of_integers))
```

这个实现是可行的，但是无论何时调用这个函数，都需要创建一个参数列表来传递给它。这可能不太方便，尤其是如果您事先不知道应该进入列表的所有值。

这就是`*args`真正有用的地方，因为它允许您传递不同数量的位置参数。举以下例子:

```py
# sum_integers_args.py
def my_sum(*args):
    result = 0
    # Iterating over the Python args tuple
    for x in args:
        result += x
    return result

print(my_sum(1, 2, 3))
```

在本例中，您不再向`my_sum()`传递列表。相反，你要传递三个不同的位置参数。`my_sum()`获取输入中提供的所有参数，并将它们打包到一个名为`args`的可迭代对象中。

注意 **`args`只是一个名字。**你不需要使用`args`这个名字。你可以选择任何你喜欢的名字，比如`integers`:

```py
# sum_integers_args_2.py
def my_sum(*integers):
    result = 0
    for x in integers:
        result += x
    return result

print(my_sum(1, 2, 3))
```

即使您将 iterable 对象作为`integers`而不是`args`传递，该函数仍然有效。这里重要的是你使用**拆包操作员** ( `*`)。

请记住，使用解包操作符`*`得到的可迭代对象是[，不是`list`，而是`tuple`](https://realpython.com/python-lists-tuples/) 。一个`tuple`和一个`list`相似，它们都支持切片和迭代。然而，元组至少在一个方面非常不同:列表是[可变的](https://realpython.com/courses/immutability-python/)，而元组不是。要对此进行测试，请运行以下代码。这个脚本试图改变一个列表的值:

```py
# change_list.py
my_list = [1, 2, 3]
my_list[0] = 9
print(my_list)
```

位于列表第一个索引处的值应更新为`9`。如果您执行这个脚本，您将看到列表确实被修改了:

```py
$ python change_list.py
[9, 2, 3]
```

第一个值不再是`0`，而是更新后的值`9`。现在，试着对一个元组做同样的事情:

```py
# change_tuple.py
my_tuple = (1, 2, 3)
my_tuple[0] = 9
print(my_tuple)
```

在这里，您可以看到相同的值，除了它们作为一个元组保存在一起。如果您尝试执行这个脚本，您将看到 Python 解释器返回一个[错误](https://realpython.com/python-exceptions/):

```py
$ python change_tuple.py
Traceback (most recent call last):
 File "change_tuple.py", line 3, in <module>
 my_tuple[0] = 9
TypeError: 'tuple' object does not support item assignment
```

这是因为 tuple 是一个不可变的对象，它的值在赋值后不能改变。在处理元组和`*args`时，请记住这一点。

## 在函数定义中使用 Python kwargs 变量

好了，现在你已经明白了`*args`是干什么的，但是`**kwargs`呢？`**kwargs`的工作方式与`*args`相似，但是它不接受位置参数，而是接受关键字(或名为的**)参数。举以下例子:**

```py
# concatenate.py
def concatenate(**kwargs):
    result = ""
    # Iterating over the Python kwargs dictionary
    for arg in kwargs.values():
        result += arg
    return result

print(concatenate(a="Real", b="Python", c="Is", d="Great", e="!"))
```

当您执行上面的脚本时，`concatenate()`将遍历 Python kwargs [字典](https://realpython.com/python-dicts/)并连接它找到的所有值:

```py
$ python concatenate.py
RealPythonIsGreat!
```

和`args`一样，`kwargs`只是一个名字，想怎么改就怎么改。同样，这里重要的是使用**拆包操作符** ( `**`)。

所以，前面的例子可以写成这样:

```py
# concatenate_2.py
def concatenate(**words):
    result = ""
    for arg in words.values():
        result += arg
    return result

print(concatenate(a="Real", b="Python", c="Is", d="Great", e="!"))
```

注意，在上面的例子中，iterable 对象是一个标准的`dict`。如果您[遍历字典](https://realpython.com/iterate-through-dictionary-python/)并想要返回它的值，如示例所示，那么您必须使用`.values()`。

事实上，如果您忘记使用这个方法，您会发现自己正在遍历 Python kwargs 字典的**键**，如下例所示:

```py
# concatenate_keys.py
def concatenate(**kwargs):
    result = ""
    # Iterating over the keys of the Python kwargs dictionary
    for arg in kwargs:
        result += arg
    return result

print(concatenate(a="Real", b="Python", c="Is", d="Great", e="!"))
```

现在，如果您尝试执行这个示例，您会注意到以下输出:

```py
$ python concatenate_keys.py
abcde
```

如您所见，如果您不指定`.values()`，您的函数将迭代您的 Python kwargs 字典的键，返回错误的结果。

[*Remove ads*](/account/join/)

## 函数中的参数排序

既然您已经了解了`*args`和`**kwargs`的用途，那么您就可以开始编写接受不同数量输入参数的函数了。但是，如果您想创建一个函数，同时接受可变数量的位置*和*命名参数，该怎么办呢？

在这种情况下，你必须记住**订单计数**。正如非默认参数必须在默认参数之前一样，`*args`必须在`**kwargs`之前。

概括地说，参数的正确顺序是:

1.  标准参数
2.  `*args`论据
3.  `**kwargs`论据

例如，这个函数定义是正确的:

```py
# correct_function_definition.py
def my_function(a, b, *args, **kwargs):
    pass
```

`*args`变量适当地列在`**kwargs`之前。但是如果你试图改变参数的顺序呢？例如，考虑以下函数:

```py
# wrong_function_definition.py
def my_function(a, b, **kwargs, *args):
    pass
```

现在，在函数定义中，`**kwargs`在`*args`之前。如果您尝试运行这个例子，您将从解释器收到一个错误:

```py
$ python wrong_function_definition.py
 File "wrong_function_definition.py", line 2
 def my_function(a, b, **kwargs, *args):
 ^
SyntaxError: invalid syntax
```

在这种情况下，由于`*args`在`**kwargs`之后，Python 解释器抛出一个 [`SyntaxError`](https://realpython.com/invalid-syntax-python/) 。

## 带星号的拆包符:`*` & `**`

现在，您可以使用`*args`和`**kwargs`来定义接受不同数量输入参数的 Python 函数。让我们更深入地了解一下**开箱操作员**。

Python 2 中引入了单星和双星号解包操作符。在 3.5 版本中，由于 [PEP 448](https://www.python.org/dev/peps/pep-0448/) ，它们变得更加强大。简而言之，解包操作符是从 Python 中的 iterable 对象解包值的操作符。单星号操作符`*`可以用在 Python 提供的任何 iterable 上，而双星号操作符`**`只能用在字典上。

让我们从一个例子开始:

```py
# print_list.py
my_list = [1, 2, 3]
print(my_list)
```

这段代码定义了一个列表，然后将它打印到标准输出中:

```py
$ python print_list.py
[1, 2, 3]
```

请注意列表是如何打印的，以及相应的括号和逗号。

现在，尝试将解包操作符`*`添加到列表名称的前面:

```py
# print_unpacked_list.py
my_list = [1, 2, 3]
print(*my_list)
```

这里，`*`操作符告诉`print()`首先解包列表。

在这种情况下，输出不再是列表本身，而是列表的内容:

```py
$ python print_unpacked_list.py
1 2 3
```

你能看出这次执行和`print_list.py`的区别吗？`print()`没有使用列表，而是使用了三个独立的参数作为输入。

您会注意到的另一件事是，在`print_unpacked_list.py`中，您使用解包操作符`*`来调用函数，而不是在函数定义中。在这种情况下，`print()`接受列表中的所有项目，就像它们是单个参数一样。

您也可以使用这个方法来调用您自己的函数，但是如果您的函数需要特定数量的参数，那么您解包的 iterable 必须具有相同数量的参数。

要测试此行为，请考虑以下脚本:

```py
# unpacking_call.py
def my_sum(a, b, c):
    print(a + b + c)

my_list = [1, 2, 3]
my_sum(*my_list)
```

这里，`my_sum()`明确声明`a`、`b`和`c`是必需参数。

如果您运行这个脚本，您将得到`my_list`中三个数字的总和:

```py
$ python unpacking_call.py
6
```

`my_list`中的 3 个元素与`my_sum()`中所需的参数完全匹配。

现在看看下面的脚本，其中`my_list`有 4 个参数，而不是 3 个:

```py
# wrong_unpacking_call.py
def my_sum(a, b, c):
    print(a + b + c)

my_list = [1, 2, 3, 4]
my_sum(*my_list)
```

在这个例子中，`my_sum()`仍然只需要三个参数，但是`*`操作符从列表中得到 4 个条目。如果您尝试执行这个脚本，您将会看到 Python 解释器无法运行它:

```py
$ python wrong_unpacking_call.py
Traceback (most recent call last):
 File "wrong_unpacking_call.py", line 6, in <module>
 my_sum(*my_list)
TypeError: my_sum() takes 3 positional arguments but 4 were given
```

当您使用`*`操作符解包一个列表并将参数传递给一个函数时，就好像您在单独传递每个参数一样。这意味着您可以使用多个解包操作符从几个列表中获取值，并将它们全部传递给一个函数。

要测试此行为，请考虑以下示例:

```py
# sum_integers_args_3.py
def my_sum(*args):
    result = 0
    for x in args:
        result += x
    return result

list1 = [1, 2, 3]
list2 = [4, 5]
list3 = [6, 7, 8, 9]

print(my_sum(*list1, *list2, *list3))
```

如果您运行这个例子，所有三个列表都将被解包。每个单独的项目被传递到`my_sum()`，产生以下输出:

```py
$ python sum_integers_args_3.py
45
```

解包操作符还有其他方便的用途。例如，假设您需要将一个列表分成三个不同的部分。输出应该显示第一个值、最后一个值以及两者之间的所有值。使用解包操作符，只需一行代码就可以完成:

```py
# extract_list_body.py
my_list = [1, 2, 3, 4, 5, 6]

a, *b, c = my_list

print(a)
print(b)
print(c)
```

本例中`my_list`包含 6 项。第一个变量分配给`a`，最后一个分配给`c`，所有其他值都被打包到一个新的列表`b`中。如果您运行[脚本](https://realpython.com/run-python-scripts/)，`print()`将向您显示，您的三个变量具有您所期望的值:

```py
$ python extract_list_body.py
1
[2, 3, 4, 5]
6
```

使用解包操作符`*`可以做的另一件有趣的事情是拆分任何可迭代对象的项目。如果您需要合并两个列表，这可能非常有用，例如:

```py
# merging_lists.py
my_first_list = [1, 2, 3]
my_second_list = [4, 5, 6]
my_merged_list = [*my_first_list, *my_second_list]

print(my_merged_list)
```

解包操作符`*`被加在`my_first_list`和`my_second_list`的前面。

如果您运行这个脚本，您会看到结果是一个合并的列表:

```py
$ python merging_lists.py
[1, 2, 3, 4, 5, 6]
```

您甚至可以使用解包操作符`**`合并两个不同的字典:

```py
# merging_dicts.py
my_first_dict = {"A": 1, "B": 2}
my_second_dict = {"C": 3, "D": 4}
my_merged_dict = {**my_first_dict, **my_second_dict}

print(my_merged_dict)
```

这里，要合并的 iterables 是`my_first_dict`和`my_second_dict`。

执行这段代码会输出一个合并的字典:

```py
$ python merging_dicts.py
{'A': 1, 'B': 2, 'C': 3, 'D': 4}
```

记住`*`操作符作用于*任何*可迭代对象。它也可以用来打开[管柱](https://realpython.com/python-strings/):

```py
# string_to_list.py
a = [*"RealPython"]
print(a)
```

在 Python 中，字符串是可迭代的对象，所以`*`会将其解包，并将所有单个值放在一个列表`a`中:

```py
$ python string_to_list.py
['R', 'e', 'a', 'l', 'P', 'y', 't', 'h', 'o', 'n']
```

前面的例子看起来很棒，但是当你使用这些操作符时，记住第七条规则很重要，Tim Peters 的[*Python 的禅宗*](https://www.python.org/dev/peps/pep-0020/):*可读性很重要*。

要了解原因，请考虑以下示例:

```py
# mysterious_statement.py
*a, = "RealPython"
print(a)
```

有一个解包操作符`*`，后面跟着一个变量、一个逗号和一个赋值。一条线装了这么多东西！实际上，这段代码与前面的例子没有什么不同。它只接受字符串`RealPython`并将所有条目分配给新列表`a`，这要感谢拆包操作符`*`。

`a`后面的逗号起了作用。将解包操作符用于变量赋值时，Python 要求结果变量要么是列表，要么是元组。通过后面的逗号，您定义了一个只有一个命名变量`a`的元组，这个变量就是列表`['R', 'e', 'a', 'l', 'P', 'y', 't', 'h', 'o', 'n']`。



您永远看不到 Python 在这个操作中创建的元组，因为您将[元组解包](https://realpython.com/python-lists-tuples/#tuple-assignment-packing-and-unpacking)与解包操作符`*`结合使用。

如果在赋值的左边命名第二个变量，Python 会将字符串的最后一个字符赋给第二个变量，同时收集列表中所有剩余的字符`a`:

>>>

```py
>>> *a, b = "RealPython"

>>> b
"n"

>>> type(b)
<class 'str'>

>>> a
["R", "e", "a", "l", "P", "y", "t", "h", o"]

>>> type(a)
<class 'list'>
```

如果您以前使用过元组解包，那么当您对第二个命名变量使用该操作时，结果可能会更熟悉，如上所示。但是，如果您想将可变长度 iterable 的所有项解包到一个变量`a`中，那么您需要添加逗号(`,` ) *，而不需要*命名第二个变量。然后 Python 会将所有项目解包到第一个命名变量中，这是一个列表。

虽然这是一个巧妙的技巧，但许多 python 爱好者并不认为这段代码可读性很强。因此，最好少用这种结构。

[*Remove ads*](/account/join/)

## 结论

您现在可以使用 **`*args`** 和 **`**kwargs`** 在函数中接受可变数量的参数。您还了解了更多关于解包操作符的知识。

你已经学会了:

*   `*args`和`**kwargs`到底是什么意思
*   如何在函数定义中使用`*args`和`**kwargs`
*   如何使用单个星号(`*`)解包可重复项
*   如何使用两个星号(`**`)解包字典

如果你还有问题，不要犹豫，在下面的评论区联系我们！要了解 Python 中星号用法的更多信息，请看一下 Trey Hunner 关于这个主题的文章。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python args 和 kwargs:揭秘**](/courses/python-kwargs-and-args/)*****