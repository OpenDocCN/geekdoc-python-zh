# 理解 Python 字典理解

> 原文：<https://www.askpython.com/python/dictionary/python-dictionary-comprehension>

在本文中，我们将了解如何使用 Python 字典理解来轻松创建字典。

与列表理解类似，python 字典理解在一行中用我们的规范构建了一个字典。这避免了创建一个字典实例，然后使用多个语句填充它的麻烦。

让我们用一些说明性的例子来看看我们可以轻松创建字典的一些方法！

* * *

## 基本的 Python 字典理解

例如，假设我们想要构建一个 **{key: value}** 对的字典，将英文字母字符映射到它们的 ascii 值。

所以，当我们调用`my_dict['a']`时，它必须输出相应的 ascii 值( **97** )。让我们为字母`a-z`这样做。

对于不熟悉 Python 代码的人来说，通常的方法是在填充之前创建一个字典，就像这样。

```py
# Create the dictionary
my_dict = {}

# Now populate it
for i in range(97, 97 + 26):
    # Map character to ascii value
    my_dict[chr(i)] = i

# Print the populated dictionary
print(my_dict)

```

**输出**

```py
{'a': 97, 'b': 98, 'c': 99, 'd': 100, 'e': 101, 'f': 102, 'g': 103, 'h': 104, 'i': 105, 'j': 106, 'k': 107, 'l': 108, 'm': 109, 'n': 110, 'o': 111, 'p': 112, 'q': 113, 'r': 114, 's': 115, 't': 116, 'u': 117, 'v': 118, 'w': 119, 'x': 120, 'y': 121, 'z': 122}

```

虽然上面的代码可以工作，但是我们可以通过调用 Python 字典理解特性使它更“Python 化”，并且只用一行代码就可以做到！

对于字典理解，语法类似于:

```py
my_diict = {key: func(key) for key in something}

```

甚至是这个

```py
my_dict = {func(val): val for val in something}

```

这里，`something`可以是 iterable，产生`key`或者`val`。然后函数`func`将键映射到值，反之亦然。您可以在一行中立即将键映射到值，同时还可以创建字典！

在我们的例子中，`val`是变量`i`，而`func(val)`是函数`chr(i)`。

所以现在，我们的例子可以简化，但仍然可读的代码！

```py
my_dict = {chr(i): i for i in range(97, 97 + 26)}
print(my_dict)

```

**输出**

```py
{'a': 97, 'b': 98, 'c': 99, 'd': 100, 'e': 101, 'f': 102, 'g': 103, 'h': 104, 'i': 105, 'j': 106, 'k': 107, 'l': 108, 'm': 109, 'n': 110, 'o': 111, 'p': 112, 'q': 113, 'r': 114, 's': 115, 't': 116, 'u': 117, 'v': 118, 'w': 119, 'x': 120, 'y': 121, 'z': 122}

```

这给出了与之前相同的输出！很神奇，不是吗？

虽然上面的例子似乎足以说明字典理解的力量，但我们还能做得更多吗？

答案是肯定的，我们甚至可以在字典理解中加入像`if`和`else`这样的条件句！让我们来看看。

## 在词典理解中使用条件句

我们可以用`if`和`else`这样的条件语句来做字典理解。

让我们以第一种情况为例，我们只想使用一个`if`条件。

字典理解的语法如下所示:

```py
my_dict = {key: value for key in iterable if condition}

```

在这种情况下，字典将只为 iterable 中满足条件的元素提供映射。

假设我们想从一个列表中构造一个字典，这个列表只将偶数映射到它的平方，我们可以使用字典理解来完成这个任务。

```py
my_list = [0, 1, 2, 3, 4, 5, 6]

my_dict = {i: i*i for i in my_list if i % 2 == 0}

print(my_dict)

```

**输出**

```py
{0: 0, 2: 4, 4: 16, 6: 36}

```

的确，我们的字典里只有偶数，作为键！

现在让我们也用一个`else`条件来做一个字典理解！

如果还需要`else`条件，我们需要修改语法以便理解。如果我们希望拥有映射到可能不同的值的相同键，语法应该是这样的:

```py
my_dict = {key: (value1 if condition1(value) is True else value2) for key, value in something}

```

让我们考虑使用现有的字典构建一个字典。如果是奇数，新字典的值将为 0。否则，它将简单地使用我们以前的字典中的旧值。

```py
old_dict = {'a': 97, 'b': 98, 'c': 99, 'd': 100}

new_dict = {k: (val if val % 2 == 0 else 0) for k, val in old_dict.items()}

print(new_dict)

```

**输出**

```py
{'a': 0, 'b': 98, 'c': 0, 'd': 100}

```

* * *

## 结论

在本文中，我们通过一些说明性的例子学习了如何理解 Python 字典。

## 参考

*   [关于 Python 字典的 AskPython 文章](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)

* * *