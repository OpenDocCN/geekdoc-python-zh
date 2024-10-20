# 如何在 Python 中检查列表、元组或字典是否为空

> 原文：<https://www.pythoncentral.io/how-to-check-if-a-list-tuple-or-dictionary-is-empty-in-python/>

在 Python 中检查任何列表、字典、集合、字符串或元组是否为空的首选方法是简单地使用一个`if`语句来检查它。

例如，如果我们这样定义一个函数:

```py

def is_empty(any_structure):

if any_structure:

print('Structure is not empty.')

return False

else:

print('Structure is empty.')

return True

```

它会神奇地检测任何内置结构是否是空的。所以如果我们运行这个:

```py

>>> d = {} # Empty dictionary

>>> l = [] # Empty list

>>> ms = set() # Empty set

>>> s = '' # Empty string

>>> t = () # Empty tuple

>>> is_empty(d)

Structure is empty.

True

>>> is_empty(l)

Structure is empty.

True

>>> is_empty(ms)

Structure is empty.

True

>>> is_empty(d)

Structure is empty.

True

>>> is_empty(s)

Structure is empty.

True

>>> is_empty(t)

Structure is empty.

True

```

**那么，如果我们给每一个加一些东西:**

```py

>>> d['element'] = 42

>>> l.append('element')

>>> ms.add('element')

>>> s = 'string'

>>> t = ('element')

```

**我们再次检查每一个，这将是结果:**

```py

>>>is_empty(d)

Structure is not empty.

False

```

如您所见，所有默认的数据结构都被检测为空，方法是在`if`语句中将该结构视为布尔值。

如果数据结构为空，当在布尔上下文中使用时，它“返回”`False`。如果数据结构有元素，当在布尔上下文中使用时，它“返回”`True`。

需要注意的一点是，在字典中有值的“空”键仍然会被算作“非空”。**例如:**

```py

>>> d = {None:'value'}

>>> is_empty(d)

Structure is not empty.

False

```

这是因为即使键是“空”的，它仍然被算作字典中的一个条目，这意味着它不是空的。

“真的这么简单吗？我还需要做什么吗？”，我听到你说。

是的，就是这么简单。Python 很神奇。处理好它。🙂

## **如何不检查列表、元组或字典是否为空**

"如果我想用不同的方法来检查这些结构是否为空呢？"，你说。

好吧，大多数其他方式都不是“错误的”，它们只是不是 Python 做事的方式。例如，如果你想通过使用 Python 的`len`函数来检查一个结构是否为空，那就不会被认为是*Python 的*。

**例如，你可以用这种方法来检查一个列表是否为空:**

```py

if len(l) == 0:

print('Empty!')

```

现在，这肯定没有错，但是为什么要输入所有额外的代码呢？当某个东西被标记为“Python”时，它通常指的是 Python 非常简洁的特性。因此，如果你可以通过缩短代码来节省空间，这通常是更 Pythonic 化的做事方式。

上面的方法可能会被称为“C”做事方法，因为它看起来很像 C 代码。

本文到此为止！

直到我们再次相遇。