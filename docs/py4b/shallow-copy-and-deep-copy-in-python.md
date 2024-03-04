# Python 中的浅层复制和深层复制

> 原文：<https://www.pythonforbeginners.com/functions/shallow-copy-and-deep-copy-in-python>

在编程时，我们需要复制现有的数据。当我们使用=操作符将一个变量赋给另一个变量时，赋值操作符不会复制对象，而只是创建一个对同一个对象的新引用。在本文中，我们将学习如何使用 python 中的浅层复制和深层复制来复制对象。我们也将实施一些项目来理解他们的工作。

## python 中如何使用浅层复制？

浅层复制是一种用于创建现有集合对象副本的功能。当我们尝试使用浅层复制来复制集合对象时，它会创建一个新的集合对象，并存储对原始对象的元素的引用。

我们可以通过复制模块在 python 中使用浅层复制。为了执行浅层复制操作，我们使用 copy()模块的 copy()方法。

copy()方法将原始集合对象作为输入，并参考原始集合对象的元素创建一个新的集合对象。然后，它返回对新集合对象的引用。

在下面的程序中，我们将使用 copy()方法创建给定的 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)的副本。

```py
import copy
myDict={1:3,4:9,6:12,5:11}
print("Original Dictionary is:")
print(myDict)
newDict=copy.copy(myDict)
print("New Dictionary is:")
print(newDict)
```

输出:

```py
Original Dictionary is:
{1: 3, 4: 9, 6: 12, 5: 11}
New Dictionary is:
{1: 3, 4: 9, 6: 12, 5: 11}
```

在输出中，我们可以看到我们创建了一个与程序中给出的原始字典相似的字典。

## 浅层复制是如何工作的？

在浅层复制中，当创建新对象时，它引用原始对象的元素。如果我们试图对新创建的对象进行任何更改，这将不会反映在原始对象中，因为对象中的元素不应引用另一个对象，即不应有任何嵌套。

```py
import copy
myDict={1:3,4:9,6:12,5:11}
print("Original Dictionary is:")
print(myDict)
newDict=copy.copy(myDict)
print("New Dictionary is:")
print(newDict)
newDict[1]=5
print("Original Dictionary after change is:")
print(myDict)
print("New Dictionary after change is:")
print(newDict)
```

输出:

```py
Original Dictionary is:
{1: 3, 4: 9, 6: 12, 5: 11}
New Dictionary is:
{1: 3, 4: 9, 6: 12, 5: 11}
Original Dictionary after change is:
{1: 3, 4: 9, 6: 12, 5: 11}
New Dictionary after change is:
{1: 5, 4: 9, 6: 12, 5: 11}
```

如果我们要对原始对象进行任何更改，这也不会反映在原始对象的副本中，因为不应进行嵌套。

```py
import copy
myDict={1:3,4:9,6:12,5:11}
print("Original Dictionary is:")
print(myDict)
newDict=copy.copy(myDict)
print("New Dictionary is:")
print(newDict)
myDict[1]=5
print("Original Dictionary after change is:")
print(myDict)
print("New Dictionary after change is:")
print(newDict)
```

输出:

```py
Original Dictionary is:
{1: 3, 4: 9, 6: 12, 5: 11}
New Dictionary is:
{1: 3, 4: 9, 6: 12, 5: 11}
Original Dictionary after change is:
{1: 5, 4: 9, 6: 12, 5: 11}
New Dictionary after change is:
{1: 3, 4: 9, 6: 12, 5: 11}
```

同样，当我们向原始对象添加任何元素时，它不会对新对象产生任何影响。

```py
import copy
myDict={1:3,4:9,6:12,5:11}
print("Original Dictionary is:")
print(myDict)
newDict=copy.copy(myDict)
print("New Dictionary is:")
print(newDict)
myDict[7]=49
print("Original Dictionary after change is:")
print(myDict)
print("New Dictionary after change is:")
print(newDict)
```

输出:

```py
Original Dictionary is:
{1: 3, 4: 9, 6: 12, 5: 11}
New Dictionary is:
{1: 3, 4: 9, 6: 12, 5: 11}
Original Dictionary after change is:
{1: 3, 4: 9, 6: 12, 5: 11, 7: 49}
New Dictionary after change is:
{1: 3, 4: 9, 6: 12, 5: 11}
```

当对象中存在嵌套时，上面讨论的场景会发生变化。即，当被复制的对象包含其他对象时，嵌套对象上发生的变化在原始对象和复制对象中都是可见的。这可以看如下。

```py
import copy
myDict={1:3,4:9,6:12,5:{10:11}}
print("Original Dictionary is:")
print(myDict)
newDict=copy.copy(myDict)
print("New Dictionary is:")
print(newDict)
myDict[5][10]=49
print("Original Dictionary after change is:")
print(myDict)
print("New Dictionary after change is:")
print(newDict)
```

输出:

```py
Original Dictionary is:
{1: 3, 4: 9, 6: 12, 5: {10: 11}}
New Dictionary is:
{1: 3, 4: 9, 6: 12, 5: {10: 11}}
Original Dictionary after change is:
{1: 3, 4: 9, 6: 12, 5: {10: 49}}
New Dictionary after change is:
{1: 3, 4: 9, 6: 12, 5: {10: 49}}
```

这是因为当我们使用 copy.copy()方法复制一个对象时，只创建了作为参数传递给 copy()方法的对象的副本。不复制对象内部的元素，只复制对这些元素的引用。因此，当只有原始数据类型(如 int、double、string)作为元素出现在原始对象中时，在原始对象中完成的更改对新对象是不可见的，因为这些数据类型是不可变的，并且对于每个更改，都会创建一个新对象。但是在嵌套对象的情况下，引用是不变的，当我们对其中一个对象进行任何更改时，它在另一个对象中是可见的。

## python 中如何使用深度复制？

为了避免在进行浅层复制时讨论的问题，我们将使用 deepcopy()方法。deepcopy()方法递归地创建对象中每个元素的副本，并且不复制引用。这可以如下进行。

```py
import copy
myDict={1:3,4:9,6:12,5:{10:11}}
print("Original Dictionary is:")
print(myDict)
newDict=copy.deepcopy(myDict)
print("New Dictionary is:")
print(newDict)
```

输出:

```py
Original Dictionary is:
{1: 3, 4: 9, 6: 12, 5: {10: 11}}
New Dictionary is:
{1: 3, 4: 9, 6: 12, 5: {10: 11}}
```

使用 deepcopy()后，即使存在嵌套，在原始对象中所做的更改也不会显示在复制的对象中。这可以看如下。

```py
import copy
myDict={1:3,4:9,6:12,5:{10:11}}
print("Original Dictionary is:")
print(myDict)
newDict=copy.deepcopy(myDict)
print("New Dictionary is:")
print(newDict)
myDict[5][10]=49
print("Original Dictionary after change is:")
print(myDict)
print("New Dictionary after change is:")
print(newDict)
```

输出:

```py
Original Dictionary is:
{1: 3, 4: 9, 6: 12, 5: {10: 11}}
New Dictionary is:
{1: 3, 4: 9, 6: 12, 5: {10: 11}}
Original Dictionary after change is:
{1: 3, 4: 9, 6: 12, 5: {10: 49}}
New Dictionary after change is:
{1: 3, 4: 9, 6: 12, 5: {10: 11}}
```

这里我们可以看到，与 copy()不同，当我们使用 deepcopy()复制对象时，在原始对象中所做的更改不会影响复制的对象，反之亦然，因为由 deepcopy()方法创建的对象不包含对原始字典元素的任何引用，而在 copy()方法的情况下，新创建的对象包含对原始对象元素的引用。

## 结论

在本文中，我们讨论了 python 中的浅层复制和深层复制。我们已经看到，当存在嵌套对象时，应该使用 deepcopy()来创建对象的副本。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，并使用异常处理来使程序更加健壮，并以系统的方式处理错误。