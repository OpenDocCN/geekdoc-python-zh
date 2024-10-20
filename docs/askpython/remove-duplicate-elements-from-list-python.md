# 在 Python 中移除列表中的重复元素

> 原文：<https://www.askpython.com/python/remove-duplicate-elements-from-list-python>

在本文中，我们将看看如何在 Python 中从[列表中删除重复的元素。解决这个问题有多种方法，我们将向您展示其中的一些。](https://www.askpython.com/python/list/python-list)

* * *

## 从列表中移除重复元素的方法–Python

## 1.使用迭代

在 Python 中，要从列表中删除重复的元素，我们可以手动遍历列表，并在新列表中添加一个不存在的元素。否则，我们跳过该元素。

代码如下所示:

```py
a = [2, 3, 3, 2, 5, 4, 4, 6]

b = []

for i in a:
    # Add to the new list
    # only if not present
    if i not in b:
        b.append(i)

print(b)

```

**输出**

```py
[2, 3, 5, 4, 6]

```

同样的代码可以使用 List Comprehension 来编写，以减少代码行数，尽管它本质上与前面的一样。

```py
a = [2 3, 4, 2, 5, 4, 4, 6]
b = []
[b.append(i) for i in a if i not in b]
print(b)

```

这种方法的问题是它有点慢，因为在遍历原始列表的同时，对新列表中的每个元素都进行了比较。

这在计算上很昂贵，我们有其他方法来处理这个问题。只有当列表不是很大时，才应该使用这个选项。否则，请参考其他方法。

## 2.使用 set()

在 Python 中，从列表中删除重复元素的一个简单而快速的方法是使用 Python 的内置`set()`方法将列表元素转换成一个唯一的集合，然后我们可以将它转换成一个删除了所有重复元素的列表。

```py
first_list = [1, 2, 2, 3, 3, 3, 4, 5, 5, 6]

# Convert to a set first
set_list = set(first_list)

# Now convert the set into a List
print(list(set_list))

second_list = [2, 3, 3, 2, 5, 4, 4, 6]

# Does the same as above, in a single line
print(list(set(second_list)))

```

**输出**

```py
[1, 2, 3, 4, 5, 6]
[2, 3, 4, 5, 6]

```

这种方法的问题是，由于我们是从一个无序的集合中创建新的列表，所以原始列表的顺序没有像第二个列表那样得到维护。因此，如果您希望仍然保持相对顺序，您必须避免这种方法。

## 3.保持顺序:使用 OrderedDict

如果您想在 Python 中删除列表中的重复元素时保留顺序，可以使用来自**集合**模块的 **OrderedDict** 类。

更具体地说，我们可以使用`OrderedDict.fromkeys(list)`来获得删除了重复元素的字典，同时仍然保持顺序。然后我们可以使用`list()`方法很容易地[将其转换成一个列表](https://www.askpython.com/python/string/python-convert-string-to-list)。

```py
from collections import OrderedDict

a = [2, 3, 3, 2, 5, 4, 4, 6]

b = list(OrderedDict.fromkeys(a))

print(b)

```

**输出**

```py
[2, 3, 5, 4, 6]

```

**注意**:如果你有 **Python 3.7** 或者更高版本，我们可以用内置的`dict.fromkeys(list)`代替。这样也会保证秩序。

正如您所观察到的，顺序确实得到了维护，因此我们得到了与第一种方法相同的输出。但是这样快多了！这是解决此问题的推荐方案。但是为了便于说明，我们将向您展示用 Python 从列表中删除重复元素的另外两种方法。

## 4.使用 list.count()

`list.count()`方法返回该值出现的次数。我们可以将它与`remove()`方法一起使用来消除任何重复的元素。但是同样，这并不是**而不是**维护了秩序。

请注意，该方法就地修改输入列表，因此更改会反映在列表中。

```py
a = [0, 1, 2, 3, 4, 1, 2, 3, 5]

for i in a:
    if a.count(i) > 1:
        a.remove(i)

print(a)

```

**输出**

```py
[0, 4, 1, 2, 3, 5]

```

一切似乎都很好，不是吗？

但是，上面的代码有一个小问题。

当我们使用 for 循环遍历列表并同时删除元素时，迭代器会跳过一个元素。所以，代码输出依赖于列表元素，如果你幸运的话，你永远不会遇到这个问题。让我们用一个简单的代码来理解这个场景。

```py
a = [1, 2, 3, 2, 5]

for i in a:
    if a.count(i) > 1:
        a.remove(i)
    print(a, i)

print(a)

```

**输出**:

```py
[1, 2, 3, 2, 5] 1
[1, 3, 2, 5] 2
[1, 3, 2, 5] 2
[1, 3, 2, 5] 5
[1, 3, 2, 5]

```

您可以看到 for 循环只执行了四次，并且跳过了 remove()调用后的下一个元素 3。如果您将输入列表作为[1，1，1，1]传递，那么最终的列表将是[1，1]。

**那么，有什么变通方法吗？**

当然有变通办法。在 for 循环中使用列表的副本，但从主列表中移除元素。创建列表副本的一个简单方法是通过切片。下面是在所有情况下都能正常工作的更新代码。

```py
a = [1, 1, 1, 1]

for i in a[:]:  # using list copy for iteration
    if a.count(i) > 1:
        a.remove(i)
    print(a, i)

print(a)

```

输出:

```py
[1, 1, 1] 1
[1, 1] 1
[1] 1
[1] 1
[1]

```

## 5.使用排序()

我们可以使用`sort()`方法对我们在方法 2 中获得的集合进行排序。这也将删除任何重复，同时保持顺序，但比`dict.fromkeys()`方法慢。

```py
a = [0, 1, 2, 3, 4, 1, 2, 3, 5]
b = list(set(a))
b.sort(key=a.index)
print(b)   

```

**输出**

```py
[0, 1, 2, 3, 4, 5]

```

## 6.使用熊猫模块

如果我们正在使用 Pandas 模块，我们可以使用`pandas.drop_duplicates()`方法删除重复项，然后将其转换为一个列表，同时还保留顺序。

```py
import pandas as pd

a = [0, 1, 2, 3, 4, 1, 2, 3, 5]

pd.Series(a).drop_duplicates().tolist()

```

**输出**

```py
[0, 1, 2, 3, 4, 5]

```

* * *

## 参考

*   JournalDev 关于删除重复列表元素的文章
*   [StackOverflow 问题](https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists?page=1&tab=votes#tab-top)

* * *