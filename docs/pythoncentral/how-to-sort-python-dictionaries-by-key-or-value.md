# 如何按键或值对 Python 字典进行排序

> 原文：<https://www.pythoncentral.io/how-to-sort-python-dictionaries-by-key-or-value/>

注意:如果你想在 Python 中对列表、元组或对象进行排序，请查看本文:[如何在 Python 中对列表或元组进行排序](https://www.pythoncentral.io/how-to-sort-a-list-tuple-or-object-with-sorted-in-python "How to Sort a List, Tuple or Object (with sorted) in Python")

Python 中的`dict` (dictionary)类对象是一个非常通用和有用的容器类型，能够存储一组值并通过键检索它们。值可以是任何类型的对象(字典甚至可以与其他字典嵌套)，键可以是任何对象，只要它是*可散列的*，基本上意味着它是不可变的(因此字符串不是唯一有效的键，但是像列表这样可变的对象永远不能用作键)。与 Python 列表或元组不同，`dict`对象中的键和值对没有任何特定的顺序，这意味着我们可以有一个像这样的`dict`:

```py

numbers = {'first': 1, 'second': 2, 'third': 3, 'Fourth': 4}

```

虽然键-值对在实例化语句中是按照一定的顺序排列的，但是通过调用它的 *`list`* 方法(这将从它的键中创建一个列表),我们可以很容易地看到它们没有按照那个顺序存储:

```py

>>> list(numbers)

['second', 'Fourth', 'third', 'first']

```

## **按关键字排序 Python 字典**

如果我们想按关键字对字典对象进行排序，最简单的方法是使用 Python 内置的 *`sorted`* 方法，该方法接受任何 iterable 并返回一个已排序的值列表(默认情况下按升序)。没有像列表那样对字典进行排序的类方法，但是`sorted`方法的工作方式完全相同。下面是它对我们的字典的作用:

```py

# This is the same as calling sorted(numbers.keys())

>>> sorted(numbers)

['Fourth', 'first', 'second', 'third']

```

我们可以看到这个方法给出了一个按升序排列的键列表，几乎是按字母顺序排列的，这取决于我们定义的“字母顺序”还要注意，我们按照键对键列表*进行了排序——如果我们想按照键对键列表*值*进行排序，或者按照值对键列表*进行排序，我们必须改变使用`sorted`方法的方式。我们一会儿将看看`sorted`的这些不同方面。**

### **按值排序 Python 字典**

与我们使用键的方式相同，我们可以使用 *`sorted`* 按照值对 Python 字典进行排序:

```py

# We have to call numbers.values() here

>>> sorted(numbers.values())

[1, 2, 3, 4]

```

这是默认顺序的值列表，按升序排列。这些都是非常简单的例子，所以现在让我们检查一些稍微复杂一点的情况，在这些情况下我们对我们的`dict`对象进行排序。

### **使用 Python 字典定制排序算法**

如果我们简单地给`sorted`方法字典的键/值作为一个参数，它将执行一个简单的排序，但是通过利用它的其他参数(即`key`和`reverse`)，我们可以让它执行更复杂的排序。

`sorted`的`key`参数(不要与字典的键混淆)允许我们定义在排序项目时使用的特定函数，作为*迭代器*(在我们的`dict`对象中)。在上面的两个例子中，键和值都是要排序的项和用于比较的项，但是如果我们想要使用我们的`dict` *值*对我们的`dict` *键*进行排序，那么我们将通过它的`key`参数告诉`sorted`这样做。比如如下:

```py

# Use the __getitem__ method as the key function

>>> sorted(numbers, key=numbers.__getitem__)

# In order of sorted values: [1, 2, 3, 4]

['first', 'second', 'third', 'Fourth']

```

通过这个语句，我们告诉`sorted`对数字`dict`(它的键)进行排序，并通过使用 numbers 的类方法来检索值来对它们进行排序——本质上我们告诉它“对于*数字*中的每个键，使用*数字*中的相应值进行比较来对其进行排序。”。

我们还可以通过关键字对数字中的*值*进行排序，但是使用`key`参数会更复杂(没有字典方法可以像使用`list.index`方法那样通过使用某个值来返回关键字)。相反，我们可以使用[列表理解](https://www.pythoncentral.io/list-comprehension-in-python/)来保持简单:

```py

# Uses the first element of each tuple to compare

>>> [value for (key, value) in sorted(numbers.items())]

[4, 1, 2, 3]

# In order of sorted keys: ['Fourth', 'first', 'second', 'third']

```

现在要考虑的另一个参数是`reverse`参数。如果这是`True`，顺序将被反转(降序)，否则如果是`False`，它将处于默认(升序)顺序，就这么简单。例如，与前两种排序一样:

```py

>>> sorted(numbers, key=numbers.__getitem__, reverse=True)

['Fourth', 'third', 'second', 'first']

>>> [value for (key, value) in sorted(numbers.items(), reverse=True)]

[3, 2, 1, 4]

```

这些排序仍然相当简单，但让我们看看一些特殊的算法，我们可能会使用字符串或数字来排序我们的字典。

### **用字符串和数字算法排序 Python 字典**

按照字母顺序对字符串进行排序是很常见的，但是使用`sorted`可能不会按照“正确的”字母顺序对我们的`dict`的键/值进行排序，即忽略大小写。为了忽略大小写，我们可以再次利用`key`参数和`str.lower`(或`str.upper`)方法，这样在比较时所有的字符串都是相同的大小写:

```py

# Won't change the items to be returned, only while sorting

>>> sorted(numbers, key=str.lower)

['first', 'Fourth', 'second', 'third']

```

为了将字符串与数字相关联，我们需要一个额外的上下文元素来正确地关联它们。让我们先创建一个新的`dict`对象:

```py

>>> month = dict(one='January',

two='February',

three='March',

four='April',

five='May')

```

它只包含字符串键和值，所以如果没有额外的上下文，就没有办法将月份按正确的顺序排列。为此，我们可以简单地创建另一个字典来将字符串映射到它们的数值，并使用该字典的`__getitem__`方法来比较我们的`month`字典中的值:

```py

>>> numbermap = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}

>>> sorted(month, key=numbermap.__getitem__)

['one', 'two', 'three', 'four', 'five']

```

正如我们所看到的，它仍然返回了第一个参数(month)的排序列表。为了按顺序返回月份，我们将再次使用[列表理解](https://www.pythoncentral.io/list-comprehension-in-python/):

```py

# Assuming the keys in both dictionaries are EXACTLY the same:

>>> [month[i] for i in sorted(month, key=numbermap.__getitem__)]

['January', 'February', 'March', 'April', 'May']

```

如果我们想根据每个字符串中重复字母的数量对键/值字符串进行排序，我们可以定义自己的自定义方法，在`sorted` *键*参数中使用:

```py

def repeats(string):

# Lower the case in the string

string = string.lower()
#获取一组唯一的字母
 uniques = set(string)
# Count 每个唯一字母的最大出现次数
counts =[string . Count(letter)for letter in uniques]
返回最大值(计数)

```

使用该功能可按如下方式进行:

```py

# From greatest to least repeats

>>> sorted(month.values(), key=repeats, reverse=True)

['February', 'January', 'March', 'April', 'May']

```

### **更高级的分类功能**

现在，假设我们有一个字典，记录每个月一个班级的学生人数，如下所示:

```py

>>> trans = dict(January=33, February=30, March=24, April=0, May=7)

```

如果我们想首先用偶数*和奇数*来组织班级规模，我们可以这样定义:

```py

def evens1st(num):

# Test with modulus (%) two

if num == 0:

return -2

# It's an even number, return the value

elif num % 2 == 0:

return num

# It's odd, return the negated inverse

else:

return -1 * (num ** -1)

```

使用`evens1st`排序函数，我们得到以下输出:

```py

# Max class size first

>>> sorted(trans.values(), key=evens1st, reverse=True)

[30, 24, 33, 7, 0]

```

同样，我们可以首先列出奇数个类的大小，然后执行许多其他算法来得到我们想要的排序。还有许多其他复杂的排序方法和技巧可以用于字典(或任何可迭代的对象)，但是现在希望这些例子已经提供了一个很好的起点。