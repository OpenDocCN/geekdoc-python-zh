# 用 Python 复制字典

> 原文：<https://www.pythonforbeginners.com/dictionary/copy-dictionary-in-python>

在编程时，可能会有我们需要精确复制字典的情况。在本文中，我们将研究用 python 复制字典的不同方法，并将实现这些方法。

## 使用 for 循环复制字典

我们可以通过遍历字典并将当前字典的条目添加到新字典中来创建一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)的副本。对于这个任务，我们可以使用 keys()方法首先获得字典中的键列表，然后我们可以将键和值对插入到另一个字典中。这可以如下进行。

```py
myDict={1:1,2:4,3:9,4:16}
print("Original Dictionary is:")
print(myDict)
newDict={}
keyList=myDict.keys()
for key in keyList:
    newDict[key]=myDict[key]
print("Copied Dictionary is:")
print(newDict)
```

输出:

```py
Original Dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
Copied Dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
```

当使用上述方法复制字典时，对字典中对象的引用被复制。因此，如果字典中有可变对象，并且我们对复制的字典进行了任何更改，那么这种更改将反映在原始字典中。如果我们有一个嵌套的字典，当我们在任何内部字典中进行更改时，该更改将会反映在复制的和原始的字典中。这可以从下面的例子中看出。

```py
myDict={1:{1:1},2:4,3:9,4:16}
print("Original Dictionary is:")
print(myDict)
newDict={}
keyList=myDict.keys()
for key in keyList:
    newDict[key]=myDict[key]
print("Copied Dictionary is:")
print(newDict)
newDict[1][1]=4
print("Copied Dictionary after change:")
print(newDict)
print("Original Dictionary after change:")
print(myDict)
```

输出:

```py
Original Dictionary is:
{1: {1: 1}, 2: 4, 3: 9, 4: 16}
Copied Dictionary is:
{1: {1: 1}, 2: 4, 3: 9, 4: 16}
Copied Dictionary after change:
{1: {1: 4}, 2: 4, 3: 9, 4: 16}
Original Dictionary after change:
{1: {1: 4}, 2: 4, 3: 9, 4: 16}
```

## 使用字典理解复制字典

当我们使用 [list comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 来复制列表时，以类似的方式，我们可以使用字典理解来复制 python 中的字典，如下所示。

```py
myDict={1:1,2:4,3:9,4:16}
print("Original Dictionary is:")
print(myDict)
newDict={key:value for (key,value) in myDict.items()}
print("Copied Dictionary is:")
print(newDict)
```

输出:

```py
Original Dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
Copied Dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
```

同样，复制的字典引用原始字典中的对象，当我们有一个嵌套字典并且我们对内部字典进行任何更改时，这些更改将在原始和复制的字典中反映出来。这可以从下面的例子中看出。

```py
myDict={1:{1:1},2:4,3:9,4:16}
print("Original Dictionary is:")
print(myDict)
newDict={key:value for (key,value) in myDict.items()}
print("Copied Dictionary is:")
print(newDict)
newDict[1][1]=4
print("Copied Dictionary after change:")
print(newDict)
print("Original Dictionary after change:")
print(myDict) 
```

输出:

```py
Original Dictionary is:
{1: {1: 1}, 2: 4, 3: 9, 4: 16}
Copied Dictionary is:
{1: {1: 1}, 2: 4, 3: 9, 4: 16}
Copied Dictionary after change:
{1: {1: 4}, 2: 4, 3: 9, 4: 16}
Original Dictionary after change:
{1: {1: 4}, 2: 4, 3: 9, 4: 16}
```

## 使用 dict.copy()方法

我们可以使用 dict.copy()方法来复制字典。在字典上调用 dict.copy()方法时，会返回原始字典的浅层副本。我们可以使用 dict.copy()复制字典，如下所示。

```py
myDict={1:1,2:4,3:9,4:16}
print("Original Dictionary is:")
print(myDict)
newDict=myDict.copy()
print("Copied Dictionary is:")
print(newDict)
```

输出:

```py
Original Dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
Copied Dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
```

由于所创建的字典副本是浅层副本，所以新字典具有对原始字典的对象的引用。当我们在任一字典中对任何具有原始数据类型的键值对进行更改时，该更改不会显示在另一个字典中，但是当字典的值中包含可变对象时，在原始和复制的字典中都可以看到更改。例如，如果我们使用 dict.copy()方法复制了一个嵌套字典，并对复制的字典中的嵌套字典进行了更改，这些更改将在原始字典中可见。同样，当我们对原始字典进行更改时，它会反映在复制的字典中。这可以从下面的例子中看出。

```py
myDict={1:{1:1},2:4,3:9,4:16}
print("Original Dictionary is:")
print(myDict)
newDict=myDict.copy()
print("Copied Dictionary is:")
print(newDict)
newDict[1][1]=4
print("Copied Dictionary after change:")
print(newDict)
print("Original Dictionary after change:")
print(myDict)
```

输出:

```py
Original Dictionary is:
{1: {1: 1}, 2: 4, 3: 9, 4: 16}
Copied Dictionary is:
{1: {1: 1}, 2: 4, 3: 9, 4: 16}
Copied Dictionary after change:
{1: {1: 4}, 2: 4, 3: 9, 4: 16}
Original Dictionary after change:
{1: {1: 4}, 2: 4, 3: 9, 4: 16}
```

## 使用 copy.copy()方法

创建字典浅层副本的另一种方法是使用 copy 模块中的 copy()方法。我们将要复制的字典作为参数传递给 copy.copy()方法，它创建作为参数传递的字典的浅层副本，并返回对它的引用。这可以如下进行。

```py
import copy
myDict={1:1,2:4,3:9,4:16}
print("Original Dictionary is:")
print(myDict)
newDict=copy.copy(myDict)
print("Copied Dictionary is:")
print(newDict)
```

输出:

```py
Original Dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
Copied Dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
```

使用 copy.copy()方法创建的字典与使用 dict.copy()方法创建的字典工作方式相似。因此，如果存在嵌套字典，并且我们对任何内部字典进行了更改，则更改将会反映在复制的和原始的字典中。这可以从下面的例子中看出。

```py
import copy
myDict={1:{1:1},2:4,3:9,4:16}
print("Original Dictionary is:")
print(myDict)
newDict=copy.copy(myDict)
print("Copied Dictionary is:")
print(newDict)
newDict[1][1]=4
print("Copied Dictionary after change:")
print(newDict)
print("Original Dictionary after change:")
print(myDict)
```

输出:

```py
Original Dictionary is:
{1: {1: 1}, 2: 4, 3: 9, 4: 16}
Copied Dictionary is:
{1: {1: 1}, 2: 4, 3: 9, 4: 16}
Copied Dictionary after change:
{1: {1: 4}, 2: 4, 3: 9, 4: 16}
Original Dictionary after change:
{1: {1: 4}, 2: 4, 3: 9, 4: 16}
```

## 使用 copy.deepcopy()方法

要创建一个不引用原始字典中对象的字典副本，我们可以使用复制模块中的 deepcopy()方法，如下所示。

```py
import copy
myDict={1:1,2:4,3:9,4:16}
print("Original Dictionary is:")
print(myDict)
newDict=copy.deepcopy(myDict)
print("Copied Dictionary is:")
print(newDict)
```

输出:

```py
Original Dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
Copied Dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
```

deepcopy()方法递归地访问字典中的所有对象，并为每个元素创建一个新对象。当我们使用 copy.deepcopy()方法复制字典时，没有引用被复制。当我们对复制的字典中的对象进行任何更改时，即使它是一个嵌套字典，也不会对原始字典进行任何更改。类似地，**当我们对原始字典进行更改时，这些更改不会反映在复制的字典**中。这可以从下面的例子中看出。

```py
import copy
myDict={1:{1:1},2:4,3:9,4:16}
print("Original Dictionary is:")
print(myDict)
newDict=copy.deepcopy(myDict)
print("Copied Dictionary is:")
print(newDict)
newDict[1][1]=4
print("Copied Dictionary after change:")
print(newDict)
print("Original Dictionary after change:")
print(myDict)
```

输出:

```py
Original Dictionary is:
{1: {1: 1}, 2: 4, 3: 9, 4: 16}
Copied Dictionary is:
{1: {1: 1}, 2: 4, 3: 9, 4: 16}
Copied Dictionary after change:
{1: {1: 4}, 2: 4, 3: 9, 4: 16}
Original Dictionary after change:
{1: {1: 1}, 2: 4, 3: 9, 4: 16}
```

## 结论

在本文中，我们看到了用 python 复制字典的不同方法。我们还看到，dict.copy()和 copy.copy()方法创建一个复制的字典，它引用原始字典的对象，而我们可以使用 copy.deepcopy()方法创建一个复制的字典，它不引用原始字典中的元素。请继续关注更多内容丰富的文章。