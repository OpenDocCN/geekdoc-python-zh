# 如何在 Python 中向集合中添加元素

> 原文：<https://www.pythonforbeginners.com/basics/how-to-add-an-element-to-a-set-in-python>

python 中的集合用于存储唯一的元素或对象。与元组或列表等其他数据结构不同，集合不允许向其中添加重复值。在本文中，我们将研究用 python 向集合中添加元素的不同方法。我们还将了解一些不允许向集合中添加特定类型对象的用例。

## 使用 Add()方法将元素添加到集合中

add()方法用于向集合中添加新值。在集合上调用时，add()方法将一个要添加到集合中的元素作为输入参数，并将其添加到集合中。当输入元素已经存在于集合中时，什么都不会发生。成功执行后，add()方法返回 None。

我们可以使用 add()方法向集合中添加一个元素，如下所示。

```py
mySet = set([1, 2, 3, 4, 5])
print("Original Set is:", mySet)
mySet.add(6)
print("Set after adding 6 to it:", mySet) 
```

输出:

```py
Original Set is: {1, 2, 3, 4, 5}
Set after adding 6 to it: {1, 2, 3, 4, 5, 6}
```

当我们向集合中添加重复元素时，集合保持不变。你可以在下面的例子中看到。

```py
mySet = set([1, 2, 3, 4, 5])
print("Original Set is:", mySet)
mySet.add(5)
print("Set after adding 5 to it:", mySet)
```

输出:

```py
Original Set is: {1, 2, 3, 4, 5}
Set after adding 5 to it: {1, 2, 3, 4, 5}
```

## 如何向集合中添加多个元素

我们将使用 update()方法向一个集合中添加多个元素。update()方法接受一个或多个 iterable 对象，如 python 字典、元组、列表或集合，并将 iterable 的元素添加到现有集合中。

我们可以将列表元素添加到集合中，如下所示。

```py
mySet = set([1, 2, 3, 4, 5])
print("Original Set is:", mySet)
myList = [6, 7, 8]
print("List of values is:", myList)
mySet.update(myList)
print("Set after adding elements of myList:", mySet)
```

输出:

```py
Original Set is: {1, 2, 3, 4, 5}
List of values is: [6, 7, 8]
Set after adding elements of myList: {1, 2, 3, 4, 5, 6, 7, 8} 
```

要添加两个或更多列表中的元素，我们只需将每个列表作为输入参数传递给 update()方法，如下所示。

```py
mySet = set([1, 2, 3, 4, 5])
print("Original Set is:", mySet)
myList1 = [6, 7, 8]
myList2 = [9, 10]
print("First List of values is:", myList1)
print("Second List of values is:", myList2)
mySet.update(myList1, myList2)
print("Set after adding elements of myList1 and myList2 :", mySet) 
```

输出:

```py
Original Set is: {1, 2, 3, 4, 5}
First List of values is: [6, 7, 8]
Second List of values is: [9, 10]
Set after adding elements of myList1 and myList2 : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
```

就像列表一样，我们可以使用与列表相同的语法从一个或多个元组或集合中添加元素。

当我们试图使用 update()方法将一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)添加到集合中时，只有字典的键被添加到集合中。这可以在下面的例子中观察到。

```py
mySet = set([1, 2, 3, 4, 5])
print("Original Set is:", mySet)
myDict = {6: 36, 7: 49, 8: 64}
print("Dictionary is:", myDict)
mySet.update(myDict)
print("Set after updating :", mySet)
```

输出:

```py
Original Set is: {1, 2, 3, 4, 5}
Dictionary is: {6: 36, 7: 49, 8: 64}
Set after updating : {1, 2, 3, 4, 5, 6, 7, 8}
```

要将字典中的值添加到集合中，我们必须使用 dict.values()方法显式传递值列表，如下所示。

```py
mySet = set([1, 2, 3, 4, 5])
print("Original Set is:", mySet)
myDict = {6: 36, 7: 49, 8: 64}
print("Dictionary is:", myDict)
mySet.update(myDict.values())
print("Set after updating :", mySet)
```

输出:

```py
Original Set is: {1, 2, 3, 4, 5}
Dictionary is: {6: 36, 7: 49, 8: 64}
Set after updating : {64, 1, 2, 3, 4, 5, 36, 49}
```

我们还可以使用解包操作符*向一个集合中添加多个元素。为此，我们将首先解包当前集合和包含要添加到集合中的元素的对象。解包后，我们可以使用所有元素创建一个新的集合，如下所示。

```py
mySet = set([1, 2, 3, 4, 5])
print("Original Set is:", mySet)
myList = [6, 7, 8]
print("List of values is:", myList)
mySet = {*mySet, *myList}
print("Set after updating :", mySet)
```

输出:

```py
Original Set is: {1, 2, 3, 4, 5}
List of values is: [6, 7, 8]
Set after updating : {1, 2, 3, 4, 5, 6, 7, 8}
```

## 将对象添加到集合

就像单个元素一样，我们也可以使用 add()方法将对象添加到集合中。唯一的条件是我们只能添加不可变的对象。例如，我们可以使用 add()方法向列表中添加一个元组，如下所示。

```py
mySet = set([1, 2, 3, 4, 5])
print("Original Set is:", mySet)
myTuple = (6, 7, 8)
print("List of values is:", myTuple)
mySet.add(myTuple)
print("Set after updating :", mySet) 
```

输出:

```py
Original Set is: {1, 2, 3, 4, 5}
List of values is: (6, 7, 8)
Set after updating : {1, 2, 3, 4, 5, (6, 7, 8)}
```

当我们试图向集合中添加一个可变对象(比如列表)时，它会引发 TypeError。这是因为可变对象是不可哈希的，不能添加到集合中。

```py
mySet = set([1, 2, 3, 4, 5])
print("Original Set is:", mySet)
myList = [6, 7, 8]
print("List of values is:", myList)
mySet.add(myList)
print("Set after updating :", mySet)
```

输出:

```py
Original Set is: {1, 2, 3, 4, 5}
List of values is: [6, 7, 8]
Least distance of vertices from vertex 0 is:
{0: 0, 1: 1, 2: 2, 3: 1, 4: 2, 5: 3}
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 5, in <module>
    mySet.add(myList)
TypeError: unhashable type: 'list'
```

TypeError 可以通过使用除了块之外的 [python try 的异常处理来处理。](https://www.pythonforbeginners.com/error-handling/python-try-and-except)

## 如何在 Python 中将字符串作为元素添加到集合中

我们可以使用 add()方法将整个字符串添加到集合中，如下所示。

```py
mySet = set([1, 2, 3, 4, 5])
print("Original Set is:", mySet)
myStr="PythonForBeginners"
print("String is:", myStr)
mySet.add(myStr)
print("Set after updating :", mySet) 
```

输出:

```py
Original Set is: {1, 2, 3, 4, 5}
String is: PythonForBeginners
Set after updating : {1, 2, 3, 4, 5, 'PythonForBeginners'}
```

为了将字符串的字符添加到集合中，我们将使用 update()方法，如下所示。

```py
mySet = set([1, 2, 3, 4, 5])
print("Original Set is:", mySet)
myStr="PythonForBeginners"
print("String is:", myStr)
mySet.update(myStr)
print("Set after updating :", mySet) 
```

输出:

```py
Original Set is: {1, 2, 3, 4, 5}
String is: PythonForBeginners
Set after updating : {1, 2, 3, 4, 5, 'P', 'y', 'F', 'r', 'g', 'B', 's', 'i', 'o', 'h', 'n', 't', 'e'}
```

## 结论

在本文中，我们看到了用 python 向集合中添加一个或多个元素的各种方法。我们还看到了如何向集合中添加字符串或其他不可变对象。请继续关注更多内容丰富的文章。