# 查找列表中元素的索引

> 原文：<https://www.pythonforbeginners.com/basics/find-the-index-of-an-element-in-a-list>

在一个列表中搜索一个元素的所有出现是一项单调乏味的任务。在本文中，我们将尝试查找列表中某个元素的索引。我们将研究不同的方法来查找列表中元素的第一个、最后一个和其他出现的位置。

## 使用 index()方法查找列表中元素的索引

我们可以使用 index()方法来查找列表中第一个出现的元素。index()方法将元素作为第一个输入参数，这是必需的。它有两个可选参数，这两个参数是列表中搜索开始和停止的索引。如果输入元素存在于指定索引之间的列表中，则 index()方法返回它第一次出现的元素的索引。我们可以在下面的例子中观察到这一点。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
myNum = 2
print("List is:", myList)
print("Number is:", myNum)
index = myList.index(myNum)
print("Index of {} is {}".format(myNum, index)) 
```

输出:

```py
List is: [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
Number is: 2
Index of 2 is 1
```

如果元素不在列表中指定的索引之间，index()方法将引发 ValueError。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
myNum = 100
print("List is:", myList)
print("Number is:", myNum)
index = myList.index(myNum)
print("Index of {} is {}".format(myNum, index)) 
```

输出:

```py
List is: [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
Number is: 100
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 5, in <module>
    index = myList.index(myNum)
ValueError: 100 is not in list 
```

我们可以通过使用除了块之外的 [python try 的异常处理来处理`ValueError`，如下所示。](https://www.pythonforbeginners.com/error-handling/python-try-and-except)

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
myNum = 100
print("List is:", myList)
print("Number is:", myNum)
try:
    index = myList.index(myNum)
    print("Index of {} is {}".format(myNum, index))
except ValueError:
    print("{} is not in the list.".format(myNum)) 
```

输出:

```py
List is: [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
Number is: 100
100 is not in the list.
```

我们可以使用 index()方法找到列表中某个元素的第一个索引，但是我们不能使用它来找到列表中任何元素的所有匹配项。我们将使用 for 循环来迭代列表，以找到列表中任何元素的所有匹配项。

## 使用 for 循环查找列表中元素的索引

要使用 for 循环查找列表中元素的索引，我们只需遍历列表并检查每个元素。在迭代过程中，如果我们找到了需要找到索引的元素，我们将简单地如下打印索引。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
myNum = 3
print("List is:", myList)
print("Number is:", myNum)
index = -1  # some invalid value
listLen = len(myList)
for i in range(listLen):
    if myList[i] == myNum:
        index = i
        break
if index == -1:
    print("{} not found in the list".format(myNum))
else:
    print("Index of {} is {}".format(myNum, index))
```

输出:

```py
List is: [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
Number is: 3
Index of 3 is 2
```

## 获取列表中最后一个出现的元素

要从元素存在的结尾找到索引，我们可以简单地遍历循环并维护最后找到元素的结果索引。只要找到元素，我们就会更新结果索引。这样，在列表的完整迭代之后，我们将拥有结果索引中元素的最后一个索引。这可以如下进行。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
myNum = 3
print("List is:", myList)
print("Number is:", myNum)
index = -1  # some invalid value
listLen = len(myList)
for i in range(listLen):
    if myList[i] == myNum:
        index = i
if index == -1:
    print("{} not found in the list".format(myNum))
else:
    print("Last index of {} is {}".format(myNum, index)) 
```

输出:

```py
List is: [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
Number is: 3
Last index of 3 is 9
```

上面的方法要求我们每次都要迭代整个列表，以找到元素在列表中出现的最后一个索引。为了避免这种情况，我们可以从末尾开始向后迭代列表。这里我们将使用索引，在找到元素后，我们将通过将负索引添加到列表的长度来获得元素的正索引，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
myNum = 3
print("List is:", myList)
print("Number is:", myNum)
index = -1  # some invalid value
listLen = len(myList)
for i in range(-1, -listLen, -1):
    if myList[i] == myNum:
        index = listLen+i
        break
if index == -1:
    print("{} not found in the list".format(myNum))
else:
    print("Last index of {} is {}".format(myNum, index)) 
```

输出:

```py
List is: [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
Number is: 3
Last index of 3 is 9
```

## 获取列表中元素的所有匹配项

要查找列表中某个元素的所有出现，我们可以使用 for 循环遍历列表，并打印该元素在列表中出现的位置的索引，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
myNum = 3
print("List is:", myList)
print("Number is:", myNum)
indices = [] 
listLen = len(myList)
for i in range(listLen):
    if myList[i] == myNum:
        indices.append(i)
if indices is None:
    print("{} not found in the list".format(myNum))
else:
    print("Indices of {} are {}".format(myNum, indices)) 
```

输出:

```py
List is: [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
Number is: 3
Indices of 3 are [2, 9]
```

不使用 for 循环，我们可以使用 [list comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 来查找元素出现的所有索引，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
myNum = 2
print("List is:", myList)
print("Number is:", myNum)
listLen = len(myList)
indices = [i for i in range(listLen) if myList[i] == myNum]
if indices is None:
    print("{} not found in the list".format(myNum))
else:
    print("Indices of {} are {}".format(myNum, indices)) 
```

输出:

```py
List is: [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 46, 67, 23]
Number is: 2
Indices of 2 are [1, 8]
```

## 结论

在本文中，我们使用了不同的方法来查找列表中元素的索引。我们还研究了查找列表中所有元素的方法。请继续关注更多内容丰富的文章。