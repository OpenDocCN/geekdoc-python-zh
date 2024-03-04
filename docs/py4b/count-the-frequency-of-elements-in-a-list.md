# 计算列表中元素的出现频率

> 原文：<https://www.pythonforbeginners.com/lists/count-the-frequency-of-elements-in-a-list>

很多时候，我们需要在 python 中对数据进行定量分析。在本文中，我们将研究一些统计列表中元素出现频率的方法。列表中元素的频率被定义为它在列表中出现的次数。

## 使用 for 循环计算列表中元素的频率

我们可以使用 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)来计算列表中元素的频率。为了执行这个操作，我们将创建一个字典，其中包含来自输入列表的惟一元素作为键，它们的计数作为值。

为了统计列表中元素的频率，首先，我们将创建一个空字典。之后，对于每一个新元素，我们会统计该元素在列表中的出现频率。在获得元素的频率后，我们将把元素及其频率作为键-值对添加到字典中。

我们还将维护一个已访问元素的列表，以过滤掉已经访问过的元素。这可以如下进行。

```py
myList = [1, 2, 3, 4, 1, 3, 46, 7, 2, 3, 5, 6, 10]
frequencyDict = dict()
visited = set()
listLength = len(myList)
for i in range(listLength):
    if myList[i] in visited:
        continue
    else:
        count = 0
        element = myList[i]
        visited.add(myList[i])
        for j in range(listLength - i):
            if myList[j+i] == element:
                count += 1
        frequencyDict[element] = count

print("Input list is:", myList)
print("Frequency of elements is:")
print(frequencyDict) 
```

输出:

```py
Input list is: [1, 2, 3, 4, 1, 3, 46, 7, 2, 3, 5, 6, 10]
Frequency of elements is:
{1: 2, 2: 2, 3: 3, 4: 1, 46: 1, 7: 1, 5: 1, 6: 1, 10: 1}
```

在上面的例子中，我们为每个唯一的元素迭代整个列表。这使得算法效率低下。在最坏的情况下，当列表包含所有唯一元素时，我们将不得不处理所有元素至少 n*(n+1)/2 次，其中 n 是列表的长度。

为了克服这个缺点，我们将修改上面的算法。作为一个改进，我们将只遍历列表一次。为了统计元素的频率，我们将遍历列表，检查每个元素是否已经作为一个键出现在字典中。如果当前元素已经作为一个键出现在字典中，我们将把与该元素相关的计数加 1。如果当前元素还没有作为一个键出现在字典中，我们将添加一个新的条目到字典中，当前元素作为键，1 作为它的关联值。

我们可以如下实现这个算法。

```py
myList = [1, 2, 3, 4, 1, 3, 46, 7, 2, 3, 5, 6, 10]
frequencyDict = dict()
visited = set()
for element in myList:
    if element in visited:
        frequencyDict[element] = frequencyDict[element] + 1
    else:
        frequencyDict[element] = 1
        visited.add(element)

print("Input list is:", myList)
print("Frequency of elements is:")
print(frequencyDict)
```

输出:

```py
Input list is: [1, 2, 3, 4, 1, 3, 46, 7, 2, 3, 5, 6, 10]
Frequency of elements is:
{1: 2, 2: 2, 3: 3, 4: 1, 46: 1, 7: 1, 5: 1, 6: 1, 10: 1}
```

可以使用除了块之外的 [python try 对上述算法进行即兴创作。在这个方法中，我们将使用 try 块来增加元素的频率。每当字典中不存在某个元素时，就会引发一个`KeyError`。这意味着该元素是唯一的元素，它的计数还没有添加到字典中。](https://www.pythonforbeginners.com/error-handling/python-try-and-except)

在 except 块中，我们将捕获`KeyError`并将一个新的键-值对添加到字典中，将当前元素作为键，将 1 作为其关联值。这将按如下方式进行。

```py
myList = [1, 2, 3, 4, 1, 3, 46, 7, 2, 3, 5, 6, 10]
frequencyDict = dict()
for element in myList:
    try:
        frequencyDict[element] = frequencyDict[element] + 1
    except KeyError:
        frequencyDict[element] = 1
print("Input list is:", myList)
print("Frequency of elements is:")
print(frequencyDict) 
```

输出:

```py
Input list is: [1, 2, 3, 4, 1, 3, 46, 7, 2, 3, 5, 6, 10]
Frequency of elements is:
{1: 2, 2: 2, 3: 3, 4: 1, 46: 1, 7: 1, 5: 1, 6: 1, 10: 1}
```

在上述两种算法中，使用 for 循环实现的解决方案对于每种类型的输入列表都具有几乎相同的效率。然而，对于只有少数元素在列表中重复多次的输入列表，使用异常处理的解决方案会执行得更快。

## 使用 counter()方法计算频率

我们可以使用集合模块中的 counter()方法来计算列表中元素的出现频率。counter()方法接受一个 iterable 对象作为输入参数。它返回一个计数器对象，该对象以键值对的形式存储所有元素的频率。我们可以使用 counter()方法来计算列表中元素的频率，如下所示。

```py
import collections

myList = [1, 2, 3, 4, 1, 3, 46, 7, 2, 3, 5, 6, 10]
frequencyDict = collections.Counter(myList)
print("Input list is:", myList)
print("Frequency of elements is:")
print(frequencyDict) 
```

输出:

```py
Input list is: [1, 2, 3, 4, 1, 3, 46, 7, 2, 3, 5, 6, 10]
Frequency of elements is:
Counter({3: 3, 1: 2, 2: 2, 4: 1, 46: 1, 7: 1, 5: 1, 6: 1, 10: 1})
```

## 结论

在本文中，我们讨论了计算列表中元素出现频率的不同方法。要阅读更多关于列表的内容，请阅读这篇关于 python 中的列表理解的文章。请继续关注更多内容丰富的文章。