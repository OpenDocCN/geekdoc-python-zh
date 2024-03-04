# Python 中的集合理解

> 原文：<https://www.pythonforbeginners.com/basics/set-comprehension-in-python>

在 python 编程中，我们使用不同的数据结构，如列表、元组、集合和字典。我们经常从程序中现有的对象创建新的列表、集合或字典。在本文中，我们将研究集合理解，并了解如何在 python 中使用它从 Python 中的现有对象创建新的集合。我们还将看一些 Python 中集合理解的例子。

## Python 中的集合理解是什么？

集合理解是一种使用列表、集合或元组等其他可迭代元素在 python 中创建集合的方法。就像我们使用[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)来创建列表一样，我们可以使用集合理解而不是 for 循环来创建一个新的集合并向其中添加元素。

## Python 中集合理解的语法

集合理解的语法如下。

newSet= **{** 表达式**为**中的元素**可迭代 **}****

**语法描述:**

*   **iterable** 可以是 Python 中的任何 iterable 对象或数据结构，我们必须使用其中的元素来创建新的集合。
*   **元素**表示 iterable 中必须包含在集合中的元素。
*   **表达式**可以是从**元素**导出的任何数学表达式。
*   **新集合**是新集合的名称，它必须从**可迭代**的元素中创建。

让我们用一个例子来讨论这个语法。在下面的例子中，我们得到了一个由 10 个整数组成的列表。我们必须创建这些整数的三元组。这可以使用集合理解来完成，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
newSet = {element*3 for element in myList}
print("The existing list is:")
print(myList)
print("The Newly Created set is:")
print(newSet) 
```

输出:

```py
The existing list is:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
The Newly Created set is:
{3, 6, 9, 12, 15, 18, 21, 24, 27, 30} 
```

在上面的例子中，我们得到了一个由 10 个整数组成的列表，并且我们已经创建了一组由给定列表中的元素组成的三元组。在语句“newSet = { element * 3 for element in my list }”中，Set comprehension 用于创建包含三个一组的 ***myList*** 中的元素的*。*

*如果您将代码与语法进行比较以理解集合，可以得出以下结论。*

1.  *mySet 是其元素已经被用来创建新集合的集合。因此 **mySet** 被用来代替 iterable。*
2.  *我们已经创建了一个新的集合，它包含了我的列表中的三个元素。因此，**元素*3** 被用来代替**表达式**。*

*我们也可以在集合理解中使用条件句。在集合理解中使用条件语句的语法如下。*

*newSet= **{** 表达式 **for** 元素 **in** iterable **if** 条件 **}***

***语法描述:***

*   ***iterable** 可以是 Python 中的任何 iterable 对象或数据结构，我们必须使用其中的元素来创建新的集合。*
*   ***条件**是一个条件表达式，使用*
*   ***元素**表示 iterable 中必须包含在集合中的元素。*
*   ***表达式**可以是从**元素**导出的任何数学表达式。*
*   ***新集合**是新集合的名称，它必须从**可迭代**的元素中创建。*

*让我们用一个例子来讨论这个语法。在下面的例子中，我们得到了一个由 10 个整数组成的列表。我们必须创建一组偶数的三元组。这可以使用集合理解来完成，如下所示。*

```py
*`myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
newSet = {element*3 for element in myList if element % 2 ==0}
print("The existing list is:")
print(myList)
print("The Newly Created set is:")
print(newSet)`* 
```

*输出:*

```py
*`The existing list is:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
The Newly Created set is:
{6, 12, 18, 24, 30}`*
```

*在上面的例子中，我们有一个 10 个整数的列表，并且我们已经创建了一个给定列表中偶数元素的三元组。在语句“newSet = { element * 3 for element in my list if element % 2 = = 0 }”中，Set comprehension 用于创建包含 ***myList*** 中偶数元素的平方的 ***newSet*** 。*

*如果我们将代码与用于集合理解的语法进行比较，可以得出以下观察结果。*

1.  *myList 是其元素已被用于创建新集合的列表。因此**我的列表**被用来代替**可重复**。*
2.  *我们必须创建一个新的列表，其中包含三个偶元素。因此，**元素*3** 被用来代替**表达式**。*
3.  *为了只选择偶数元素，我们在**条件**处使用了条件语句“element % 2 == 0”。*

## *集合理解的例子*

*现在我们已经理解了 python 中集合理解的语法，我们将通过一些例子来更好地理解这个概念。*

### *从另一个集合的元素创建一个集合*

*如果必须使用另一个集合的元素创建一个集合，可以通过创建新的集合来实现。创建新集合后，可以使用 add()方法和 for 循环向新集合添加元素。在下面的例子中，我们创建了一个新的集合，它包含了一个现有集合的元素的平方。*

```py
*`mySet = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
newSet = set()
for element in mySet:
   newSet.add(element**2)
print("The existing set is:")
print(mySet)
print("The Newly Created set is:")
print(newSet)`* 
```

*输出:*

```py
*`The existing set is:
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
The Newly Created set is:
{64, 1, 4, 36, 100, 9, 16, 49, 81, 25}`*
```

*在上面的例子中，初始化一个空集，然后使用 add()方法向其中添加元素是低效的。取而代之的是，我们可以使用集合理解直接初始化包含所有元素的新集合，如下所示。*

```py
*`mySet = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
newSet = {element ** 2 for element in mySet}
print("The existing set is:")
print(mySet)
print("The Newly Created set is:")
print(newSet)`* 
```

*输出:*

```py
*`The existing set is:
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
The Newly Created set is:
{64, 1, 4, 36, 100, 9, 16, 49, 81, 25}`*
```

### *基于条件从集合中筛选元素*

*通过对元素应用一些条件，我们可以从旧集合的元素创建一个新集合。要使用 for 循环实现这一点，可以使用条件语句和 add()方法，如下所示。在下面的例子中，我们从一个集合中过滤了偶数。*

```py
*`mySet = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
newSet = set()
for element in mySet:
    if element % 2 == 0:
        newSet.add(element)
print("The existing set is:")
print(mySet)
print("The Newly Created set is:")
print(newSet)`* 
```

*输出:*

```py
*`The existing set is:
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
The Newly Created set is:
{2, 4, 6, 8, 10}`* 
```

*除了 for 循环，您还可以使用集合理解从旧集合中过滤出元素来创建一个新集合，如下所示。*

```py
*`mySet = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
newSet = {element for element in mySet if element % 2 == 0}
print("The existing set is:")
print(mySet)
print("The Newly Created set is:")
print(newSet)`* 
```

*输出:*

```py
*`The existing set is:
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
The Newly Created set is:
{2, 4, 6, 8, 10}`*
```

### *从集合中删除元素*

*如果您必须从集合中删除一些元素，您可以从尚未删除的元素创建一个新集合。之后，您可以将新集合分配给旧集合变量，如下所示。在下面的例子中，我们删除了集合中所有的奇数元素。*

```py
*`mySet = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
print("The existing set is:")
print(mySet)
mySet = {element for element in mySet if element % 2 == 0}
print("The modified set is:")
print(mySet)`* 
```

*输出:*

```py
*`The existing set is:
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
The modified set is:
{2, 4, 6, 8, 10}`*
```

### *更改集合元素的数据类型*

*我们还可以使用集合理解来更改集合元素的数据类型，如下例所示。这里，我们已经将集合中的所有整数元素转换为字符串。*

```py
*`mySet = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
newSet = {str(element) for element in mySet }
print("The existing set is:")
print(mySet)
print("The Newly Created set is:")
print(newSet)`* 
```

*输出:*

```py
*`The existing set is:
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
The Newly Created set is:
{'2', '4', '8', '6', '3', '7', '1', '10', '5', '9'}`*
```

## *结论*

*在本文中，我们讨论了 Python 中的集合理解。我们还查看了它的语法和示例，以便更好地理解这个概念。要了解更多关于其他数据结构的知识，可以阅读这篇关于 Python 中的[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)*