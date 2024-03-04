# 比较 Python 中的两个列表

> 原文：<https://www.pythonforbeginners.com/basics/compare-two-lists-in-python>

在 python 中编程时，为了检查不同的条件，必须经常进行比较。对于单个条件检查，我们可能需要比较两个变量或两组变量。在本文中，我们将尝试用不同的方法来比较 python 中的两个列表。在比较时，我们必须检查两个列表是否包含相同的元素，而不考虑元素在列表中出现的顺序。因此，我们必须打印结果。

## 使用 sort()方法比较两个列表

为了检查两个列表是否包含相同的元素，我们可以首先使用 sort()方法对列表的元素进行排序。然后，我们可以比较两个列表。

为了比较，首先我们将检查列表的长度是否相等。如果长度不相等，列表将被自动视为不同。

如果列表的长度相同，我们将对两个列表进行排序。然后，我们将使用==操作符比较这些列表，以检查这些列表是否相等。如果排序后的列表相等，将确定两个原始列表包含相同的元素。这可以如下实现。

```py
# function to compare lists
def compare(l1, l2):
    # here l1 and l2 must be lists
    if len(l1) != len(l2):
        return False
    l1.sort()
    l2.sort()
    if l1 == l2:
        return True
    else:
        return False

list1 = [1, 2, 3, 4]
list2 = [1, 4, 3, 2]
list3 = [2, 3, 4, 5]
print("list1 is:",list1)
print("list2 is:",list2)
print("list3 is:",list3)

# comparing list1 and list 2
print("list1 and list2 contain same elements:",compare(list1, list2))
#comparing list2 and list3
print("list1 and list3 contain same elements:",compare(list1, list3))
```

输出:

```py
list1 is: [1, 2, 3, 4]
list2 is: [1, 4, 3, 2]
list3 is: [2, 3, 4, 5]
list1 and list2 contain same elements: True
list1 and list3 contain same elements: False
```

## 使用 Python 中的集合进行比较

为了在 python 中比较两个列表，我们可以使用集合。python 中的集合只允许包含唯一值。我们可以利用集合的这个特性来判断两个列表是否有相同的元素。

为了比较，首先我们将检查列表的长度是否相等。如果长度不相等，列表将自动标记为不同。

之后，我们将使用 set()构造函数将列表转换成集合。我们可以使用==运算符来比较这两个集合，以检查这两个集合是否相等。如果两个集合相等，将确定两个列表包含相等的值。下面的例子说明了这个概念。

```py
# function to compare lists
def compare(l1, l2):
    # here l1 and l2 must be lists
    if len(l1) != len(l2):
        return False
    set1 = set(l1)
    set2 = set(l2)
    if set1 == set2:
        return True
    else:
        return False

list1 = [1, 2, 3, 4]
list2 = [1, 4, 3, 2]
list3 = [2, 3, 4, 5]
print("list1 is:", list1)
print("list2 is:", list2)
print("list3 is:", list3)

# comparing list1 and list 2
print("list1 and list2 contain same elements:", compare(list1, list2))
# comparing list2 and list3
print("list1 and list3 contain same elements:", compare(list1, list3))
```

输出:

```py
list1 is: [1, 2, 3, 4]
list2 is: [1, 4, 3, 2]
list3 is: [2, 3, 4, 5]
list1 and list2 contain same elements: True
list1 and list3 contain same elements: False
```

## 使用频率计数器比较两个列表

我们也可以在不比较长度的情况下比较两个列表。为此，首先我们必须为每个列表创建一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)，它将跟踪列表中元素的频率。在创建了将列表元素存储为键并将它们的频率存储为值的字典之后，我们可以比较两个字典中每个元素的频率。如果每个元素的频率变得相等，将确认两个列表包含相等的元素。

对于这个任务，我们可以使用 counter()方法。当在列表上调用 counter()方法时，它会创建一个 python 字典，并将元素存储为键，将元素的频率存储为值。在调用 counter()方法后，我们可以使用==操作符来比较创建的字典，以检查每个元素的频率是否相等。如果结果为真，则列表包含相等的元素。否则不会。从下面的例子可以看出这一点。

```py
import collections
# function to compare lists
def compare(l1, l2):
    # here l1 and l2 must be lists
    if len(l1) != len(l2):
        return False
    counter1 = collections.Counter(l1)
    counter2=collections.Counter(l2)
    if counter1 == counter2:
        return True
    else:
        return False

list1 = [1, 2, 3, 4]
list2 = [1, 4, 3, 2]
list3 = [2, 3, 4, 5]
print("list1 is:", list1)
print("list2 is:", list2)
print("list3 is:", list3)

# comparing list1 and list 2
print("list1 and list2 contain same elements:", compare(list1, list2))
# comparing list2 and list3
print("list1 and list3 contain same elements:", compare(list1, list3)) 
```

输出:

```py
list1 is: [1, 2, 3, 4]
list2 is: [1, 4, 3, 2]
list3 is: [2, 3, 4, 5]
list1 and list2 contain same elements: True
list1 and list3 contain same elements: False
```

## 结论

在本文中，我们看到了 python 中比较两个列表的三种不同方法，并检查了它们是否包含相同的元素，而不考虑元素的位置。要阅读更多关于列表的内容，请阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。

在示例中使用的 compare()函数中，用户可能传递两个其他对象而不是列表。在这种情况下，程序可能会出错。为了避免这种情况，我们可以使用使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 的异常处理，通过在 try-except 块中使用 type()方法应用类型检查来检查作为参数传递的对象是否是列表，从而避免运行时错误。