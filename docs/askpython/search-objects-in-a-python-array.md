# 在 Python 数组中查找对象–查找数组中对象的第一个、最后一个和所有出现的位置

> 原文：<https://www.askpython.com/python/array/search-objects-in-a-python-array>

今天在本教程中，我们将借助递归查找数组中某个元素的第一个、最后一个和所有出现的位置。

在进入任何问题陈述之前，让我们首先理解什么是递归。如果你想了解递归，这里提供了一个了解递归的链接。

***了解一下递归这里: [Python 递归](https://www.askpython.com/python/python-recursion-function)***

* * *

## 查找元素的第一个匹配项

让我们从寻找元素在一个 [Python 数组](https://www.askpython.com/python/array/python-array-declaration)中的第一次出现开始。我们的目标是**找到元素在元素列表(数组)中出现的第一个位置**。

**例如:** 数组给定= =>【1，2，3，4，2】
第一次出现== > 2

为了找到问题的解决方案，我们将采取以下步骤:

```py
Step 1 :  Check if list is empty then return that list is empty
Step 2 : Check if there is only one element then check the first element with X and return the answer if found
Step 3 : For more than one element, we will check if the first element is equal to X if found then return 
Step 4 : Otherwise recursively go by slicing the array and incrementing and decremementing the itrerator and n value (size of array ) respectively
Step 5 :  Repeat until the element is found or not

```

**上述步骤的代码实现如下所示:**

```py
def find_first(arr,n,x,itr):

    # check if list is empty
    if(n==0):
        print("List empty!")
        return

    # Only one element
    elif(n==1):
        if(arr[0]==x):
            print("Element present at position 1")
        else:
            print("Element not found")
        return

    # More than one element
    else:
        if(arr[0] == x):
            print("Found at position: ", itr+1)
        else:
            find_first(arr[1:],n-1,x,itr+1)
        return

arr = [1,2,3,4,5,2,10,10]
n  = len(arr)
x = 10
itr = 0
find_first(arr,n,x,itr)

```

**输出:**

```py
Found at position:  7

```

* * *

## 查找对象的最后一次出现

接下来，我们将尝试使用 Python 查找该元素的最后一次出现。我们的目标是**找到元素在元素列表(数组)中出现的最后一个位置**。

例如:
数组给定= =>【1，2，3，4，2】
最后一次出现== > 5

为了找到问题的解决方案，我们将采取以下步骤:

```py
Step 1 :  Check if list is empty then return that list is empty
Step 2 : Check if there is only one element then check the first element with X and return the answer if found
Step 3 : For more than one element, we will check if the last element is equal to X if found then return 
Step 4 : Otherwise recursively go by slicing the array and decremementing both the iterator and n value (size of array ) 
Step 5 :  Repeat until the element is found or not

```

**用 Python 实现上述步骤**

```py
def find_first(arr,n,x,itr):

    # check if list is empty
    if(n==0):
        print("List empty!")
        return

    # Only one element
    elif(n==1):
        if(arr[0]==x):
            print("Element present at position 1")
        else:
            print("Element not found")
        return

    # More than one element
    else:
        if(arr[n-1] == x):
            print("Found at position: ", itr+1)
        else:
            find_first(arr[:-1],n-1,x,itr-1)
        return

arr = [1,2,3,4,5,2,3,2,3,2,10,10]
n  = len(arr)
x = 2
itr = n - 1
find_first(arr,n,x,itr)

```

**输出**:

```py
Found at position:  10

```

* * *

## 查找对象的所有出现

这里我们的目标是**找到**元素在元素列表(数组)中出现的所有位置。出现的位置包括数组中元素的第一个、最后一个和任何中间位置。

例如:
数组给定= =>【1，2，3，4，2】
所有出现次数== > 2 5

为了找到问题的解决方案，我们将采取以下步骤:

```py
Step 1 :  Check if list is empty then return that list is empty
Step 2 : Check if there is only one element then print the position of the element and return
Step 3 : For more than one element, we will check if the first element is equal to X if found then print and keep on recursively calling the function again by slicing the array and decremementing n value (size of array ) and incrementing the value of iterator
Step 5 :  Repeat until all the elements are encountered.

```

**用 Python 实现上述步骤**

```py
def find_first(arr,n,x,itr):

    # check if list is empty
    if(n==0):
        print("List empty!")
        return

    # Only one element
    elif(n==1):
        if(arr[0]==x):
            print(itr+1,end=" ")
        else:
            print("Element not found")

    # More than one element
    else:
        if(arr[0] == x):
            print(itr+1,end=" ")
        find_first(arr[1:],n-1,x,itr+1)

arr = [1,2,10,3,4,10,5,2,10,2,3,10]
n  = len(arr)
x = 10
itr = 0
print("Found at position: ",end="") 
find_first(arr,n,x,itr)

```

**输出:**

```py
Found at position: 3 6 9 12 

```

* * *

## 结论

到本教程结束时，我们已经熟悉了在给定的数组中查找元素的第一个、最后一个和所有出现的位置。希望你明白其中的逻辑！

感谢您的阅读！快乐学习！😇

* * *