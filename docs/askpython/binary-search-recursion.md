# 在 Python 中使用递归的二分搜索法

> 原文：<https://www.askpython.com/python/examples/binary-search-recursion>

在本教程中，我们将了解如何借助递归实现二分搜索法。我希望现在你已经熟悉了二分搜索法和递归。

为了让你更简单，我们将简要地介绍它们。

* * *

## 什么是二分搜索法？

**二分搜索法**是一种高效快速的算法，用于在元素的**排序列表**中查找元素。

它通过重复**将数组一分为二**来查找元素，然后比较除法的中间部分来识别元素可能出现在哪个除法中。

为了实现二分搜索法，我们需要三个指针，即下界、上界、**和一个中间指针**。

子阵列的划分由下限和上限定义，而中间指针值与需要定位的元素的值进行比较。

***在这里阅读更多关于二分搜索法:[Python 中的二分搜索法算法](https://www.askpython.com/python/examples/binary-search-algorithm-in-python)***

* * *

## 什么是递归？

现在，二分搜索法可以用许多方式来实现，下面提到了其中的一些:

1.  [使用循环的二分搜索法算法](https://www.askpython.com/python/examples/binary-search-algorithm-in-python)
2.  [使用二叉查找树的二分搜索法算法](https://www.askpython.com/python/examples/binary-search-tree)

在本教程中，我们将借助递归实现二分搜索法。

当一个函数调用本身可以是一个直接或间接的调用，以解决一个较小的问题或同一类型的一个较大的问题时，这种技术被称为**递归**。

***在这里阅读更多关于递归的内容:[Python 中的递归](https://www.askpython.com/python/python-recursion-function)***

* * *

## 使用递归的二分搜索法实现

让我们在这里用 Python 实现递归的二分搜索法算法。我添加了带有注释的代码，以帮助您理解每一行的作用。

```py
def Binary_Search(arr,n,lb,ub,X):

    # 1\. List is empty
    if(n==0):
        print("ERROR!")

    # 2\. If element is not found lb exceeds ub    
    elif(lb>ub):
        print("Not found!")

    # 3\. Keep searching for the element in array
    else:
        mid = int((lb+ub)/2)
        if(arr[mid]==X):
            print(mid+1)
        elif(arr[mid]>X):
            Binary_Search(arr,n,lb,mid,X);
        elif(arr[mid]<X):
            Binary_Search(arr,n,mid+1,ub,X);

arr = [1,2,3,4,5,6,7,8,9]
n = len(arr)
X = int(input())
Binary_Search(arr,n,0,n-1,X)

```

* * *

## 输出

```py
Original List is:  [1, 2, 3, 4, 5, 6, 7, 8, 9]
Element to Search for:  90
Result: Not found!

```

```py
Original List is:  [1, 2, 3, 4, 5, 6, 7, 8, 9]
Element to Search for:  5
Result: 5

```

* * *

## 结论

在本教程中，我们了解了如何借助递归以及二分搜索法和递归的一些基础知识来实现二分搜索法。

希望你喜欢这个教程！感谢您的阅读！

敬请关注更多此类教程！😇

* * *