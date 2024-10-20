# Python 中的砖块排序算法[易于实现]

> 原文：<https://www.askpython.com/python/examples/brick-sort-algorithm>

在本教程中，我们将学习如何实现砖块排序算法，以便对数组中的元素进行排序。这在编码界是未知的，但学习一种新的排序技术没有坏处。

在用 python 编程语言实现 brick sort 之前，让我们先了解一下什么是 brick sort。

***也读作:[在 Python 中插入排序](https://www.askpython.com/python/examples/insertion-sort-in-python)***

* * *

## 砖块排序算法简介

**砖块排序**，又称**奇偶排序**，是**泡泡排序**的修改版本。排序算法分为两个阶段，即**奇数阶段和偶数阶段**。通过确保在每次迭代中执行偶数和奇数阶段，控制将一直运行，直到对数组进行排序。

现在你可能会问这些奇数和偶数阶段是什么意思？当控件执行奇数阶段时，我们将只对奇数索引处的元素进行排序。在事件阶段的执行过程中，以类似的模式，控件将只对偶数索引处的元素进行排序。

* * *

## 砖块排序算法实现

为了实现砖块排序，我们将遵循一些步骤，下面也将提到。

*   声明**砖块排序函数**来执行排序，并取一个变量到**在奇数和偶数阶段**之间切换。
*   创建一个变量 **isSort** ，初始值为 0。此变量的目的是跟踪当前阶段。
*   运行 while 循环进行迭代，直到 isSort 等于 1。
    1.  创建一个内部 for 循环来对奇数条目进行排序。
    2.  类似地，创建另一个内部 for 循环来对偶数条目进行排序。
*   一旦排序完成，我们**返回结果**。

* * *

## 在 Python 中实现砖块排序

让我们直接进入 Python 中的块排序算法的实现，并确保我们可以获得预期的输出。

```py
def brickSort(array, n): 
    isSort = 0
    while isSort == 0: 
        isSort = 1
        for i in range(1, n-1, 2): 
            if array[i] > array[i+1]: 
                array[i], array[i+1] = array[i+1], array[i] 
                isSort = 0
        for i in range(0, n-1, 2): 
            if array[i] > array[i+1]: 
                array[i], array[i+1] = array[i+1], array[i] 
                isSort = 0
    return

array = [31, 76, 18, 2, 90, -6, 0, 45, -3] 
n = len(array)
print("Array input by user is: ", end="")
for i in range(0, n): 
    print(array[i], end =" ")   
brickSort(array, n);
print("\nArray after brick sorting is: ", end="")
for i in range(0, n): 
    print(array[i], end =" ") 

```

上面提到的代码如下所示。您可以看到数组排序成功。

```py
Array input by user is: 31 76 18 2 90 -6 0 45 -3 
Array after brick sorting is: -6 -3 0 2 18 31 45 76 90 

```

* * *

## 结论

我希望今天你学到了一种新的排序算法。虽然要求不多，但是口袋里有一些额外的知识总是好的！

感谢您的阅读！快乐学习！😇

* * *