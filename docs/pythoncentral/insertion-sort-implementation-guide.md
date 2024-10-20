# 插入排序:快速教程和实现指南

> 原文：<https://www.pythoncentral.io/insertion-sort-implementation-guide/>

## 先决条件

要了解插入排序，您必须知道:

1.  Python 3
2.  Python 数据结构-列表

## 什么是插入排序？

我们正在学习排序系列的第二个教程。前面的教程讲了冒泡排序，这是一个非常简单的排序算法。如果你还没有读过，你可以在这里找到它。像所有的排序算法一样，我们认为一个列表只有在升序时才被排序。降序被认为是最坏的未排序情况。

顾名思义，在插入排序中，一个元素被比较并插入到列表中正确的位置。要应用这种排序，必须考虑列表的一部分要排序，另一部分不排序。首先，考虑第一个元素是排序的部分，列表中的其他元素是未排序的。现在，将未排序部分中的每个元素与排序部分中的元素进行比较。然后将其插入已分类零件的正确位置。 **记住，列表的排序部分始终保持排序。**

让我们用一个例子来看看，list = [5，9，1，2，0]

第一轮:

```py
sorted - 5 and unsorted -  9,1,2,0

5<9: nothing happens

alist =  [5,9,1,2,0]
```

第二轮:

```py
sorted - 5, 9 and unsorted - 1,2,0

1<9: do nothing

1<5: insert 1 before 5

alist = [1,5,9,2,7,0]
```

第三轮:

```py
sorted -1, 5, 9 and unsorted - 2,0

2<9: do nothing 

2<5: do nothing

2>1: do nothing

insert 2 before 5

alist = [1,2,5,9,0]
```

**注:** 即使我们很清楚 2 小于 5 但大于 1，算法也无从得知这一点。因此，只有在与排序部分中的所有元素进行比较后，它才会使插入位置正确。

第四轮:

```py
sorted -1, 2, 5, 9 and unsorted - 0

0<9: do nothing

0<5: do nothing

0<2: do nothing

0<1: insert 0 before 1

alist = [0,1,2,5,9]
```

为了更好的理解，请看这个动画。

## 如何实现插入排序？

现在你已经对插入排序有了一个很好的理解，让我们来看看这个算法和它的代码。

### 算法

1.  考虑要排序的第一个元素，其余元素不排序
2.  与第二个元素比较:
    1.  如果第二个元素<第一个元素，将该元素插入排序部分的正确位置
    2.  否则，保持原样
3.  重复 1 和 2，直到所有元素排序完毕

**注意:** 随着排序部分中元素数量的增加，来自未排序部分的新元素在插入前必须与排序列表中的 **所有** 元素进行比较。

### 代码

```py
def insertionSort(alist):

   for i in range(1,len(alist)):

       #element to be compared
       current = alist[i]

       #comparing the current element with the sorted portion and swapping
       while i>0 and alist[i-1]>current:
           alist[i] = alist[i-1]
           i = i-1
          alist[i] = current

       #print(alist)

   return alist

print(insertionSort([5,2,1,9,0,4,6]))
```

取消对 *print* 语句的注释，查看列表是如何排序的。

## 结论

插入排序最适用于少量元素。插入排序的最坏情况运行时间复杂度是 o(n2 ),类似于冒泡排序。然而，插入排序被认为比冒泡排序更好。探究原因并在下面评论。本教程到此为止。快乐的蟒蛇！