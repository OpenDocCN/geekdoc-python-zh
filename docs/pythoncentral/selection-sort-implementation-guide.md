# 选择排序:快速教程和实施指南

> 原文：<https://www.pythoncentral.io/selection-sort-implementation-guide/>

## 先决条件

要了解选择排序，您必须知道:

1.  Python 3
2.  Python 数据结构-列表

## 什么是选择排序？

我们正在学习排序系列的第三个教程。前面的教程讲的是[冒泡排序](https://www.pythoncentral.io/bubble-sort-implementation-guide/)和[插入排序](https://www.pythoncentral.io/Insertion-sort-implementation-guide/)。如果你还没有读过，请照着我们将要建立的那些概念去做。像所有的排序算法一样，我们认为一个列表只有在升序时才被排序。降序被认为是最坏的未排序情况。

类似于插入排序，我们首先考虑要排序的第一个元素，其余的元素不排序。随着算法的进行，列表中已排序的部分将增长，未排序的部分将继续收缩。

选择排序是从列表中挑选最小的元素，并将其放入列表的排序部分。最初，第一个元素被认为是最小值，并与其他元素进行比较。在这些比较过程中，如果找到一个较小的元素，则认为它是新的最小值。完成一整轮后，找到的最小元素与第一个元素交换。这个过程一直持续到所有元素都被排序。

让我们用一个例子来看看，列表= [5，9，1，2，7，0]

5 被认为是已排序的，而元素 9，1，2，7，0 被认为是未排序的。

第一轮:

```py
5 is the minimum

5<9: nothing happens

5>1: 1 is new minimum

1<2: nothing happens

1<7: nothing happens

1>0: 0 is the new minimum

Swap 0 and 5

alist =  [0,9,1,2,7,5]
```

第二轮:

```py
9 is the minimum

9>1: 1 is the new minimum

1<2: nothing happens

1<7: nothing happens

1<5: nothing happens

Swap 1 and 9

alist = [0,1,9,2,7,5]
```

第三轮:

```py
9 is considered minimum

9>2: 2 is the new minimum

2<7: nothing happens

2<5: nothing happens

Swap 2 and 9

alist = [0,1,2,9,7,5]
```

第 4 轮:

```py
9 is considered minimum

9>7: 7 is the new minimum

7>5: 5 is the new minimum

Swap 9 and 5

alist = [0.1.2.5,7,9]
```

第五轮:

```py
7 is considered minimum

7<9: nothing happens

alist = [0.1.2.5,7,9]
```

**注意:** 即使在第 4 轮之后我们可以看到列表已经排序，算法也没有办法知道这一点。因此，只有在完全遍历了整个列表之后，算法才会停止。

为了更好的理解，请看这个动画。

## 如何实现选择排序？

现在你对什么是选择排序有了一个很好的理解，让我们来看看这个算法和它的代码。

### 算法

1.  考虑要排序的第一个元素，其余元素不排序
2.  假设第一个元素是最小的元素。
3.  检查第一个元素是否小于其他元素:
    1.  如果是，什么都不做
    2.  如果否，选择另一个较小的元素作为最小值，并重复步骤 3
4.  在完成列表的一次迭代后，用列表的第一个元素交换最小的元素。
5.  现在认为列表中的第二个元素是最小的，以此类推，直到列表中的所有元素都被覆盖。

**注意:** 一旦一个元素被添加到列表的排序部分，它就决不能被接触和或比较。

### 代码

```py
def selectionSort(alist):

   for i in range(len(alist)):

      # Find the minimum element in remaining
       minPosition = i

       for j in range(i+1, len(alist)):
           if alist[minPosition] > alist[j]:
               minPosition = j

       # Swap the found minimum element with minPosition       
       temp = alist[i]
       alist[i] = alist[minPosition]
       alist[minPosition] = temp

   return alist

print(selectionSort([5,2,1,9,0,4,6]))

```

## 结论

选择排序最适用于少量元素。插入排序的最坏情况运行时间复杂度是 o(n2 ),类似于插入和冒泡排序。本教程到此为止。快乐的蟒蛇！