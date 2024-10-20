# 合并排序:快速教程和实施指南

> 原文：<https://www.pythoncentral.io/merge-sort-implementation-guide/>

## 先决条件

要了解合并排序，您必须知道:

1.  Python 3
2.  Python 数据结构-列表
3.  递归

## 什么是合并排序？

我们正在学习排序系列的第四个教程。之前的教程涵盖了[冒泡排序](https://www.pythoncentral.io/bubble-sort-implementation-guide/)、[插入排序](http://: https://www.pythoncentral.io/insertion-sort-implementation-guide/)和[选择排序](http://: https://www.pythoncentral.io/selection-sort-implementation-guide/)。如果你还没有读过这些，请照着我们将要建立的那些概念去做。像所有的排序算法一样，我们认为一个列表只有在升序排列时才被排序。降序被认为是最坏的未排序情况。

合并排序与我们目前所见的其他排序技术有很大不同。合并排序可用于对未排序的列表进行排序，或者合并两个已排序的列表。

### 对未排序的列表进行排序

这个想法是将未排序的列表分成更小的组，直到一个组中只有一个元素。然后，按排序后的顺序将两个元素分组，并逐渐增加组的大小。每次发生合并时，必须逐个比较组中的元素，并按照排序的顺序组合成一个列表。这个过程一直持续到所有元素被合并和排序。注意，当重组发生时，排序的顺序必须 **始终** 保持不变。

让我们用一个例子来看看，列表= [5，9，1，2，7，0]

分裂

```py
Step 1: [5,9,1] [2,7,0]

Step 2: [5] [9,1] [2] [7,0]

Step 3: [5] [9] [1] [2] [7] [0]
```

**注意:** 在第二步中，我们正在处理组中的奇数个元素，所以我们任意拆分它们。所以，[5，9] [1] [2，7] [0]也是正确的。

合并

```py
Step 4: [5,9] [1,2] [0,7]

Step 5: [1,2,5,9][0,7]

Step 6: [0,1,2,5,7,9]
```

为了更好的理解，请看这个动画。

### 合并两个排序列表

你可能认为交替地从两个排序列表中取出元素并把它们放在一起会产生一个排序列表。这是一个非常错误的想法。让我们看看为什么。

```py
a =  [1,3,4,9]  b = [2,5,7,8] 
```

如上所述，在合并两个列表时，我们得到了:[1，2，3，5，4，7，9，8]

很明显，合并后的列表是没有排序的。所以我们需要在制作一个大的排序列表之前比较这些元素。让我们看一个例子。

```py
a =  [1,3,4,9]  b = [2,5,7,8] 

Step 1: 1<2 new list → [1] a =  [3,4,9]  b = [2,5,7,8] 

Step 2: 3>2  new list → [1,2] a =  [3,4,9]  b = [5,7,8] 

Step 3: 3<5 new list → [1,2,3] a =  [4,9]  b = [5,7,8] 

Step 4: 4<5 new list → [1,2,3,4] a = [9] b = [5,7,8]

Step 5: 9>5 new list → [1,2,3,4,5] a = [9] b = [7,8]

Step 6:9>7 new list → [1,2,3,4,5,7] a = [9] b = [8]

Step 7: 9>8 new list → [1,2,3,4,5,7,8] a = [9] b = []

Step 8: new list → [1,2,3,4,5,7,8,9]
```

## 如何实现归并排序？

现在你已经对什么是合并排序有了一个很好的理解，让我们来看看如何对一个列表进行排序的算法及其代码。

### 算法

1.  递归地将未排序的列表分成组，直到每组有一个元素
2.  比较每个元素，然后将它们分组
3.  重复步骤 2，直到整个列表在过程中被合并和排序

### 代码

```py
def mergeSort(alist):

   print("Splitting ",alist)

   if len(alist)>1:
       mid = len(alist)//2
       lefthalf = alist[:mid]
       righthalf = alist[mid:]

       #recursion
       mergeSort(lefthalf)
       mergeSort(righthalf)

       i=0
       j=0
       k=0

       while i < len(lefthalf) and j < len(righthalf):
           if lefthalf[i] < righthalf[j]:
               alist[k]=lefthalf[i]
               i=i+1
           else:
               alist[k]=righthalf[j]
               j=j+1
           k=k+1

       while i < len(lefthalf):
           alist[k]=lefthalf[i]
           i=i+1
           k=k+1

       while j < len(righthalf):
           alist[k]=righthalf[j]
           j=j+1
           k=k+1

   print("Merging ",alist)

alist = [54,26,93,17,77,31,44,55,20]
mergeSort(alist)
print(alist)
```

代码由互动 Python 提供。

## 结论

合并排序对大量和少量元素都有效，比冒泡、插入和选择排序更有效。这是有代价的，因为合并排序使用额外的空间来产生一个排序列表。归并排序的最坏情况运行时复杂度为*o(nlog(n))*，空间复杂度为 *n* 。尝试合并两个排序列表并在下面评论。本教程到此为止。快乐的蟒蛇！