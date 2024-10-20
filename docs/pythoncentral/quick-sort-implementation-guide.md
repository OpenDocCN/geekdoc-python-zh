# 快速排序:教程和实施指南

> 原文：<https://www.pythoncentral.io/quick-sort-implementation-guide/>

## 先决条件

要了解快速排序，您必须知道:

1.  Python 3
2.  Python 数据结构-列表
3.  递归

## 什么是快速排序？

我们正在学习排序系列的第五个也是最后一个教程。前面的教程讲的是[冒泡排序](https://www.pythoncentral.io/bubble-sort-implementation-guide/)、[插入排序](https://www.pythoncentral.io/insertion-sort-implementation-guide/ ‎)、[选择排序](https://www.pythoncentral.io/selection-sort-implementation-guide/‎)和[归并排序](https://www.pythoncentral.io/merge-sort-implementation-guide/‎)。如果你还没有读过，请照着我们将要建立的那些概念去做。像所有的排序算法一样，我们认为一个列表只有在升序时才被排序。降序被认为是最坏的未排序情况。

快速排序与我们目前看到的排序技术有很大的不同，也很复杂。在这种技术中，我们选择第一个元素，并将其称为。其思想是对元素进行分组，使枢轴左边的元素比枢轴小，而枢轴右边的元素比枢轴大。这是通过维护两个指针 *左* 和 *右* 来实现的。 *左边的* 指针指向枢轴后的第一个元素。让我们把 *左边* 所指的元素称为 *lelement。* 同理，右侧的指针指向列表右侧最远的元素。让我们称这个元素为*relement*。在每一步，比较 *元素* 与 *枢轴* 与 *元素* 与 *枢轴。* 记住 *元素<枢轴* 和 *元素>枢轴* 。如果不满足这些条件，则 *元素* 和 *元素* 被交换。否则， *左* 指针递增 1， *右* 指针递减 1。当 *左> =右* 时， *枢轴* 与 *元素* 或 *元素交换。**枢轴* 元素将在其正确的位置。然后在列表的左半部分和右半部分继续快速排序。

让我们用一个例子来看看，列表= [5，9，6，2，7，0]

| 枢轴 | 左 | 右 | 比较 | 动作 | 列表 |
| 5 | 9 [1] | 0 [5] | 9>50 < 5 | Swap 0 and 9将 *向左* 增加 1递减 *右 1* | 【5，0，6，2，7，9】 |
| 5 | 6 [2] | 7【4】 | 6>57 > 5 | 将 *右* 指针减少 1 个 指针，将 *左* 指针原样保留 | 【5，0，6，2，7，9】 |
| 5 | 6 [2] | 2【3】 | 6>52 < 5 | Swap 2 and 6将 *向左* 增加 1递减 *右 1* | 【5，0，2，6，7，9】 |
| 5 | 6【3】 | 2【2】 | 停止(自 *左>右* ) | 交换 5 和 2 | 【2，0，5，6，7，9】 |

同样，对左半部分[2，0]和右半部分[6，7，9]执行快速排序。重复此过程，直到整个列表排序完毕。即使我们可以看到右半部分已经排序，算法也无法知道这一点。

为了更好的理解，请看这个动画。

## 如何实现快速排序？

现在你已经对快速排序有了一个相当好的理解，让我们来看看这个算法及其代码。

### 算法

1.  选择一个支点
2.  设置左指针和右指针
3.  比较一下指针元素(*lelement)*与 pivot指针元素(*relelement)*与*pivot*。
4.  检查*lelement<pivot*和*relement>*pivot:
    1.  如果是，则左指针递增，右 指针指针递减
    2.  如果没有，互换 *lelement* 和 *relement*
5.  当 *左> =右、* 将 *支点* 与或 *右* 指针互换。
6.  在列表的左半部分和右半部分重复步骤 1 - 5，直到整个列表排序完毕。

### 代码

```py
def quickSort(alist):

  quickSortHelper(alist,0,len(alist)-1)

def quickSortHelper(alist,first,last):

  if first<last:
      splitpoint = partition(alist,first,last)
      quickSortHelper(alist,first,splitpoint-1)
      quickSortHelper(alist,splitpoint+1,last)

def partition(alist,first,last):

  pivotvalue = alist[first]
  leftmark = first+1
  rightmark = last
  done = False

  while not done:
      while leftmark <= rightmark and alist[leftmark] <= pivotvalue:
          leftmark = leftmark + 1

      while alist[rightmark] >= pivotvalue and rightmark >= leftmark:
          rightmark = rightmark -1

      if rightmark < leftmark:
          done = True
      else:
          temp = alist[leftmark]
          alist[leftmark] = alist[rightmark]
          alist[rightmark] = temp

  temp = alist[first]
  alist[first] = alist[rightmark]
  alist[rightmark] = temp

  return rightmark

alist = [54,26,93,17,77,31,44,55,20]
quickSort(alist)
print(alist)
```

代码由互动 Python 提供。

## 如何选择支点？

选择 *中枢* 非常关键，因为它决定了这个算法的效率。如果我们得到一个已经排序的列表，我们选择第一个元素为 *pivot* 那么这将是一场灾难！这是因为没有比 *中枢* 更大的元素了，这大大降低了 *的性能。* 为了避免这种情况，有几种方法可以用来选择 *枢轴* 的值。一种这样的方法是中位数法。在这里，我们选择第一个元素、最后一个元素和中间的元素。在比较了这三个元素之后，我们选择具有中间值的元素。让我们用一个例子来看看，list = [5，6，9，0，3，1，2]

第一个元素是 5，最后一个元素是 2，中间元素是 0。比较这些值，很明显，2 是中间值，这个选为。诸如此类的方法确保我们不会以最差的选择作为我们的 *支点。* 记住最差选择为是列表中最小或最大的值。

## 结论

快速排序最适用于小数量和大数量的元素。快速排序在最坏情况下的运行时间复杂度是 O(n2)，类似于插入和冒泡排序，但可以改进为 O(nlog(n))，如前一节所述。与合并排序不同，它没有使用额外内存或空间的缺点。这就是为什么这是最常用的排序技术之一。本教程到此为止。快乐的蟒蛇！