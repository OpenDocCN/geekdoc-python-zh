# 循环队列:实现教程

> 原文：<https://www.pythoncentral.io/circular-queue/>

## 先决条件

要了解循环队列，您首先应该很好地理解以下内容:

1.  Python 3
2.  线性队列(你可以在这里了解更多)
3.  基本 Python 数据结构概念-列表
4.  基本数学运算-模数(%)

## 什么是循环队列？

在你继续阅读本教程之前，我强烈推荐你阅读我们之前的关于队列的教程，因为我们将建立在这些概念之上。循环队列被广泛使用，并且经常在工作面试中被测试。循环队列可以看作是对线性队列的改进，因为:

1.  由于头尾指针会自行复位，因此无需复位。这意味着一旦头部或尾部到达队列的末尾，它会将自己重置为 0。
2.  尾部和头部可以指向同一个位置——这意味着队列是空的
3.  头可以比尾大，反之亦然。这是可能的，因为头指针和尾指针允许相互交叉。

看看 [这个](https://www.cs.usfca.edu/~galles/visualization/QueueArray.html) 的动画可以更好地理解环形队列。

基于上述动画的观察:

1.  头指针——指向队列的最前面。或者换句话说，如果调用出列操作，它指向要删除的元素。
2.  尾指针指向下一个可以插入新元素的空白点。在上面的动画中，如果您试图完全填满队列，您将无法在第 13 个位置之后排队。这是因为在第 14 个位置插入一个元素后，尾部没有空的点可以指向。即使还有一个空位，也认为队列已满。您还应该尝试执行三次或四次出列操作，然后将一个元素入队。在这里，您将看到元素从第 14 个位置插入，然后从 0 重新开始。正是由于这个原因，它被称为循环队列。
3.  元素数量:
    1.  尾>=头:元素个数=尾-头。例如，如果 Head = 2，Tail = 5，那么元素的数量将是 5 - 2 = 3
    2.  头>尾:元素个数=(队列大小)-(头尾)=(队列大小)-头+尾。例如，头= 14，尾= 5，队列大小= 15，那么元素数= 15 - (14 - 5) = 6

## 如何实现循环队列？

我希望你现在有信心知道什么是循环队列。让我们 看看如何使用语言不可知的方法来实现它。为此，我们需要像对待数组一样对待列表，因此我们将限制它的大小。

**注意:** 在出队操作期间，头指针将增加 1，但实际上不会从队列中移除任何元素。这是因为一旦删除了一个元素，列表会自动将所有其他元素向左移动一个位置。这意味着位置 0 将总是包含一个元素，该元素*不是实际队列/循环队列的工作方式*。

### 算法

以下步骤可以看作是循环队列操作的流程图:

1.  初始化队列、队列大小(maxSize)、头指针和尾指针
2.  排队:
    1.  检查元素的数量(size)是否等于队列的大小(maxSize):
        1.  如果是，抛出错误消息“队列已满！”
        2.  如果没有，则追加新元素并递增尾指针
3.  出列:
    1.  检查元素数量(大小)是否等于 0:
        1.  如果是，抛出错误消息“队列为空！”
        2.  如果没有，则增加头指针
4.  尺寸:
    1.  如果尾部> =头部，则大小=尾部-头部
    2.  如果 head>tail，size = maxSize -(头尾)

**注意:**头和尾指针的范围应该在 0 和 maxSize - 1 之间，因此我们使用的逻辑是，如果我们将 x 除以 5，那么余数永远不会大于 5。换句话说，应该在 0 到 4 之间。因此，将此逻辑应用于公式 tail = (tail+1)%maxSize 和 head = (head+1)%maxSize。请注意，这有助于我们避免在队列变满时将 tail 和 head 重新初始化为 0。

### 程序

```py
class CircularQueue:

    #Constructor
    def __init__(self):
        self.queue = list()
        self.head = 0
        self.tail = 0
        self.maxSize = 8

    #Adding elements to the queue
    def enqueue(self,data):
        if self.size() == self.maxSize-1:
            return ("Queue Full!")
        self.queue.append(data)
        self.tail = (self.tail + 1) % self.maxSize
        return True

    #Removing elements from the queue
    def dequeue(self):
        if self.size()==0:
            return ("Queue Empty!") 
        data = self.queue[self.head]
        self.head = (self.head + 1) % self.maxSize
        return data

    #Calculating the size of the queue
    def size(self):
        if self.tail>=self.head:
            return (self.tail-self.head)
        return (self.maxSize - (self.head-self.tail))

q = CircularQueue()
print(q.enqueue(1))
print(q.enqueue(2))
print(q.enqueue(3))
print(q.enqueue(4))
print(q.enqueue(5))
print(q.enqueue(6))
print(q.enqueue(7))
print(q.enqueue(8))
print(q.enqueue(9))
print(q.dequeue())
print(q.dequeue())
print(q.dequeue())
print(q.dequeue())
print(q.dequeue())
print(q.dequeue())
print(q.dequeue())
print(q.dequeue())
print(q.dequeue())
```

## 应用

循环队列有多种用途，例如:

1.  计算机架构(调度器)
2.  磁盘驱动器
3.  视频缓冲
4.  打印机作业调度

## 结论

开始时，循环队列可能看起来有点混乱，但掌握它的唯一方法就是不断练习。在上面提供的动画链接中尝试不同的入队和出队操作，看看它是如何工作的。本教程到此为止。快乐的蟒蛇！