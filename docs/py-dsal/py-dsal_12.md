# python 数据结构与算法 13 队列的抽象数据类型

队列的抽象数据类型

队列的抽象数据类型由下面的操作定义。队列是结构化，有序的数据集，前端删除数据，后端加入数据，保持 FIFO 属性：

*   Queue() 定义一个空队列，无参数，返回值是空队列。

*   enqueue(item)  在队列尾部加入一个数据项，参数是数据项，无返回值。

*   dequeue()  删除队列头部的数据项，不需要参数，返回值是被删除的数据，队列本身有变化。

*   isEmpty()  检测队列是否为空。无参数，返回布尔值。

*   size() 返回队列数据项的数量。无参数，返回一个整数。

举例说明，q 是一个刚创建的空队列，表 1 分别显示了操作、表内数据和返回值。4 是第一个加入队列的，所以也是第一个出队的。

| **Table 1: Example Queue Operations** |
| **Queue Operation** | **Queue Contents** | **Return Value** |
| q.isEmpty() | [] | True |
| q.enqueue(4) | [4] |  |
| q.enqueue('dog') | ['dog',4] |  |
| q.enqueue(True) | [True,'dog',4] |  |
| q.size() | [True,'dog',4] | 3 |
| q.isEmpty() | [True,'dog',4] | False |
| q.enqueue(8.4) | [8.4,True,'dog',4] |  |
| q.dequeue() | [8.4,True,'dog'] | 4 |
| q.dequeue() | [8.4,True] | 'dog' |
| q.size() | [8.4,True] | 2 |