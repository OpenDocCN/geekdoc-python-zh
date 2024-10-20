# Python 中的生产者-消费者问题

> 原文：<https://www.askpython.com/python/producer-consumer-problem>

大家好！在本教程中，我们将学习生产者-消费者问题，这是一个经典的并发性问题，以及如何使用 Python 线程解决它。所以让我们开始吧。

***也读:[如何用 Python 创建自定义数据集？](https://www.askpython.com/python-modules/pytorch-custom-datasets)***

## 什么是生产者-消费者问题？

**生产者-消费者问题**由 3 个部分组成:

### 1.**有界缓冲区**

缓冲区是可由不同线程访问的临时存储。缓冲区的一个简单例子是数组。多个线程可以从缓冲区读取数据，也可以同时将数据写入缓冲区。有界缓冲区的容量有限，不能存储超出其容量的数据。

### 2.**生产者线程**

生产者线程是这样一种线程，它生成一些数据，将其放入缓冲区，然后再次启动，直到没有生成所有需要的数据。这方面的一个例子是一个线程，它通过网络下载一些数据，并将其临时存储在缓冲区中

### 3.**消费者线程**

消费者线程是这样一种线程，它消费缓冲区中存在的数据，将其用于某个任务，然后再次启动，直到分配给该线程的任务没有完成。这方面的一个例子是一个线程，它读取通过互联网下载的数据，并将其存储在数据库中。

## 当线程的运行速率不同时会发生什么？

由线程完成的操作的速度可以根据所分配的任务而变化。因此，在我们的例子中，或者我们的生产者线程可能比消费者线程慢，或者生产者线程可能比消费者线程消耗数据的速度快。

如果线程运行的速率不同，可能会有一些问题，这就是生产者-消费者问题。

1.  如果生产者线程试图将数据生成到缓冲区中，并且发现缓冲区已经满了，则生产者线程既不能在缓冲区中添加更多的数据，也不能覆盖消费者尚未消费的现有数据。因此，生产者线程应该停止自己，直到一些数据没有从缓冲区消费。如果生产者线程很快，这种情况是可能的。
2.  如果消费者线程试图消费来自缓冲区的数据，但是发现缓冲区是空的，则消费者线程不能获取数据，并且它应该停止自己，直到一些数据没有被添加到缓冲区中。如果消费者线程很快，这种情况是可能的。
3.  由于缓冲区是由不同的线程共享的，这些线程可以同时访问缓冲区中的数据，因此可能会出现竞争情况，并且两个线程不应该同时访问共享的缓冲区。生产者线程应该将数据添加到缓冲区，而消费者线程应该等待，或者生产者线程应该等待，而消费者线程正在共享缓冲区上工作以读取数据。

## 使用信号量解决问题

我们可以借助**信号量**来解决这个问题，信号量是线程间同步的工具。为了解决生产者-消费者问题的问题陈述中定义的 3 个问题，我们维护了 3 个信号量。

1.  **empty:** 这个信号量存储缓冲区中为空的槽的数量。这个信号量的初始值是我们的有界缓冲区的大小。在缓冲区中添加任何数据之前，生产者线程将尝试获取这个信号量，并将它的值减 1。如果这个信号量的值已经是 0，这意味着缓冲区已满，我们的空信号量将阻塞生产者线程，直到空信号量的值变得大于 0。类似地，在消费者线程消耗完缓冲区中的数据后，它将释放这个信号量，并将信号量的值增加 1。
2.  **full:** 这个信号量存储缓冲区中已满的槽的数量。这个信号量的初始值是 0。在使用缓冲区中的数据之前，使用者线程将尝试获取这个信号量。如果这个信号量的值已经是 0，这意味着缓冲区已经是空的，我们的满信号量将阻塞消费者线程，直到满信号量的值变得大于 0。类似地，生产者线程将在添加一个项目后释放这个信号量。
3.  互斥:这个信号量将通过一次只允许一个信号量在共享缓冲区上操作来处理竞争情况。这个信号量的初始值是 1。在对共享缓冲区进行操作之前，两个线程都将尝试获取这个信号量。如果任何线程发现这个信号量的值为 0，这意味着另一个线程正在缓冲区上操作，它将被这个信号量阻塞。在缓冲区上操作之后，工作线程将释放这个信号量，以便另一个线程可以在缓冲区上操作。

我们还维护了两个指针来帮助我们的线程添加或获取数据。

*   **in pointer:** 这个指针将告诉我们的生产者线程，在生产者生成的缓冲区中，在哪里添加下一个数据。添加后，指针增加 1。
*   **out 指针:**这个指针将告诉我们的消费者线程从缓冲区的哪里读取下一个数据。读取后，指针增加 1。

## 用 Python 实现生产者-消费者问题

让我们检查一下如何用 Python 解决这个问题的实现。假设我们有一个容量为 10 的有界缓冲区。生产者线程将生产 20 个项目，消费者线程将消费生产者生产的这 20 个项目。在生产者中添加`time.sleep(1)`和在消费者中添加`time.sleep(2.5)`使得我们的生产者线程比消费者线程运行得更快。即使我们首先启动我们的消费者线程，它也会一直等到缓冲区中没有数据。

```py
import threading
import time

# Shared Memory variables
CAPACITY = 10
buffer = [-1 for i in range(CAPACITY)]
in_index = 0
out_index = 0

# Declaring Semaphores
mutex = threading.Semaphore()
empty = threading.Semaphore(CAPACITY)
full = threading.Semaphore(0)

# Producer Thread Class
class Producer(threading.Thread):
  def run(self):

    global CAPACITY, buffer, in_index, out_index
    global mutex, empty, full

    items_produced = 0
    counter = 0

    while items_produced < 20:
      empty.acquire()
      mutex.acquire()

      counter += 1
      buffer[in_index] = counter
      in_index = (in_index + 1)%CAPACITY
      print("Producer produced : ", counter)

      mutex.release()
      full.release()

      time.sleep(1)

      items_produced += 1

# Consumer Thread Class
class Consumer(threading.Thread):
  def run(self):

    global CAPACITY, buffer, in_index, out_index, counter
    global mutex, empty, full

    items_consumed = 0

    while items_consumed < 20:
      full.acquire()
      mutex.acquire()

      item = buffer[out_index]
      out_index = (out_index + 1)%CAPACITY
      print("Consumer consumed item : ", item)

      mutex.release()
      empty.release()      

      time.sleep(2.5)

      items_consumed += 1

# Creating Threads
producer = Producer()
consumer = Consumer()

# Starting Threads
consumer.start()
producer.start()

# Waiting for threads to complete
producer.join()
consumer.join()

```

**输出:**

```py
Producer produced :  1
Consumer consumed item :  1
Producer produced :  2
Producer produced :  3
Consumer consumed item :  2
Producer produced :  4
Producer produced :  5
Consumer consumed item :  3
Producer produced :  6
Producer produced :  7
Producer produced :  8
Consumer consumed item :  4
Producer produced :  9
Producer produced :  10
Consumer consumed item :  5
Producer produced :  11
Producer produced :  12
Producer produced :  13
Consumer consumed item :  6
Producer produced :  14
Producer produced :  15
Consumer consumed item :  7
Producer produced :  16
Producer produced :  17
Consumer consumed item :  8
Producer produced :  18
Consumer consumed item :  9
Producer produced :  19
Consumer consumed item :  10
Producer produced :  20
Consumer consumed item :  11
Consumer consumed item :  12
Consumer consumed item :  13
Consumer consumed item :  14
Consumer consumed item :  15
Consumer consumed item :  16
Consumer consumed item :  17
Consumer consumed item :  18
Consumer consumed item :  19
Consumer consumed item :  20

```

## 结论

恭喜你！现在你知道如何解决经典的生产者-消费者问题了。现实生活中有许多可能发生类似情况的例子，例如打印一个文档，其中多个应用程序想要打印一个文档，通过网络下载数据并存储在数据库中，等等。

感谢阅读！！