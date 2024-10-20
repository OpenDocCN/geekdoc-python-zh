# Python 中的同步–Python 中的同步线程

> 原文：<https://www.askpython.com/python/examples/synchronization-in-python>

我们来谈谈 Python 中的同步。[多线程](https://www.askpython.com/python-modules/multithreading-in-python)允许您的计算机利用系统上的多核/多 CPU 并行执行操作。但是，当同时读取和更新共享变量时，可能会导致错误的结果。我们将学习如何同步线程以给出正确的结果。

## 理解多线程中的竞争条件

当两个或更多线程试图同时访问共享资源并更改数据时，这些变量的最终值是不可预测的。这是因为线程调度算法可以随时在线程之间交换，并且您不知道哪个线程将首先执行。这种情况称为竞争条件。

让我们举一个例子，我们使用线程将一些金额从一个银行帐户转移到另一个帐户。我们将创建 100 个线程来将 1 个单元从 account1 转移到 account2。

```py
import threading
import time

class BankAccount():
  def __init__(self, name, balance):
    self.name = name
    self.balance = balance

  def __str__(self):
    return self.name

# These accounts are our shared resources
account1 = BankAccount("account1", 100)
account2 = BankAccount("account2", 0)

class BankTransferThread(threading.Thread):
  def __init__(self, sender, receiver, amount):
    threading.Thread.__init__(self)
    self.sender = sender
    self.receiver = receiver
    self.amount = amount

  def run(self):
    sender_initial_balance = self.sender.balance
    sender_initial_balance -= self.amount
    # Inserting delay to allow switch between threads
    time.sleep(0.001)
    self.sender.balance = sender_initial_balance

    receiver_initial_balance = self.receiver.balance
    receiver_initial_balance += self.amount
    # Inserting delay to allow switch between threads
    time.sleep(0.001)
    self.receiver.balance = receiver_initial_balance

if __name__ == "__main__":

  threads = []

  for i in range(100):
    threads.append(BankTransferThread(account1, account2, 1))

  for thread in threads:
    thread.start()

  for thread in threads:
    thread.join()

  print(account1.balance)
  print(account2.balance)

```

```py
account1 98
account2 3

```

最初，帐户 1 有 100 个单位，帐户 2 有 0 个单位。

在 100 次转移 1 个单位后，账户 1 应该有 0 个单位，账户 2 应该有 100 个单位。然而，我们得到了不同的结果。如果我们运行多次，我们会得到不同的结果。

## Python 中的同步——同步线程的不同方法

让我们看看如何同步线程以避免竞争情况。

### 1.锁定对象

锁对象是最基本的同步原语，它在被锁定时不属于特定的线程。锁对象不保存关于哪个线程拥有锁的许可以及任何线程都可以释放锁的信息。

锁定对象处于两种状态之一:“锁定”和“解锁”。当创建锁对象时，它处于“解锁”状态。锁定对象中只有 3 种方法:

*   ***acquire():*** 该方法将锁对象从“解锁”状态改为“锁定”状态，并允许调用线程继续执行。如果锁对象已经处于“锁定”状态，调用线程将被阻塞，直到锁进入“解锁”状态。
*   ***release():*** 该方法将锁定对象的状态从“锁定”变为“解锁”状态。如果锁定对象已经处于“解锁”状态，则引发`RuntimeError`。该方法可以从任何线程调用，而不仅仅是获得锁的线程。
*   ***locked():*** 如果获取了锁对象，则该方法返回 true。

让我们看看如何使用 Lock 对象将 Python 中的同步添加到银行转帐示例中。

```py
import threading
import time

class BankAccount():
  def __init__(self, name, balance):
    self.name = name
    self.balance = balance

  def __str__(self):
    return self.name

# These accounts are our shared resources
account1 = BankAccount("account1", 100)
account2 = BankAccount("account2", 0)

# Creating lock for threads
lock = threading.Lock()

class BankTransferThread(threading.Thread):
  def __init__(self, sender, receiver, amount):
    threading.Thread.__init__(self)
    self.sender = sender
    self.receiver = receiver
    self.amount = amount

  def run(self):
    lock.acquire()

    sender_initial_balance = self.sender.balance
    sender_initial_balance -= self.amount
    # Inserting delay to allow switch between threads
    time.sleep(0.001)
    self.sender.balance = sender_initial_balance

    receiver_initial_balance = self.receiver.balance
    receiver_initial_balance += self.amount
    # Inserting delay to allow switch between threads
    time.sleep(0.001)
    self.receiver.balance = receiver_initial_balance

    lock.release()

if __name__ == "__main__":

  threads = []

  for i in range(100):
    threads.append(BankTransferThread(account1, account2, 1))

  for thread in threads:
    thread.start()

  for thread in threads:
    thread.join()

  print(account1.name, account1.balance)
  print(account2.name, account2.balance)

```

```py
account1 0
account2 100

```

锁对象不知道哪个线程调用了`acquire()`方法，任何线程都可以对锁调用`release()`，锁可以从调用`acquire()`的线程那里获取许可。

同样，如果同一个线程在没有`release()`的情况下再次调用`acquire()`方法，该线程将处于死锁状态。

```py
import threading

lock = threading.Lock()

def funcA():
  print("In A, acquiring lock")
  lock.acquire()

  print("In A, lock acquired")

  print("In A, lock acquiring again and entering into deadlock")
  lock.acquire()

  print("In A, releasing lock")
  lock.release()

  print("In A, lock released")

def funcB():
  print("In B, releasing lock acquired by A")
  lock.release()

  print("In B, lock released")

if __name__ == "__main__":
  thread1 = threading.Thread(target=funcA)
  thread2 = threading.Thread(target=funcB)

  thread1.start()
  thread2.start()

  thread1.join()
  thread2.join()

```

```py
In A, acquiring lock
In A, lock acquired
In A, lock acquiring again and entering into deadlock
In B, releasing lock acquired by A
In B, lock released
In A, releasing lock
In A, lock released

```

### 2.锁定对象

可重入锁(RLock)是另一种同步原语，可以被同一个线程多次获取，而不会进入死锁状态。RLock 对象知道哪个线程拥有锁的权限，并且同一个线程可以解锁它。

RLock 对象处于两种状态之一:“锁定”和“解锁”。创建 RLock 对象时，它处于“解锁”状态。RLock 对象中只有两种方法:

*   ***acquire():*** 该方法将锁对象从“解锁”状态改为“锁定”状态，并允许调用线程继续执行。如果同一个线程再次调用此方法，它会将递归级别增加一级。为了完全释放锁，同一个线程需要调用`release()`相同的次数。如果另一个线程在“锁定”状态下调用此方法，该线程将被阻塞。
*   ***release():*** 这个方法释放锁，递归层次减一。如果递减后递归级别变为 0，则锁定状态变为“未锁定”状态。如果递减后递归级别仍然是非零的，锁保持“锁定”状态，并由调用线程拥有。如果 RLock 对象已经处于“解锁”状态，则引发`RuntimeError`。

```py
import threading

lock = threading.RLock()

def funcA():
  print("In A, acquiring lock")
  lock.acquire()

  print("In A, lock acquired, recursion level = 1")

  print("In A, acquiring lock again")
  lock.acquire()

  print("In A, lock acquired again, recursion level = 2")

  print("In A, releasing lock")
  lock.release()

  print("In A, lock released, recursion level = 1")

def funcB():
  print("In B, trying to acquire lock, but A released only once, so entering in deadlock state")
  lock.acquire()
  print("This statement won't be executed")

if __name__ == "__main__":
  thread1 = threading.Thread(target=funcA)
  thread2 = threading.Thread(target=funcB)

  thread1.start()
  thread2.start()

  thread1.join()
  thread2.join()

```

```py
In A, acquiring l
In A, lock acquired, recursion level = 1
In A, acquiring lock again
In A, lock acquired again, recursion level = 2
In A, releasing lock
In A, lock released, recursion level = 1
In B, trying to acquire lock, but A released only once, so entering in deadlock state

```

### 3.信号灯

信号量只是一个非负的变量，在线程间共享。虽然`Lock`和`RLock`对象只允许一个线程执行，但 Semaphore 允许一次执行多个线程。信号量用于通过指定创建信号量对象时允许执行的线程数量来保护容量有限的资源。如果这个初始计数是 1，信号量可以帮助线程同步。

*   ***创建信号量:*** 创建信号量对象，在线程模块中调用`Semaphore(count)`，其中`count`是允许同时访问的线程数。计数的默认值是 1。
*   ***acquire():*** 当线程调用此方法时
    *   如果信号量的计数值为 0，线程将被阻塞，直到被调用`release()`唤醒。
    *   如果信号量的计数值大于 0，则它递减 1，线程继续执行。
*   ***release():*** 该方法将计数值递增 1。如果任何线程在`acquire()`上被阻塞，它会解除其中一个线程的阻塞。

让我们举一个例子，10 个线程试图读取一个共享资源，但是我们使用信号量将对共享资源的并发读取限制为 3 个。

```py
import threading
import time

read_mutex = threading.Semaphore(3)

# Our shared resource
data = "A Data Stream"

class ReaderThread(threading.Thread):
  def __init__(self):
    threading.Thread.__init__(self)

  def run(self):

    read_mutex.acquire()

    output = self.getName() + " starts reading"
    print(output)

    # threads take time to read a data
    time.sleep(0.5)
    some_data = data

    output = self.getName() + " ends reading"
    print(output)

    read_mutex.release()

if __name__ == "__main__":

  threads = []
  for i in range(10):
    threads.append(ReaderThread())

  for thread in threads:
    thread.start()

  for thread in threads:
    thread.join()

```

```py
Thread-6 starts reading
Thread-7 starts reading
Thread-8 starts reading
Thread-8 ends readingThread-7 ends readingThread-6 ends reading

Thread-10 starts reading
Thread-11 starts reading
Thread-9 starts reading
Thread-11 ends readingThread-10 ends reading
Thread-12 starts reading

Thread-13 starts reading
Thread-9 ends reading
Thread-14 starts reading
Thread-13 ends readingThread-12 ends reading

Thread-15 starts reading
Thread-14 ends reading
Thread-15 ends reading

```

## 结论

在本教程中，我们学习了 Python 中的同步，通过使用 Python 中的[线程模块来避免竞争情况。我们使用 Lock、RLock 和信号量来实现 Python 中的同步。感谢阅读！！](https://www.askpython.com/python-modules/multiprocessing-in-python)