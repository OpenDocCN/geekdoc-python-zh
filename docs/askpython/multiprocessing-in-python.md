# Python 中的多重处理

> 原文：<https://www.askpython.com/python-modules/multiprocessing-in-python>

嘿伙计们！在本文中，我们将学习 Python 中的多重处理。那么，我们开始吧。

## 什么是多重处理？

多重处理是 python 中的一个包，它支持生成利用 Python API 的进程的能力。它类似于 Python 中的线程模块。

## 理解 Python 中的多重处理

多处理器是指计算机有一个以上的中央处理器。如果一台计算机只有一个多核处理器，可以使用 Python 中的[多线程并行运行任务。](https://www.askpython.com/python-modules/multithreading-in-python)

多处理器系统能够同时支持多个处理器。为了找到我们系统中可用的 CPU 内核的数量，我们使用了 ***mp.cpu_count()*** 函数。

在本文中，我们将使用 Python 的多重处理模块

**下面是使用多处理模块在 Python 中查找处理器数量的示例代码:**

```py
import multiprocessing as mp

print(mp.cpu_count())

```

输出: **12**

这里的计数是多个处理器之间的内核总数。

**该模块的四个最重要的类是-**

*   流程类
*   锁定级
*   队列类别
*   池类

让我们分别看看这些类中的每一个…

### 1.流程类

进程是当前进程的分叉副本。它创建一个新的进程标识符，任务作为独立的子进程运行。

***start()*** 和 ***join()*** 函数都属于这个类。为了在进程中传递一个参数，我们使用 ***args*** 关键字。

start()函数示例-

在这里，我们创建了一个函数 *calc_square* 和 *calc_cube* ，分别用于求数字的平方和立方。在主函数中，我们创建了对象 *p1* 和 *p2* 。 *p1.start()* 和 *p2.start()* 将启动函数，调用 *p1.join()和 p2.join* 将终止进程。

```py
import time
import multiprocessing

def calc_square(numbers):
	for n in numbers:
		print('square ' + str(n*n))

def calc_cube(numbers):
	for n in numbers:
		print('cube '+ str(n*n*n))

if __name__ == "__main__":
	arr=[2,3,8,9]
	p1=multiprocessing.Process(target=calc_square,args=(arr,))
	p2=multiprocessing.Process(target=calc_cube,args=(arr,))

	p1.start()
	p2.start()

	p1.join()
	p2.join()

	print("Done")

```

输出:

```py
square 4
square 9
square 64
square 81
cube 8
cube 27
cube 512
cube 729
Done

```

### 2.锁定级

lock 类允许代码被锁定，以确保没有其他进程可以执行类似的代码，直到它被释放。

要认领锁， ***使用获取()*** 函数，要释放锁， ***使用释放()*** 函数。

```py
from multiprocessing import Process, Lock

lock=Lock()
def printer(data):
  lock.acquire()
  try:
      print(data)
  finally:
      lock.release()

if __name__=="__main__":
  items=['mobile','computer','tablet']
  for item in items:
     p=Process(target=printer,args=(item,))
     p.start()

```

输出

```py
mobile
computer
tablet

```

### 3.队列类别

队列是一种使用先进先出(FIFO)技术的数据结构。它帮助我们使用本地 Python 对象执行进程间通信。

当作为参数传递时，队列使进程能够使用共享数据。

***put()*** 函数用于向队列中插入数据， ***get()*** 函数用于从队列中消耗数据。

```py
import multiprocessing as mp

def sqr(x,q):
	q.put(x*x)

if __name__ == "__main__":
	q=mp.Queue() # Instance of queue class created
	processes=[mp.Process(target=sqr,args=(i,q))for i in range (2,10)] # List of processes within range 2 to 10
	for p in processes:
		p.start()

	for p in processes:
		p.join()

	result = [q.get() for p in processes]
	print(result)

```

输出:

```py
[4, 9, 16, 25, 36, 64, 49, 81]

```

### 4.池类

pool 类帮助我们针对多个输入值并行执行一个函数。这个概念叫做数据并行。

这里，数组[5，9，8]被映射为函数调用中的输入。pool.map()函数用于传递多个参数的列表。

```py
import multiprocessing as mp

def my_func(x):
  print(x**x)

def main():
  pool = mp.Pool(mp.cpu_count())
  result = pool.map(my_func, [5,9,8])

if __name__ == "__main__":
  main()

```

输出:

```py
3125
387420489
16777216

```

## 结论

在本文中，我们学习了 Python 中多处理的四个最重要的类——进程、锁、队列和池，它们可以更好地利用 CPU 内核并提高性能。

## 参考

[官方模块文档](https://docs.python.org/3/library/multiprocessing.html)