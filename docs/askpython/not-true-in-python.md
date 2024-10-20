# Python 中什么是“不真实”？

> 原文：<https://www.askpython.com/python/examples/not-true-in-python>

在本文中，我们将讨论 Python 中的**非真**概念。总的来说，这不是一个无聊的讲座，而是我们只是在完善我们的基础知识。

## Python 中的运算符和关键字

Python 有大量的操作符。这就是为什么与其他编程语言不同，它的语法相对简单。下面是 Python 中的[操作符列表:](https://www.askpython.com/course/python-course-operators)

1.  +:加号
2.  –:减号
3.  =:赋值运算符
4.  ==:等于运算符
5.  ！=:不等于
6.  < =:小于或等于
7.  > =:大于或等于
8.  %:模数
9.  //:楼层划分
10.  -=:减量
11.  +=:增量
12.  /=:除
13.  %=:模数

这些支持算术运算，但最重要的是，我们还有一些其他的逻辑运算:

1.  &:和
2.  |:或者
3.  不

另外，看看布尔运算:

1.  **真**
2.  **假**

我们可以像使用关键字一样使用逻辑操作符。但是，在 Python 中，我们没有任何运算符用于 **not** 或 **complement** 。显然，还有**！="** 但适合小型操作。对于复杂的操作，我们可以使用**“not”**关键字使事情变得简单。

## Python 中“非”的意义

光是这个例子就足以证明**而不是**是多么有用:

**预测 while 循环是否运行**

**代码:**

```py
condition = not True
while(condition):
    print("Hello world")

```

代码不会运行。当且仅当括号内的条件为真时， [while 循环](https://www.askpython.com/course/python-course-while-loop)迭代代码。这里条件不为真意味着它为假。如果您在空闲状态下运行这个小代码片段，那么输出也将为 False。

```py
>>> not True

```

```py
False

```

所以，这就是 not 运算符的意义。

## Python 中“真”的意义

**True** 是 Python 中的布尔运算符。其意义在于，人们可以设置标志、运行循环以及用它做更多的事情。让我们看一个例子:

在屏幕上打印 n 次“hello”。

```py
while True:
    print("Hello")

```

**输出:**

```py
Hello
Hello
Hello
Hello
...
...
Runtime Error occurred

```

最后一条消息是**“发生运行时错误”**。这意味着当我们使用 **True** 无限运行循环，并且没有循环控制语句时，它会继续执行这段代码 n 次。这是需要注意的。

## not 和 True 一起使用

这里我们将构建一个代码来检查每个数字，并打印出它是否是质数。

```py
num = int(input("Enter a number: "))
isPrime = not True
num_sqrt = int(num**0.5)

if(num > 1):
	for i in range(2, num_sqrt + 1):
		if (num % i == 0):
			isPrime = True
			break
	if (isPrime == (not True)):
		print("%d is prime" %(num))
	else:
		print("%d is composite" %(num))
else:
	print("%d is composite" %(num))

```

**输出:**

```py
>>> Enter a number: 39
39 is not prime

>> Enter a number: 17
17 is prime

```

**说明:**

1.  首先，接受 num 的输入。
2.  然后设置一个名为 isPrime 的变量。这只是一个指示器，最初指示一个值**不为真**。
3.  然后我们取这个数的平方根。
4.  然后我们放一个条件，如果数大于 1。它运行一个循环，从 **2 迭代到(数字的平方根+1)。**
5.  然后对于每次迭代，我们检查这个数是否能被它自己整除。如果是，则指示符 isPrime 被设置为真。这意味着这个数是质数。
6.  如果不是这种情况，那么数字是合成的。

在这里，不真实和虚假一起起作用。解释的主要动机是我们可以用它来代替 **False** 。

## 结论

这样，我们可以一起使用不真实的概念，希望这篇文章是有帮助的，我们开始知道，我们可以灵活地使用 Python 的概念，为我们的利益。