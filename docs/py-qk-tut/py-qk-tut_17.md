## Python 进阶 08 错误处理

[`www.cnblogs.com/vamei/archive/2012/07/10/2582787.html`](http://www.cnblogs.com/vamei/archive/2012/07/10/2582787.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

在项目开发中，错误处理是不可或缺的。错误处理帮助人们 debug，通过更加丰富的信息，让人们更容易找到 bug 的所在。错误处理还可以提高程序的容错性。

我们之前在讲循环对象的时候，曾提到一个 StopIteration 的错误，该错误是在循环对象穷尽所有元素时的报错。

我们以它为例，来说明基本的错误处理。

一个包含错误的程序:

```py
re = iter(range(5)) for i in range(100): print re.next() print 'HaHaHaHa'

```

首先，我们定义了一个循环对象 re，该循环对象将进行 5 次循环，每次使用序列的一个元素。

在随后的 for 循环中，我们手工调用 next()函数。当循环进行到第 6 次的时候，re.next()不会再返回元素，而是举出(raise)StopIteration 的错误。整个程序将会中断。

我们可以修改以上错误程序，直到完美的没有 bug。但另一方面，如果我们在写程序的时候，知道这里可能犯错以及可能的犯错类型，我们可以针对该错误类型定义好”应急预案“。

```py
re = iter(range(5)) try: for i in range(100): print re.next() except StopIteration: print 'here is end ',i print 'HaHaHaHa'

```

在 try 程序段中，我们放入容易犯错的部分。我们可以跟上 except，来说明如果在 try 部分的语句发生 StopIteration 时，程序该做的事情。如果没有发生错误，则 except 部分被跳过。

随后，程序将继续运行，而不是彻底中断。

完整的语法结构如下：

```py
try:
    ... except error1:
    ... except error2:
    ... else:
    ... finally:
    ...

```

else 是指所有其它的错误。finally 是无论何种情况，最后都要做的一些事情。流程如下，try->except/else->finally

我们也可以自己写一个举出错误的例子:

```py
print 'Lalala'
raise StopIteration print 'Hahaha'

```

(注意，这个例子不具备任何实际意义。读者可探索更多有意义的例子。)

StopIteration 是一个类。当我们 raise 它的时候，有一个中间环节，就是 Python 利用 StopIteration 生成一个该类的一个对象。Python 实际上举出的，是这一个对象。当然，也可以直接写成:

总结:

try: ... except error: ... else: ... finally: ...
raise error