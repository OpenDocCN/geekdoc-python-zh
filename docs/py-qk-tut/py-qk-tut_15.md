## Python 进阶 06 循环对象

[`www.cnblogs.com/vamei/archive/2012/07/09/2582499.html`](http://www.cnblogs.com/vamei/archive/2012/07/09/2582499.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

通过上面一讲，我们再次熟悉了 Python 里的循环控制。现在，我们将接触循环对象(iterable object)。

这一讲的主要目的是为了大家在读 Python 程序的时候对循环对象有一个基本概念。

循环对象的并不是随着 Python 的诞生就存在的，但它的发展迅速，特别是 Python 3x 的时代，从 zip()或者 map()的改变来看，循环对象正在成为循环的标准形式。

1\. 什么是循环对象

循环对象是这样一个对象，它包含有一个 next()方法(__next__()方法，在 python 3x 中)， 这个方法的目的是进行到下一个结果，而在结束一系列结果之后，举出 StopIteration 错误。

当一个循环结构（比如 for）调用循环对象时，它就会每次循环的时候调用 next()方法，直到 StopIteration 出现，for 循环接收到，就知道循环已经结束，停止调用 next()。

假设我们有一个 test.txt 的文件:

我们运行一下 python 命令行：

>>> f = open('test.txt')

>>> f.next()

>>> f.next()

...

不断地输入 f.next()，直到最后出现 StopIteration

open()返回的实际上是一个循环对象，包含有 next()方法。而该 next()方法每次返回的就是新的一行的内容，到达文件结尾时举出 StopIteration。这样，我们相当于手工进行了循环。

自动进行的话，就是：

```py
for line in open('test.txt'): print line

```

在这里，for 结构自动调用 next()方法，将该方法的返回值赋予给 line。循环知道出现 StopIteration 的时候结束。

相对于序列，用循环对象来控制循环的好处在于：可以不用在循环还没有开始的时候，就生成每次要使用的元素。所使用的元素在循环过程中逐次生成。这样，就节省了空间，提高了效率，并提高编程的灵活性。

2\. iter()函数和循环器(iterator)

从技术上来说，循环对象和 for 循环调用之间还有一个中间层，就是要将循环对象转换成循环器(iterator)。这一转换是通过使用 iter()函数实现的。但从逻辑层面上，常常可以忽略这一层，所以循环对象和循环器常常相互指代对方。

3\. 生成器(generator)

生成器的主要目的是构成一个用户自定义的循环对象。

生成器的编写方法和函数定义类似，只是在 return 的地方改为 yield。生成器中可以有多个 yield。当生成器遇到一个 yield 时，会暂停运行生成器，返回 yield 后面的值。当再次调用生成器的时候，会从刚才暂停的地方继续运行，直到下一个 yield。生成器自身又构成一个循环器，每次循环使用一个 yield 返回的值。

下面是一个生成器:

```py
def gen():
    a = 100
    yield a
    a = a*8
    yield a yield 1000

```

该生成器共有三个 yield， 如果用作循环器时，会进行三次循环。

再考虑如下一个生成器:

```py
def gen(): for i in range(4): yield i

```

它又可以写成生成器表达式(Generator Expression):

```py
G = (x for x in range(4))

```

生成器表达式是生成器的一种简便的编写方式。读者可进一步查阅。

4\. 表理解(list comprehension)

表理解是快速生成表的方法。假设我们生成表 L：

```py
L = [] for x in range(10):
    L.append(x**2)

```

以上产生了表 L，但实际上有快捷的写法，也就是表理解的方式:

```py
L = [x**2 for x in range(10)]

```

这与生成器表达式类似，只不过用的是中括号。

（表理解的机制实际上是利用循环对象，有兴趣可以查阅。）

考虑下面的表理解会生成什么？

```py
xl = [1,3,5]
yl = [9,12,13]
L = [ x**2 for (x,y) in zip(xl,yl) if y > 10]

```

总结：

循环对象

生成器

表理解