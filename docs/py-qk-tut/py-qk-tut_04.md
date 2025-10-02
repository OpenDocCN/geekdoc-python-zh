## Python 基础 05 缩进和选择

[`www.cnblogs.com/vamei/archive/2012/05/29/2524706.html`](http://www.cnblogs.com/vamei/archive/2012/05/29/2524706.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

python 最具特色的就是用缩进来写模块。我们下面以 if 选择结构来举例。

先看 C 语言的表达方式（注意，这是 C，不是 Python!）

```py
if ( i > 0 )
{
    x = 1;
    y = 2;
}

```

这个语句是说，如果 i>1 的话，我们将进行括号中所包括的两个赋值操作。

括号中包含的就是块操作，它表明了其中的语句隶属于 if 

在 python 中，同样的目的，这段话是这样的

在 python 中， 去除了 i > 0 周围的括号，去除了每个语句句尾的分号，还去除了表示块的花括号。

多出来了 if ...之后的:(冒号), 还有就是 x = 1 和 y =2 前面有四个空格的缩进。通过这些缩进，python 可以识别这两个语句是隶属于 if 的。

（python 这样设计的理由是：程序会看起来很好看）

我们写一个完整的程序，命名为 ifDemo.py

```py
i = 1 x = 1
if i > 0:
    x = x+1
print x

```

$python ifDemo.py  # 运行

这个程序在顺序运行到 if 的时候，检测到真值（True），执行 x = x+1, 在此之后，print x 语句没有缩进，那么就是 if 之外。

如果将第一句改成 i = -1，那么 if 遇到假值 (False), 由于 x = x+1 隶属于 if, 这一句就跳过。 print x 没有缩进，所以是 if 之外，不会跳过，继续执行。

这种以四个空格的缩进来表示隶属关系的书写方式，我们以后还会经常看到。Python 很强调程序的可读性，这种强制的缩进要求实际上是在帮程序员写出整洁的程序。

复杂一些的选择的例子：

```py
i = 1
if i > 0:
    print 'positive i'
    i = i + 1
elif i == 0:
    print 'i is 0'
    i = i * 10
else:
    print 'negative i'
    i = i - 1
print 'new i:',i

```

```py
这里有三个块，分别以 if, elif, else 引领。
python 顺序检测所跟随的条件，如果发现为假，那么跳过后面紧跟的块，检测下一个条件； 如果发现为真，那么执行后面紧跟的块，跳过剩下的块
(else 等同于 elif True)
整个 if 可以做为一句语句放在另一个 if 语句的块中 

```

```py
i  = 5
if i > 1: print 'i bigger than 1'
    print 'good'
    if i > 2: print 'i bigger than 2'
        print 'even better'

```

我们可以看到， if i > 2 后面的块相对于该 if 缩进了四个空格，以表明其隶属于该 if

总结：

if 语句之后的冒号

以四个空格的缩进来表示隶属关系, python 中不能随意缩进

if  <条件 1>:

    statement

elif <条件 2>:

    statement

elif <条件 3>：

    statement

else:

    statement