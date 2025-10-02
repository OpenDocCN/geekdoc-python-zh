# 六、错误和异常

## 错误和异常 (1)

虽然在前面的学习中，已经遇到了错误和异常问题，但是一直没有很认真的研究它。现在来近距离观察错误和异常。

### 错误

Python 中的错误之一是语法错误(syntax errors)，比如：

```py
>>> for i in range(10)
  File "<stdin>", line 1
    for i in range(10)
                     ^
SyntaxError: invalid syntax 
```

上面那句话因为缺少冒号`:`，导致解释器无法解释，于是报错。这个报错行为是由 Python 的语法分析器完成的，并且检测到了错误所在文件和行号（`File "<stdin>", line 1`），还以向上箭头`^`标识错误位置（后面缺少`:`），最后显示错误类型。

错误之二是在没有语法错误之后，会出现逻辑错误。逻辑错误可能会由于不完整或者不合法的输入导致，也可能是无法生成、计算等，或者是其它逻辑问题。

当 Python 检测到一个错误时，解释器就无法继续执行下去，于是抛出异常。

### 异常

看一个异常（让 0 做分母了，这是小学生都相信会有异常的）：

```py
>>> 1/0
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ZeroDivisionError: integer division or modulo by zero 
```

当 Python 抛出异常的时候，首先有“跟踪记录(Traceback)”，还可以给它取一个更优雅的名字“回溯”。后面显示异常的详细信息。异常所在位置（文件、行、在某个模块）。

最后一行是错误类型以及导致异常的原因。

下表中列出常见的异常

| 异常 | 描述 |
| --- | --- |
| NameError | 尝试访问一个没有申明的变量 |
| ZeroDivisionError | 除数为 0 |
| SyntaxError | 语法错误 |
| IndexError | 索引超出序列范围 |
| KeyError | 请求一个不存在的字典关键字 |
| IOError | 输入输出错误（比如你要读的文件不存在） |
| AttributeError | 尝试访问未知的对象属性 |

为了能够深入理解，依次举例，展示异常的出现条件和结果。

#### NameError

```py
>>> bar
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'bar' is not defined 
```

Python 中变量需要初始化，即要赋值。虽然不需要像某些语言那样声明，但是要赋值先。因为变量相当于一个标签，要把它贴到对象上才有意义。

#### ZeroDivisionError

```py
>>> 1/0
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ZeroDivisionError: integer division or modulo by zero 
```

貌似这样简单的错误时不会出现的，但在实际情境中，可能没有这么容易识别，所以，依然要小心为妙。

#### SyntaxError

```py
>>> for i in range(10)
  File "<stdin>", line 1
    for i in range(10)
                     ^
SyntaxError: invalid syntax 
```

这种错误发生在 Python 代码编译的时候，当编译到这一句时，解释器不能讲代码转化为 Python 字节码，就报错。只有改正才能继续。所以，它是在程序运行之前就会出现的（如果有错）。现在有不少编辑器都有语法校验功能，在你写代码的时候就能显示出语法的正误，这多少会对编程者有帮助。

#### IndexError

```py
>>> a = [1,2,3]
>>> a[4]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: list index out of range

>>> d = {"python":"itdiffer.com"}
>>> d["java"]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'java' 
```

这两个都属于“鸡蛋里面挑骨头”类型，一定得报错了。不过在编程实践中，特别是循环的时候，常常由于循环条件设置不合理出现这种类型的错误。

#### IOError

```py
>>> f = open("foo")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IOError: [Errno 2] No such file or directory: 'foo' 
```

如果你确认有文件，就一定要把路径写正确，因为你并没有告诉 Python 对你的 computer 进行全身搜索，所以，Python 会按照你指定位置去找，找不到就异常。

#### AttributeError

```py
>>> class A(object): pass
... 
>>> a = A()
>>> a.foo
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'A' object has no attribute 'foo' 
```

属性不存在。这种错误前面多次见到。

其实，Python 内建的异常也不仅仅上面几个，上面只是列出常见的异常中的几个。比如还有：

```py
>>> range("aaa")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: range() integer end argument expected, got str. 
```

总之，如果读者在调试程序的时候遇到了异常，不要慌张，这是好事情，是 Python 在帮助你修改错误。只要认真阅读异常信息，再用 `dir()`，`help()` 或者官方网站文档、google 等来协助，一定能解决问题。

### 处理异常

在一段程序中，为了能够让程序健壮，必须要处理异常。举例：

```py
#!/usr/bin/env Python
# coding=utf-8

while 1:
    print "this is a division program."
    c = raw_input("input 'c' continue, otherwise logout:")
    if c == 'c':
        a = raw_input("first number:")
        b = raw_input("second number:")
        try:
            print float(a)/float(b)
            print "*************************"
        except ZeroDivisionError:
            print "The second number can't be zero!"
            print "*************************"
    else:
        break 
```

运行这段程序，显示如下过程：

```py
$ python 21601.py 
this is a division program.
input 'c' continue, otherwise logout:c
first number:5
second number:2
2.5
*************************
this is a division program.
input 'c' continue, otherwise logout:c
first number:5
second number:0
The second number can't be zero!
*************************
this is a division program.
input 'c' continue, otherwise logout:d
$ 
```

从运行情况看，当在第二个数，即除数为 0 时，程序并没有因为这个错误而停止，而是给用户一个友好的提示，让用户有机会改正错误。这完全得益于程序中“处理异常”的设置，如果没有“处理异常”，异常出现，就会导致程序终止。

处理异常的方式之一，使用 `try...except...`。

对于上述程序，只看 try 和 except 部分，如果没有异常发生，except 子句在 try 语句执行之后被忽略；如果 try 子句中有异常可，该部分的其它语句被忽略，直接跳到 except 部分，执行其后面指定的异常类型及其子句。

except 后面也可以没有任何异常类型，即无异常参数。如果这样，不论 try 部分发生什么异常，都会执行 except。

在 except 子句中，可以根据异常或者别的需要，进行更多的操作。比如：

```py
#!/usr/bin/env Python
# coding=utf-8

class Calculator(object):
    is_raise = False
    def calc(self, express):
        try:
            return eval(express)
        except ZeroDivisionError:
            if self.is_raise:
                print "zero can not be division."
            else:
                raise 
```

在这里，应用了一个函数 `eval()`，它的含义是：

```py
eval(...)
    eval(source[, globals[, locals]]) -> value

    Evaluate the source in the context of globals and locals.
    The source may be a string representing a Python expression
    or a code object as returned by compile().
    The globals must be a dictionary and locals can be any mapping,
    defaulting to the current globals and locals.
    If only globals is given, locals defaults to it. 
```

例如：

```py
>>> eval("3+5")
8 
```

另外，在 except 子句中，有一个 `raise`，作为单独一个语句。它的含义是将异常信息抛出。并且，except 子句用了一个判断语句，根据不同的情况确定走不同分支。

```py
if __name__ == "__main__":
    c = Calculator()
    print c.calc("8/0") 
```

这时候 `is_raise = False`，则会：

```py
$ python 21602.py 
Traceback (most recent call last):
  File "21602.py", line 17, in <module>
    print c.calc("8/0")
  File "21602.py", line 8, in calc
    return eval(express)
  File "<string>", line 1, in <module>
ZeroDivisionError: integer division or modulo by zero 
```

如果将 `is_raise` 的值改为 True，就是这样了：

```py
if __name__ == "__main__":
    c = Calculator()
    c.is_raise = True    #通过实例属性修改
    print c.calc("8/0") 
```

运行结果：

```py
$ python 21602.py 
zero can not be division.
None 
```

最后的 None 是 `c.calc("8/0")`的返回值，因为有 `print c.calc("8/0")`，所以被打印出来。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 错误和异常 (2)

try...except...是处理异常的基本方式。在原来的基础上，还可有扩展。

### 处理多个异常

处理多个异常，并不是因为同时报出多个异常。程序在运行中，只要遇到一个异常就会有反应，所以，每次捕获到的异常一定是一个。所谓处理多个异常的意思是可以容许捕获不同的异常，有不同的 except 子句处理。

```py
#!/usr/bin/env Python
# coding=utf-8

while 1:
    print "this is a division program."
    c = raw_input("input 'c' continue, otherwise logout:")
    if c == 'c':
        a = raw_input("first number:")
        b = raw_input("second number:")
        try:
            print float(a)/float(b)
            print "*************************"
        except ZeroDivisionError:
            print "The second number can't be zero!"
            print "*************************"
        except ValueError:
            print "please input number."
            print "************************"
    else:
        break 
```

将上节的一个程序进行修改，增加了一个 except 子句，目的是如果用户输入的不是数字时，捕获并处理这个异常。测试如下：

```py
$ python 21701.py 
this is a division program.
input 'c' continue, otherwise logout:c
first number:3
second number:"hello"        #输入了一个不是数字的东西
please input number.         #对照上面的程序，捕获并处理了这个异常
************************
this is a division program.
input 'c' continue, otherwise logout:c
first number:4
second number:0
The second number can't be zero!
*************************
this is a division program.
input 'c' continue, otherwise logout:4
$ 
```

如果有多个 except，在 try 里面如果有一个异常，就转到相应的 except 子句，其它的忽略。如果 except 没有相应的异常，该异常也会抛出，不过这是程序就要中止了，因为异常“浮出”程序顶部。

除了用多个 except 之外，还可以在一个 except 后面放多个异常参数，比如上面的程序，可以将 except 部分修改为：

```py
except (ZeroDivisionError, ValueError):
    print "please input rightly."
    print "********************" 
```

运行的结果就是：

```py
$ python 21701.py 
this is a division program.
input 'c' continue, otherwise logout:c
first number:2
second number:0           #捕获异常
please input rightly.
********************
this is a division program.
input 'c' continue, otherwise logout:c
first number:3
second number:a           #异常
please input rightly.
********************
this is a division program.
input 'c' continue, otherwise logout:d
$ 
```

需要注意的是，except 后面如果是多个参数，一定要用圆括号包裹起来。否则，后果自负。

突然有一种想法，在对异常的处理中，前面都是自己写一个提示语，发现自己写的不如内置的异常错误提示更好。希望把它打印出来。但是程序还能不能中断。Python 提供了一种方式，将上面代码修改如下：

```py
while 1:
    print "this is a division program."
    c = raw_input("input 'c' continue, otherwise logout:")
    if c == 'c':
        a = raw_input("first number:")
        b = raw_input("second number:")
        try:
            print float(a)/float(b)
            print "*************************"
        except (ZeroDivisionError, ValueError), e:
            print e
            print "********************"
    else:
        break 
```

运行一下，看看提示信息。

```py
$ python 21702.py 
this is a division program.
input 'c' continue, otherwise logout:c
first number:2
second number:a                         #异常
could not convert string to float: a
********************
this is a division program.
input 'c' continue, otherwise logout:c
first number:2
second number:0                         #异常
float division by zero
********************
this is a division program.
input 'c' continue, otherwise logout:d
$ 
```

> 在 Python3.x 中，常常这样写：`except (ZeroDivisionError, ValueError) as e:`

以上程序中，之处理了两个异常，还可能有更多的异常呢？如果要处理，怎么办？可以这样：`execpt:` 或者 `except Exception, e`，后面什么参数也不写就好了。

### else 子句

有了 `try...except...`，在一般情况下是够用的，但总有不一般的时候出现，所以，就增加了一个 else 子句。其实，人类的自然语言何尝不是如此呢？总要根据需要添加不少东西。

```py
>>> try:
...     print "I am try"
... except:
...     print "I am except"
... else:
...     print "I am else"
... 
I am try
I am else 
```

这段演示，能够帮助读者理解 else 的执行特点。如果执行了 try，则 except 被忽略，但是 else 被执行。

```py
>>> try:
...     print 1/0
... except:
...     print "I am except"
... else:
...     print "I am else"
... 
I am except 
```

这时候 else 就不被执行了。

理解了 else 的执行特点，可以写这样一段程序，还是类似于前面的计算，只不过这次要求，如果输入的有误，就不断要求从新输入，知道输入正确，并得到了结果，才不再要求输入内容，程序结束。

在看下面的参考代码之前，读者是否可以先自己写一段呢？并调试一下，看看结果如何。

```py
#!/usr/bin/env Python
# coding=utf-8
while 1:
    try:
        x = raw_input("the first number:")
        y = raw_input("the second number:")

        r = float(x)/float(y)
        print r
    except Exception, e:
        print e
        print "try again."
    else:
        break 
```

先看运行结果：

```py
$ python 21703.py
the first number:2
the second number:0        #异常，执行 except
float division by zero
try again.                 #循环
the first number:2
the second number:a        #异常 
could not convert string to float: a
try again.
the first number:4
the second number:2        #正常，执行 try
2.0                        #然后 else：break，退出程序
$ 
```

相当满意的执行结果。

需要对程序中的 except 简单说明，这次没有像前面那样写，而是 `except Exception, e`，意思是不管什么异常，这里都会捕获，并且传给变量 e，然后用 `print e` 把异常信息打印出来。

### finally

finally 子句，一听这个名字，就感觉它是做善后工作的。的确如此，如果有了 finally，不管前面执行的是 try，还是 except，它都要执行。因此一种说法是用 finally 用来在可能的异常后进行清理。比如：

```py
>>> x = 10

>>> try:
...     x = 1/0
... except Exception, e:
...     print e
... finally:
...     print "del x"
...     del x
... 
integer division or modulo by zero
del x 
```

看一看 x 是否被删除？

```py
>>> x
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'x' is not defined 
```

当然，在应用中，可以将上面的各个子句都综合起来使用，写成如下样式：

```py
try:
    do something
except:
    do something
else:
    do something
finally
    do something 
```

### 和条件语句相比

`try...except...`在某些情况下能够替代 `if...else..` 的条件语句。这里我无意去比较两者的性能，因为看到有人讨论这个问题。我个人觉得这不是主要的，因为它们之间性能的差异不大。主要是你的选择。一切要根据实际情况而定，不是说用一个就能包打天下。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 错误和异常 (3)

按照一般的学习思路，掌握了前两节内容，已经足够编程所需了。但是，我还想再多一步，还是因为本教程的读者是要 from beginner to master。

### assert

```py
>>> assert 1==1
>>> assert 1==0
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AssertionError 
```

从上面的举例中可以基本了解了 assert 的特点。

assert，翻译过来是“断言”之意。assert 是一句等价于布尔真的判定，发生异常就意味着表达式为假。

assert 的应用情景就有点像汉语的意思一样，当程序运行到某个节点的时候，就断定某个变量的值必然是什么，或者对象必然拥有某个属性等，简单说就是断定什么东西必然是什么，如果不是，就抛出错误。

```py
#!/usr/bin/env Python
# coding=utf-8

class Account(object):
    def __init__(self, number):
        self.number = number
        self.balance = 0

    def deposit(self, amount):
        assert amount > 0
        self.balance += balance

    def withdraw(self, amount):
        assert amount > 0
        if amount <= self.balance:
            self.balance -= amount
        else:
            print "balance is not enough." 
```

上面的程序中，deposit() 和 withdraw() 方法的参数 amount 值必须是大于零的，这里就用断言，如果不满足条件就会报错。比如这样来运行：

```py
if __name__ == "__main__":
    a = Account(1000)
    a.deposit(-10) 
```

出现的结果是：

```py
$ python 21801.py
Traceback (most recent call last):
  File "21801.py", line 22, in <module>
    a.deposit(-10)
  File "21801.py", line 10, in deposit
    assert amount > 0
AssertionError 
```

这就是断言 assert 的引用。什么是使用断言的最佳时机？有文章做了总结：

如果没有特别的目的，断言应该用于如下情况：

*   防御性的编程
*   运行时对程序逻辑的检测
*   合约性检查（比如前置条件，后置条件）
*   程序中的常量
*   检查文档

(上述要点来自：[Python 使用断言的最佳时机](http://www.oschina.net/translate/when-to-use-assert) )

不论是否理解，可以先看看，请牢记，在具体开发过程中，有时间就回来看看本教程，不断加深对这些概念的理解，这也是 master 的成就之法。

最后，引用危机百科中对“异常处理”词条的说明，作为对“错误和异常”部分的总结（有所删改）：

> 异常处理，是编程语言或计算机硬件里的一种机制，用于处理软件或信息系统中出现的异常状况（即超出程序正常执行流程的某些特殊条件）。
> 
> 各种编程语言在处理异常方面具有非常显著的不同点（错误检测与异常处理区别在于：错误检测是在正常的程序流中，处理不可预见问题的代码，例如一个调用操作未能成功结束）。某些编程语言有这样的函数：当输入存在非法数据时不能被安全地调用，或者返回值不能与异常进行有效的区别。例如，C 语言中的 atoi 函数（ASCII 串到整数的转换）在输入非法时可以返回 0。在这种情况下编程者需要另外进行错误检测（可能通过某些辅助全局变量如 C 的 errno），或进行输入检验（如通过正则表达式），或者共同使用这两种方法。
> 
> 通过异常处理，我们可以对用户在程序中的非法输入进行控制和提示，以防程序崩溃。
> 
> 从进程的视角，硬件中断相当于可恢复异常，虽然中断一般与程序流本身无关。
> 
> 从子程序编程者的视角，异常是很有用的一种机制，用于通知外界该子程序不能正常执行。如输入的数据无效（例如除数是 0），或所需资源不可用（例如文件丢失）。如果系统没有异常机制，则编程者需要用返回值来标示发生了哪些错误。
> 
> 一段代码是异常安全的，如果这段代码运行时的失败不会产生有害后果，如内存泄露、存储数据混淆、或无效的输出。
> 
> Python 语言对异常处理机制是非常普遍深入的，所以想写出不含 try, except 的程序非常困难。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。