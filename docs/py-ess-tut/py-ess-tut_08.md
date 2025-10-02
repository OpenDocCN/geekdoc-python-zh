# 第八章 异常

> 来源：[`www.cnblogs.com/Marlowes/p/5428641.html`](http://www.cnblogs.com/Marlowes/p/5428641.html)
> 
> 作者：Marlowes

在编写程序的时候，程序员通常需要辨别事件的正常过程和异常(非正常)的情况。这类异常事件可能是错误(比如试图除以`0`)，或者是不希望经常发生的事情。为了能够处理这些异常事件，可以在所有可能发生这类事件的地方都使用条件语句(比如让程序检查除法的分母是否为零)。但是，这么做可能不仅会没效率和不灵活，而且还会让程序难以阅读。你可能会想直接忽略这些异常事件，期望它们永不发生，但 Python 的异常对象提供了非常强大的替代解决方案。

本章介绍如何创建和引发自定义的异常，以及处理异常的各种方法。

## 8.1 什么是异常

Python 用*异常对象*(exception object)来表示异常情况。遇到错误后，会引发异常。如果异常对象并未被处理或捕捉，程序就会用所谓的*回溯*(traceback， 一种错误信息)终止执行：

```py
>>> 1 / 0
Traceback (most recent call last):
  File "<stdin>", line 1, in <module> 
  ZeroDivisionError: integer division or modulo by zero 
```

如果这些错误信息就是异常的全部功能，那么它也就不必存在了。事实上，每个异常都是一些类(本例中是`ZeroDivisionError`)的实例，这些实例可以被引发，并且可以用很多种方法进行捕捉，使得程序可以捉住错误并且对其进行处理，而不是让整个程序失效。

## 8.2 按自己的方式出错

异常可以在某些东西出错的时候自动引发。在学习如何处理异常之前，先看一下自己如何引发异常，以及创建自己的异常类型。

### 8.2.1 `raise`语句

为了引发异常，可以使用一个类(应该是`Exception`的子类)或者实例参数调用`raise`语句。使用类时，程序会自动创建类的一个实例。下面是一些简单的例子，使用了内建的`Exception`的异常类：

```py
>>> raise Exception
Traceback (most recent call last):
  File "<stdin>", line 1, in <module> 
  Exception 
>>> raise Exception("hyperdrive overload")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module> \
  Exception: hyperdrive overload 
```

第一个例子 raise Exception 引发了一个没有任何有关错误信息的普通异常。后一个例子中，则添加了错误信息 hyperdrive overload。

内建的异常类有很多。Python 库参考手册的 Built-in Exceptions 一节中有关与它们的描述。用交互式解释器也可以分析它们，这些内建异常都可以在`exceptions`模块(和内建的命名空间)中找到。可以使用`dir`函数列出模块内容，这部分会在第十章中讲到：

```py
>>> import exceptions 
>>> dir(exceptions)
['ArithmeticError', 'AssertionError', 'AttributeError', ...] 
```

读者的解释器中，这个名单可能要长得多——出于对易读性的考虑，这里删除了大部分名字，所有这些异常都可以用在`raise`语句中：

```py
>>> raise ArithmeticError
Traceback (most recent call last):
  File "<stdin>", line 1, in <module> 
  ArithmeticError 
```

表 8-1 描述了一些最重要的内建异常类：

表 8-1 一些内建异常类

```py
Exception　　　　　　　　　　　　所有异常的基类
AttributeError　　　　　　　　　 特性引用或赋值失败时引发
IOError　　　　　　　　　　　　　试图打开不存在文件(包括其他情况)时引发
IndexError                       在使用序列中不存在的索引时引发
KeyError　　　　　　　　　　　　 在使用映射中不存在的键时引发
NameError　　　　　　　　　　　  在找不到名字(变量)时引发
SyntaxError　　　　　　　　　　　在代码为错误形式时引发
TypeError　　　　　　　　　　　　在内建操作或者函数应用于错误类型的对象时引发
ValueError　　　　　　　　　　　 在内建操作或者函数应用于正确类型的对象，但是该对象使用不合适的值时引发
ZeroDivisionError　　　　　　　　在除法或者模除操作的第二个参数为 0 时引发 
```

### 8.2.2 自定义异常类

尽管内建的异常类已经包括了大部分的情况，而且对于很多要求都已经足够了，但是有些时候还是需要创建自己的异常类。比如在超光速推进装置过载(hyperdrive overload)的例子中，如果能有个具体的`HyperDriveError`类来表示超光速推进装置的错误状况是不是更自然一些？错误信息是足够了，但是会在 8.3 节中看到，可以根据异常所在的类，选择性地处理当前类型的异常。所以如果想要使用特殊的错误处理代码处理超光速推进装置的错误，那么就需要一个独立于`exceptions`模块的异常类。

那么如何创建自己的异常类呢？就像其他类一样，只是要确保从`Exception`类继承(不管是间接还是直接，也就是说继承其他的内建异常类也是可以的)。那么编写一个自定义异常类基本上就像下面这样：

```py
class SomeCustomException(Exception): 
    pass 
```

还不能做太多事，对吧？(如果你愿意，也可以向你的异常类中增加方法)

## 8.3 捕捉异常

前面曾经提到过，关于异常的最有意思的地方就是可以处理它们(通常叫做诱捕或者捕捉异常)。这个功能可以使用`try/except`语句来实现。假设创建了一个让用户输入两个数，然后进行相除的程序，像下面这样：

```py
x = input("Enter the first number: ")
y = input("Enter the second number: ") 
print x / y 
```

程序工作正常，假如用户输入 0 作为第二个数

```py
Enter the first number: 10 Enter the second number: 0
Traceback (most recent call last):
  File "/home/marlowes/MyPython/My_Exception.py", line 6, in <module>
    print x / y
ZeroDivisionError: integer division or modulo by zero 
```

为了捕捉异常并且做出一些错误处理(本例中只是输出一些更友好的错误信息)，可以这样重写程序：

```py
try:
    x = input("Enter the first number: ")
    y = input("Enter the second number: ") 
    print x / y 
except ZeroDivisionError: 
    print "The second number can't be zero!" 
```

看起来用`if`语句检查`y`值会更简单一些，本例中这样做的确很好。但是如果需要给程序加入更多除法，那么就得给每个除法加个 if 语句。而且使用`try/except`的话只需要一个错误处理器。

*注：如果没有捕捉异常，它就会被“传播”到调用的函数中。如果在那里依然没有捕获，这些异常就会“浮”到程序的最顶层，也就是说你可以捕捉到在其他人的函数中所引发的异常。有关这方面的更多信息，请参见 8.10 节。*

**看，没参数**

如果捕捉到了异常，但是又想重新引发它(也就是说要传递异常，不进行处理)，那么可以调用不带参数的`raise`(还能在捕捉到异常时显式地提供具体异常，在 8.6 节会对此进行解释)。

举个例子吧，看看这么做多么有用：考虑一下一个能“屏蔽”`ZeroDivisionError`(除零错误)的计算器类。如果这个行为被激活，那么计算器就打印错误信息，而不是让异常传播。如果在与用户交互的过程中使用，那么这就有用了，但是如果是在程序内部使用，引发异常会更好些。因此“屏蔽”机制就可以关掉了，下面是这样一个类的代码：

```py
class MuffledCalculator():

    muffled = False 
    def calc(self, expr): 
        try: 
            return eval(expr) 
        except ZeroDivisionError: 
            if self.muffled: 
                print "Division by zero is illegal"
            else: 
                raise 
```

*注：如果除零行为发生而屏蔽机制被打开，那么`calc`方法会(隐式地)返回`None`。换句话说，如果打开了屏蔽机制，那么就不应该依赖返回值。*

下面是这个类的用法示例，分别打开和关闭了屏蔽：

```py
>>> calculator = MuffledCalculator() 
>>> calculator.calc("10 / 2") 
5
>>> calculator.calc("10 / 0")
Traceback (most recent call last):
  File "/home/marlowes/MyPython/My_Exception.py", line 28, in <module> calculator.calc("10 / 0")
  File "/home/marlowes/MyPython/My_Exception.py", line 19, in calc return eval(expr)
  File "<string>", line 1, in <module> ZeroDivisionError: integer division or modulo by zero >>> calculator.muffled = True >>> calculator.calc("10 / 0")
Division by zero is illegal 
```

当计算器没有打开屏蔽机制时，`ZeroDivisionError`被捕捉但已传递了。

## 8.4 不止一个`except`子句

如果运行上一节的程序并且在提示符后面输入非数字类型的值，就会产生另一个异常：

```py
Enter the first number: 10 
Enter the second number: "Hello, world!" 
Traceback (most recent call last):
  File "/home/marlowes/MyPython/My_Exception.py", line 8, in <module>
    print x / y
TypeError: unsupported operand type(s) for /: 'int' and 'str' 
```

因为`except`子句只寻找`ZeroDivisionError`异常，这次的错误就溜过了检查并导致程序终止。为了捕捉这个异常，可以直接在同一个`try/except`语句后面加上另一个`except`子句：

```py
try:
    x = input("Enter the first number: ")
    y = input("Enter the second number: ") 
    print x / y 
except ZeroDivisionError: 
    print "The second number can't be zero!"
except TypeError: 
    print "That wasn't a number, was it?" 
```

这次用`if`语句实现可就复杂了。怎么检查一个值是否能被用在除法中？方法很多，但是目前最好的方式是直接将值用来除一下看看是否奏效。

还应该注意到，异常处理并不会搞乱原来的代码，而增加一大堆`if`语句检查可能的错误情况会让代码相当难读。

## 8.5 用一个块捕捉两个异常

如果需要用一个块捕捉多个类型异常，那么可以将它们作为元组列出，像下面这样：

```py
try:
    x = input("Enter the first number: ")
    y = input("Enter the second number: ") 
    print x / y 
except (ZeroDivisionError, TypeError, NameError): 
    print "Your numbers were bogus..." 
```

上面的代码中，如果用户输入字符串或者其他类型的值，而不是数字，或者第 2 个数为 0，都会打印同样的错误信息。当然，只打印一个错误信息并没有什么帮助。另外一个方法就是继续要求输入数字直到可以进行除法运算为止。8.8 节中会介绍如何实现这一功能。

注意，`except`子句中异常对象外面的圆括号很重要。忽略它们是一种常见的错误，那样你会得不到想要的结果。关于这方面的解释，请参见 8.6 节。

## 8.6 捕捉对象

如果希望在`except`子句中访问异常对象本身，可以使用两个参数(注意，就算要捕捉到多个异常，也只需向`except`子句提供一个参数——一个元组)。比如，如果想让程序继续运行，但是又因为某种原因想记录下错误(比如只是打印给用户看)，这个功能就很有用。下面的示例程序会打印异常(如果发生的话)，但是程序会继续运行：

```py
try:
    x = input("Enter the first number: ")
    y = input("Enter the second number: ") 
    print x / y 
except (ZeroDivisionError, TypeError), e: 
    print e 
```

(在这个小程序中，`except`子句再次捕捉了两种异常，但是因为你可以显式地捕捉对象本身，所以异常可以打印出来，用户就能看到发生什么(8.8 节会介绍一个更有用的方法)。——译者注)

*注：在 Python3.0 中，`except`子句会被写作`except (ZeroDivisionError, TypeError) as e`。*

## 8.7 真正的捕捉

就算程序能处理好几种类型的异常，但是有些异常还会从眼皮地下溜走。比如还用那个除法程序来举例，在提示符下面直接按回车，不输入任何东西，会的到一个类似下面这样的错误信息(*栈跟踪*)：

```py
Traceback (most recent call last):
  File "/home/marlowes/MyPython/My_Exception.py", line 33, in <module> x = input("Enter the first number: ")
  File "<string>", line 0 
  ^ SyntaxError: unexpected EOF while parsing 
```

这个异常逃过了`try/except`语句的检查，这很正常。程序员无法预测会发生什么，也不能对其进行准备。在这些情况下，与其用那些并非捕捉这些异常的`try/except`语句隐藏异常，还不如让程序立刻崩溃。

但是如果真的想用一段代码捕捉所有异常，那么可以在 except 子句中忽略所有的异常类：

```py
try:
    x = input("Enter the first number: ")
    y = input("Enter the second number: ") 
    print x / y 
except: 
    print "Something wrong happened..." 
```

现在可以做任何事情了：

```py
Enter the first number: "This" is *completely* illegal 123 Something wrong happened... 
```

*警告：像这样捕捉所有异常是危险的，因为它会隐藏所有程序员未想到并且未做好准备处理的错误。它同样会捕捉用户终止执行的 Ctrl+C 企图，以及用`sys.exit`函数终止程序的企图，等等。这时使用`except Exception, e`会更好些，或者对异常对象`e`进行一些检查。*

## 8.8 万事大吉

有些情况中，没有坏事发生时执行一段代码是很有用的；可以像对条件和循环语句那样，给`try/except`语句加个`else`子句：

```py
try: 
    print "A simple task"
except: 
    print "What? Something went wrong?"
else: 
    print "Ah... It went as planned." 
```

运行之后会的到如下输出：

```py
A simple task
Ah... It went as planned. 
```

使用`else`子句可以实现在 8.5 节中提到的循环：

```py
while True: 
    try:
        x = input("Enter the first number: ")
        y = input("Enter the second number: ")
        value = x / y 
        print "x / y is", value 
    except: 
        print "Invalid input. Please try again."
    else: 
        break 
```

这里的循环只有在没有异常引发的情况下才会退出(由`else`子句中的`break`语句退出)。换句话说，只要有错误发生，程序会不断要求重新输入。下面是一个例子的运行情况：

```py
Enter the first number: 1 
Enter the second number: 0
Invalid input. Please try again.
Enter the first number: "foo" 
Enter the second number: "bar" 
Invalid input. Please try again.
Enter the first number: baz
Invalid input. Please try again.
Enter the first number: 10 
Enter the second number: 2 
x / y is 5 
```

之前提到过，可以使用空的`except`子句来捕捉所有`Exception`类的异常(也会捕捉其所有子类的异常)。百分之百捕捉到所有的异常是不可能的，因为`try/except`语句中的代码可能会出现问题，比如使用旧风格的字符串异常或者自定义的异常类不是`Exception`类的子类。不过如果需要使用`except Exception`的话，可以使用 8.6 节中的技巧在除法程序中打印更加有用的错误信息：

```py
while True: 
    try:
        x = input("Enter the first number: ")
        y = input("Enter the second number: ")
        value = x / y 
        print "x / y is", value 
    except Exception, e: 
        print "Invalid input:", e 
        print "Please try again"
    else: 
        break 
```

下面是示例运行：

```py
Enter the first number: 1 Enter the second number: 0
Invalid input: integer division or modulo by zero
Please try again
Enter the first number: "x" 
Enter the second number: "y" 
Invalid input: unsupported operand type(s) for /: 'str' and 'str' Please try again
Enter the first number: quuux
Invalid input: name 'quuux' is not defined
Please try again
Enter the first number: 10 
Enter the second number: 2 
x / y is 5 
```

## 8.9 最后······

最后，是`finally`子句。它可以用来在可能的异常后进行清理。它和`try`子句联合使用：

```py
x = None 
try:
    x = 1 / 0 
finally: 
    print "Cleaning up..."
    del x 
```

上面的代码中，`finally`子句肯定会被执行，不管`try`子句中是否发生异常(在`try`子句之前初始化`x`的原因是如果不这样做，由于`ZeroDivisionError`的存在，`x`就永远不会被赋值。这样就会导致在`finally`子句中使用`del`删除它的时候产生异常，而且这个异常是无法捕捉的)。

运行这段代码，在程序崩溃之前，对于变量`x`的清理就完成了：

```py
Cleaning up...
  File "/home/marlowes/MyPython/My_Exception.py", line 36, in <module> x = 1 / 0
ZeroDivisionError: integer division or modulo by zero 
```

*注：在 Python2.5 之前的版本内，`finally`子句需要独立使用，而不能作为`try`语句的`except`子句使用。如果都要使用的话，那么需要两条语句。但在 Python2.5 及其之后的版本中，可以尽情地组合这些子句。*

## 8.10 异常和函数

异常和函数能很自然地一起工作。如果异常在函数内引发而不被处理，它就会传播至(浮到)函数调用的地方。如果在那里也没有处理异常，它就会继续传播，一直到达主程序(全局作用域)。如果那里没有异常处理程序，程序会带着栈跟踪中止。看个例子：

```py
>>> def faulty():
...     raise Exception("Something is wrong")
... 
>>> def ignore_exception():
...     faulty()
... 
>>> def handle_exception():
...     try:
...         faulty()
...     except:
...         print "Exception handled" 
... 
>>> ignore_exception()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module> File "<stdin>", line 2, in ignore_exception
  File "<stdin>", line 2, in faulty
Exception: Something is wrong 
>>> handle_exception()
Exception handled 
```

可以看到，`faulty`中产生的异常通过`faulty`和`ignore_exception`传播，最终导致了栈跟踪。同样地，它也传播到了`handle_exception`，但在这个函数中被`try/except`语句处理。

## 8.11 异常之禅

异常处理并不是很复杂。如果知道某段代码可能会导致某种异常，而又不希望程序以堆栈跟踪的形式终止，那么就根据需要添加`try/except`或者`try/finally`语句(或者它们的组合)进行处理。

有些时候，条件语句可以实现和异常处理同样的功能，但是条件语句可能在自然性和可读性上差些。而从另一方面来看，某些程序中使用`if/else`实现会比使用`try/except`要好。让我们看几个例子。

假设有一个字典，我们希望打印出存储在特定的键下面的值。如果该键不存在，那么什么也不做。代码可能像下面这样写：

```py
def describePerson(person): 
    print "Description of", person["name"] 
    print "Age:", person["age"] 
    if "occupation" in person: 
        print "Occupation:", person["occupation"] 
```

如果给程序提供包含名字`Throatwobbler Mangrove`和年龄`42`(没有职业)的字典的函数，会得到如下输出：

```py
Description of Throatwobbler Mangrove
Age: 42 
```

如果添加了职业`camper`，会的到如下输出：

```py
Description of Throatwobbler Mangrove
Age: 42 
Occupation: camper 
```

代码非常直观，但是效率不高(尽管这里主要关心的是代码的简洁性)。程序会两次查找`"occupation"`键，其中一次用来检查键是否存在(在条件语句中)，另外一次获得值(打印)。另外一个解决方案如下：

```py
def describePerson(person): 
    print "Description of", person["name"] 
    print "Age:", person["age"] 
    try: 
        print "Occupation: " + person["occupation"] 
    except KeyError: 
        pass 
```

*注：这里在打印职业时使用加号而不是逗号。否则字符串`"Occupation:"`在异常引发之前就会被输出。*

这个程序直接假定`"occupation"`键存在。如果它的确存在，那么就会省事一些。直接取出它的值再打印输出即可——不用额外检查它是否真的存在。如果该键不存在，则会引发`KeyError`异常，而被`except`子句捕捉到。

在查看对象是否存在特定特性时，`try/except`也很有用。假设想要查看某对象是否有`write`特性，那么可以使用如下代码：

```py
try:
    obj.write 
except AttributeError: 
    print "The object is not writeable"
else: 
    print "The object is writeable" 
```

这里的`try`子句仅仅访问特性而不用对它做别的有用的事情。如果`AttributeError`异常引发，就证明对象没有这个特性，反之存在该特性。这是实现第七章中介绍的`getattr`(7.2.8 节)方法的替代方法，至于更喜欢哪种方法，完全是个人喜好。其实在内部实现`getattr`时也是使用这种方法：它试着访问特性并且捕捉可能引发的`AttributeError`异常。

注意，这里所获得的效率提高并不多(微乎其微)，一般来说(除非程序有性能问题)程序开发人员不用过多担心这类优化问题。在很多情况下，使用`try/except`语句比使用`if/else`会更自然一些(更“Python 化”)，应该养成尽可能使用`try/except`语句的习惯。

## 8.12 小结

本章的主题如下。

☑ 异常对象：异常情况(比如发生错误)可以用异常对象表示。它们可以用几种方法处理，但是如果忽略的话，程序就会中止。

☑ 警告：警告类似于异常，但是(一般来说)仅仅打印错误信息。

☑ 引发异常：可以使用`raise`语句引发异常。它接受异常类或者异常实例作为参数。还能提供两个参数(异常和错误信息)。如果在 except 子句中不使用参数调用`raise`，它就会“重新引发”该子句捕捉到的异常。

☑ 自定义异常类：用继承`Exception`类的方法可以创建自己的异常类。

☑ 捕捉异常：使用`try`语句的`except`子句捕捉异常。如果在`except`子句中不特别指定异常类，那么所有的异常都会被捕捉。异常可以放在元组中以实现多个异常的指定。如果给`except`提供两个参数，第二个参数就会绑定到异常对象上。同样，在一个`try/except`语句中能包含多个`except`子句，用来分别处理不同的异常。

☑ `else`子句：除了`except`子句，可以使用`else`子句。如果主`try`块中没有引发异常，`else`子句就会被执行。

☑ `finally`：如果需要确保某些代码不管是否有异常引发都要执行(比如清理代码)，那么这些代码可以放置在`finally`(注意，在 Python2.5 以前，在一个`try`语句中不能同时使用`except`和`finally`子句——但是一个子句可以放置在另一个子句中)子句中。

☑ 异常和函数：在函数内引发异常时，它就会被传播到函数调用的地方(对于方法也是一样)。

### 8.12.1 本章的新函数

本章涉及的新函数如表 8-2 所示。

表 8-2 本章的新函数

```py
warnings,filterwarnings(action, ...)  用于过滤警告 
```

### 8.12.2 接下来学什么

本章讲异常，内容可能有些意外(双关语)，而下一章的内容真的很不可思议，恩，近乎不可思议。