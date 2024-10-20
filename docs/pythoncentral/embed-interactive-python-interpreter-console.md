# 如何嵌入交互式 Python 解释器控制台

> 原文：<https://www.pythoncentral.io/embed-interactive-python-interpreter-console/>

你可以在应用程序中使用解释器做一些事情。其中最有趣的是让你的用户能够在运行时编写你的应用程序，就像 [GIMP](https://www.gimp.org/ "The GNU Image Manipulation Program") 和 [Scribus](https://www.scribus.net/ "Scribus: Open Source Desktop Publishing") 所做的那样，但它也可以用来构建增强的 Python shells，像 [IPython](https://ipython.org/ "IPython Interactive Computing") 。

首先:首先从标准库的`code`模块导入`InteractiveConsole`类，然后子类化它，这样你就可以添加一些钩子。Python 2 和 3 之间的`code`模块没有真正的变化，所以本文对两者都有效。

```py

from code import InteractiveConsole
类控制台(InteractiveConsole):
def _ _ init _ _(* args):interactive console。__init__(*args) 

```

到目前为止，代码只是样板文件；它创建了一个名为`Console`的类，它只是`InteractiveConsole`的子类。

下面的代码演示了该类的工作方式。第 4 行调用了从`InteractiveConsole`继承而来的`runcode`方法。`runcode`方法获取一串源代码并在控制台内部执行，在本例中，将`1`分配给`a`，然后打印出来。

```py

a = 0

code = 'a = 1; print(a)'

console = Console()

console.runcode(code) # prints 1

print(a)              # prints 0

```

第 5 行打印了`0`,因为控制台有自己的名称空间。注意，`console`对象是一个常规的运行时对象；它在与初始化它的代码相同的线程和进程中运行代码，所以对`runcode`的调用通常会被阻塞。

Python 的`code`模块还提供了一个`InteractiveInterpreter`类，它会自动对表达式求值，就像 Python 的交互模式一样，但是使用多行输入会更复杂。`InteractiveConsole.runcode`接受任意有效的 Python 块。通常，当您需要使用终端时，您应该使用`InteractiveInterpreter`类，当您的输入将是完整的 Python 块时，通常作为文件或来自图形用户界面的输入字符串，您应该使用`InteractiveConsole`。

## 处理用户输入

你经常想要处理用户的输入，可能需要转换编译一个语法扩展，比如 IPython Magic，或者执行更复杂的宏。

向名为`preprocess`的`Console`添加一个新的静态方法，该方法只接受一个字符串并返回它。添加另一个名为`enter`的新方法，它接受用户的输入，通过`preprocess`运行，然后传递给`runcode`。这样做可以很容易地重新定义处理器，要么通过子类化`Console`，要么在运行时简单地给实例的`preprocess`属性分配一个新的 callable。

```py

class Console(InteractiveConsole):
def _ _ init _ _(* args):interactive console。__init__(*args)
def enter(self，source):
source = self . preprocess(source)
self . runcode(source)
@staticmethod 
 def 预处理(源):返回源
Console = Console()
Console . preprocess = lambda source:source[4:]
Console . enter(>>>print(1)’)

```

## 灌注命名空间

`InteractiveConsole`类构造函数接受一个可选参数，这是一个字典，用于在创建控制台的名称空间时填充它。

```py

names = {'a': 1, 'b': 2}

console = Console(names)      # prime the console

console.runcode('print(a+b)') # prints 3

```

当你创建一个新的`Console`实例时，传入对象允许你将任何对象放入用户编写应用程序可能需要的名称空间中。重要的是，这些可以是对对象的特定运行时实例的引用，而不仅仅是来自库导入的类和函数定义。

现在，您已经有了充实控制台所需的钩子。添加一个在新线程中调用`enter`的`spawn`方法，可以让输入阻塞或并行运行。添加一个允许编写阻塞和非阻塞输入的预处理器的额外好处。

## 访问名称空间

要在控制台的名称空间创建后访问它，可以引用它的`locals`属性。

```py

console.locals['a'] = 1

console.runcode('print(a)') # prints 1

console.runcode('a = 2')

print(console.locals['a'])  # prints 2

```

因为这有点难看，您可以将一个空的模块对象传入控制台，在外层空间保存对它的引用，然后使用该模块共享对象。

标准库的`imp`模块提供了一个`new_module`函数，让我们在不影响`sys.modules`(或者不需要实际文件)的情况下创建一个新的模块对象。您需要将新模块的名称传递给函数，并取回空模块。

```py

from imp import new_module

superspace = new_module('superspace')

```

现在您可以在控制台创建时将`superspace`模块传递到控制台的名称空间中。在控制台内部也称它为`superspace`,这使得它们是同一个对象变得更加明显，但是您可以使用不同的名称。

```py

console = Console({'superspace': superspace})

```

现在`superspace`是一个单独的空模块对象，在两个名称空间中都被这个名称引用。

```py

superspace.a = 1

console.enter('print(superspace.a)')    # prints 1

console.enter('superspace.a = 2')

print(superspace.a)                     # prints 2

```

## 舍入

将共享模块绑定到控制台实例是有意义的，因此每个控制台实例都有自己的共享模块。`__init__`方法需要扩展来更直接地处理它的参数，所以它仍然能够接受可选的名称空间字典。

最好将预处理器的钩子传递到控制台的名称空间，这样用户就可以将自己的处理器绑定到这个名称空间。为了简单起见，下面的例子只是将对`self`的引用作为`console`传递到它自己的名称空间中，不是因为它是元的，只是因为它更容易阅读代码。

```py

from code import InteractiveConsole

from imp import new_module
类控制台(InteractiveConsole):
def __init__(self，names=None): 
 names = names 或者{} 
 names['控制台'] = self 
 InteractiveConsole。__init__(self，names)
self . super space = new _ module(' super space ')
def enter(self，source):
source = self . preprocess(source)
self . runcode(source)
@staticmethod 
 def 预处理(源):
返回源
console = Console() 

```

如果你运行这段代码，现在在外部空间和控制台内部都有一个名为`console`的全局变量引用相同的东西，所以`console.superspace`在两者中都是相同的空模块。

在实践中，如果你允许用户编写你的应用程序，你会想要写一个你想要公开的运行时对象的包装器，这样用户就有一个干净的 API，他们可以在不崩溃的情况下入侵。然后将这些包装对象传递到控制台。