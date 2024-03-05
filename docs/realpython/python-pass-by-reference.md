# Python 中的引用传递:背景和最佳实践

> 原文：<https://realpython.com/python-pass-by-reference/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**顺便引用 Python 中的:最佳实践**](/courses/pass-by-reference-python-best-practices/)

熟悉 Python 之后，您可能会注意到函数没有像您预期的那样修改参数的情况，尤其是在您熟悉其他编程语言的情况下。一些语言将函数参数作为对现有[变量](https://realpython.com/python-variables/)的**引用**来处理，这被称为**通过引用传递**。其他语言将它们作为**独立值**来处理，这种方法被称为**按值传递**。

如果你是一名中级 Python 程序员，希望理解 Python 处理函数参数的特殊方式，那么本教程适合你。您将在 Python 中实现引用传递构造的真实用例，并学习一些最佳实践来避免函数参数中的陷阱。

在本教程中，您将学习:

*   通过引用传递意味着什么以及为什么你想这样做
*   通过引用传递与**通过值传递**和 **Python 独特的方法**有何不同
*   Python 中**函数参数**的行为方式
*   如何在 Python 中使用某些**可变类型**进行引用传递
*   Python 中通过引用复制传递的最佳实践是什么

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 参照定义路径

在深入研究按引用传递的技术细节之前，通过将术语分解为几个部分来更仔细地了解它本身是有帮助的:

*   **传递**的意思是给函数提供一个自变量。
*   **通过引用**意味着你传递给函数的参数是对内存中已经存在的变量的**引用**，而不是该变量的独立副本。

因为您给了函数一个对现有变量的引用，所以对这个引用执行的所有操作都会直接影响它所引用的变量。让我们看一些例子来说明这在实践中是如何工作的。

下面，您将看到如何在 C#中通过引用传递变量。注意在突出显示的行中使用了关键字[`ref`:](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/ref)

```py
using  System; // Source:
// https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/classes-and-structs/passing-parameters
class  Program { static  void  Main(string[]  args) { int  arg; // Passing by reference.
  // The value of arg in Main is changed.
  arg  =  4; squareRef(ref  arg);   Console.WriteLine(arg); // Output: 16
  } static  void  squareRef(ref  int  refParameter)   { refParameter  *=  refParameter; } }
```

如你所见，`squareRef()`的`refParameter`必须用`ref`关键字声明，在调用函数时也必须使用关键字。然后参数将通过引用传入，并可以就地修改。

Python 没有`ref`关键字或任何与之等同的东西。如果您尝试在 Python 中尽可能接近地复制上面的例子，那么您会看到不同的结果:

>>>

```py
>>> def main():
...     arg = 4
...     square(arg)
...     print(arg)
...
>>> def square(n):
...     n *= n
...
>>> main()
4
```

在这种情况下，`arg`变量是*而不是*被改变了位置。Python 似乎将您提供的参数视为独立的值，而不是对现有变量的引用。这是否意味着 Python 通过值而不是通过引用传递参数？

不完全是。Python 既不通过引用也不通过值来传递参数，而是通过赋值来传递**。下面，您将快速探索按值传递和按引用传递的细节，然后更仔细地研究 Python 的方法。之后，您将浏览一些[最佳实践](#replicating-pass-by-reference-with-python)，以实现 Python 中的等效引用传递。**

[*Remove ads*](/account/join/)

## 对照按引用传递和按值传递

当您通过引用传递函数参数时，这些参数只是对现有值的引用。相反，当您通过值传递参数时，这些参数将成为原始值的独立副本。

让我们重温一下 C#的例子，这次没有使用`ref`关键字。这将导致程序使用默认的按值传递行为:

```py
using  System; // Source:
// https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/classes-and-structs/passing-parameters
class  Program { static  void  Main(string[]  args) { int  arg; // Passing by value.
  // The value of arg in Main is not changed.
  arg  =  4; squareVal(arg);   Console.WriteLine(arg); // Output: 4
  } static  void  squareVal(int  valParameter)   { valParameter  *=  valParameter; } }
```

在这里，你可以看到`squareVal()`没有修改原始变量。更确切地说，`valParameter`是原始变量`arg`的独立副本。虽然这符合您在 Python 中看到的行为，但请记住 Python 并不完全通过值传递。我们来证明一下。

Python 的内置`id()`返回一个整数，代表所需对象的内存地址。使用`id()`，您可以验证以下断言:

1.  函数参数最初引用与其原始变量相同的地址。
2.  在函数中重新分配参数会给它一个新的地址，而原始变量保持不变。

在下面的例子中，注意到`x`的地址最初与`n`的地址匹配，但是在重新分配后发生了变化，而`n`的地址从未改变:

>>>

```py
>>> def main():
...     n = 9001
...     print(f"Initial address of n: {id(n)}")
...     increment(n)
...     print(f"  Final address of n: {id(n)}")
...
>>> def increment(x):
...     print(f"Initial address of x: {id(x)}")
...     x += 1
...     print(f"  Final address of x: {id(x)}")
...
>>> main()
Initial address of n: 140562586057840
Initial address of x: 140562586057840
 Final address of x: 140562586057968 Final address of n: 140562586057840
```

当您调用`increment()`时，`n`和`x`的初始地址是相同的，这一事实证明了`x`参数不是通过值传递的。否则，`n`和`x`将会有不同的内存地址。

在学习 Python 如何处理参数的细节之前，我们先来看一些引用传递的实际用例。

## 使用引用传递构造

通过引用传递变量是实现特定编程模式的几种策略之一。虽然很少需要，但通过引用传递可能是一个有用的工具。

在这一节中，您将看到三种最常见的模式，对于这些模式，通过引用传递是一种实用的方法。然后您将看到如何用 Python 实现这些模式。

### 避免重复对象

如您所见，通过值传递变量将导致创建该值的副本并存储在内存中。在默认通过值传递的语言中，您可能会发现通过引用传递变量会提高性能，特别是当变量包含大量数据时。当您的代码在资源受限的机器上运行时，这一点会更加明显。

然而，在 Python 中，这从来都不是问题。你会在下一节的[中看到原因。](#passing-arguments-in-python)

### 返回多个值

通过引用传递的最常见应用之一是创建一个函数，该函数在返回不同值的同时改变引用参数的值。您可以修改按引用传递的 C#示例来说明这种技术:

```py
using  System; class  Program { static  void  Main(string[]  args) { int  counter  =  0; // Passing by reference.
  // The value of counter in Main is changed.
  Console.WriteLine(greet("Alice",  ref  counter)); Console.WriteLine("Counter is {0}",  counter); Console.WriteLine(greet("Bob",  ref  counter)); Console.WriteLine("Counter is {0}",  counter); // Output:
  // Hi, Alice!
  // Counter is 1
  // Hi, Bob!
  // Counter is 2
  } static  string  greet(string  name,  ref  int  counter) { string  greeting  =  "Hi, "  +  name  +  "!"; counter++; return  greeting; } }
```

在上面的例子中，`greet()`返回一个问候[字符串](https://realpython.com/python-strings/)，并修改`counter`的值。现在尝试用 Python 尽可能地再现这一点:

>>>

```py
>>> def main():
...     counter = 0
...     print(greet("Alice", counter))
...     print(f"Counter is {counter}")
...     print(greet("Bob", counter))
...     print(f"Counter is {counter}")
...
>>> def greet(name, counter):
...     counter += 1
...     return f"Hi, {name}!"
...
>>> main()
Hi, Alice!
Counter is 0
Hi, Bob!
Counter is 0
```

在上面的例子中,`counter`没有递增，因为正如您之前了解到的，Python 无法通过引用传递值。那么，如何才能获得与 C#相同的结果呢？

本质上，C#中的引用参数不仅允许函数返回值，还允许对附加参数进行操作。这相当于返回多个值！

幸运的是，Python 已经支持返回多个值。严格来说，返回多个值的 Python 函数实际上返回一个包含每个值的[元组](https://realpython.com/python-lists-tuples/):

>>>

```py
>>> def multiple_return():
...     return 1, 2
...
>>> t = multiple_return()
>>> t  # A tuple
(1, 2)

>>> # You can unpack the tuple into two variables:
>>> x, y = multiple_return()
>>> x
1
>>> y
2
```

正如您所看到的，要返回多个值，您可以简单地使用 [`return`关键字](https://realpython.com/python-keywords/#returning-keywords-return-yield)，后跟逗号分隔的值或变量。

有了这种技术，您可以将`greet()`中的 [`return`语句](https://realpython.com/python-return-statement/)从之前的 Python 代码中修改为既返回问候又返回计数器:

>>>

```py
>>> def main():
...     counter = 0
...     print(greet("Alice", counter))
...     print(f"Counter is {counter}")
...     print(greet("Bob", counter))
...     print(f"Counter is {counter}")
...
>>> def greet(name, counter):
...     return f"Hi, {name}!", counter + 1 ...
>>> main()
('Hi, Alice!', 1)
Counter is 0
('Hi, Bob!', 1)
Counter is 0
```

这看起来还是不对。虽然`greet()`现在返回多个值，但是它们被[打印为`tuple`。此外，原来的`counter`变量保持在`0`。](https://realpython.com/python-print/)

为了清理你的输出并得到想要的结果，你必须在每次调用`greet()`时**重新分配**你的`counter`变量:

>>>

```py
>>> def main():
...     counter = 0
...     greeting, counter = greet("Alice", counter) ...     print(f"{greeting}\nCounter is {counter}")
...     greeting, counter = greet("Bob", counter) ...     print(f"{greeting}\nCounter is {counter}")
...
>>> def greet(name, counter):
...     return f"Hi, {name}!", counter + 1
...
>>> main()
Hi, Alice!
Counter is 1
Hi, Bob!
Counter is 2
```

现在，通过调用`greet()`重新分配每个变量后，您可以看到想要的结果！

将返回值赋给变量是获得与 Python 中通过引用传递相同结果的最佳方式。在关于[最佳实践](#replicating-pass-by-reference-with-python)的章节中，您将了解到原因以及一些额外的方法。

[*Remove ads*](/account/join/)

### 创建条件多重返回函数

这是返回多个值的一个特定用例，其中该函数可以在一个[条件语句](https://realpython.com/python-conditional-statements/)中使用，并具有额外的副作用，如修改作为参数传入的外部变量。

考虑一下标准的 [Int32。C#中的 TryParse](https://docs.microsoft.com/en-us/dotnet/api/system.int32.tryparse?view=netcore-3.1#System_Int32_TryParse_System_String_System_Int32__) 函数，返回一个[布尔值](https://realpython.com/python-boolean/)，同时对一个整数参数的引用进行操作:

```py
public  static  bool  TryParse  (string  s,  out  int  result);
```

该函数试图使用 [`out`关键字](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/out-parameter-modifier)将`string`转换为 32 位有符号整数。有两种可能的结果:

1.  **如果解析成功**，那么输出参数将设置为结果整数，函数将返回`true`。
2.  **如果解析失败**，那么输出参数将被设置为`0`，函数将返回`false`。

您可以在下面的示例中看到这一点，该示例尝试转换许多不同的字符串:

```py
using  System; // Source:
// https://docs.microsoft.com/en-us/dotnet/api/system.int32.tryparse?view=netcore-3.1#System_Int32_TryParse_System_String_System_Int32__
public  class  Example  { public  static  void  Main()  { String[]  values  =  {  null,  "160519",  "9432.0",  "16,667", "   -322   ",  "+4302",  "(100);",  "01FA"  }; foreach  (var  value  in  values)  { int  number; if  (Int32.TryParse(value,  out  number))  { Console.WriteLine("Converted '{0}' to {1}.",  value,  number); } else  { Console.WriteLine("Attempted conversion of '{0}' failed.", value  ??  "<null>"); } } } }
```

上面的代码试图通过`TryParse()`将不同格式的字符串转换成整数，输出如下:

```py
Attempted conversion of '<null>' failed.
Converted '160519' to 160519.
Attempted conversion of '9432.0' failed.
Attempted conversion of '16,667' failed.
Converted '   -322   ' to -322.
Converted '+4302' to 4302.
Attempted conversion of '(100);' failed.
Attempted conversion of '01FA' failed.
```

要在 Python 中实现类似的函数，您可以使用多个返回值，如前所述:

```py
def tryparse(string, base=10):
    try:
        return True, int(string, base=base)
    except ValueError:
        return False, None
```

这个`tryparse()`返回两个值。第一个值指示转换是否成功，第二个值保存结果(如果失败，则保存 [`None`](https://realpython.com/null-in-python/) )。

然而，使用这个函数有点笨拙，因为您需要在每次调用时解包返回值。这意味着您不能在 [`if`语句](https://realpython.com/python-conditional-statements/)中使用该函数:

>>>

```py
>>> success, result = tryparse("123")
>>> success
True
>>> result
123

>>> # We can make the check work
>>> # by accessing the first element of the returned tuple,
>>> # but there's no way to reassign the second element to `result`:
>>> if tryparse("456")[0]:
...     print(result)
...
123
```

尽管它通常通过返回多个值来工作，但是`tryparse()`不能用于条件检查。这意味着你有更多的工作要做。

您可以利用 Python 的灵活性并简化函数，根据转换是否成功返回不同类型的单个值:

```py
def tryparse(string, base=10):
    try:
        return int(string, base=base)
    except ValueError:
        return None
```

由于 Python 函数能够返回不同的数据类型，现在可以在条件语句中使用该函数。但是怎么做呢？难道您不需要先调用函数，指定它的返回值，然后检查值本身吗？

通过利用 Python 在对象类型方面的灵活性，以及 Python 3.8 中新的[赋值表达式](https://realpython.com/python38-new-features/#the-walrus-in-the-room-assignment-expressions)，您可以在条件`if`语句*中调用这个简化的函数，如果检查通过，*将获得返回值:

>>>

```py
>>> if (n := tryparse("123")) is not None:
...     print(n)
...
123
>>> if (n := tryparse("abc")) is None:
...     print(n)
...
None

>>> # You can even do arithmetic!
>>> 10 * tryparse("10")
100

>>> # All the functionality of int() is available:
>>> 10 * tryparse("0a", base=16)
100

>>> # You can also embed the check within the arithmetic expression!
>>> 10 * (n if (n := tryparse("123")) is not None else 1)
1230
>>> 10 * (n if (n := tryparse("abc")) is not None else 1)
10
```

哇！这个 Python 版本的`tryparse()`甚至比 C#版本更强大，允许您在条件语句和算术表达式中使用它。

通过一点小聪明，您复制了一个特定且有用的按引用传递模式，而实际上没有按引用传递参数。事实上，当使用赋值表达式操作符(`:=`)并在 Python 表达式中直接使用返回值时，您再次**为返回值**赋值。

到目前为止，您已经了解了通过引用传递意味着什么，它与通过值传递有何不同，以及 Python 的方法与这两者有何不同。现在您已经准备好仔细研究 Python 如何处理函数参数了！

[*Remove ads*](/account/join/)

## 用 Python 传递参数

Python 通过赋值传递参数。也就是说，当您调用 Python 函数时，每个函数参数都变成一个变量，传递的值被分配给该变量。

因此，通过理解赋值机制本身是如何工作的，甚至是在函数之外，您可以了解 Python 如何处理函数参数的重要细节。

### 理解 Python 中的赋值

[赋值语句](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements)的 Python 语言参考提供了以下细节:

*   如果赋值目标是一个标识符或变量名，那么这个名字就被绑定到对象上。比如在`x = 2`中，`x`是名字，`2`是对象。
*   如果名称已经绑定到一个单独的对象，那么它将重新绑定到新对象。比如说，如果`x`已经是`2`了，你发出`x = 3`，那么变量名`x`被重新绑定到`3`。

所有的 [Python 对象](https://realpython.com/pointers-in-python/)都在一个特定的结构中实现。这个结构的属性之一是一个计数器，它记录有多少个名字被绑定到这个对象。

**注意:**这个计数器被称为**引用计数器**，因为它跟踪有多少引用或名称指向同一个对象。不要混淆引用计数器和按引用传递的概念，因为这两者是不相关的。

Python 文档提供了关于[引用计数](https://docs.python.org/3/extending/extending.html#reference-counts)的更多细节。

让我们继续看`x = 2`的例子，看看当你给一个新变量赋值时会发生什么:

1.  如果表示值`2`的对象已经存在，那么就检索它。否则，它被创建。
2.  该对象的引用计数器递增。
3.  在当前的[名称空间](https://realpython.com/python-namespaces-scope/)中添加一个条目，将标识符`x`绑定到表示`2`的对象。这个条目实际上是存储在字典中的一个[键-值对！由`locals()`或`globals()`返回该字典的表示。](https://realpython.com/python-namespaces-scope/#python-namespace-dictionaries)

现在，如果您将`x`重新赋值为不同的值，会发生以下情况:

1.  代表`2`的对象的引用计数器递减。
2.  表示新值的对象的引用计数器递增。
3.  当前名称空间的字典被更新，以将`x`与表示新值的对象相关联。

Python 允许您使用函数`sys.getrefcount()`获得任意值的引用计数。您可以用它来说明赋值如何增加和减少这些引用计数器。请注意，交互式解释器采用的行为会产生不同的结果，因此您应该从文件中运行以下代码:

```py
from sys import getrefcount

print("--- Before  assignment ---")
print(f"References to value_1: {getrefcount('value_1')}")
print(f"References to value_2: {getrefcount('value_2')}")
x = "value_1"
print("--- After   assignment ---")
print(f"References to value_1: {getrefcount('value_1')}")
print(f"References to value_2: {getrefcount('value_2')}")
x = "value_2"
print("--- After reassignment ---")
print(f"References to value_1: {getrefcount('value_1')}")
print(f"References to value_2: {getrefcount('value_2')}")
```

此脚本将显示赋值前、赋值后和重新赋值后每个值的引用计数:

```py
--- Before  assignment ---
References to value_1: 3
References to value_2: 3
--- After   assignment ---
References to value_1: 4
References to value_2: 3
--- After reassignment ---
References to value_1: 3
References to value_2: 4
```

这些结果说明了标识符(变量名)和代表不同值的 Python 对象之间的关系。当您将多个变量赋给同一个值时，Python 会增加现有对象的引用计数器，并更新当前名称空间，而不是在内存中创建重复的对象。

在下一节中，您将通过探索 Python 如何处理函数参数来建立对赋值操作的理解。

### 探索函数参数

Python 中的函数参数是**局部变量**。那是什么意思？**局部**是 Python 的[作用域](https://realpython.com/python-scope-legb-rule/)之一。这些范围由上一节提到的名称空间字典表示。您可以使用`locals()`和`globals()`分别检索本地和全局名称空间字典。

执行时，每个函数都有自己的本地名称空间:

>>>

```py
>>> def show_locals():
...     my_local = True
...     print(locals())
...
>>> show_locals()
{'my_local': True}
```

使用`locals()`，您可以演示函数参数在函数的本地名称空间中成为常规变量。让我们给函数添加一个参数`my_arg`:

>>>

```py
>>> def show_locals(my_arg):
...     my_local = True
...     print(locals())
...
>>> show_locals("arg_value")
{'my_arg': 'arg_value', 'my_local': True}
```

您还可以使用`sys.getrefcount()`来展示函数参数如何增加对象的引用计数器:

>>>

```py
>>> from sys import getrefcount

>>> def show_refcount(my_arg):
...     return getrefcount(my_arg)
...
>>> getrefcount("my_value")
3
>>> show_refcount("my_value")
5
```

上面的脚本首先输出`"my_value"`外部的引用计数，然后输出`show_refcount()`内部的引用计数，显示引用计数增加了两个，而不是一个！

那是因为，除了`show_refcount()`本身之外，`show_refcount()`内部对`sys.getrefcount()`的调用也接收`my_arg`作为参数。这将`my_arg`放在`sys.getrefcount()`的本地名称空间中，增加了对`"my_value"`的额外引用。

通过检查函数内部的名称空间和引用计数，您可以看到函数参数的工作方式与赋值完全一样:Python 在函数的本地名称空间中创建标识符和表示参数值的 Python 对象之间的绑定。这些绑定中的每一个都会增加对象的引用计数器。

现在您可以看到 Python 是如何通过赋值传递参数的了！

[*Remove ads*](/account/join/)

## 用 Python 复制按引用传递

在上一节中检查了名称空间之后，您可能会问为什么没有提到 [`global`](https://realpython.com/python-scope-legb-rule/) 作为一种修改变量的方法，就好像它们是通过引用传递的一样:

>>>

```py
>>> def square():
...     # Not recommended!
...     global n
...     n *= n
...
>>> n = 4
>>> square()
>>> n
16
```

使用`global`语句通常会降低代码的清晰度。这可能会产生许多问题，包括以下问题:

*   自由变量，看似与任何事物无关
*   对于所述变量没有显式参数的函数
*   不能与其他变量或参数一起使用的函数，因为它们依赖于单个全局变量
*   使用全局变量时缺少[线程安全](https://en.wikipedia.org/wiki/Thread_safety)

将前面的示例与下面的示例进行对比，下面的示例显式返回值:

>>>

```py
>>> def square(n):
...     return n * n
...
>>> square(4)
16
```

好多了！你避免了全局变量的所有潜在问题，通过要求一个参数，你使你的函数更加清晰。

尽管既不是按引用传递的语言，也不是按值传递的语言，Python 在这方面没有缺点。它的灵活性足以应对挑战。

### 最佳实践:返回并重新分配

您已经谈到了从函数返回值并将它们重新赋值给一个变量。对于操作单个值的函数，返回值比使用引用要清楚得多。此外，由于 Python 已经在幕后使用指针，即使它能够通过引用传递参数，也不会有额外的性能优势。

旨在编写返回一个值的专用函数，然后将该值(重新)赋给变量，如下例所示:

```py
def square(n):
    # Accept an argument, return a value.
    return n * n

x = 4
...
# Later, reassign the return value:
x = square(x)
```

返回和赋值也使您的意图更加明确，代码更容易理解和测试。

对于操作多个值的函数，您已经看到 Python 能够[返回一组值](#returning-multiple-values)。你甚至超越了 Int32 的[的优雅。C#中的 try parse()](#creating-conditional-multiple-return-functions)感谢 Python 的灵活性！

如果您需要对多个值进行操作，那么您可以编写返回多个值的单用途函数，然后将这些值赋给变量。这里有一个例子:

```py
def greet(name, counter):
    # Return multiple values
    return f"Hi, {name}!", counter + 1

counter = 0
...
# Later, reassign each return value by unpacking.
greeting, counter = greet("Alice", counter)
```

当调用返回多个值的函数时，可以同时分配多个变量。

### 最佳实践:使用对象属性

对象属性在 Python 的赋值策略中有自己的位置。Python 对[赋值语句](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements)的语言参考声明，如果目标是支持赋值的对象属性，那么对象将被要求对该属性执行赋值。如果您将对象作为参数传递给函数，那么它的属性可以就地修改。

编写接受具有属性的对象的函数，然后直接对这些属性进行操作，如下例所示:

>>>

```py
>>> # For the purpose of this example, let's use SimpleNamespace.
>>> from types import SimpleNamespace

>>> # SimpleNamespace allows us to set arbitrary attributes.
>>> # It is an explicit, handy replacement for "class X: pass".
>>> ns = SimpleNamespace()

>>> # Define a function to operate on an object's attribute. >>> def square(instance):
...     instance.n *= instance.n
...
>>> ns.n = 4
>>> square(ns)
>>> ns.n
16
```

请注意，`square()`需要被编写为直接操作一个属性，该属性将被修改，而不需要重新分配返回值。

值得重复的是，您应该确保属性支持赋值！下面是与`namedtuple`相同的例子，它的属性是只读的:

>>>

```py
>>> from collections import namedtuple
>>> NS = namedtuple("NS", "n")
>>> def square(instance):
...     instance.n *= instance.n
...
>>> ns = NS(4)
>>> ns.n
4
>>> square(ns)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in square
AttributeError: can't set attribute
```

试图修改不允许修改的属性会导致`AttributeError`。

此外，您应该注意类属性。它们将保持不变，并将创建和修改一个实例属性:

>>>

```py
>>> class NS:
...     n = 4
...
>>> ns = NS()
>>> def square(instance):
...     instance.n *= instance.n
...
>>> ns.n
4
>>> square(ns)
>>> # Instance attribute is modified.
>>> ns.n
16
>>> # Class attribute remains unchanged.
>>> NS.n
4
```

因为类属性在通过类实例修改时保持不变，所以您需要记住引用实例属性。

[*Remove ads*](/account/join/)

### 最佳实践:使用字典和列表

Python 中的字典是一种不同于所有其他内置类型的对象类型。它们被称为**映射类型**。Python 关于映射类型的文档提供了对该术语的一些见解:

> 一个[映射](https://docs.python.org/3/glossary.html#term-mapping)对象将[的散列值](https://docs.python.org/3/glossary.html#term-hashable)映射到任意对象。映射是可变的对象。目前只有一种标准的映射类型，即字典。([来源](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict))

本教程没有介绍如何实现自定义映射类型，但是您可以使用简单的字典通过引用来复制传递。下面是一个使用直接作用于字典元素的函数的示例:

>>>

```py
>>> # Dictionaries are mapping types.
>>> mt = {"n": 4}
>>> # Define a function to operate on a key:
>>> def square(num_dict):
...     num_dict["n"] *= num_dict["n"]
...
>>> square(mt)
>>> mt
{'n': 16}
```

因为您是在给字典键重新赋值，所以对字典元素的操作仍然是一种赋值形式。使用字典，您可以通过同一个字典对象访问修改后的值。

虽然列表不是映射类型，但是您可以像使用字典一样使用它们，因为有两个重要的特性:**可订阅性**和**可变性**。这些特征值得再解释一下，但是让我们先来看看使用 Python 列表模仿按引用传递的最佳实践。

要使用列表复制引用传递，请编写一个直接作用于列表元素的函数:

>>>

```py
>>> # Lists are both subscriptable and mutable.
>>> sm = [4]
>>> # Define a function to operate on an index:
>>> def square(num_list):
...     num_list[0] *= num_list[0]
...
>>> square(sm)
>>> sm
[16]
```

因为您是在给列表中的元素重新赋值，所以对列表元素的操作仍然是一种赋值形式。与字典类似，列表允许您通过同一个列表对象访问修改后的值。

现在让我们来探索可订阅性。当一个对象的结构子集可以通过索引位置访问时，该对象是可订阅的:

>>>

```py
>>> subscriptable = [0, 1, 2]  # A list
>>> subscriptable[0]
0
>>> subscriptable = (0, 1, 2)  # A tuple
>>> subscriptable[0]
0
>>> subscriptable = "012"  # A string
>>> subscriptable[0]
'0'
>>> not_subscriptable = {0, 1, 2}  # A set
>>> not_subscriptable[0]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'set' object is not subscriptable
```

列表、元组和字符串是可下标的，但集合不是。试图访问一个不可订阅的对象元素会引发一个`TypeError`。

可变性是一个更广泛的主题，需要[额外的探索](https://realpython.com/pointers-in-python/#immutable-vs-mutable-objects)和[文档参考](https://docs.python.org/3/library/stdtypes.html#immutable-sequence-types)。简而言之，如果一个对象的结构可以就地改变而不需要重新分配，那么它就是**可变的**:

>>>

```py
>>> mutable = [0, 1, 2]  # A list
>>> mutable[0] = "x"
>>> mutable
['x', 1, 2]

>>> not_mutable = (0, 1, 2)  # A tuple
>>> not_mutable[0] = "x"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment

>>> not_mutable = "012"  # A string
>>> not_mutable[0] = "x"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'str' object does not support item assignment

>>> mutable = {0, 1, 2}  # A set
>>> mutable.remove(0)
>>> mutable.add("x")
>>> mutable
{1, 2, 'x'}
```

列表和集合是可变的，就像字典和其他映射类型一样。字符串和元组是不可变的。试图修改一个不可变对象的元素会引发一个`TypeError`。

## 结论

Python 的工作方式不同于支持通过引用或值传递参数的语言。函数参数成为分配给传递给函数的每个值的局部变量。但是这并不妨碍您在其他语言中通过引用传递参数时获得预期的相同结果。

**在本教程中，您学习了:**

*   Python 如何处理**给变量赋值**
*   Python 中函数参数如何通过赋值函数传递
*   为什么**返回值**是通过引用复制传递的最佳实践
*   如何使用**属性**、**字典**和**列表**作为备选最佳实践

您还学习了一些在 Python 中复制按引用传递构造的其他最佳实践。您可以使用这些知识来实现传统上需要支持按引用传递的模式。

为了继续您的 Python 之旅，我鼓励您更深入地研究您在这里遇到的一些相关主题，例如[可变性](https://realpython.com/pointers-in-python/#immutable-vs-mutable-objects)、[赋值表达式](https://realpython.com/python38-new-features/#the-walrus-in-the-room-assignment-expressions)和 [Python 名称空间和范围](https://realpython.com/python-namespaces-scope/)。

保持好奇，下次见！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**顺便引用 Python 中的:最佳实践**](/courses/pass-by-reference-python-best-practices/)*******