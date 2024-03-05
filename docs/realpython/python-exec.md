# Python 的 exec():执行动态生成的代码

> 原文：<https://realpython.com/python-exec/>

Python 内置的 **`exec()`** 函数可以让你从一个字符串或者编译后的代码输入中执行任意的 Python 代码。

当您需要运行动态生成的 Python 代码时,`exec()`函数会很方便，但是如果您不小心使用它，它会非常危险。在本教程中，您不仅将学习如何使用`exec()`，而且同样重要的是，何时可以在您的代码中使用该函数。

**在本教程中，您将学习如何:**

*   使用 Python 内置的 **`exec()`** 函数
*   使用`exec()`执行作为**字符串**或**编译代码**对象的**代码**
*   评估并最小化与在代码中使用`exec()`相关的**安全风险**

此外，您将编写一些使用`exec()`解决与动态代码执行相关的不同问题的例子。

为了充分利用本教程，您应该熟悉 Python 的[名称空间](https://realpython.com/python-namespaces-scope/)和[范围](https://realpython.com/python-scope-legb-rule/)，以及[字符串](https://realpython.com/python-strings/)。你也应该熟悉 Python 的一些[内置函数](https://docs.python.org/3/library/functions.html)。

**示例代码:** [单击此处下载免费的示例代码](https://realpython.com/bonus/python-exec-code/)，您将使用它来探索 exec()函数的用例。

## 了解 Python 的`exec()`

Python 内置的 [`exec()`](https://docs.python.org/3/library/functions.html#exec) 函数可以让你执行任何一段 Python 代码。通过这个函数，你可以执行**动态生成的代码**。这是您在程序执行期间读取、自动生成或获取的代码。正常情况下是字符串。

`exec()`函数获取一段代码，并像 Python 解释器一样执行它。Python 的`exec()`就像 [`eval()`](https://realpython.com/python-eval-function/) 但更强大，也容易出现安全问题。虽然`eval()`只能评估[表达式](https://realpython.com/python-operators-expressions/)，`exec()`可以执行语句序列，以及[导入](https://realpython.com/python-import/)，函数调用和定义，类定义和实例化，等等。本质上，`exec()`可以执行一个完整的全功能 Python 程序。

`exec()`的签名形式如下:

```py
exec(code [, globals [, locals]])
```

函数执行`code`，可以是包含有效 Python 代码的*字符串*，也可以是*编译的*代码对象。

**注:** Python 是[解释的](https://en.wikipedia.org/wiki/Interpreter_(computing))语言，而不是[编译的](https://en.wikipedia.org/wiki/Compiled_language)语言。然而，当你运行一些 Python 代码时，解释器将其翻译成[字节码](https://docs.python.org/3/glossary.html#term-bytecode)，这是一个 Python 程序在 [CPython](https://realpython.com/cpython-source-code-guide/) 实现中的内部表示。这个中间翻译也被称为**编译代码**，并且是 Python 的[虚拟机](https://docs.python.org/3/glossary.html#term-virtual-machine)执行的。

如果`code`是一个[字符串](https://realpython.com/python-strings/)，那么它就被*解析为一套 Python 语句*，然后在内部*编译成字节码*，最后*执行*，除非在解析或编译步骤中出现语法错误。如果`code`持有一个编译过的代码对象，那么它将被直接执行，从而使这个过程更加高效。

`globals`和`locals`参数允许你提供代表[全局](https://realpython.com/python-scope-legb-rule/#modules-the-global-scope)和[局部](https://realpython.com/python-scope-legb-rule/#functions-the-local-scope)名称空间的字典，其中`exec()`将运行目标代码。

`exec()`函数的[返回](https://realpython.com/python-return-statement/)值为 [`None`](https://realpython.com/null-in-python/) ，大概是因为并不是每一段代码都有一个最终的、唯一的、具体的结果。可能只是有些[副作用](https://en.wikipedia.org/wiki/Side_effect_(computer_science))。这种行为与`eval()`明显不同，后者返回计算表达式的结果。

为了初步感受一下`exec()`是如何工作的，您可以用两行代码创建一个基本的 Python 解释器:

>>>

```py
>>> while True:
...     exec(input("->> "))
...

->> print("Hello, World!")
Hello, World!

->> import this
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
 ...

->> x = 10
->> if 1 <= x <= 10: print(f"{x} is between 1 and 10")
10 is between 1 and 10
```

在这个例子中，您使用一个无限的 [`while`](https://realpython.com/python-while-loop/) 循环来模仿 Python [解释器](https://realpython.com/interacting-with-python/#starting-the-interpreter)或 [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop) 的行为。在循环内部，您使用 [`input()`](https://docs.python.org/3/library/functions.html#input) 在命令行获取用户的输入。然后使用`exec()`来处理和运行输入。

这个例子展示了`exec()` : *执行以字符串形式出现的代码的主要用例。*

**注意:**你已经知道使用`exec()`可能意味着安全风险。现在你已经看到了`exec()`的主要用例，你认为那些安全风险可能是什么？您将在本教程的后面找到答案。

当需要动态运行字符串形式的代码时，通常会使用`exec()`。例如，您可以编写一个程序来生成包含有效 Python 代码的字符串。您可以从程序执行的不同时刻获得的部分构建这些字符串。您还可以使用用户输入或任何其他输入源来构造这些字符串。

一旦将目标代码构建成字符串，就可以使用`exec()`来执行它们，就像执行任何 Python 代码一样。

在这种情况下，您很难确定您的字符串将包含什么。这就是为什么`exec()`意味着严重的安全风险的一个原因。如果您在构建代码时使用不可信的输入源，比如用户的直接输入，这一点尤其正确。

在编程中，像`exec()`这样的函数是一个非常强大的工具，因为它允许你编写能够动态生成和执行新代码的程序。为了生成这个新代码，您的程序将只使用运行时可用的信息。为了运行代码，您的程序将使用`exec()`。

然而，权力越大，责任越大。这个`exec()`函数意味着严重的[安全风险](#uncovering-and-minimizing-the-security-risks-behind-exec)，你很快就会知道。所以，大部分时间应该避免使用`exec()`。

在接下来的几节中，您将了解到`exec()`是如何工作的，以及如何使用这个函数来执行以字符串或编译后的代码对象形式出现的代码。

[*Remove ads*](/account/join/)

## 从字符串输入运行代码

调用`exec()`最常见的方式是使用来自基于字符串的输入的代码。要构建这个基于字符串的输入，可以使用:

*   单行代码或**单行代码片段**
*   用分号**分隔的多行代码**
*   由**换行符**分隔的多行代码
*   在三重引用的字符串中有多行代码，并且有适当的缩进

一行程序由一行代码组成，一次执行多个动作。假设您有一个数字序列，并且您想要构建一个包含输入序列中所有偶数的平方和的新序列。

要解决这个问题，您可以使用下面的一行代码:

>>>

```py
>>> numbers = [2, 3, 7, 4, 8]

>>> sum(number**2 for number in numbers if number % 2 == 0) 84
```

在突出显示的行中，您使用一个[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)来计算输入值序列中所有偶数的平方值。然后你用 [`sum()`](https://realpython.com/python-sum-function/) 来计算总平方和。

要用`exec()`运行这段代码，您只需要将您的单行代码转换成一个单行字符串:

>>>

```py
>>> exec("result = sum(number**2 for number in numbers if number % 2 == 0)")
>>> result
84
```

在本例中，您将一行代码表示为一个字符串。然后你把这个字符串送入`exec()`执行。原始代码和字符串之间的唯一区别是，后者将计算结果存储在一个变量中，供以后访问。记住`exec()`返回的是`None`，而不是具体的执行结果。为什么？因为不是每一段代码都有一个最终唯一的结果。

Python 允许你在一行代码中编写多个[语句](https://docs.python.org/3/reference/simple_stmts.html#simple-statements)，用分号分隔。即使不鼓励这种做法，也没有什么能阻止你这样做:

>>>

```py
>>> name = input("Your name: "); print(f"Hello, {name}!")
Your name: Leodanis
Hello, Leodanis!
```

您可以使用分号来分隔多个语句，并构建一个单行字符串作为`exec()`的参数。方法如下:

>>>

```py
>>> exec("name = input('Your name: '); print(f'Hello, {name}!')")
Your name: Leodanis
Hello, Leodanis!
```

这个例子的思想是，通过使用分号分隔多个 Python 语句，可以将它们组合成一个单行字符串。在这个例子中，第一个语句接受用户的输入，而第二个语句[将问候消息打印到屏幕上。](https://realpython.com/python-print/)

您还可以使用换行符`\n`在一个单行字符串中聚合多个语句:

>>>

```py
>>> exec("name = input('Your name: ')\nprint(f'Hello, {name}!')")
Your name: Leodanis
Hello, Leodanis!
```

换行符使`exec()`将单行字符串理解为多行 Python 语句集。然后`exec()`在一行中运行聚合语句，这就像一个多行代码文件。

构建用于输入`exec()`的基于字符串的输入的最后一种方法是使用三重引号字符串。这种方法可以说更加灵活，允许您生成基于字符串的输入，其外观和工作方式与普通 Python 代码相似。

值得注意的是，这种方法要求您使用正确的缩进和代码格式。考虑下面的例子:

>>>

```py
>>> code = """
... numbers = [2, 3, 7, 4, 8]
... ... def is_even(number):
...     return number % 2 == 0
... ... even_numbers = [number for number in numbers if is_even(number)]
... ... squares = [number**2 for number in even_numbers]
... ... result = sum(squares)
... ... print("Original data:", numbers)
... print("Even numbers:", even_numbers)
... print("Square values:", squares)
... print("Sum of squares:", result)
... """

>>> exec(code)
Original data: [2, 3, 7, 4, 8]
Even numbers: [2, 4, 8]
Square values: [4, 16, 64]
Sum of squares: 84
```

在这个例子中，您使用一个三重引号字符串向`exec()`提供输入。注意，这个字符串看起来像任何一段普通的 Python 代码。它使用适当的缩进、命名风格和格式。`exec()`函数将理解并执行这个字符串作为一个常规的 Python 代码文件。

您应该注意到，当您将一个带有代码的字符串传递给`exec()`时，该函数将解析目标代码并将其编译成 Python 字节码。在所有情况下，输入字符串都应该包含有效的 Python 代码。

如果`exec()`在解析和编译步骤中发现任何[无效语法](https://realpython.com/invalid-syntax-python/)，那么输入代码将不会运行:

>>>

```py
>>> exec("print('Hello, World!)")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    File "<string>", line 1
 print('Hello, World!)            ^
SyntaxError: unterminated string literal (detected at line 1)
```

在这个例子中，目标代码包含对`print()`的调用，该调用将一个字符串作为参数。这个字符串没有正确地以结束单引号结束，所以`exec()`抛出一个`SyntaxError`指出这个问题，并且不运行输入代码。请注意，Python 将错误定位在字符串的开头，而不是应该放在结束单引号的结尾。

运行字符串形式的代码，就像你在上面的例子中所做的那样，可以说是使用`exec()`的自然方式。但是，如果您需要多次运行输入代码，那么使用字符串作为参数将使函数每次都运行解析和编译步骤。这种行为会使您的代码在执行速度方面效率低下。

在这种情况下，最方便的方法是预先编译目标代码，然后根据需要用`exec()`多次运行最终的字节码。在下一节中，您将学习如何对编译后的代码对象使用`exec()`。

[*Remove ads*](/account/join/)

## 执行编译后的代码

实际上，`exec()`在处理包含代码的字符串时会非常慢。如果您需要不止一次地动态运行一段给定的代码，那么预先编译它将是最有效和推荐的方法。为什么？因为您将只运行一次解析和编译步骤，然后重用编译后的代码。

要编译一段 Python 代码，可以使用 [`compile()`](https://docs.python.org/3/library/functions.html#compile) 。这个内置函数将一个字符串作为参数，并在其上运行一次性的字节码编译，生成一个代码对象，然后可以传递给`exec()`执行。

`compile()`的签名形式如下:

```py
compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1)
```

在本教程中，您将只使用`compile()`的前三个参数。`source`参数保存了需要编译成字节码的代码。`filename`参数将保存从中读取代码的文件。要读取一个字符串对象，必须将`filename`设置为`"<string>"`值。

**注意:**要深入了解`compile()`的其余参数，请查看函数的官方[文档](https://docs.python.org/3/library/functions.html?highlight=built#compile)。

最后，`compile()`可以生成代码对象，您可以使用`exec()`或`eval()`来执行这些代码对象，这取决于`mode`参数的值。根据目标执行功能，该参数应设置为`"exec"`或`"eval"`:

>>>

```py
>>> string_input = """
... def sum_of_even_squares(numbers):
...     return sum(number**2 for number in numbers if number % 2 == 0)
... ... print(sum_of_even_squares(numbers))
... """

>>> compiled_code = compile(string_input, "<string>", "exec")
>>> exec(compiled_code)

>>> numbers = [2, 3, 7, 4, 8]
>>> exec(compiled_code)
84

>>> numbers = [5, 3, 9, 6, 1]
>>> exec(compiled_code)
36
```

用`compile()`预先编译经常重复的代码可以帮助您稍微提高代码的性能，方法是在每次调用`exec()`时跳过解析和字节码编译步骤。

## 从 Python 源文件运行代码

你也可以使用`exec()`来运行你从你的文件系统或者其他地方的一个可靠的`.py` [文件](https://realpython.com/working-with-files-in-python/)中读取的代码。为此，您可以使用内置的 [`open()`](https://realpython.com/working-with-files-in-python/#pythons-with-open-as-pattern) 函数以字符串形式读取文件内容，然后您可以将其作为参数传递给`exec()`。

例如，假设您有一个名为`hello.py`的 Python 文件，其中包含以下代码:

```py
# hello.py

print("Hello, Pythonista!")
print("Welcome to Real Python!")

def greet(name="World"):
    print(f"Hello, {name}!")
```

这个示例脚本将问候语和欢迎消息打印到屏幕上。它还定义了一个用于测试的示例函数`greet()`。该函数将一个名称作为参数，并将定制的问候语打印到屏幕上。

现在回到 Python 交互式会话，运行以下代码:

>>>

```py
>>> with open("hello.py", mode="r", encoding="utf-8") as hello:
...     code = hello.read()
...

>>> exec(code)
Hello, Pythonista!
Welcome to Real Python!

>>> greet()
Hello, World!

>>> greet("Pythonista")
Hello, Pythonista!
```

在这个例子中，首先使用 [`with`语句](https://realpython.com/python-with-statement/)中的内置`open()`函数将目标`.py`文件作为常规文本文件打开。然后你调用 file 对象上的 [`.read()`](https://docs.python.org/3/library/io.html#io.TextIOBase.read) 将文件的内容读入`code` [变量](https://realpython.com/python-variables/)。对`.read()`的调用将文件内容作为字符串返回。最后一步是用这个字符串作为参数调用`exec()`。

该示例运行代码，并使位于`hello.py`中的`greet()`函数和对象在当前名称空间中可用。这就是为什么你可以直接使用`greet()`。这种行为背后的秘密与`globals`和`locals`参数有关，这将在下一节中介绍。

使用上例中的技术，您可以打开、读取和执行任何包含 Python 代码的文件。当您事先不知道将运行哪些源文件时，这种技术可能会起作用。所以，你不能写`import module`，因为当你写代码的时候，你不知道模块的名字。

**注意:**在 Python 中，您会发现获得类似结果的更安全的方法。例如，您可以使用导入系统。要更深入地了解这种替代方案，请查看[动态导入](https://realpython.com/python-import/#dynamic-imports)。

如果您选择使用这种技术，那么请确保您只执行来自可信源文件的代码。理想情况下，最可靠的源文件是那些你有意识地创建来动态运行的文件。在没有检查代码之前，决不能运行来自外部来源(包括您的用户)的代码文件。

[*Remove ads*](/account/join/)

## 使用`globals`和`locals`自变量

您可以使用`globals`和`locals`参数将执行上下文传递给`exec()`。这些参数可以接受字典对象，这些字典对象将作为全局和局部的[名称空间](https://realpython.com/python-namespaces-scope/)，这些名称空间`exec()`将用来运行目标代码。

这些参数是可选的。如果你省略了它们，那么`exec()`将执行当前[作用域](https://realpython.com/python-scope-legb-rule/)中的输入代码，这个作用域中的所有名字和对象都将对`exec()`可用。同样，在调用`exec()`之后，您在输入代码中定义的所有名称和对象都将在当前作用域中可用。

考虑下面的例子:

>>>

```py
>>> code = """
... z = x + y
... """

>>> # Global names are accessible from exec()
>>> x = 42
>>> y = 21

>>> z
Traceback (most recent call last):
    ...
NameError: name 'z' is not defined

>>> exec(code)
>>> # Names in code are available in the current scope
>>> z
63
```

这个例子表明，如果您调用`exec()`而没有为`globals`和`locals`参数提供特定的值，那么函数将在当前范围内运行输入代码。在这种情况下，当前范围是全局范围。

注意，在调用`exec()`之后，输入代码中定义的名字在当前作用域中也是可用的。这就是为什么你可以在最后一行代码中访问`z`。

如果只给`globals`提供一个值，那么这个值必须是一个字典。`exec()`函数将把这个字典用于全局名和本地名。此行为将限制对当前作用域中大多数名称的访问:

>>>

```py
>>> code = """
... z = x + y
... """

>>> x = 42
>>> y = 21

>>> exec(code, {"x": x})
Traceback (most recent call last):
    ...
NameError: name 'y' is not defined

>>> exec(code, {"x": x, "y": y})

>>> z
Traceback (most recent call last):
    ...
NameError: name 'z' is not defined
```

在对`exec()`的第一次调用中，您使用一个字典作为`globals`参数。因为您的字典没有提供保存`y`名字的键，所以对`exec()`的调用不能访问这个名字，并引发了一个`NameError`异常。

在对`exec()`的第二次调用中，您向`globals`提供了一个不同的字典。在这种情况下，字典包含两个变量，`x`和`y`，这允许函数正确工作。然而，这一次您在调用`exec()`后无法访问`z`。为什么？因为您正在使用一个定制的字典来为`exec()`提供一个执行范围，而不是回到您当前的范围。

如果您用一个没有明确包含`__builtins__`键的`globals`字典调用`exec()`，那么 Python 将自动在该键下插入一个对[内置范围](https://realpython.com/python-scope-legb-rule/#builtins-the-built-in-scope)或名称空间的引用。因此，所有内置对象都可以从目标代码中访问:

>>>

```py
>>> code = """
... print(__builtins__)
... """

>>> exec(code, {})
{'__name__': 'builtins', '__doc__': "Built-in functions, ...}
```

在本例中，您已经为`globals`参数提供了一个空字典。注意，`exec()`仍然可以访问内置的名称空间，因为这个名称空间会自动插入到提供的字典中的`__builtins__`键下。

如果您为`locals`参数提供一个值，那么它可以是任何一个[映射](https://docs.python.org/3/glossary.html#term-mapping)对象。当`exec()`运行您的目标代码时，这个映射对象将保存本地名称空间:

>>>

```py
>>> code = """
... z = x + y
... print(f"{z=}")
... """

>>> x = 42  # Global name

>>> def func():
...     y = 21  # Local name
...     exec(code, {"x": x}, {"y": y})
...

>>> func()
z=63

>>> z
Traceback (most recent call last):
    ...
NameError: name 'z' is not defined
```

在这个例子中，对`exec()`的调用嵌入在一个函数中。因此，您有一个全局(模块级)作用域和一个局部(函数级)作用域。`globals`参数提供来自全局范围的`x`名称，而`locals`参数提供来自局部范围的`y`名称。

注意，在运行`func()`之后，您不能访问`z`,因为这个名称是在`exec()`的本地作用域下创建的，从外部是不可用的。

使用`globals`和`locals`参数，您可以调整`exec()`运行代码的上下文。当谈到最小化与`exec()`相关的安全风险时，这些争论非常有用，但是您仍然应该确保您只运行来自可信来源的代码。在下一节中，您将了解这些安全风险以及如何应对它们。

## 揭露并最大限度减少`exec()` 背后的安全风险

到目前为止，您已经了解到，`exec()`是一个强大的工具，它允许您执行以字符串形式出现的任意代码。你应该非常小心谨慎地使用`exec()`，因为它能够运行*任何一段*代码。

通常，提供给`exec()`的代码是在运行时动态生成的。这段代码可能有许多输入源，包括您的程序用户、其他程序、数据库、数据流和网络连接等等。

在这种情况下，您不能完全确定输入字符串将包含什么。因此，面对不可信和恶意的输入代码源的可能性非常高。

与`exec()`相关的安全问题是许多 Python 开发者建议完全避免这个函数的最常见原因。找到更好、更快、更健壮、更安全的解决方案几乎总是可能的。

但是，如果您必须在代码中使用`exec()`，那么通常推荐的方法是使用显式的`globals`和`locals`字典。

`exec()`的另一个关键问题是它打破了编程中的一个基本假设:*你当前正在读或写的代码就是你启动程序时将要执行的代码。*`exec()`如何打破这个假设？它让你的程序运行动态生成的新的未知代码。这种新代码可能难以遵循、维护，甚至难以控制。

在接下来的章节中，如果您需要在代码中使用`exec()`，您将深入了解一些您应该应用的建议、技术和实践。

[*Remove ads*](/account/join/)

### 避免来自不可信来源的输入

如果您的用户可以在运行时为您的程序提供任意 Python 代码，那么如果他们输入违反或破坏您的安全规则的代码，就会出现问题。为了说明这个问题，回到使用`exec()`执行代码的 Python 解释器示例:

>>>

```py
>>> while True:
...     exec(input("->> "))
...
->> print("Hello, World!")
Hello, World!
```

现在假设您想使用这种技术在一个 Linux web 服务器上实现一个交互式 Python 解释器。如果您允许您的用户将任意代码直接传递到您的程序中，那么恶意用户可能会提供类似于`"import os; os.system('rm -rf *')"`的东西。这段代码可能会删除你服务器磁盘上的所有内容，所以*不要运行它*。

为了防止这种风险，您可以利用`globals`字典来限制对`import`系统的访问:

>>>

```py
>>> exec("import os", {"__builtins__": {}}, {})
Traceback (most recent call last):
    ...
ImportError: __import__ not found
```

`import`系统内部使用内置的`__import__()`函数。所以，如果你禁止访问内置的名称空间，那么`import`系统将无法工作。

尽管您可以按照上面的例子调整`globals`字典，但是有一点您绝对不能做，那就是在您自己的计算机上使用`exec()`来运行外部的和潜在的不安全代码。即使您仔细清理和验证了输入，您也有被黑客攻击的风险。所以，你最好避免这种做法。

### 限制`globals`和`locals`以最小化风险

如果您想在使用`exec()`运行代码时微调对全局和本地名称的访问，您可以提供自定义字典作为`globals`和`locals`参数。例如，如果您将空字典传递给`globals`和`locals`，那么`exec()`将无法访问您当前的全局和本地名称空间:

>>>

```py
>>> x = 42
>>> y = 21

>>> exec("print(x + y)", {}, {})
Traceback (most recent call last):
    ...
NameError: name 'x' is not defined
```

如果你用空字典调用`globals`和`locals`来调用`exec()`，那么你就禁止访问全局和本地名字。这个调整允许您在使用`exec()`运行代码时限制可用的名称和对象。

然而，这种技术不能保证安全使用`exec()`。为什么？因为该函数仍然可以访问 Python 的所有内置名称，正如您在[部分](#using-the-globals-and-locals-arguments)中了解到的`globals`和`locals`参数:

>>>

```py
>>> exec("print(min([2, 3, 7, 4, 8]))", {}, {})
2

>>> exec("print(len([2, 3, 7, 4, 8]))", {}, {})
5
```

在这些例子中，您为`globals`和`locals`使用了空字典，但是`exec()`仍然可以访问内置函数，如 [`min()`](https://realpython.com/python-min-and-max/) 、 [`len()`](https://realpython.com/len-python-function/) 和 [`print()`](https://realpython.com/python-print/) 。你如何阻止`exec()`访问内置名字？这是下一节的主题。

### 决定允许的内置名称

正如您已经了解到的，如果您将一个自定义字典传递给没有`__builtins__`键的`globals`，那么 Python 将自动使用新的`__builtins__`键下内置范围内的所有名称更新该字典。为了限制这种隐式行为，您可以使用一个包含具有适当值的`__builtins__`键的`globals`字典。

例如，如果您想完全禁止访问内置名称，那么您可以像下面这样调用`exec()`:

>>>

```py
>>> exec("print(min([2, 3, 7, 4, 8]))", {"__builtins__": {}}, {})
Traceback (most recent call last):
    ...
NameError: name 'print' is not defined
```

在这个例子中，您将`globals`设置为一个包含一个`__builtins__`键的自定义字典，一个空字典作为它的关联值。这种做法可以防止 Python 将对内置名称空间的引用插入到`globals`中。通过这种方式，可以确保`exec()`在执行代码时无法访问内置名称。

如果您只需要`exec()`访问某些内置名称，您也可以调整您的`__builtins__`键:

>>>

```py
>>> allowed_builtins = {"__builtins__": {"min": min, "print": print}} >>> exec("print(min([2, 3, 7, 4, 8]))", allowed_builtins, {})
2

>>> exec("print(len([2, 3, 7, 4, 8]))", allowed_builtins, {})
Traceback (most recent call last):
    ...
NameError: name 'len' is not defined
```

在第一个例子中，`exec()`成功地运行了您的输入代码，因为`min()`和`print()`出现在与`__builtins__`键相关联的字典中。在第二个例子中，`exec()`引发了一个`NameError`，并且不运行您的输入代码，因为`len()`不在提供的`allowed_builtins`中。

上面例子中的技术允许你最小化使用`exec()`的安全隐患。然而，这些技术并不是完全安全的。所以，每当你觉得需要使用`exec()`的时候，试着去想另一个不使用该功能的解决方案。

[*Remove ads*](/account/join/)

## 将`exec()`付诸行动

至此，您已经了解了内置的`exec()`函数是如何工作的。您知道可以使用`exec()`来运行基于字符串或编译代码的输入。您还了解了这个函数可以接受两个可选参数，`globals`和`locals`，这允许您调整`exec()`的执行名称空间。

此外，您已经了解到使用`exec()`意味着一些严重的安全问题，包括允许用户在您的计算机上运行任意 Python 代码。您研究了一些推荐的编码实践，它们有助于最小化与代码中的`exec()`相关的安全风险。

在接下来的部分中，您将编写几个实际的例子，帮助您发现适合使用`exec()`的用例。

### 从外部来源运行代码

使用`exec()`执行来自用户或任何其他来源的字符串代码可能是`exec()`最常见也是最危险的用例。对于您来说，该函数是接受字符串形式的代码并在给定程序的上下文中将其作为常规 Python 代码运行的最快方式。

你绝不能使用`exec()`在你的机器上运行任意的外部代码，因为没有安全的方法可以做到这一点。如果你打算使用`exec()`，那么就用它来让你的用户在*他们自己的机器*上运行*他们自己的代码*。

标准库有一些模块使用`exec()`来执行用户以字符串形式提供的代码。一个很好的例子就是 [`timeit`](https://docs.python.org/3/library/timeit.html#module-timeit) 模块，[吉多·范·罗苏姆](https://twitter.com/gvanrossum)原来自己写的。

`timeit`模块提供了一种快速的方法来计时以字符串形式出现的小块 Python 代码。查看模块文档中的以下示例:

>>>

```py
>>> from timeit import timeit

>>> timeit("'-'.join(str(n) for n in range(100))", number=10000)
0.1282792080000945
```

`timeit()`函数将代码片段作为字符串，运行代码，并返回执行时间的测量值。该函数还接受其他几个参数。例如，`number`允许您提供想要执行目标代码的次数。

在这个函数的核心，你会发现 [`Timer`](https://docs.python.org/3/library/timeit.html#timeit.Timer) 类。`Timer`使用`exec()`运行提供的代码。如果你检查 [`timeit`](https://github.com/python/cpython/blob/main/Lib/timeit.py) 模块中`Timer`的源代码，那么你会发现这个类的[初始化器](https://realpython.com/python-class-constructor/#object-initialization-with-__init__)，`.__init__()`，包括以下代码:

```py
# timeit.py

# ...

class Timer:
    """Class for timing execution speed of small code snippets."""

    def __init__(
        self,
        stmt="pass",
        setup="pass",
        timer=default_timer,
        globals=None
    ):
        """Constructor.  See class doc string."""
        self.timer = timer
        local_ns = {}
        global_ns = _globals() if globals is None else globals
        # ...
        src = template.format(stmt=stmt, setup=setup, init=init)
        self.src = src  # Save for traceback display
        code = compile(src, dummy_src_name, "exec")
 exec(code, global_ns, local_ns)        self.inner = local_ns["inner"]

    # ...
```

高亮行中对`exec()`的调用使用`global_ns`和`local_ns`作为全局和局部名称空间来执行用户代码。

当你为你的用户提供一个工具时，这种使用`exec()`的方式是合适的，用户必须提供他们自己的目标代码。这些代码将在用户的机器上运行，因此他们将负责保证输入代码的安全运行。

使用`exec()`运行字符串代码的另一个例子是 [`doctest`](https://realpython.com/python-doctest/) 模块。这个模块检查文档字符串，寻找看起来像 Python [交互式](https://realpython.com/interacting-with-python/)会话的文本。如果`doctest`发现任何类似交互式会话的文本，那么它将该文本作为 Python 代码执行，以检查它是否如预期那样工作。

例如，假设您有以下将两个数字相加的函数:

```py
# calculations.py

def add(a, b):
    """Return the sum of two numbers.

 Tests:
 >>> add(5, 6)
 11
 >>> add(2.3, 5.4)
 7.7
 >>> add("2", 3)
 Traceback (most recent call last):
 TypeError: numeric type expected for "a" and "b"
 """
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError('numeric type expected for "a" and "b"')
    return a + b

# ...
```

在这个代码片段中，`add()`用几个测试定义了一个 docstring，这些测试检查函数应该如何工作。注意，这些测试代表了在一个假设的交互会话中使用有效和无效参数类型对`add()`的调用。

一旦在 docstrings 中有了这些交互式测试和它们的预期输出，就可以使用`doctest`来运行它们，并检查它们是否发出预期的结果。

**注意:**doctest 模块提供了一个令人惊叹的有用工具，您可以在编写代码时使用它来测试代码。

转到命令行，在包含您的`calculations.py`文件的目录中运行以下命令:

```py
$ python -m doctest calculations.py
```

如果所有测试都按预期进行，这个命令不会发出任何输出。如果至少有一个测试失败，那么您将得到一个指出问题的异常。要确认这一点，您可以在函数的 docstring 中更改一个预期的输出，并再次运行上面的命令。

`doctest`模块使用`exec()`来执行任何交互式嵌入 docstring 的代码，你可以在模块的[源代码](https://github.com/python/cpython/blob/main/Lib/doctest.py)中确认:

```py
# doctest.py

class DocTestRunner:
    # ...

    def __run(self, test, compileflags, out):
        # ...
        try:
            # Don't blink!  This is where the user's code gets run.
 exec( compile(example.source, filename, "single", compileflags, True), test.globs )            self.debugger.set_continue() # ==== Example Finished ====
            exception = None
        except KeyboardInterrupt:
        # ...
```

正如您在这个代码片段中可以确认的，用户的代码在一个`exec()`调用中运行，该调用使用`compile()`来编译目标代码。为了运行这段代码，`exec()`使用`test.globs`作为它的`globals`参数。注意，在调用`exec()`之前的注释开玩笑地说这是用户代码运行的地方。

同样，在这个用例`exec()`中，提供安全代码示例的责任在用户身上。`doctest`维护者不负责确保对`exec()`的调用不会造成任何损害。

需要注意的是，`doctest`并不能防止与`exec()`相关的安全风险。换句话说，`doctest`将运行任何 Python 代码。例如，有人可以修改您的`add()`函数，在 docstring 中包含以下代码:

```py
# calculations.py

def add(a, b):
    """Return the sum of two numbers.

 Tests:
 >>> import os; os.system("ls -l") 0 """
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError('numeric type expected for "a" and "b"')
    return a + b
```

如果在这个文件上运行`doctest`，那么`ls -l`命令将成功运行。在这个例子中，嵌入的命令基本上是无害的。然而，恶意用户可以修改您的 docstring 并嵌入类似`os.system("rm -rf *")`或任何其他危险命令的东西。

同样，你必须小心使用`exec()`和使用该功能的工具，就像`doctest`一样。在`doctest`的具体案例中，只要你知道你的嵌入式测试代码来自哪里，这个工具就会相当安全和有用。

[*Remove ads*](/account/join/)

### 将 Python 用于配置文件

可以使用`exec()`运行代码的另一种情况是当您有一个使用有效 Python 语法的配置文件时。您的文件可以定义几个具有特定值的配置参数。然后，您可以读取该文件并用`exec()`处理其内容，以构建一个包含所有配置参数及其值的字典对象。

例如，假设您正在使用的文本编辑器应用程序有以下配置文件:

```py
# settings.conf

font_face = ""
font_size = 10
line_numbers = True
tab_size = 4
auto_indent = True
```

这个文件具有有效的 Python 语法，所以您可以像处理常规的`.py`文件一样使用`exec()`来执行它的内容。

**注意:**你会发现几种比使用`exec()`更好、更安全的方法来处理配置文件。在 Python 标准库中，您有 [`configparser`](https://docs.python.org/3/library/configparser.html#module-configparser) 模块，它允许您处理使用 [INI 文件格式](https://en.wikipedia.org/wiki/INI_file)的配置文件。

下面的函数读取您的`settings.conf`文件并构建一个配置字典:

>>>

```py
>>> from pathlib import Path

>>> def load_config(config_file):
...     config_file = Path(config_file)
...     code = compile(config_file.read_text(), config_file.name, "exec")
...     config_dict = {}
...     exec(code, {"__builtins__": {}}, config_dict)
...     return config_dict
...

>>> load_config("settings.conf")
{
 'font_face': '',
 'font_size': 10,
 'line_numbers': True,
 'tab_size': 4,
 'auto_indent': True
}
```

`load_config()`函数获取配置文件的路径。然后，它将目标文件作为文本读取，并将该文本传递给`exec()`以供执行。在`exec()`运行期间，该函数将配置参数注入到`locals`字典中，该字典稍后将返回给调用者代码。

**注:**本节的技术是*大概*是`exec()`的一个安全用例。在这个例子中，您的系统上运行着一个应用程序，特别是一个文本编辑器。

如果你修改应用程序的配置文件以包含恶意代码，那么你只会伤害自己，而你很可能不会这么做。然而，您仍然有可能在应用程序的配置文件中意外包含潜在的危险代码。所以，如果你不小心的话，这项技术可能会变得不安全。

当然，如果您自己编写应用程序，并且发布了带有恶意代码的配置文件，那么您将会损害整个社区。

就是这样！现在，您可以从生成的字典中读取所有配置参数及其相应的值，并使用这些参数来设置编辑器项目。

## 结论

您已经学习了如何使用内置的 **`exec()`** 函数从字符串或字节码输入中执行 Python 代码。这个函数为执行动态生成的 Python 代码提供了一个快捷的工具。您还了解了如何最小化与`exec()`相关的安全风险，以及何时可以在代码中使用该函数。

**在本教程中，您已经学会了如何:**

*   使用 Python 内置的 **`exec()`** 函数
*   使用 Python 的`exec()`来运行**基于字符串的**和**编译代码的**输入
*   评估并最小化与使用`exec()`相关的**安全风险**

此外，您编写了一些实用的例子，帮助您更好地理解何时以及如何在 Python 代码中使用`exec()`。

**示例代码:** [单击此处下载免费的示例代码](https://realpython.com/bonus/python-exec-code/)，您将使用它来探索 exec()函数的用例。******