# Python eval():动态评估表达式

> 原文：<https://realpython.com/python-eval-function/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 Python eval()**](/courses/evaluate-expressions-dynamically-python-eval/) 动态计算表达式

Python 的 **`eval()`** 允许您从基于字符串或基于[编译代码的](https://docs.python.org/3/library/stdtypes.html#code-objects)输入中计算任意 Python [**表达式**](https://realpython.com/python-operators-expressions/) 。当您试图从任何作为[字符串](https://realpython.com/python-strings/)或编译的代码对象的输入中动态计算 Python 表达式时，这个函数会很方便。

虽然 Python 的`eval()`是一个非常有用的工具，但是这个函数有一些重要的安全隐患，您应该在使用它之前考虑一下。在本教程中，您将学习`eval()`是如何工作的，以及如何在您的 Python 程序中安全有效地使用它。

在本教程中，您将学习:

*   Python 的 **`eval()`** 是如何工作的
*   如何使用`eval()`到**动态评估**任意基于字符串或基于编译代码的输入
*   如何使你的代码不安全，如何最小化相关的安全风险

 **此外，您将学习如何使用 Python 的`eval()`编写一个应用程序，以交互方式计算数学表达式。在这个例子中，你将把你所学到的关于`eval()`的一切应用到现实世界的问题中。如果你想得到这个应用程序的代码，那么你可以点击下面的方框:

**下载示例代码:** [单击此处获取代码，您将在本教程中使用](https://realpython.com/bonus/python-eval-project/)来学习 Python 的 eval()。

## 了解 Python 的`eval()`

您可以使用内置的 Python [**`eval()`**](https://docs.python.org/3/library/functions.html#eval) 从基于字符串或基于编译代码的输入中动态计算表达式。如果您将一个[字符串](https://realpython.com/courses/python-strings/)传递给`eval()`，那么函数会解析它，将其编译成[字节码](https://docs.python.org/3/glossary.html#term-bytecode)，并将其作为一个 Python 表达式进行求值。但是如果你用一个编译过的代码对象调用`eval()`，那么这个函数只执行评估步骤，如果你用相同的输入多次调用`eval()`，这是非常方便的。

Python 的`eval()`的签名定义如下:

```py
eval(expression[, globals[, locals]])
```

该函数有一个名为`expression`的第一个参数，它保存了需要计算的表达式。`eval()`还带有两个可选参数:

1.  `globals`
2.  `locals`

在接下来的三节中，您将了解这些参数是什么，以及`eval()`如何使用它们来动态计算 Python 表达式。

**注意:**还可以使用 [**`exec()`**](https://realpython.com/python-exec/) 动态执行 Python 代码。`eval()`和`exec()`的主要区别在于`eval()`只能执行或计算表达式，而`exec()`可以执行任何一段 Python 代码。

[*Remove ads*](/account/join/)

### 第一个参数:`expression`

`eval()`的第一个自变量叫做 **`expression`** 。这是一个必需的参数，用于保存函数的基于字符串的**或基于编译代码的**输入。当您调用`eval()`时，`expression`的内容被评估为一个 Python 表达式。查看以下使用基于字符串的输入的示例:****

>>>

```py
>>> eval("2 ** 8")
256
>>> eval("1024 + 1024")
2048
>>> eval("sum([8, 16, 32])")
56
>>> x = 100
>>> eval("x * 2")
200
```

当您使用一个字符串作为参数调用`eval()`时，该函数返回对输入字符串求值的结果。默认情况下，`eval()`可以访问全局名称，比如上面例子中的`x`。

为了评估基于字符串的`expression`，Python 的`eval()`运行以下步骤:

1.  **解析** `expression`
2.  **将**编译成字节码
3.  **将**评估为 Python 表达式
4.  **返回**评估的结果

`eval()`的第一个参数的名称`expression`强调了该函数只适用于表达式，不适用于[复合语句](https://docs.python.org/3/reference/compound_stmts.html)。 [Python 文档](https://docs.python.org/3/)将**表达式**定义如下:

> **表情**
> 
> 一段可以被赋值的语法。换句话说，表达式是像文字、名称、属性访问、操作符或函数调用这样的表达式元素的集合，它们都返回值。与许多其他语言相比，并不是所有的语言结构都是表达式。还有不能做表达式的语句，比如`while`。赋值也是语句，不是表达式。([来源](https://docs.python.org/3/glossary.html#term-expression))

另一方面，Python **语句**具有以下定义:

> **声明**
> 
> 一个语句是一个套件(一个“代码块”)的一部分。一个语句可以是一个表达式，也可以是带有关键字的几个结构之一，比如`if`、`while`或`for`。([来源](https://docs.python.org/3/glossary.html#term-statement))

如果你试图将一个复合语句传递给`eval()`，那么你将得到一个 [`SyntaxError`](https://realpython.com/invalid-syntax-python/) 。看看下面的例子，在这个例子中，您试图使用`eval()`执行一个 [`if`语句](https://realpython.com/python-conditional-statements/):

>>>

```py
>>> x = 100
>>> eval("if x: print(x)")
  File "<string>", line 1
    if x: print(x)
    ^
SyntaxError: invalid syntax
```

如果您尝试使用 Python 的`eval()`来评估一个复合语句，那么您将得到一个`SyntaxError`，就像上面的[回溯](https://realpython.com/python-traceback/)一样。那是因为`eval()`只接受表情。任何其他语句，如`if`、[、`for`、](https://realpython.com/python-for-loop/)[、`while`、](https://realpython.com/python-while-loop/)、[、`import`、](https://realpython.com/absolute-vs-relative-python-imports/)、[、`def`、](https://realpython.com/defining-your-own-python-function/)或[、`class`、](https://realpython.com/courses/intro-object-oriented-programming-oop-python/)，都会引发错误。

**注意:**一个`for`循环是一个复合语句，但是`for` [关键字](https://realpython.com/python-keywords/)也可以用在[综合](https://realpython.com/courses/using-list-comprehensions-effectively/)中，被认为是表达式。你可以使用`eval()`来评估理解，即使他们使用了`for`关键字。

也不允许使用`eval()`进行赋值操作:

>>>

```py
>>> eval("pi = 3.1416")
  File "<string>", line 1
    pi = 3.1416
       ^
SyntaxError: invalid syntax
```

如果您试图将赋值操作作为参数传递给 Python 的`eval()`，那么您将得到一个`SyntaxError`。赋值操作是语句而不是表达式，语句不允许使用`eval()`。

每当解析器不理解输入表达式时，您也会得到一个`SyntaxError`。请看以下示例，在该示例中，您尝试对违反 Python 语法的表达式求值:

>>>

```py
>>> # Incomplete expression
>>> eval("5 + 7 *")
  File "<string>", line 1
    5 + 7 *
          ^
SyntaxError: unexpected EOF while parsing
```

您不能向`eval()`传递违反 Python 语法的表达式。在上面的例子中，您试图计算一个不完整的表达式(`"5 + 7 *"`)并得到一个`SyntaxError`，因为解析器不理解表达式的语法。

也可以将[编译后的代码对象](https://docs.python.org/3/library/stdtypes.html#code-objects)传递给 Python 的`eval()`。为了编译你将要传递给`eval()`的代码，你可以使用 [`compile()`](https://docs.python.org/3/library/functions.html#compile) 。这是一个内置函数，它可以将输入字符串编译成一个[代码对象](https://docs.python.org/3/library/stdtypes.html#code-objects)或一个 [AST 对象](https://docs.python.org/3/library/ast.html#ast.AST)，这样您就可以用`eval()`对其进行评估。

如何使用`compile()`的细节已经超出了本教程的范围，但是这里快速浏览一下它的前三个必需参数:

1.  **`source`** 保存着你要编译的源代码。该参数接受普通字符串、[字节字符串](https://docs.python.org/3/library/stdtypes.html#bytes-objects)和 AST 对象。
2.  **`filename`** 给出从中读取代码的文件。如果要使用基于字符串的输入，那么这个参数的值应该是`"<string>"`。
3.  **`mode`** 指定你想要得到哪种编译后的代码。如果你想用`eval()`处理编译后的代码，那么这个参数应该设置为`"eval"`。

**注:**关于`compile()`的更多信息，查看[官方文档](https://docs.python.org/3/library/functions.html#compile)。

你可以使用`compile()`向`eval()`提供代码对象，而不是普通的字符串。看看下面的例子:

>>>

```py
>>> # Arithmetic operations
>>> code = compile("5 + 4", "<string>", "eval")
>>> eval(code)
9
>>> code = compile("(5 + 7) * 2", "<string>", "eval")
>>> eval(code)
24
>>> import math
>>> # Volume of a sphere
>>> code = compile("4 / 3 * math.pi * math.pow(25, 3)", "<string>", "eval")
>>> eval(code)
65449.84694978735
```

如果你使用`compile()`来编译你要传递给`eval()`的表达式，那么`eval()`会经历以下步骤:

1.  **评估**编译后的代码
2.  **返回**评估的结果

如果您使用基于编译代码的输入调用 Python 的`eval()`,那么该函数将执行评估步骤并立即返回结果。当您需要多次计算同一个表达式时，这非常方便。在这种情况下，最好预编译表达式，并在后续调用`eval()`时重用得到的字节码。

如果你预先编译输入表达式，那么对`eval()`的连续调用将运行得更快，因为你不会重复**解析**和**编译**的步骤。如果计算复杂的表达式，不必要的重复会导致高 CPU 时间和过多的内存消耗。

[*Remove ads*](/account/join/)

### 第二个论点:`globals`

`eval()`的第二个自变量叫做 **`globals`** 。它是可选的，拥有一个为`eval()`提供全局[名称空间](https://realpython.com/python-scope-legb-rule/#python-scope-vs-namespace)的[字典](https://realpython.com/python-dicts/)。有了`globals`，你可以告诉`eval()`在评估`expression`时使用哪些全局名称。

全局名称是在您的[当前全局范围或名称空间](https://realpython.com/python-scope-legb-rule/#modules-the-global-scope)中可用的所有名称。您可以从代码中的任何地方访问它们。

字典中传递给`globals`的所有名字在执行时都可以被`eval()`使用。看看下面的例子，它展示了如何使用一个定制的字典为`eval()`提供一个全局的[名称空间](https://realpython.com/python-namespaces-scope/):

>>>

```py
>>> x = 100  # A global variable
>>> eval("x + 100", {"x": x})
200
>>> y = 200  # Another global variable
>>> eval("x + y", {"x": x})
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 1, in <module>
NameError: name 'y' is not defined
```

如果你给`eval()`的`globals`参数提供一个自定义字典，那么`eval()`将只把这些名字作为全局变量。在这个自定义字典之外定义的任何全局名称都不能从`eval()`内部访问。这就是为什么当你试图访问上面代码中的`y`时，Python 会抛出一个`NameError`:传递给`globals`的字典不包含`y`。

您可以通过在字典中列出名称来将它们插入到`globals`中，然后这些名称将在评估过程中可用。例如，如果您将`y`插入到`globals`，那么上述示例中对`"x + y"`的求值将按预期进行:

>>>

```py
>>> eval("x + y", {"x": x, "y": y})
300
```

因为您将`y`添加到您的自定义`globals`字典中，所以对`"x + y"`的求值是成功的，并且您得到了期望的返回值`300`。

您也可以提供当前全局范围中不存在的名称。为此，您需要为每个名称提供一个具体的值。运行时，`eval()`会将这些名称解释为全局名称:

>>>

```py
>>> eval("x + y + z", {"x": x, "y": y, "z": 300})
600
>>> z
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'z' is not defined
```

即使`z`没有在您当前的全局作用域中定义，变量[的值](https://realpython.com/python-variables/)仍然存在于`globals`中，其值为`300`。在这种情况下，`eval()`可以访问`z`，就像它是一个全局变量一样。

`globals`背后的机制相当灵活。可以将任何可见变量(全局、[局部](https://realpython.com/python-scope-legb-rule/#nested-functions-the-enclosing-scope)或[非局部](https://realpython.com/python-scope-legb-rule/#nested-functions-the-enclosing-scope)传递给`globals`。你也可以像上面例子中的`"z": 300`一样传递自定义的键值对。`eval()`会把它们都当作全局变量。

关于`globals`重要的一点是，如果你向它提供一个不包含键`"__builtins__"`值的自定义字典，那么在`expression`被解析之前，对 [`builtins`](https://docs.python.org/3/library/builtins.html#module-builtins) 字典的引用将自动插入到`"__builtins__"`下。这确保了在评估`expression`时`eval()`可以完全访问 Python 的所有内置名称。

下面的例子表明，即使您向`globals`提供一个空字典，对`eval()`的调用仍然可以访问 Python 的内置名称:

>>>

```py
>>> eval("sum([2, 2, 2])", {})
6
>>> eval("min([1, 2, 3])", {})
1
>>> eval("pow(10, 2)", {})
100
```

在上面的代码中，您提供了一个空字典(`{}`)到`globals`。由于该字典不包含名为`"__builtins__"`的键，Python 自动插入一个引用了`builtins`中的名字的键。这样，`eval()`在解析`expression`时就可以完全访问 Python 的所有内置名称。

如果调用`eval()`而没有将自定义字典传递给`globals`，那么参数将默认为调用`eval()`的环境中 [`globals()`](https://realpython.com/python-scope-legb-rule/#globals) 返回的字典:

>>>

```py
>>> x = 100  # A global variable
>>> y = 200  # Another global variable
>>> eval("x + y")  # Access both global variables
300
```

当您调用`eval()`而没有提供`globals`参数时，该函数使用`globals()`返回的字典作为其全局名称空间来计算`expression`。所以，在上面的例子中，你可以自由地访问`x`和`y`，因为它们是包含在你当前[全局作用域](https://realpython.com/python-scope-legb-rule/#modules-the-global-scope)中的全局变量。

[*Remove ads*](/account/join/)

### 第三个论点:`locals`

Python 的`eval()`带第三个参数，叫做 **`locals`** 。这是另一个保存字典的可选参数。在这种情况下，字典包含了`eval()`在评估`expression`时用作本地名称的变量。

局部名称是您在给定函数中定义的那些名称([变量](https://realpython.com/python-variables/)、[函数](https://realpython.com/defining-your-own-python-function/)、[类](https://realpython.com/courses/intro-object-oriented-programming-oop-python/)等等)。局部名称仅在封闭函数内部可见。当你写一个函数的时候，你可以定义这些类型的名字。

因为`eval()`已经写好了，所以你不能给它的代码或者[局部范围](https://realpython.com/python-scope-legb-rule/#functions-the-global-scope)添加局部名字。但是，您可以将一个字典传递给`locals`,`eval()`会将这些名称视为本地名称:

>>>

```py
>>> eval("x + 100", {}, {"x": 100})
200
>>> eval("x + y", {}, {"x": 100})
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 1, in <module>
NameError: name 'y' is not defined
```

对`eval()`的第一次调用中的第二个字典保存变量`x`。该变量被`eval()`解释为局部变量。换句话说，它被视为定义在`eval()`主体中的变量。

你可以在`expression`中使用`x`，`eval()`将可以访问它。相反，如果您尝试使用`y`，那么您将得到一个`NameError`，因为`y`既没有在`globals`名称空间中定义，也没有在`locals`名称空间中定义。

像使用`globals`一样，您可以将任何可见变量(全局、局部或非局部)传递给`locals`。你也可以像上面例子中的`"x": 100`一样传递自定义的键值对。`eval()`会把它们都当作局部变量。

注意，要向`locals`提供字典，首先需要向`globals`提供字典。不能在`eval()`中使用关键字参数:

>>>

```py
>>> eval("x + 100", locals={"x": 100})
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: eval() takes no keyword arguments
```

如果你试图在调用`eval()`时使用关键字参数，那么你会得到一个`TypeError`解释说`eval()`没有关键字参数。因此，在提供`locals`字典之前，您需要提供一个`globals`字典。

如果您没有将字典传递给`locals`，那么它默认为传递给`globals`的字典。这里有一个例子，你将一个空字典传递给`globals`，而没有传递给`locals`:

>>>

```py
>>> x = 100
>>> eval("x + 100", {})
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 1, in <module>
NameError: name 'x' is not defined
```

假设您没有向`locals`提供自定义字典，那么参数默认为传递给`globals`的字典。在这种情况下，`eval()`不能访问`x`，因为`globals`持有一个空字典。

`globals`和`locals`的主要实际区别在于，如果一个`"__builtins__"`键不存在，Python 会自动将该键插入到`globals`中。无论您是否向`globals`提供自定义词典，都会发生这种情况。另一方面，如果您向`locals`提供一个自定义词典，那么该词典将在`eval()`执行期间保持不变。

## 用 Python 的`eval()` 对表达式求值

您可以使用 Python 的`eval()`来计算任何类型的 Python 表达式，但不能计算 Python 语句，比如基于关键字的复合语句或赋值语句。

当您需要动态评估表达式时，使用其他 Python 技术或工具会大大增加您的开发时间和工作量，这会非常方便。在本节中，您将了解如何使用 Python 的`eval()`来计算布尔、数学和通用 Python 表达式。

### 布尔表达式

[**布尔表达式**](https://realpython.com/python-boolean/) 是 Python 表达式，当解释器对其求值时，返回一个真值(`True`或`False`)。它们通常用在`if`语句中，以检查某个条件是真还是假。因为布尔表达式不是复合语句，所以可以使用`eval()`来计算它们:

>>>

```py
>>> x = 100
>>> y = 100
>>> eval("x != y")
False
>>> eval("x < 200 and y > 100")
False
>>> eval("x is y")
True
>>> eval("x in {50, 100, 150, 200}")
True
```

您可以将`eval()`与使用以下任何 Python 运算符的布尔表达式一起使用:

*   [**值比较运算符**](https://docs.python.org/3/reference/expressions.html#value-comparisons) : `<`、`>`、`<=`、`>=`、`==`、`!=`
*   [**逻辑(布尔)运算符**](https://docs.python.org/3/reference/expressions.html#boolean-operations) : `and`、[、`or`、](https://realpython.com/python-or-operator/)、`not`
*   [**会员测试操作员**](https://docs.python.org/3/reference/expressions.html#membership-test-operations) : `in`，`not in`
*   [**身份符**](https://docs.python.org/3/reference/expressions.html#is-not) : [`is`，`is not`](https://realpython.com/python-is-identity-vs-equality/)

在所有情况下，该函数返回您正在评估的表达式的真值。

现在，你可能在想，我为什么要用`eval()`而不是直接用布尔表达式呢？好吧，假设你需要实现一个条件语句，但是你想动态地改变条件:

>>>

```py
>>> def func(a, b, condition):
...     if eval(condition):
...         return a + b
...     return a - b
...
>>> func(2, 4, "a > b")
-2
>>> func(2, 4, "a < b")
6
>>> func(2, 2, "a is b")
4
```

在`func()`中，您使用`eval()`对提供的`condition`进行评估，并根据评估结果返回`a + b`或`a - b`。在上面的例子中，您只使用了几个不同的条件，但是如果您坚持使用您在`func()`中定义的名称`a`和`b`，您可以使用任何数量的其他条件。

现在想象一下，如果不使用 Python 的`eval()`，你将如何实现这样的东西。这会花费更少的代码和时间吗？不会吧！

[*Remove ads*](/account/join/)

### 数学表达式

Python 的`eval()`的一个常见用例是从基于字符串的输入中计算数学表达式。例如，如果您想创建一个 [Python 计算器](https://realpython.com/python-pyqt-gui-calculator/)，那么您可以使用`eval()`来评估[用户的输入](https://realpython.com/python-input-output/)并返回计算结果。

以下示例显示了如何使用`eval()`和 [`math`](https://realpython.com/python-math-module/) 来执行数学运算:

>>>

```py
>>> # Arithmetic operations
>>> eval("5 + 7")
12
>>> eval("5 * 7")
35
>>> eval("5 ** 7")
78125
>>> eval("(5 + 7) / 2")
6.0
>>> import math
>>> # Area of a circle
>>> eval("math.pi * pow(25, 2)")
1963.4954084936207
>>> # Volume of a sphere
>>> eval("4 / 3 * math.pi * math.pow(25, 3)")
65449.84694978735
>>> # Hypotenuse of a right triangle
>>> eval("math.sqrt(math.pow(10, 2) + math.pow(15, 2))")
18.027756377319946
```

当使用`eval()`计算数学表达式时，可以传入任何种类或复杂度的表达式。`eval()`将解析它们，对它们进行评估，如果一切正常，将给出预期的结果。

### 通用表达式

到目前为止，您已经学习了如何在布尔和数学表达式中使用`eval()`。然而，您可以将`eval()`用于更复杂的 Python 表达式，包括函数调用、对象创建、属性访问、[理解](https://realpython.com/list-comprehension-python/)等等。

例如，您可以调用内置函数或通过标准或第三方模块导入的函数:

>>>

```py
>>> # Run the echo command
>>> import subprocess
>>> eval("subprocess.getoutput('echo Hello, World')")
'Hello, World'
>>> # Launch Firefox (if available)
>>> eval("subprocess.getoutput('firefox')")
''
```

在这个例子中，您使用 Python 的`eval()`来执行一些系统命令。你可以想象，你可以用这个特性做很多*有用的事情。然而，`eval()`也可能让您面临严重的安全风险，比如允许恶意用户在您的机器上运行系统命令或任意代码。*

在下一节中，您将了解解决与 eval()相关的一些安全风险的方法。

## 最大限度地减少`eval()` 的安全问题

尽管 Python 的用途几乎是无限的，但它的`eval()`也有重要的**安全隐患**。`eval()`被认为是不安全的，因为它允许您(或您的用户)动态执行任意 Python 代码。

这被认为是糟糕的编程实践，因为你正在读(或写)的代码是*而不是*你将要执行的代码。如果您计划使用`eval()`来评估来自用户或任何其他外部来源的输入，那么您将无法确定将要执行什么代码。如果您的应用程序运行在错误的人手中，这将是一个严重的安全风险。

出于这个原因，良好的编程实践通常建议不要使用`eval()`。但是如果你选择使用这个函数，那么经验法则是*永远不要*用**不可信的输入**来使用它。这条规则的棘手之处在于弄清楚哪种输入可以信任。

作为不负责任地使用`eval()`会使您的代码不安全的一个例子，假设您想要构建一个在线服务来评估任意的 Python 表达式。您的用户将引入表达式，然后单击`Run`按钮。该应用程序将获得用户的输入，并将其传递给`eval()`进行评估。

该应用程序将在您的个人服务器上运行。是的，就是你保存所有有价值文件的那台服务器。如果您运行的是 Linux 机器，并且应用程序的进程具有正确的权限，那么恶意用户可能会引入如下的危险字符串:

```py
"__import__('subprocess').getoutput('rm –rf *')"
```

上面的代码将删除应用程序当前目录中的所有文件。那太可怕了，不是吗？

**注意:** [`__import__()`](https://docs.python.org/3/library/functions.html#__import__) 是一个内置函数，以模块名为字符串，返回对模块对象的引用。`__import__()`是一个函数，与`import`语句完全不同。你不能用`eval()`来评估一个`import`语句。

当输入不可信时，没有完全有效的方法来避免与`eval()`相关的安全风险。但是，您可以通过限制`eval()`的执行环境来最小化您的风险。在接下来的几节中，您将学习一些这样做的技巧。

[*Remove ads*](/account/join/)

### 限制`globals`和`locals`

您可以通过将自定义字典传递给`globals`和`locals`参数来限制`eval()`的执行环境。例如，您可以将空字典传递给两个参数，以防止`eval()`访问调用者的[当前作用域或名称空间](https://realpython.com/python-scope-legb-rule/#python-scope-vs-namespace)中的名称:

>>>

```py
>>> # Avoid access to names in the caller's current scope
>>> x = 100
>>> eval("x * 5", {}, {})
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 1, in <module>
NameError: name 'x' is not defined
```

如果您将空字典(`{}`)传递给`globals`和`locals`，那么`eval()`在计算字符串`"x * 5"`时，无论是在其全局名称空间还是其本地名称空间中都不会找到名称`x`。结果，`eval()`会抛出一个`NameError`。

不幸的是，像这样限制`globals`和`locals`参数并不能消除与使用 Python 的`eval()`相关的所有安全风险，因为您仍然可以访问 Python 的所有内置名称。

### 限制内置名称的使用

正如您之前看到的，Python 的`eval()`在解析`expression`之前会自动将对`builtins`的字典的引用插入到`globals`中。恶意用户可以利用这种行为，通过使用内置函数`__import__()`来访问标准库和您系统上安装的任何第三方模块。

以下示例显示，即使在限制了`globals`和`locals`之后，您也可以使用任何内置函数和任何标准模块，如`math`或`subprocess`:

>>>

```py
>>> eval("sum([5, 5, 5])", {}, {})
15
>>> eval("__import__('math').sqrt(25)", {}, {})
5.0
>>> eval("__import__('subprocess').getoutput('echo Hello, World')", {}, {})
'Hello, World'
```

即使你限制`globals`和`locals`使用空字典，你仍然可以使用任何内置函数，就像你在上面的代码中使用`sum()`和`__import__()`一样。

你可以使用`__import__()`来导入任何标准或第三方模块，就像你在上面用`math`和`subprocess`所做的一样。使用这种技术，您可以访问在`math`、`subprocess`或任何其他模块中定义的任何函数或类。现在想象一下恶意用户使用`subprocess`或标准库中任何其他强大的模块会对您的系统做什么。

为了最小化这种风险，您可以通过覆盖`globals`中的`"__builtins__"`键来限制对 Python 内置函数的访问。良好的实践建议使用包含键值对`"__builtins__": {}`的定制字典。看看下面的例子:

>>>

```py
>>> eval("__import__('math').sqrt(25)", {"__builtins__": {}}, {})
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 1, in <module>
NameError: name '__import__' is not defined
```

如果你传递一个包含键值对`"__builtins__": {}`到`globals`的字典，那么`eval()`将不能直接访问 Python 的内置函数，比如`__import__()`。然而，正如您将在下一节看到的，这种方法仍然不能使`eval()`完全安全。

### 限制输入中的名称

即使您可以使用自定义的`globals`和`locals`字典来限制 Python 的`eval()`的执行环境，该函数仍然容易受到一些花哨技巧的攻击。例如，您可以使用**类型的文字**来访问类 [`object`](https://docs.python.org/3/library/functions.html#object) ，比如`""`、`[]`、`{}`或`()`以及一些特殊的属性:

>>>

```py
>>> "".__class__.__base__
<class 'object'>
>>> [].__class__.__base__
<class 'object'>
>>> {}.__class__.__base__
<class 'object'>
>>> ().__class__.__base__
<class 'object'>
```

一旦你可以访问`object`，你可以使用特殊的方法 [`.__subclasses__()`](https://docs.python.org/3/library/stdtypes.html#class.__subclasses__) 来访问所有从`object`继承的类。它是这样工作的:

>>>

```py
>>> for sub_class in ().__class__.__base__.__subclasses__():
...     print(sub_class.__name__)
...
type
weakref
weakcallableproxy
weakproxy
int
...
```

这段代码将[打印](https://realpython.com/python-print/)一个大的类列表到你的屏幕上。其中一些职业非常强大，如果落入坏人之手会非常危险。这打开了另一个重要的安全漏洞，仅仅限制`eval()`的执行环境是无法弥补的:

>>>

```py
>>> input_string = """[
...     c for c in ().__class__.__base__.__subclasses__()
...     if c.__name__ == "range"
... ][0](10)"""
>>> list(eval(input_string, {"__builtins__": {}}, {}))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

上面代码中的 list comprehension 过滤从`object`继承的类，返回一个包含类 [`range`](https://realpython.com/python-range/) 的`list`。第一个索引(`[0]`)返回类别`range`。一旦访问了`range`，就调用它来生成一个`range`对象。然后在`range`对象上调用`list()`来生成一个包含十个整数的列表。

在这个例子中，您使用`range`来说明`eval()`中的一个安全漏洞。现在想象一下，如果你的系统暴露了像 [`subprocess.Popen`](https://docs.python.org/3/library/subprocess.html#subprocess.Popen) 这样的类，恶意用户会做什么。

**注:**要更深入地了解`eval()`的漏洞，请查看 Ned Batchelder 的文章， [Eval 真的很危险。](http://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html)

这个漏洞的一个可能的解决方案是限制输入中名字的使用，要么限制为一堆安全的名字，要么限制为完全没有名字的名字。要实现这一技术，您需要完成以下步骤:

1.  **创建**一个字典，其中包含您想在`eval()`中使用的名字。
2.  **使用模式`"eval"`中的`compile()`将输入字符串编译成字节码。**
3.  **检查字节码对象上的** `.co_names`以确保它只包含允许的名字。
4.  如果用户试图输入一个不允许的名字，引发 a `NameError`。

看看下面的函数，您在其中实现了所有这些步骤:

>>>

```py
>>> def eval_expression(input_string):
...     # Step 1
...     allowed_names = {"sum": sum}
...     # Step 2
...     code = compile(input_string, "<string>", "eval")
...     # Step 3
...     for name in code.co_names:
...         if name not in allowed_names:
...             # Step 4
...             raise NameError(f"Use of {name} not allowed")
...     return eval(code, {"__builtins__": {}}, allowed_names)
```

在`eval_expression()`中，您实现了之前看到的所有步骤。这个函数将您可以与`eval()`一起使用的名字限制为字典`allowed_names`中的那些名字。为此，该函数使用了`.co_names`，它是一个代码对象的属性，返回一个包含代码对象中名称的[元组](https://realpython.com/python-lists-tuples/#python-tuples)。

以下示例展示了`eval_expression()`在实践中是如何工作的:

>>>

```py
>>> eval_expression("3 + 4 * 5 + 25 / 2")
35.5
>>> eval_expression("sum([1, 2, 3])")
6
>>> eval_expression("len([1, 2, 3])")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 10, in eval_expression
NameError: Use of len not allowed
>>> eval_expression("pow(10, 2)")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 10, in eval_expression
NameError: Use of pow not allowed
```

如果您调用`eval_expression()`来计算算术运算，或者如果您使用包含允许名称的表达式，那么您将得到预期的结果。否则你会得到一个`NameError`。在上面的例子中，你唯一允许的名字是`sum()`。像`len()`和`pow()`这样的名字是不允许的，所以当你试图使用它们时，这个函数会抛出一个`NameError`。

如果你想完全禁止使用名字，那么你可以重写`eval_expression()`如下:

>>>

```py
>>> def eval_expression(input_string):
...     code = compile(input_string, "<string>", "eval")
...     if code.co_names:
...         raise NameError(f"Use of names not allowed")
...     return eval(code, {"__builtins__": {}}, {})
...
>>> eval_expression("3 + 4 * 5 + 25 / 2")
35.5
>>> eval_expression("sum([1, 2, 3])")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 4, in eval_expression
NameError: Use of names not allowed
```

现在你的函数不允许*在输入字符串中有任何*名字。为了实现这一点，您检查`.co_names`中的名字，如果找到一个名字，就抛出一个`NameError`。否则，你评估`input_string`并返回评估结果。在这种情况下，您也使用一个空字典来限制`locals`。

您可以使用这种技术来最小化`eval()`的安全问题，并增强您抵御恶意攻击的能力。

[*Remove ads*](/account/join/)

### 将输入限制为文字

Python 的`eval()`的一个常见用例是评估包含标准 Python 文字的字符串，并将它们转换成具体的对象。

标准库提供了一个名为 [`literal_eval()`](https://docs.python.org/3/library/ast.html#ast.literal_eval) 的函数，可以帮助实现这个目标。该函数不支持运算符，但它支持[列表、元组](https://realpython.com/python-lists-tuples/)、[数字](https://realpython.com/python-numbers/)、字符串等:

>>>

```py
>>> from ast import literal_eval
>>> # Evaluating literals
>>> literal_eval("15.02")
15.02
>>> literal_eval("[1, 15]")
[1, 15]
>>> literal_eval("(1, 15)")
(1, 15)
>>> literal_eval("{'one': 1, 'two': 2}")
{'one': 1, 'two': 2}
>>> # Trying to evaluate an expression
>>> literal_eval("sum([1, 15]) + 5 + 8 * 2")
Traceback (most recent call last):
  ...
ValueError: malformed node or string: <_ast.BinOp object at 0x7faedecd7668>
```

注意`literal_eval()`只适用于标准类型的文字。它不支持使用运算符或名称。如果你试图给`literal_eval()`输入一个表达式，那么你会得到一个`ValueError`。这个函数还可以帮助您最小化与使用 Python 的`eval()`相关的安全风险。

## 使用 Python 的`eval()`与`input()`

在 [Python 3.x](https://realpython.com/products/python-basics-book) 中，内置的 [**`input()`**](https://docs.python.org/3/library/functions.html#input) 在命令行读取用户输入，将其转换为字符串，剥离尾部换行符，并将结果返回给调用者。由于`input()`的结果是一个字符串，您可以将它提供给`eval()`，并将其作为一个 Python 表达式进行计算:

>>>

```py
>>> eval(input("Enter a math expression: "))
Enter a math expression: 15 * 2
30
>>> eval(input("Enter a math expression: "))
Enter a math expression: 5 + 8
13
```

你可以用 Python 的`eval()`包装`input()`来自动评估用户的输入。这是`eval()`的一个常见用例，因为它模拟了 Python 2.x 中 [`input()`的行为，其中`input()`将用户的输入作为 Python 表达式进行评估并返回结果。](https://docs.python.org/2/library/functions.html#input)

Python 2.x 中`input()`的这种行为在 Python 3.x 中被改变了，因为它有安全隐患。

## 构建数学表达式评估器

到目前为止，您已经了解了 Python 的`eval()`如何工作，以及如何在实践中使用它。您还了解了`eval()`具有重要的安全含义，并且通常认为在代码中避免使用`eval()`是一种好的做法。但是，有些情况下 Python 的`eval()`可以帮你节省很多时间和精力。

在本节中，您将编写一个应用程序来动态计算数学表达式。如果你想在不使用`eval()`的情况下解决这个问题，那么你需要完成以下步骤:

1.  **解析**输入的表达式。
2.  **将**表达式的组成部分改为 Python 对象(数字、运算符、函数等)。
3.  **把**一切都组合成一个表达式。
4.  **确认**该表达式在 Python 中有效。
5.  **评估**最终表达式并返回结果。

考虑到 Python 可以处理和评估的各种可能的表达式，这将是一项繁重的工作。幸运的是，您可以使用`eval()`来解决这个问题，并且您已经学习了几种技术来降低相关的安全风险。

您可以通过单击下面的框来获取您将在本节中构建的应用程序的源代码:

**下载示例代码:** [单击此处获取代码，您将在本教程中使用](https://realpython.com/bonus/python-eval-project/)来学习 Python 的 eval()。

首先，启动你最喜欢的代码编辑器。创建一个名为`mathrepl.py`的新 [Python 脚本](https://realpython.com/run-python-scripts/#scripts-vs-modules)，然后添加以下代码:

```py
 1import math
 2
 3__version__ = "1.0"
 4
 5ALLOWED_NAMES = {
 6    k: v for k, v in math.__dict__.items() if not k.startswith("__")
 7}
 8
 9PS1 = "mr>>"
10
11WELCOME = f"""
12MathREPL {__version__}, your Python math expressions evaluator!
13Enter a valid math expression after the prompt "{PS1}".
14Type "help" for more information.
15Type "quit" or "exit" to exit.
16"""
17
18USAGE = f"""
19Usage:
20Build math expressions using numeric values and operators.
21Use any of the following functions and constants:
22
23{', '.join(ALLOWED_NAMES.keys())} 24"""
```

在这段代码中，首先导入 Python 的`math`模块。这个模块将允许你使用预定义的函数和常量来执行数学运算。常量`ALLOWED_NAMES`保存了一个字典，其中包含了`math`中的非特殊名称。这样，你就可以用`eval()`来使用它们了。

您还定义了另外三个字符串常量。您将使用它们作为脚本的用户界面，并根据需要将它们打印到屏幕上。

现在，您已经准备好编写应用程序的核心功能了。在这种情况下，您希望编写一个接收数学表达式作为输入并返回其结果的函数。为此，您编写了一个名为`evaluate()`的函数:

```py
26def evaluate(expression):
27    """Evaluate a math expression."""
28    # Compile the expression
29    code = compile(expression, "<string>", "eval")
30
31    # Validate allowed names
32    for name in code.co_names:
33        if name not in ALLOWED_NAMES:
34            raise NameError(f"The use of '{name}' is not allowed")
35
36    return eval(code, {"__builtins__": {}}, ALLOWED_NAMES)
```

该函数的工作原理如下:

1.  在**行`26`** 中，你定义`evaluate()`。这个函数将字符串`expression`作为一个参数，并返回一个[浮点数](https://realpython.com/lessons/floats/)，它将字符串的计算结果表示为一个数学表达式。

2.  在**行`29`** 中，你用`compile()`把输入的字符串`expression`变成编译好的 Python 代码。如果用户输入一个无效的表达式，编译操作将引发一个`SyntaxError`。

3.  在**行`32`** 中，你启动一个`for`循环来检查`expression`中包含的名字，并确认它们可以在最终的表达式中使用。如果用户提供了一个不在允许名称列表中的名称，那么您将引发一个`NameError`。

4.  在**行`36`** 中，执行数学表达式的实际求值。注意，按照良好实践的建议，您将自定义词典传递给了`globals`和`locals`。`ALLOWED_NAMES`保存在`math`中定义的函数和常数。

**注意:**因为这个应用程序使用了在`math`中定义的函数，你需要考虑到当你用一个无效的输入值调用这些函数时，其中的一些函数会引发一个`ValueError`。

例如，`math.sqrt(-10)`会产生一个错误，因为`-10`的[平方根](https://realpython.com/python-square-root-function/)未定义。稍后，您将看到如何在您的客户机代码中捕捉这个错误。

使用`globals`和`locals`参数的自定义值，以及检查**行`33`和**中的名称，可以最大限度地降低与使用`eval()`相关的安全风险。

当您在 [`main()`](https://realpython.com/python-main-function/) 中编写其客户端代码时，您的数学表达式计算器就完成了。在这个函数中，您将定义程序的主循环，并结束读取和评估用户在命令行中输入的表达式的循环。

对于此示例，应用程序将:

1.  **打印**给用户的欢迎信息
2.  **显示**准备读取用户输入的提示
3.  **提供**选项以获取使用说明并终止应用程序
4.  **读取**用户的数学表达式
5.  **评估**用户的数学表达式
6.  **将**评估结果打印到屏幕上

查看下面的`main()`实现:

```py
38def main():
39    """Main loop: Read and evaluate user's input."""
40    print(WELCOME)
41    while True:
42        # Read user's input
43        try:
44            expression = input(f"{PS1} ")
45        except (KeyboardInterrupt, EOFError):
46            raise SystemExit()
47
48        # Handle special commands
49        if expression.lower() == "help":
50            print(USAGE)
51            continue
52        if expression.lower() in {"quit", "exit"}:
53            raise SystemExit()
54
55        # Evaluate the expression and handle errors
56        try:
57            result = evaluate(expression)
58        except SyntaxError:
59            # If the user enters an invalid expression
60            print("Invalid input expression syntax")
61            continue
62        except (NameError, ValueError) as err:
63            # If the user tries to use a name that isn't allowed
64            # or an invalid value for a given math function
65            print(err)
66            continue
67
68        # Print the result if no error occurs
69        print(f"The result is: {result}")
70
71if __name__ == "__main__":
72    main()
```

在`main()`中，首先打印`WELCOME`消息。然后在一个 [`try`语句](https://docs.python.org/3/reference/compound_stmts.html#the-try-statement)中读取用户的输入，以捕捉`KeyboardInterrupt`和`EOFError`。如果出现这两种异常中的任何一种，就要终止应用程序。

如果用户输入`help`选项，则应用程序显示您的`USAGE`指南。同样，如果用户输入`quit`或`exit`，那么应用程序终止。

最后，您使用`evaluate()`来评估用户的数学表达式，然后您将结果打印到屏幕上。需要注意的是，调用`evaluate()`会引发以下异常:

*   **`SyntaxError`** :当用户输入不符合 Python 语法的表达式时，就会出现这种情况。
*   **`NameError`** :当用户试图使用不允许的名称(函数、类或属性)时，就会发生这种情况。
*   **`ValueError`** :当用户试图使用一个不允许作为`math`中给定函数输入的值时，就会发生这种情况。

注意，在`main()`中，您捕获了所有这些异常，并相应地向用户打印消息。这将允许用户检查表达式，修复问题，并再次运行程序。

就是这样！您已经使用 Python 的`eval()`用大约 70 行代码构建了一个数学表达式求值器。为了[运行应用程序](https://realpython.com/run-python-scripts/)，打开您系统的命令行并键入以下命令:

```py
$ python3 mathrepl.py
```

该命令将启动数学表达式计算器的[命令行界面](https://realpython.com/python-command-line-arguments/#the-command-line-interface) (CLI)。您会在屏幕上看到类似这样的内容:

```py
MathREPL 1.0, your Python math expressions evaluator!
Enter a valid math expression after the prompt "mr>>".
Type "help" for more information.
Type "quit" or "exit" to exit.

mr>>
```

一旦你到了那里，你就可以输入并计算任何数学表达式。例如，键入以下表达式:

```py
mr>> 25 * 2
The result is: 50
mr>> sqrt(25)
The result is: 5.0
mr>> pi
The result is: 3.141592653589793
```

如果您输入一个有效的数学表达式，那么应用程序将对其进行计算，并将结果打印到您的屏幕上。如果你的表达式有任何问题，应用程序会告诉你:

```py
mr>> 5 * (25 + 4
Invalid input expression syntax
mr>> sum([1, 2, 3, 4, 5])
The use of 'sum' is not allowed
mr>> sqrt(-15)
math domain error
mr>> factorial(-15)
factorial() not defined for negative values
```

在第一个例子中，您错过了右括号，所以您得到一条消息，告诉您语法不正确。然后你调用`sum()`，这是不允许的，你得到一个解释性的错误信息。最后，您用一个无效的输入值调用一个`math`函数，应用程序生成一条消息，指出您的输入中的问题。

这就是你要做的——你的数学表达式计算器已经准备好了！随意添加一些额外的功能。让您开始的一些想法包括扩大允许名称的字典，并添加更详细的警告消息。试一试，让我们在评论中了解它的进展。

[*Remove ads*](/account/join/)

## 结论

您可以使用 Python 的 **`eval()`** 从基于字符串或基于代码的输入中计算 Python 的**表达式**。当您尝试动态计算 Python 表达式，并且希望避免从头开始创建自己的表达式计算器的麻烦时，这个内置函数非常有用。

在本教程中，您已经学习了`eval()`如何工作，以及如何安全有效地使用它来计算任意 Python 表达式。

**您现在能够:**

*   使用 Python 的`eval()`来动态地**评估**基本的 Python 表达式
*   使用`eval()`运行更复杂的语句，如**函数调用**、**对象创建**，以及**属性访问**
*   最小化与使用 Python 的`eval()`相关的**安全风险**

此外，您已经编写了一个应用程序，它使用[命令行界面](https://realpython.com/command-line-interfaces-python-argparse/)使用`eval()`来交互式地评估数学表达式。您可以点击下面的链接下载该应用程序的代码:

**下载示例代码:** [单击此处获取代码，您将在本教程中使用](https://realpython.com/bonus/python-eval-project/)来学习 Python 的 eval()。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 Python eval()**](/courses/evaluate-expressions-dynamically-python-eval/) 动态计算表达式***********