# 在 Python 中将字符串转换为变量名

> 原文：<https://www.pythonforbeginners.com/basics/convert-string-to-variable-name-in-python>

在用 python 编程时，有几种情况下我们可能需要将字符串转换成变量名。例如，假设我们需要将一些用户数据作为输入，用户需要输入一些字段名称和它们相应的值。我们需要将字段名转换成一个变量，这样我们就可以给它们赋值。在本文中，我们将讨论在 python 中将输入字符串转换为变量名的不同方法。

## Python 中的字符串和变量

python 中的[变量是对内存中对象的引用。我们在 python 中使用变量来处理不同类型的值。当我们给一个 python 变量赋值时，解释器为该值创建一个 python 对象。之后，变量名指的是内存位置。您可以在 python 中定义一个变量，如以下 python 程序所示。](https://www.pythonforbeginners.com/basics/python-variables)

```py
myVar = 5
print("The value in myVar is:", myVar)
```

输出:

```py
The value in myVar is: 5
```

python 字符串是包含在单引号或双引号中的文字。我们也可以使用三重引号来定义字符串。我们可以定义一个字符串值，并将其赋给一个字符串变量，如下例所示。

```py
myStr = "PythonForBeginners"
print("The value in myStr is:", myStr)
```

输出:

```py
The value in myStr is: PythonForBeginners
```

在上面的例子中，我们创建了一个变量`myStr`。然后，我们将字符串“`PythonForBeginners`”赋给了`myStr`。

## 如何在 Python 中访问变量名？

我们可以使用`globals()`函数和`locals()`函数来访问 python 中的变量名。

执行时，`globals()`函数返回一个字典，其中包含所有字符串形式的变量名及其对应的值。它是一个全局符号表，包含程序全局范围内定义的所有名称。您可以在下面的 python 代码中观察到这一点。

```py
myStr = "PythonForBeginners"
myNum = 5
print("The variables in global scope and their values are:")
myVars = globals()
print(myVars)
```

输出:

```py
The variables in global scope and their values are:
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7fe3bfa934c0>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/home/aditya1117/PycharmProjects/pythonProject/string12.py', '__cached__': None, 'myStr': 'PythonForBeginners', 'myNum': 5, 'myVars': {...}} 
```

在上面的例子中，您可以观察到由`globals()`函数返回的字典包含一些默认值以及我们定义的变量。

`globals()`函数返回只包含全局变量作为键的字典。当我们在函数或其他内部作用域中定义一个变量时，我们不能使用`globals()`函数访问在该作用域中定义的变量。例如，看看下面的代码。

```py
def myFun():
    funvar1 = "Aditya"
    funVar2 = 1117
    print("I am in myFun. The variables in myFun are:")
    print(funvar1, funVar2)

myStr = "PythonForBeginners"
myNum = 5
myFun()
print("The variables in global scope and their values are:")
myVars = globals()
print(myVars)
```

输出:

```py
I am in myFun. The variables in myFun are:
Aditya 1117
The variables in global scope and their values are:
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7f9d18bb24c0>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/home/aditya1117/PycharmProjects/pythonProject/string12.py', '__cached__': None, 'myFun': <function myFun at 0x7f9d18b6f280>, 'myStr': 'PythonForBeginners', 'myNum': 5, 'myVars': {...}}

Process finished with exit code 0
```

在上面的例子中，我们已经在函数 myFun 中定义了 funvar1 和 funvar2。然而，这些变量不存在于全局符号表中。

即使我们执行函数`myFun`中的`globals()`函数，在`myFun`中定义的变量也不会包含在全局符号表中。您可以在下面的示例中观察到这一点。

```py
def myFun():
    funvar1 = "Aditya"
    funVar2 = 1117
    print("I am in myFun. The variables in myFun are:")
    print(funvar1, funVar2)
    print("The variables in global scope and their values are:")
    myVars = globals()
    print(myVars)

myStr = "PythonForBeginners"
myNum = 5
myFun()
```

输出:

```py
I am in myFun. The variables in myFun are:
Aditya 1117
The variables in global scope and their values are:
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7eff3f70d4c0>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/home/aditya1117/PycharmProjects/pythonProject/string12.py', '__cached__': None, 'myFun': <function myFun at 0x7eff3f6ca280>, 'myStr': 'PythonForBeginners', 'myNum': 5} 
```

要打印函数中定义的变量名，我们可以使用 `locals()` 函数。

当在函数或其他内部作用域中调用时,`locals()`函数返回一个字典，其中变量名及其相关值作为一个键-值对出现。

您可以使用 print 语句来打印字典，如下所示。

```py
def myFun():
    funvar1 = "Aditya"
    funVar2 = 1117
    print("I am in myFun. The variables in myFun are:")
    print(funvar1, funVar2)
    print("The variables in local scope of myFun and their values are:")
    myVars = locals()
    print(myVars)

myStr = "PythonForBeginners"
myNum = 5
myFun() 
```

输出:

```py
I am in myFun. The variables in myFun are:
Aditya 1117
The variables in local scope of myFun and their values are:
{'funvar1': 'Aditya', 'funVar2': 1117} 
```

在上面的例子中，您可以观察到由`locals()`函数返回的字典包含了在`myFun`内部定义的变量。

`locals()`函数在全局范围内执行时，打印包含全局变量及其值的字典。您可以在下面的示例中观察到这一点。

```py
def myFun():
    funvar1 = "Aditya"
    funVar2 = 1117
    print("I am in myFun. The variables in myFun are:")
    print(funvar1, funVar2)

myStr = "PythonForBeginners"
myNum = 5
myFun()
print("The variables in the global scope and their values are:")
myVars = locals()
print(myVars)
```

输出:

```py
I am in myFun. The variables in myFun are:
Aditya 1117
The variables in the global scope and their values are:
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7fd4fec2e4c0>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/home/aditya1117/PycharmProjects/pythonProject/string12.py', '__cached__': None, 'myFun': <function myFun at 0x7fd4febeb280>, 'myStr': 'PythonForBeginners', 'myNum': 5, 'myVars': {...}} 
```

既然我们已经讨论了如何在 python 中访问变量名，现在让我们讨论如何在 python 中创建动态变量和定义动态变量名。对于这一点，有各种方法，我们将逐一讨论。

## 使用 locals()方法将字符串转换为 Python 中的变量名

正如我们在上一节中看到的，python 解释器将变量名和它们的值以字典的形式存储在一个符号表中。如果在我们的程序中给我们一个字符串作为输入，我们可以通过将输入字符串作为一个键添加到符号表中，用该字符串定义一个变量名。我们可以添加单个字符、数值或字符串作为变量的关联值。

要将字符串转换为变量名，我们将遵循以下步骤。

*   首先，我们将使用`locals()`函数获得包含符号表的字典。`locals()`函数在执行时返回当前作用域的符号表。
*   一旦我们获得了符号表，我们将使用下标符号添加字符串名称作为键，变量的值作为关联值。
*   将键-值对添加到符号表后，将使用给定的字符串名称和关联的值创建变量。

您可以通过这个简单的例子观察到这一点。

```py
myStr = "domain"
print("The string is:",myStr)
myVars = locals()
myVars[myStr] = "pythonforbeginners.com"
print("The variables are:")
print(myVars)
print("{} is {}".format(myStr, domain)) 
```

输出:

```py
The string is: domain
The variables are:
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7f2fb9cb44c0>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/home/aditya1117/PycharmProjects/pythonProject/string12.py', '__cached__': None, 'myStr': 'domain', 'myVars': {...}, 'domain': 'pythonforbeginners.com'}
domain is pythonforbeginners.com
```

在上面的例子中，我们使用了下标符号对符号表进行了修改。您可以使用`__setitem__()` 方法，而不是使用下标符号将新的字符串值作为键添加到符号表中。

当在 [python 字典](https://www.pythonforbeginners.com/dictionary/python-dictionary-quick-guide)上调用 `__setitem__()`方法时，该方法将一个字符串文字作为第一个参数，并将与新变量名相关联的值作为第二个参数。执行后，字符串和值作为字典中的键-值对添加到字典中。

由于`locals()`方法返回的符号表也是一个字典，我们可以使用`__setitem__()` 方法将字符串转换成 python 中的变量名，如下所示。

```py
myStr = "domain"
print("The string is:", myStr)
myVars = locals()
myVars.__setitem__(myStr, "pythonforbeginners.com")
print("The variables are:")
print(myVars)
print("{} is {}".format(myStr, domain))
```

输出:

```py
The string is: domain
The variables are:
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7f77830734c0>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/home/aditya1117/PycharmProjects/pythonProject/string12.py', '__cached__': None, 'myStr': 'domain', 'myVars': {...}, 'domain': 'pythonforbeginners.com'}
domain is pythonforbeginners.com
```

上面使用`locals()`方法的方法只在当前作用域中进行修改，因此，当我们想把一个字符串转换成一个像函数一样的局部作用域中的变量名时，这是很有用的。如果你只想改变一个函数的符号表，可以使用`locals()` 函数将一个字符串转换成 python 中的变量名，如下所示。

```py
def myFun():
    myStr = "domain"
    print("The string is:", myStr)
    myVars = locals()
    myVars.__setitem__(myStr, "pythonforbeginners.com")
    print("The variables are:")
    print(myVars)

myFun()
```

输出:

```py
The string is: domain
The variables are:
{'myStr': 'domain', 'domain': 'pythonforbeginners.com'}
```

如果您想在全局符号表中进行更改，将一个 [python 字符串](https://www.pythonforbeginners.com/basics/python-string-methods-for-string-manipulation)转换成一个全局变量名，您可以在全局范围内执行`locals()`函数。之后，您可以使用下标符号或前面示例中所示的`__setitem__()`方法添加变量。

## 使用 globals()方法将字符串转换为 Python 中的变量名

如果你想在函数中把一个字符串转换成一个全局变量，你不能使用`locals()`函数。对于此任务，您可以使用`globals()`功能。

执行时，`globals()`函数返回全局符号表。您可以在任何范围内对全局符号表进行更改，以将字符串转换为全局变量名。为此，我们将执行以下步骤。

*   首先，我们将使用`globals()`函数获得全局符号表。`globals()`函数在执行时，将全局符号表作为字典返回。
*   一旦我们获得了符号表，我们将使用字典的下标符号添加字符串名称作为键，变量值作为关联值。
*   将键-值对添加到符号表后，将使用给定的字符串名称和关联的值创建变量。

执行以上步骤后，我们可以将一个字符串转换成一个全局变量。您可以在下面的示例中观察到这一点。

```py
myStr = "domain"
print("The string is:",myStr)
myVars = globals()
myVars[myStr] = "pythonforbeginners.com"
print("The variables are:")
print(myVars)
print("{} is {}".format(myStr, domain))
```

输出:

```py
The string is: domain
The variables are:
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7ff717bd34c0>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/home/aditya1117/PycharmProjects/pythonProject/string12.py', '__cached__': None, 'myStr': 'domain', 'myVars': {...}, 'domain': 'pythonforbeginners.com'}
domain is pythonforbeginners.com
```

不使用下标符号，您可以使用带有`globals()`函数的`__setitem__()`方法将字符串转换成 python 中的全局变量名，如下所示。

```py
myStr = "domain"
print("The string is:", myStr)
myVars = globals()
myVars.__setitem__(myStr, "pythonforbeginners.com")
print("The variables are:")
print(myVars)
print("{} is {}".format(myStr, domain))
```

输出:

```py
The string is: domain
The variables are:
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7fc4c62ba4c0>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/home/aditya1117/PycharmProjects/pythonProject/string12.py', '__cached__': None, 'myStr': 'domain', 'myVars': {...}, 'domain': 'pythonforbeginners.com'}
domain is pythonforbeginners.com
```

## 使用 vars()函数将字符串转换成 Python 中的变量名

在 python 中，除了使用`locals()` 和`globals()`函数将字符串转换为变量名，我们还可以使用 `vars()`函数。当在全局范围内执行时，`vars()`函数的行为就像`globals()`函数一样。当在函数或内部作用域中执行时，`vars()`函数的行为类似于`locals()` 函数。

要在全局范围内使用`vars()`函数将字符串转换成变量名，我们将使用以下步骤。

*   使用`vars()`函数，我们将获得包含全局范围内变量名的字典。
*   获得字典后，我们将使用字典的下标符号添加字符串名称作为键，变量的值作为关联值。
*   一旦我们将字符串和相关值添加到字典中，就在全局范围内创建了变量。

下面是在 python 中执行上述步骤将字符串转换为变量名的示例代码。

```py
myStr = "domain"
print("The string is:",myStr)
myVars = vars()
myVars[myStr] = "pythonforbeginners.com"
print("The variables are:")
print(myVars)
print("{} is {}".format(myStr, domain))
```

输出:

```py
The string is: domain
The variables are:
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7fb9c6d614c0>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/home/aditya1117/PycharmProjects/pythonProject/string12.py', '__cached__': None, 'myStr': 'domain', 'myVars': {...}, 'domain': 'pythonforbeginners.com'}
domain is pythonforbeginners.com
```

您也可以使用字典上的`__setitem__()`方法而不是下标符号来创建变量，如下例所示。

```py
myStr = "domain"
print("The string is:", myStr)
myVars = vars()
myVars.__setitem__(myStr, "pythonforbeginners.com")
print("The variables are:")
print(myVars)
print("{} is {}".format(myStr, domain))
```

输出:

```py
The string is: domain
The variables are:
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7fb5e21444c0>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/home/aditya1117/PycharmProjects/pythonProject/string12.py', '__cached__': None, 'myStr': 'domain', 'myVars': {...}, 'domain': 'pythonforbeginners.com'}
domain is pythonforbeginners.com
```

要使用`vars()`函数将字符串转换为局部范围内的变量名，就像函数一样，您可以执行与我们使用`locals()`函数创建全局变量相同的步骤。您可以在下面的示例中观察到这一点。

```py
def myFun():
    myStr = "domain"
    print("The string is:", myStr)
    myVars = vars()
    myVars.__setitem__(myStr, "pythonforbeginners.com")
    print("The variables are:")
    print(myVars)

myFun()
```

输出:

```py
The string is: domain
The variables are:
{'myStr': 'domain', 'domain': 'pythonforbeginners.com'}
```

在前面的章节中，我们已经直接更改了符号表，将字符串转换为变量名。但是，这不是执行任务的最佳方式。

现在让我们讨论如何在不直接改变符号表的情况下，在 python 中将字符串转换为变量名。

## 使用 exec()函数将字符串转换成 Python 中的变量名

我们可以使用 exec()函数来动态执行 python 语句。exec()函数将字符串形式的 python 语句作为输入参数。然后，执行 python 语句，就像它是用代码编写的普通 python 语句一样。例如，我们可以使用如下所示的`exec()`函数定义一个值为 5 的变量 x。

```py
myStr = "x=5"
exec(myStr)
print(x)
```

输出:

```py
5
```

要使用 exec()函数将字符串转换为变量名，我们将使用字符串格式。我们可以使用以下步骤来完成整个过程。

*   首先，我们将定义一个变量 myStr，它包含我们需要转换成变量名的原始字符串。
*   之后，我们将创建一个格式为"`{} =` " `{}"`"的字符串`myTemplate`。这里，我们将使用第一个占位符表示字符串名称，第二个占位符表示我们将从字符串变量创建的变量值。
*   在创建了带有占位符的字符串之后，我们将调用`myTemplate`上的`format()`方法，将`myStr`作为第一个输入参数，将从`myStr`创建的变量的值作为第二个输入参数。
*   一旦执行，`format()`方法将返回一个类似 python 语句的字符串，并将`myStr`作为变量名，给定的值被赋给它。
*   在获得包含 python 语句的字符串后，我们将把它传递给`exec()`函数。
*   一旦执行了`exec()`函数，就会创建一个以字符串`myStr`作为变量名的变量。

您可以在下面的示例中观察到这一点。

```py
myStr = "domain"
myVal = "pythonforbeginners.com"
myTemplate = "{} = \"{}\""
statement = myTemplate.format(myStr, myVal)
exec(statement)
print(domain) 
```

输出:

```py
pythonforbeginners.com 
```

## 使用 setattr()函数将字符串转换成 Python 中的变量名

除了使用 `exec()`函数，我们还可以使用`setattr()`函数将字符串转换成 python 中的变量名。

`setattr()`函数将 python 对象作为第一个输入参数，将属性(变量)名称作为第二个输入参数，将属性值作为第三个输入参数。执行后，它将属性添加到对象中。

要使用`setattr()`函数将字符串转换为变量名，我们首先需要获得 python 对象的当前作用域，这样我们就可以将变量作为属性添加到其中。为此，我们必须执行两项任务。

*   首先，我们需要获得程序中当前加载的模块的名称。
*   之后，我们需要找到当前正在执行的模块，即当前作用域。

为了找到当前加载到内存中的模块的名称，我们将使用`sys.modules`属性。属性包含一个字典，该字典将模块名映射到已经加载的模块。

获得字典后，我们需要找到当前模块。为此，我们将使用`__name__`属性。`__name__`是一个内置属性，计算结果为当前模块的名称。

`__name__`属性也出现在符号表中。您可以使用如下所示的`__name__`属性找到当前模块的名称。

```py
print("The current module name is:")
print(__name__)
```

输出:

```py
The current module name is:
__main__
```

在这里，您可以看到我们目前处于`__main__`模块。

在使用`__name__`属性获得当前模块的名称之后，我们将使用`sys.modules`属性上的下标符号获得当前模块对象。

获得当前模块后，我们将使用`setattr()`函数将字符串转换为变量名。为此，我们将当前模块作为第一个输入参数，字符串作为第二个输入参数，变量的值作为第三个输入参数传递给`setattr()`函数。执行完 `setattr()`函数后，将在当前范围内创建变量，输入字符串作为变量名。

您可以在下面的示例中观察到这一点。

```py
import sys

myStr = "domain"
myVal = "pythonforbeginners.com"
moduleName = __name__
currModule = sys.modules[moduleName]
setattr(currModule, myStr,myVal)
print(domain)
```

输出:

```py
pythonforbeginners.com
```

## 结论

在本文中，我们讨论了在 python 中将字符串转换为变量名的不同方法。在本文讨论的所有方法中，我建议您使用带有 exec()方法的方法。这是因为有几个保留的关键字不能用作变量名。关键字用于运行程序的不同任务。然而，如果我们直接改变符号表，我们可能会改变与关键字相关的值。在这种情况下，程序会出错。因此，尝试使用带有 `exec()` 函数的方法来[将字符串](https://www.pythonforbeginners.com/basics/convert-a-list-to-string-in-python)转换成 python 中的变量名。