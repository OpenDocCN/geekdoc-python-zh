# Python 中的变量和常量

> 原文：<https://www.pythonforbeginners.com/basics/variables-and-constants-in-python>

在学习 python 编程时，您一定遇到过某些短语，如关键字、变量、常量和文字。在本文中，我们将学习变量和常量，并研究它们在 python 中的定义和用法的基本概念。

## Python 中有哪些变量？

python 中的变量是一个名称，用于引用内存中的对象。变量也称为引用，因为它们引用内存中的特定对象。

例如，以下代码片段中给出的变量`myNum`将引用一个 integer 类型的对象，该对象包含 1117 作为其值。

```py
myNum = 1117
```

我们可以给变量赋值。同样，我们也可以将变量赋值给变量，如下所示。

```py
myNum=1117
anotherNum=myNum
```

当我们给一个变量赋值时，两个变量都开始指向内存中的同一个对象。这可以使用 id()方法进行验证，该方法为每个对象提供一个惟一的标识符。

```py
myNum = 1117
anotherNum = myNum
print("id of myNum is:", id(myNum))
print("id of anotherNum is:", id(anotherNum))
```

输出:

```py
id of myNum is: 140209154920336
id of anotherNum is: 140209154920336
```

在上面的例子中，我们可以看到两个变量的 id 是相同的，这证实了两个变量引用了同一个对象。

我们也可以修改变量中的值。在修改可变数据类型的值时，新值被分配给同一个对象。例如，如果我们修改下面例子中给出的 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中的一个键的值，字典将保持不变。这可以从修改前后的字典 id 来验证。

```py
myDict = {1: 2}
print("Id of myDict before modification:", id(myDict))
myDict[1] = 4
print("Id of myDict after modification:", id(myDict))
```

输出:

```py
Id of myDict before modification: 140457374471040
Id of myDict after modification: 140457374471040
```

当我们修改赋给不可变数据类型(如整数)的值时，会创建一个新对象并将其赋给变量。这可以从下面的例子中看出。

```py
myNum=1117
print("Id of myNum before modification:", id(myNum))
myNum = 1118
print("Id of myNum after modification:", id(myNum))
```

输出:

```py
Id of myNum before modification: 140485377224528
Id of myNum after modification: 140485378289104
```

在上面的例子中，我们可以看到 myNum 变量的 id 发生了变化。这确认了在给 myNum 赋值的过程中创建了一个新对象。

## Python 中定义变量的约定

python 中的变量名总是以字母开头。我们可以用大写或小写字母开始一个变量，但是它不应该以数字或任何特殊字符开始。

```py
name="PythonForBeginners" #correct variable name
Name="PythonForBeginners" #correct variable name
2name="PythonForBeginners" #incorrect variable name
#name="PythonForBeginners" #incorrect variable name
```

变量可以有下划线字符，但不能有任何其他特殊字符，如#、@、$、%、&、！。

```py
my_name="PythonForBeginners" #correct variable name
my&name="PythonForBeginners" #incorrect variable name
```

当变量名由多个单词组成时，我们可以使用 camelCase 命名约定。在 camelCase 约定中，我们以一个小写字母开始变量，变量中的每个单词都以大写字母开始，如下所示。

```py
myName="PythonForBeginners" #correct variable name
```

我们还可以使用下划线来分隔变量名中的不同单词，如下所示。

```py
my_name="PythonForBeginners" #correct variable name
```

Python 是一种区分大小写的语言。这意味着拼写相同但大小写不同的变量名将引用不同的对象。

```py
myNum=1117
MyNum = 1118
print("myNum is:",myNum)
print("MyNum is:",MyNum)
```

输出:

```py
myNum is: 1117
MyNum is: 1118
```

在 python 中，关键字不应用作变量名，否则在程序执行过程中会出现错误。

```py
in =1117 #incorrect variable name
```

## Python 中的常量有哪些？

常量是文字，它包含一个在程序执行过程中不应该改变的值。

在 python 中，解释器不会区分变量和常量。在程序中，我们通过使用命名约定来区分变量和常量。

python 中的常量只使用大写字母和下划线定义。通常，常量是在一个模块中定义的，当我们需要使用它们的时候，它们就会被导入到程序中。例如，我们可以使用 cmath 模块中的常量 PI，在导入后如下所示。

```py
import cmath
print("Value of PI is:",cmath.pi)
```

输出:

```py
Value of PI is: 3.141592653589793 
```

我们也可以在导入常量后修改它，如下所示。

```py
import cmath

print("Value of PI is:", cmath.pi)
cmath.pi = 1117
print("modified value of PI is:", cmath.pi) 
```

输出:

```py
Value of PI is: 3.141592653589793
modified value of PI is: 1117
```

能够修改常量中的值完全违背了常量背后的思想。为了避免这种情况，应该在模块中定义 getter 函数来访问常量。这将限制用户修改这些值。

## 结论

在本文中，我们研究了 python 中的变量和常量。我们还看到了 python 中命名变量和常量的约定。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，并使用异常处理来使程序更加健壮，并以系统的方式处理错误。请继续关注更多内容丰富的文章。