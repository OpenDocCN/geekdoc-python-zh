# 功能

> 原文：<https://www.pythonforbeginners.com/basics/python-functions-cheat-sheet>

## Python 中的函数是什么？

函数是你可以调用的东西(可能有一些参数，你放在括号里的东西)，它执行一个动作并返回值。

## 我为什么要使用函数？

*   将代码任务简化为简单的任务
*   可以更容易地在开发人员之间分割代码
*   消除重复代码
*   重用代码
*   获得良好的代码结构
*   更容易调试。

## 函数的规则是什么？

*   Python 中的函数必须在使用前定义。
*   使用关键字“def ”,后跟函数名和括号()来创建函数。
*   函数必须被命名，并指定它有什么参数(如果有的话)。
*   一个函数可以使用多个实参，每个实参都响应函数中的一个参数。
*   一个函数可以使用多个实参，每个实参都响应函数中的一个参数。
*   关键字“def”是必需的，并且必须是小写字母。
*   名字可以是你喜欢的任何东西。
*   行尾必须以冒号(:)结尾
*   该函数通常以使用 return 返回值结束。
*   函数内部的代码必须缩进
*   调用时使用该函数。

## 参数(自变量)

参数(也称为自变量)是函数的输入。Python 语言中的所有参数(自变量)都是通过引用传递的。有一些不同类型的参数，其中两个是:

### 位置

位置参数没有关键字，首先被赋值。

### 关键字

关键字参数有关键字，在位置参数之后第二个赋值。当你调用一个函数时，你决定使用位置、关键字或者两者的混合。如果你愿意，你可以选择做所有的关键字。

## 打电话

函数、过程或函数的调用必须有括号。在括号之间，可以有一个或多个参数值，但也可以为空。

首先发生的是函数参数获得它们的值，然后继续函数中的其余代码。当一个函数值完成时，它将它返回给调用。

**单参数函数调用:**

正常=摄氏度至华氏度(摄氏度温度)

**不带参数的函数调用:**

x =输入()

**带两个参数的过程调用:**

矩形(20，10)

**不带参数的过程调用:**

说你好()

请记住，当 Python 进行调用时，必须已经定义了函数。

## 返回

参数是函数的输入，返回值是输出。

return 关键字用于从函数中返回值。该函数将根据 return 命令退出。(之后的所有代码都将被忽略)

函数可能会也可能不会返回值。如果函数没有 return 关键字，它将发送一个 None 值。

## 在 Python 中创建函数

在 Python 中创建一个函数的第一件事是定义它并给它一个名字(可能在括号中有一些参数)

定义它并给它一个名称> > def name()

为函数>>命令创建方向

调用函数> > name()

通过在定义中创建变量，可以向函数发送值。(这些变量只在这个特定的函数内部起作用)

让我们看一个例子:

第一行定义了函数号()

该函数有两个参数 num1 和 num2

第二行将 num1 和 num2 相加

```py
def numbers(num1, num2): 

    print num1+num2 
```

如果这个定义在程序的开始，我们要做的就是写 def 数字(1，2)来把值发送给函数。

我们通过在函数调用中赋值来实现。你也可以定义数学函数。这需要一个数的平方根:def square(x): return x*x

让我们看一个例子，如何创建一个简单的函数的任何参数。

```py
def name():
    # Get the user's name.
    name = raw_input('Enter your name: ') 

    # Return the name.
    return name         

name() 
```

在第二个示例中，显示了如何将参数传递给函数:

```py
def even(number):        
    if number % 2 == 0:
        return True

    else:
        return False

print even(10) 
```

## 例子

如果你还没有读过 Python 的[非程序员教程，读一读吧。这是](https://en.wikibooks.org/wiki/Non-Programmer's_Tutorial_for_Python_2.6/Defining_Functions "non-programmers")[学习 Python](https://www.pythonforbeginners.com/learn-python) 的绝佳资源。

这个转换温度的例子是一个如何使用函数的好例子。

```py
def print_options():
    print "Options:"
    print " 'p' print options"
    print " 'c' convert from celsius"
    print " 'f' convert from fahrenheit"
    print " 'q' quit the program"

def celsius_to_fahrenheit(c_temp):
    return 9.0 / 5.0 * c_temp + 32

def fahrenheit_to_celsius(f_temp):
    return (f_temp - 32.0) * 5.0 / 9.0

choice = "p"

while choice != "q":

    if choice == "c":
        temp = input("Celsius temperature: ")
        print "Fahrenheit:", celsius_to_fahrenheit(temp)

    elif choice == "f":
        temp = input("Fahrenheit temperature: ")
        print "Celsius:", fahrenheit_to_celsius(temp)

    elif choice != "q":
        print_options()

    choice = raw_input("option: ") 
```

我希望你喜欢这个小抄，希望你今天学到了一些东西。