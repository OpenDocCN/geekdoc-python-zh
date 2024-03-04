# 如何在 Python 中接受用户输入

> 原文：<https://www.pythonforbeginners.com/basics/how-to-take-user-input-in-python>

在编程时，我们经常需要在程序中接受用户输入。在这篇文章中，我们将看看在 python 中获取用户输入的不同方法。我们将讨论将不同的 [python 文字](https://www.pythonforbeginners.com/basics/python-literals)如字符串、整数和十进制值作为程序输入的方法。

## 如何在 Python 中获取输入

我们可以使用 input()方法获取 python 中的用户输入。

执行 input()方法时，它接受一个选项字符串参数，作为提示显示给用户。接受输入后，input()方法将用户输入的值作为字符串返回。即使用户输入一个数字或标点符号，input()方法也总是将输入作为字符串读取。您可以在下面的示例中观察到这一点。

```py
value = input("Please input a number:")
print("The data type of {} is {}".format(value, type(value)))
```

输出:

```py
Please input a number:1243
The data type of 1243 is <class 'str'>
```

在执行 input()方法时，程序的执行会暂停，直到用户输入后按下 enter 键。一旦用户按下 enter 键，input()方法就完成了它的执行。

## 如何在 Python 中接受整数输入

我们刚刚看到 input()方法将每个值都作为一个字符串读取。如果我们想验证用户输入了一个整数，我们可以使用正则表达式。

我们将使用 re.match()方法来验证给定的输入是否为整数。为此，我们将传递整数的正则表达式，即“[-+]？\d+$ "和输入字符串，以 re.match()方法作为输入。如果输入字符串包含除开头带有–或+符号的十进制数字以外的字符，re.match()方法将返回 **None** 。否则，match()方法返回一个 match 对象。要检查用户输入是否只包含整数，我们可以使用 re.match()方法，如下所示。

```py
import re

flag = True
input_value = None
while flag:
    input_value = input("Please input a number:")
    match_val = re.match("[-+]?\\d+$", input_value)
    if match_val is None:
        print("Please enter a valid integer.")
    else:
        flag = False
number = int(input_value)
print("The input number is:", number)
```

输出:

```py
Please input a number:Aditya
Please enter a valid integer.
Please input a number:PFB
Please enter a valid integer.
Please input a number:123.4
Please enter a valid integer.
Please input a number:+1234
The input number is: 1234
```

这里，我们使用了 while 循环反复提示用户输入整数。我们已经使用 re.match()方法检查了输入值。如果输入不正确，将再次执行 while 循环。否则，当**标志**变为**假**时，while 循环终止。类似地，我们可以使用下面讨论的 re.match()方法要求用户输入十进制数。

## 在 Python 中如何将十进制数作为输入

同样，我们将使用 re.match()方法来验证用户是否输入了有效的十进制数。为此，我们将传递十进制数的正则表达式，即“[-+]？\\d+([/。]\\d+)？$ "和 re.match()方法的输入字符串作为输入。如果输入字符串包含除十进制数字以外的字符，开头有–和+以及可选的小数点“.”在数字之间，re.match()方法将返回 **None** 。否则，match()方法返回一个 match 对象。要检查用户输入是否只包含一个十进制数，我们可以使用 re.match()方法，如下所示。

```py
import re

flag = True
input_value = None
while flag:
    input_value = input("Please input a number:")
    match_val = re.match("[-+]?\\d+([/.]\\d+)?$", input_value)
    if match_val is None:
        print("Please enter a valid decimal number.")
    else:
        flag = False
number = float(input_value)
print("The input number is:", number) 
```

输出:

```py
Please input a number:Aditya
Please enter a valid decimal number.
Please input a number:PFB
Please enter a valid decimal number.
Please input a number:-123.456
The input number is: -123.456
```

## 结论

在本文中，我们看到了使用 input()方法在 python 中获取用户输入的各种方法。我们还讨论了使用 re.match()方法验证输入。要学习更多关于在 python 中验证字符串的知识，您可以阅读这篇关于 python 中的[正则表达式的文章。](https://www.pythonforbeginners.com/regex/regular-expressions-in-python)