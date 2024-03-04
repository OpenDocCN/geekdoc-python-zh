# 如何在 Python 中获得一个列表作为用户输入

> 原文：<https://www.pythonforbeginners.com/basics/how-to-get-a-list-as-user-input-in-python>

我们可以使用 input()函数从用户那里获取一个值作为输入。如果我们必须得到一个值列表作为输入呢？在本文中，我们将讨论在 Python 中获取列表作为用户输入的两种方法。

## 使用 For 循环获取一个列表作为用户输入

我们可以使用 python 中的 for [循环获得一个值列表。为此，我们可以首先创建一个空列表和一个计数变量。之后，我们将询问用户列表中值的总数。获得值的总数后，我们可以使用 for 循环运行 input()函数 count 次，并使用 append()方法将输入添加到列表中，如下所示。](https://www.pythonforbeginners.com/basics/loops)

```py
input_count = int(input("Input the Total number of elements in the list:"))
input_list = list()
for i in range(input_count):
    input_value = input("Enter value {}:".format(i + 1))
    input_list.append(input_value)
print("The list given as input by the user is :", input_list)
```

输出:

```py
Input the Total number of elements in the list:5
Enter value 1:12
Enter value 2:345
Enter value 3:PFB
Enter value 4:12345
Enter value 5:Aditya
The list given as input by the user is : ['12', '345', 'PFB', '12345', 'Aditya']
```

## 使用 While 循环获取一个列表作为用户输入

当用户想要停止输入值时，我们可以要求用户输入一个特殊的值，而不是要求用户输入值的总数。之后，我们可以继续接受输入，直到用户给出特定的值作为输入信号，表明已经没有值了。我们可以使用 python 中的 [while 循环来做到这一点。](https://www.pythonforbeginners.com/loops/python-while-loop)

在这个方法中，我们将首先创建一个空列表和一个布尔变量**标志**。我们将初始化**标志**为真。这里，flag 将作为 while 循环中的决策变量。之后，我们将开始在 while 循环中接受用户的输入，并将它们添加到列表中。如果用户输入特定的值，表明没有剩余的值，我们将把**假**赋给**标志**变量。当**标志的值变为假时，它强制 while 循环终止。**

在 while 循环停止执行后，我们获得了作为输入给出的所有值的列表，如下例所示。

```py
flag = True
input_list = list()
while flag:
    input_value = input("Enter the value in the list. To finish, press enter key without any input:\n")
    if input_value == "":
        flag = False
        continue
    input_list.append(input_value)
print("The list given as input by the user is :", input_list)
```

输出:

```py
Enter the value in the list. To finish, press enter key without any input:
12
Enter the value in the list. To finish, press enter key without any input:
23
Enter the value in the list. To finish, press enter key without any input:
Aditya
Enter the value in the list. To finish, press enter key without any input:
567
Enter the value in the list. To finish, press enter key without any input:

The list given as input by the user is : ['12', '23', 'Aditya', '567']
```

## 只需使用 Input()方法一次，就可以获得一个列表作为用户输入

我们知道接受用户输入在时间和资源方面是很昂贵的，因为程序在接受用户输入时必须进行系统调用。因此，为了最大化程序的效率，我们可以避免多次使用 input()函数，同时将一个列表作为用户输入。

为此，我们将要求用户输入列表中的所有值，用空格字符分隔它们。将空格分隔的值作为输入后，我们将使用 [python 字符串分割](https://www.pythonforbeginners.com/dictionary/python-split)操作来获得所有输入值的列表。这可以在下面的例子中观察到。

```py
input_values = input("Enter the values in the list separated by space:\n")
input_list = input_values.split()
print("The list given as input by the user is :", input_list) 
```

输出:

```py
Enter the values in the list separated by space:
12 345 Aditya PFB 123345
The list given as input by the user is : ['12', '345', 'Aditya', 'PFB', '123345']
```

## 结论

在本文中，我们看到了在 python 中获取列表作为用户输入的不同方法。要了解如何在 Python 中从文件中获取输入，您可以阅读这篇关于 Python 中的[文件处理的文章。](https://www.pythonforbeginners.com/filehandling/file-handling-in-python)