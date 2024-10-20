# Python 101 -了解循环

> 原文：<https://www.blog.pythonlibrary.org/2020/05/27/python-101-learning-about-loops/>

很多时候，当你写代码的时候，你需要找到一种迭代的方法。也许您需要迭代字符串中的字母或`list`中的对象。迭代的过程是通过一个循环来完成的。

循环是一种编程结构，它允许你迭代块。这些块可能是字符串中的字母，也可能是文件中的行。

在 Python 中，有两种类型的循环结构:

*   `for`循环
*   `while`循环

除了对序列进行迭代，你还可以使用一个循环多次做同样的事情。一个例子是 web 服务器，它基本上是一个无限循环。服务器等待，监听客户端发送给它的消息。当收到消息时，循环将调用一个函数作为响应。

再比如游戏循环。当你赢了一局或输了一局，游戏通常不会退出。而是会问你要不要再玩一次。这也是通过将整个程序包装在一个循环中来实现的。

在本章中，您将学习如何:

*   创建一个`for`循环
*   在字符串上循环
*   循环查阅字典
*   从元组中提取多个值
*   通过循环使用`enumerate`
*   创建一个`while`循环
*   打破循环
*   使用`continue`
*   循环和`else`语句
*   嵌套循环

让我们从看`for`循环开始吧！

### 创建 for 循环

`for`循环是 Python 中最流行的循环结构。使用以下语法创建一个`for`循环:

```py
for x in iterable:
    # do something
```

现在上面的代码什么也不做。因此，让我们编写一个`for`循环来遍历一个列表，一次一项:

```py
>>> my_list = [1, 2, 3]
>>> for item in my_list:
...     print(item)
... 
1
2
3
```

在这段代码中，您创建了一个包含三个整数的`list`。接下来，创建一个`for`循环，表示“对于我列表中的每一项，打印出该项”。

当然，大多数时候你实际上想要对物品做些什么。例如，您可能想要加倍它:

```py
>>> my_list = [1, 2, 3]
>>> for item in my_list:
...     print(f'{item * 2}')
... 
2
4
6
```

或者您可能只想打印偶数编号的项目:

```py
>>> my_list = [1, 2, 3]
>>> for item in my_list:
...     if item % 2 == 0:
...         print(f'{item} is even')
... 
2 is even
```

这里使用模数运算符`%`来计算该项除以 2 的余数。如果余数是 0，那么你就知道一个项是偶数。

您可以使用循环和条件以及任何其他 Python 构造来创建复杂的代码片段，这些代码片段只受您的想象力的限制。

让我们来学习除了列表之外你还能循环什么。

### 在字符串上循环

Python 中的`for`循环与其他编程语言的一个不同之处在于，您可以迭代任何序列。因此可以迭代其他数据类型。

让我们来看看字符串的迭代:

```py
>>> my_str = 'abcdefg'
>>> for letter in my_str:
...     print(letter)
... 
a
b
c
d
e
f
g
```

这向您展示了迭代一个字符串是多么容易。

现在让我们尝试迭代另一种常见的数据类型！

### 在字典上循环

Python 字典也允许循环。默认情况下，当您在字典上循环时，您将在它的键上循环:

```py
>>> users = {'mdriscoll': 'password', 'guido': 'python', 'steve': 'guac'}
>>> for user in users:
...     print(user)
... 
mdriscoll
guido
steve
```

如果使用字典的`items()`方法，可以循环遍历字典的键和值:

```py
>>> users = {'mdriscoll': 'password', 'guido': 'python', 'steve': 'guac'}
>>> for user, password in users.items():
...     print(f"{user}'s password is {password}")
... 
mdriscoll's password is password
guido's password is python
steve's password is guac
```

在这个例子中，您指定您想要在每次迭代中提取`user`和`password`。您可能还记得，`items()`方法返回一个格式类似元组列表的视图。因此，您可以从这个视图中提取每个`key: value`对并打印出来。

这导致我们在元组上循环，并在循环时从元组中取出单个项目！

### 循环时提取元组中的多个值

有时，您需要循环遍历元组列表，并获取元组中的每一项。这听起来有点奇怪，但是你会发现这是一个相当常见的编程任务。

```py
>>> list_of_tuples = [(1, 'banana'), (2, 'apple'), (3, 'pear')]
>>> for number, fruit in list_of_tuples:
...     print(f'{number} - {fruit}')
... 
1 - banana
2 - apple
3 - pear
```

为了实现这一点，您要利用这样一个事实，即您知道每个元组中有两个条目。因为您事先知道元组列表的格式，所以您知道如何提取值。

如果您没有从元组中单独提取项目，您可能会得到这样的输出:

```py
>>> list_of_tuples = [(1, 'banana'), (2, 'apple'), (3, 'pear')]
>>> for item in list_of_tuples:
...     print(item)
... 
(1, 'banana')
(2, 'apple')
(3, 'pear')
```

这可能不是你所期望的。您通常希望从`tuple`中提取一个项目或者多个项目，而不是提取整个`tuple`。

现在让我们发现另一种有用的循环方式！

### 通过循环使用`enumerate`

Python 自带一个名为`enumerate`的内置函数。这个函数接受一个迭代器或序列，比如一个字符串或列表，并以`(position, item)`的形式返回一个元组。

这可让您在序列中循环时轻松了解项目在序列中的位置。

这里有一个例子:

```py
>>> my_str = 'abcdefg'
>>> for pos, letter in enumerate(my_str):
...     print(f'{pos} - {letter}')
... 
0 - a
1 - b
2 - c
3 - d
4 - e
5 - f
6 - g
```

现在让我们看看 Python 支持的另一种类型的循环！

### 创建一个`while`循环

Python 还有另一种类型的循环结构，叫做`while`循环。用关键字`while`后跟一个表达式创建一个`while`循环。换句话说，`while`循环将一直运行，直到满足特定条件。

让我们来看看这些循环是如何工作的:

```py
>>> count = 0
>>> while count < 10:
...     print(count)
...     count += 1
```

这个循环的表述方式与条件语句非常相似。您告诉 Python，只要`count`小于 10，您就希望循环运行。在循环内部，打印出当前的`count`，然后将`count`加 1。

如果您忘记增加`count`，循环将一直运行，直到您停止或终止 Python 进程。

您可以通过犯这种错误来创建一个无限循环，或者您可以这样做:

由于表达式总是`True`，这段代码将打印出字符串“程序正在运行”，直到您终止该进程。

### 打破循环

有时你想提前停止一个循环。例如，您可能想要循环，直到找到特定的内容。一个很好的用例是遍历文本文件中的行，当找到第一个出现的特定字符串时停止。

要提前停止循环，可以使用关键字`break`:

```py
>>> count = 0
>>> while count < 10:
...     if count == 4:
...         print(f'{count=}')
...         break
...     print(count)
...     count += 1
... 
0
1
2
3
count=4
```

在本例中，您希望在计数达到 4 时停止循环。为了实现这一点，您添加了一个条件语句来检查`count`是否等于 4。当它出现时，您打印出计数等于 4，然后使用`break`语句退出循环。

您也可以在`for`循环中使用`break`:

```py
>>> list_of_tuples = [(1, 'banana'), (2, 'apple'), (3, 'pear')]
>>> for number, fruit in list_of_tuples:
...     if fruit == 'apple':
...         print('Apple found!')
...         break
...     print(f'{number} - {fruit}')
... 
1 - banana
Apple found!
```

对于这个例子，当你找到一个苹果的时候，你想跳出这个循环。否则你打印出你发现了什么水果。因为苹果在第二个元组中，所以你永远不会得到第三个元组。

当使用`break`时，循环将只从`break`语句所在的最内层循环中跳出。

你可以使用`break`来帮助控制程序的流程。事实上，条件语句和`break`一起被称为`flow control`语句。

另一个可以用来控制代码流的语句是 **continue** 。接下来让我们来看看！

### 使用`continue`

`continue`语句用于继续循环中的下一次迭代。你可以用`continue`跳过一些东西。

让我们编写一个跳过偶数的循环:

```py
>>> for number in range(2, 12):
...     if number % 2 == 0:
...         continue
...     print(number)
... 
3
5
7
9
11
```

在这段代码中，您对从 2 开始到 11 结束的一系列数字进行循环。对于此范围内的每个数字，使用模数运算符`%`，得到该数字除以 2 的余数。如果余数是零，它就是一个偶数，您可以使用`continue`语句继续到序列中的下一个值。这实际上跳过了偶数，所以只打印出奇数。

通过使用`continue`语句，您可以使用巧妙的条件语句跳过序列中任意数量的内容。

### 循环和`else`语句

关于 Python 循环，一个鲜为人知的事实是，您可以像处理`if/else`语句一样向它们添加一个`else`语句。`else`语句只有在没有`break`语句出现时才会被执行。

从另一个角度来看，`else`语句只有在循环成功完成时才会执行。

循环中的`else`语句的主要用例是在序列中搜索一个项目。如果没有找到条目，您可以使用`else`语句来引发一个异常。

让我们看一个简单的例子:

```py
>>> my_list = [1, 2, 3]
>>> for number in my_list:
...     if number == 4:
...         print('Found number 4!')
...         break
...     print(number)
... else:
...     print('Number 4 not found')
... 
1
2
3
Number 4 not found
```

这个例子在三个整数的`list`上循环。它会寻找数字 4，如果找到了，就会跳出循环。如果没有找到那个号码，那么`else`语句将会执行并通知您。

尝试将数字 4 添加到`list`中，然后重新运行代码:

```py
>>> my_list = [1, 2, 3, 4]
>>> for number in my_list:
...     if number == 4:
...         print('Found number 4')
...         break
...     print(number)
... else:
...     print('Number 4 not found')
... 
1
2
3
Found number 4
```

更合适的方法是引发一个异常，而不是打印一条消息。

### 嵌套循环

循环也可以相互嵌套。嵌套循环有很多原因。最常见的原因之一是解开嵌套的数据结构。

让我们用一个嵌套的`list`作为例子:

```py
>>> nested = [['mike', 12], ['jan', 15], ['alice', 8]]
>>> for lst in nested:
...     print(f'List = {lst}')
...     for item in lst:
...         print(f'Item -> {item}')
```

外部循环将提取每个嵌套的`list`并打印出来。然后在内部循环中，您的代码将提取嵌套列表中的每一项并打印出来。

如果您运行这段代码，您应该会看到如下所示的输出:

```py
List = ['mike', 12]
Item -> mike
Item -> 12
List = ['jan', 15]
Item -> jan
Item -> 15
List = ['alice', 8]
Item -> alice
Item -> 8
```

当嵌套列表的长度不同时，这种类型的代码特别有用。例如，您可能需要对包含额外数据或数据不足的列表进行额外的处理。

### 包扎

循环对于迭代数据非常有帮助。在本文中，您了解了 Python 的两个循环结构:

*   `for`循环
*   `while`循环

您还学习了使用`break`和`continue`语句进行流控制。最后，你学会了如何在循环中使用`else`,以及为什么要嵌套循环。

只要稍加练习，你很快就会非常熟练地在自己的代码中使用循环！