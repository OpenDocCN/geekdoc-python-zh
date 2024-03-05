# 如何在 Python 中模拟 Do-While 循环？

> 原文：<https://realpython.com/python-do-while/>

如果你从像 [C](https://realpython.com/c-for-python-programmers/) 、 [C++](https://realpython.com/python-vs-cpp/) 、 [Java](https://realpython.com/java-vs-python/) 或 [JavaScript](https://realpython.com/python-vs-javascript/) 这样的语言来到 Python，那么你可能会错过它们的 [do-while](https://en.wikipedia.org/wiki/Do_while_loop) 循环结构。do-while 循环是一个常见的[控制流](https://en.wikipedia.org/wiki/Control_flow)语句，它至少执行其代码块一次，不管**循环条件**是真还是假。这种行为依赖于在每次迭代结束时评估循环条件这一事实。所以，第一次迭代总是运行。

这种类型的循环最常见的用例之一是接受和处理用户的输入。考虑以下用 C 编写的示例:

```py
#include  <stdio.h> int  main()  { int  number; do  {   printf("Enter a positive number: "); scanf("%d",  &number); printf("%d\n",  number); }  while  (number  >  0);   return  0; }
```

这个小程序运行一个`do` … `while`循环，要求用户输入一个正数。然后输入被存储在`number`中并打印到屏幕上。循环一直运行这些操作，直到用户输入一个非正数。

如果您编译并运行这个程序，那么您将得到以下行为:

```py
Enter a positive number: 1
1
Enter a positive number: 4
4
Enter a positive number: -1
-1
```

循环条件`number > 0`在循环结束时被评估，这保证了循环的**主体**将至少运行一次。这个特性将 do-while 循环与常规的 [while](https://en.wikipedia.org/wiki/While_loop) 循环区分开来，后者在开始时评估循环条件。在 while 循环中，不能保证运行循环体。如果循环条件一开始就是假的，那么肉体根本不会运行。

**注意:**在本教程中，你将把控制 while 或 do-while 循环的条件称为**循环条件**。这个概念不应该与**循环的主体**混淆，后者是在 C 等语言中夹在花括号之间的代码块，或者在 Python 中缩进。

使用 do-while 循环结构的一个原因是效率。例如，如果循环条件意味着高成本操作，并且循环必须运行 *n* 次( *n* ≥ 1)，那么该条件将在 do-while 循环中运行 *n* 次。相反，常规的 while 循环将运行代价高昂的条件 *n* + 1 次。

Python 没有 do-while 循环结构。为什么？显然，核心开发人员从来没有为这种类型的循环找到一个好的语法。很可能，这就是[吉多·范·罗苏姆](https://twitter.com/gvanrossum) [拒绝](https://mail.python.org/pipermail/python-ideas/2013-June/021610.html) PEP [315](https://peps.python.org/pep-0315/) 的原因，这是一种在语言中添加 do-while 循环的尝试。一些核心开发人员更喜欢 do-while 循环，并期待[围绕这个话题](https://twitter.com/raymondh/status/1528772337306419200)重新展开讨论。

同时，您将探索 Python 中可用的替代方法。简而言之，**如何在 Python 中模拟 do-while 循环？**在本教程中，你将学习如何使用`while`创建类似 do-while 循环的循环。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 简而言之:使用一个`while`循环和`break`语句

在 Python 中模拟 do-while 循环最常见的技术是使用无限的 [`while`循环](https://realpython.com/python-while-loop/)，其中的 [`break`](https://realpython.com/python-keywords/#iteration-keywords-for-while-break-continue-else) 语句包装在 [`if`语句](https://realpython.com/python-conditional-statements/)中，该语句检查给定的条件，如果该条件为真，则中断迭代:

```py
while True:
    # Do some processing...
    # Update the condition...
    if condition:
        break
```

这个循环使用`True`作为它的形式条件。这个技巧把循环变成了无限循环。在条件语句之前，循环运行所有需要的处理并更新中断条件。如果这个条件评估为真，那么`break`语句将跳出循环，程序执行将继续其正常路径。

**注意:**使用无限循环和`break`语句可以模拟 do-while 循环。这种技术是 Python 社区一般[推荐](https://twitter.com/raymondh/status/1528772339818717185)使用的，但并不完全安全。

例如，如果在`break`语句之前引入一个`continue`语句，那么循环可能会错过中断条件，并进入一个不受控制的无限循环。

下面是如何编写与您在本教程介绍中编写的 C 程序等效的 Python:

>>>

```py
>>> while True:
...     number = int(input("Enter a positive number: "))
...     print(number)
...     if not number > 0:
...         break
...
Enter a positive number: 1
1
Enter a positive number: 4
4
Enter a positive number: -1
-1
```

这个循环使用内置的 [`input()`](https://docs.python.org/3/library/functions.html#input) 函数接受用户的输入。然后使用 [`int()`](https://docs.python.org/3/library/functions.html#int) 将输入转换成整数。如果用户输入一个小于或等于`0`的数字，那么`break`语句运行，循环终止。

有时，您会遇到需要保证循环至少运行一次的情况。在那些情况下，你可以像上面一样使用`while`和`break`。在下一节中，您将编写一个猜数字游戏，该游戏使用这样一个 do-while 循环来接受和处理用户在命令行中的输入。

[*Remove ads*](/account/join/)

## Do-While 循环在实践中是如何工作的？

do-while 循环最常见的用例是*接受并处理用户的输入*。作为一个实际的例子，假设您有一个用 [JavaScript](https://realpython.com/python-vs-javascript/) 实现的猜数字游戏。代码使用一个`do` … `while`循环来处理用户的输入:

```py
 1// guess.js 2
 3const  LOW  =  1; 4const  HIGH  =  10; 5
 6let  secretNumber  =  Math.floor(Math.random()  *  HIGH)  +  LOW; 7let  clue  =  ''; 8let  number  =  null; 9
10do  {  11  let  guess  =  prompt(`Guess a number between ${LOW} and ${HIGH}  ${clue}`); 12  number  =  parseInt(guess); 13  if  (number  >  secretNumber)  { 14  clue  =  `(less than ${number})`; 15  }  else  if  (number  <  secretNumber)  { 16  clue  =  `(greater than ${number})`; 17  } 18}  while  (number  !=  secretNumber);  19
20alert(`You guessed it! The secret number is ${number}`);
```

这个脚本做了几件事。下面是正在发生的事情的分类:

*   **第 3 行和第 4 行**定义了两个常数来界定秘密数字将存在的间隔。

*   **第 6 行到第 8 行**定义了[变量](https://realpython.com/python-variables/)来存储秘密数字、线索消息和`number`的初始值，它将保存用户的输入。

*   **第 10 行**开始一个`do` … `while`循环来处理用户的输入，并确定用户是否已经猜出了密码。

*   **第 11 行**定义了一个本地变量`guess`，用来存储命令行提供的用户输入。

*   **第 12 行**使用`parseInt()`将输入值转换成整数。

*   **第 13 行**定义了一个条件语句，检查输入数字是否大于秘密数字。如果是这种情况，那么`clue`被设置为适当的消息。

*   **第 15 行**检查输入的数字是否小于密码，然后相应地设置`clue`。

*   **第 18 行**定义循环条件，检查输入的数字是否与密码不同。在这个具体的例子中，循环将继续运行，直到用户猜出密码。

*   **第 20 行**最后启动一个警告框，通知用户猜测成功。

现在说你想把上面的例子翻译成 Python 代码。Python 中一个等价的猜数字游戏看起来像这样:

```py
# guess.py

from random import randint

LOW, HIGH = 1, 10

secret_number = randint(LOW, HIGH)
clue = ""

while True:
    guess = input(f"Guess a number between {LOW} and {HIGH}  {clue} ")
    number = int(guess)
    if number > secret_number:
        clue = f"(less than {number})"
    elif number < secret_number:
        clue = f"(greater than {number})"
    else:
 break 
print(f"You guessed it! The secret number is {number}")
```

这段 Python 代码的工作方式就像它的等效 JavaScript 代码一样。主要区别在于，在这种情况下，您使用的是常规的`while`循环，因为 Python 没有`do` … `while`循环。在这个 Python 实现中，当用户猜出秘密数字时，就会运行`else`子句，从而打破循环。代码的最后一行[打印](https://realpython.com/python-print/)成功的猜测消息。

使用一个无限循环和一个`break`语句，就像你在上面的例子中所做的那样，是在 Python 中模拟 do-while 循环最广泛使用的方法。

## Do-While 和 While 循环有什么区别？

简而言之，do-while 循环和 while 循环的主要区别在于，前者至少执行一次循环体，因为循环条件是在最后检查的。另一方面，如果条件评估为 true，则执行常规 while 循环的主体，这在循环开始时进行测试。

下表总结了这两种循环的主要区别:

| 在…期间 | 做一会儿 |
| --- | --- |
| 是一个入口控制循环 | 是一个出口控制循环 |
| 仅在循环条件为真时运行 | 运行，直到循环条件变为假 |
| 首先检查条件，然后执行循环体 | 执行循环体，然后检查条件 |
| 如果循环条件最初为假，则执行循环体零次 | 不管循环条件的真值是多少，至少执行一次循环体 |
| 对于 *n* 次迭代，检查循环条件 *n* + 1 次 | 检查循环条件 *n* 次，其中 *n* 为迭代次数 |

while 循环是一个控制流结构，它提供了一个通用的通用循环。它允许您在给定条件保持为真的情况下重复运行一组语句。do-while 循环的用例更加具体。它主要用于只有在循环体至少已经运行过一次的情况下，检查循环条件才有意义。

## 在 Python 中，可以使用什么替代方法来模拟 Do-While 循环？

至此，您已经了解了在 Python 中模拟 do-while 循环的推荐或最常用的方法。然而，Python 在模拟这种类型的循环时非常灵活。一些程序员总是使用无限的`while`循环和`break`语句。其他程序员使用他们自己的公式。

在本节中，您将了解一些模拟 do-while 循环的替代技术。第一种方法是在循环开始之前运行第一个操作*。第二种选择意味着使用一个循环条件，在循环开始之前，该条件被初始设置为真值。*

### 循环前的第一个操作

正如您已经了解到的，do-while 循环最相关的特性是循环体总是至少运行一次。要使用一个`while`循环来模拟这个功能，您可以在循环开始之前获取循环体并运行它。然后你可以在循环中重复这个物体。

这个解决方案听起来很重复，如果你不使用某种技巧的话。幸运的是，您可以使用一个函数来打包循环体并防止重复。使用这种技术，您的代码将如下所示:

```py
condition = do_something()

while condition:
    condition = do_something()
```

对`do_something()`的第一次调用保证了所需的功能至少运行一次。只有当`condition`为真时，循环内部对`do_something()`的调用才会运行。注意，您需要在每次迭代中更新循环条件，以使该模式正确工作。

下面的代码展示了如何使用这种技术实现猜数字游戏:

```py
# guess.py

from random import randint

LOW, HIGH = 1, 10

secret_number = randint(LOW, HIGH)
clue = ""

def process_move(clue):
    user_input = input(f"Guess a number between {LOW} and {HIGH}  {clue} ")
    number = int(user_input)
    if number > secret_number:
        clue = f"(less than {number})"
    elif number < secret_number:
        clue = f"(greater than {number})"
    return number, clue

number, clue = process_move(clue)  # First iteration 
while number != secret_number:
 number, clue = process_move(clue) 
print(f"You guessed it! The secret number is {number}")
```

在这个新版本的猜数字游戏中，你将所有循环的功能都打包到`process_move()`中。这个函数返回当前数字，您将在以后检查循环条件时使用它。它还返回线索消息。

注意`process_move()`在循环开始前运行一次，模仿 do-while 循环的主要特征，至少运行一次它的主体。

在循环内部，您调用函数来运行游戏的主要功能，并相应地更新循环条件。

[*Remove ads*](/account/join/)

### 使用初始真循环条件

使用初始设置为`True`的循环条件是模拟 do-while 循环的另一种选择。在这种情况下，您只需要在循环开始运行之前将循环条件设置为`True`。这种做法可以确保循环体至少运行一次:

```py
do = True

while do:
    do_something()
    if condition:
        do = False
```

这个替代构造与您在上一节中使用的非常相似。主要区别在于循环条件是一个在循环内部更新的[布尔](https://realpython.com/python-boolean/)变量。

这种技术也类似于使用无限`while`循环和`break`语句的技术。然而，这种方法更显而易见，可读性更好，因为它允许您使用描述性的变量名，而不是简单的`break`语句和像`True`这样的硬编码条件。

**注意:**有时给布尔变量命名更自然，这样你就可以把它设置为`True`来打破循环。在这些情况下，您可以用类似于`while not done:`的东西开始循环，并在循环内将`done`设置为`True`。

您可以使用这种技术来重写您的数字猜测游戏，如下面的代码示例所示:

```py
# guess.py

from random import randint

LOW, HIGH = 1, 10

secret_number = randint(LOW, HIGH)
clue = ""
number_guessed = False 
while not number_guessed:
    user_input = input(f"Guess a number between {LOW} and {HIGH}  {clue} ")
    number = int(user_input)
    if number > secret_number:
        clue = f"(less than {number})"
    elif number < secret_number:
        clue = f"(greater than {number})"
    else:
 number_guessed = True 
print(f"You guessed it! The secret number is {number}")
```

在本例中，首先定义一个布尔变量`number_guessed`，它允许您控制循环。在循环内部，您像往常一样处理用户的输入。如果用户猜出了密码，那么`number_guessed`被设置为`True`，程序跳出循环执行。

## 结论

在本教程中，你已经学会了用 Python 模拟一个 do-while 循环。这种语言没有这种循环结构，这种结构在 C、C++、Java 和 JavaScript 等语言中可以找到。您了解了在常规 while 循环的帮助下，您总是可以编写 do-while 循环，并且可以使用几种不同模式中的一种来完成它。

模拟 do-while 循环最常见的技术是创建一个[无限`while`循环](https://realpython.com/python-while-loop/#infinite-loops)，在循环体的末尾添加一个条件语句。该条件控制循环，并使用 [`break`](https://realpython.com/python-keywords/#iteration-keywords-for-while-break-continue-else) 语句跳出循环。

您还了解了如何使用一些替代技术来提供与 do-while 循环相同的功能。您的选择包括在循环之前进行第一组操作，或者使用一个初始设置为`True`的布尔变量来控制循环。

有了这些知识，您就可以开始在自己的 Python 代码中模拟 do-while 循环了。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。**