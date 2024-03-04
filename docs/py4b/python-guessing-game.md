# Python 中猜谜游戏的实现

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-guessing-game>

用 python 编写简单的游戏是练习条件语句和循环的好方法。在本文中，我们将使用 if-else 块和 while 循环在 python 中实现一个猜谜游戏。

## 猜谜游戏是什么？

我们要用 python 实现的猜谜游戏有简单的规则。

*   首先，程序生成一个介于 1 和 99 之间的随机数。
*   然后，它让用户猜这个数字。
*   如果用户输入的数字小于系统生成的数字，系统会告诉用户猜测值较低。然后，它要求用户再次猜测号码。
*   如果用户输入的数字大于系统生成的数字，系统告诉用户猜测的数字更大。然后，它要求用户再次猜测号码。
*   如果用户猜对了数字，系统通知用户，游戏结束。

## 如何用 Python 实现猜谜游戏？

我们将使用以下步骤来创建猜谜游戏。

*   首先，我们将使用 python 中的 [random 模块中的 randint()函数来生成一个介于 1 和 99 之间的随机数。](https://www.pythonforbeginners.com/random/how-to-use-the-random-module-in-python)
*   接下来，我们将使用 input()函数将用户猜测的数字作为输入。
*   之后，我们将使用 while 循环来实现程序逻辑。在 while 循环中，我们将使用 if-else 块来检查用户输入的条件。
*   如果用户猜对了数字，我们将使用 break 语句来退出 while 循环并结束程序。

下面是用 Python 实现猜谜游戏的完整代码。

```py
import random
n = random.randint(1, 99)
guess = int(input("Enter an integer from 1 to 99: "))
while True:
    if guess < n:
        print ("guess is low")
        guess = int(input("Enter an integer from 1 to 99: "))
    elif guess > n:
        print ("guess is high")
        guess = int(input("Enter an integer from 1 to 99: "))
    else:
        print ("you guessed it right! Bye!")
        break
```

输出:

```py
Enter an integer from 1 to 99: 23
guess is low
Enter an integer from 1 to 99: 45
guess is low
Enter an integer from 1 to 99: 67
guess is low
Enter an integer from 1 to 99: 89
guess is low
Enter an integer from 1 to 99: 98
you guessed it right! Bye!
```

## 结论

在本文中，我们讨论了如何用 python 创建一个猜谜游戏。要了解更多关于 python 编程的知识，你可以阅读这篇关于 python 中 [hangman 游戏的文章。您可能也会喜欢这篇关于 Python](https://www.pythonforbeginners.com/code-snippets-source-code/game-hangman) 中的[字符串操作的文章。](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！