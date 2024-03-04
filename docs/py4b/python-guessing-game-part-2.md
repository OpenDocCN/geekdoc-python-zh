# Python:猜谜游戏第 2 部分

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-guessing-game-part-2>

## 概观

这个小程序扩展了我之前在这个
中写过的猜谜游戏:post[“Python 猜谜游戏”](https://www.pythonforbeginners.com/code-snippets-source-code/python-guessing-game "Guessing Game")。

## 猜谜

在这个游戏中，我们将添加一个计数器来计算用户可以猜多少次。

计数器初始设置为零。

只要猜测次数少于 5 次，while 循环就会运行。

如果用户在此之前猜中了正确的数字，脚本将会中断，并向用户显示猜对数字需要猜多少次。

该脚本中的变量可以更改为任何值。

为了便于阅读，我将把程序分成几块

首先，我们导入随机模块

```py
import random 
```

然后我们给“数字”变量一个 1 到 99 之间的随机数。

```py
number = random.randint(1, 99) 
```

将 guests 变量设置为 0，这将对猜测进行计数

```py
guesses = 0 
```

只要猜测的次数少于 5 次，就让用户猜一个数字。

然后将猜测计数器加 1。

打印出一条消息给用户猜测的次数。

```py
while guesses < 5:
    guess = int(raw_input("Enter an integer from 1 to 99: "))
    guesses +=1
    print "this is your %d guess" %guesses 
```

检查猜测值是低于、高于还是等于我们的随机数，并打印结果消息
。

如果猜测和我们的数字一样，退出程序。

```py
 if guess < number:
        print "guess is low"
    elif guess > number:
        print "guess is high"
    elif guess == number:
        break 
```

打印出用户的猜测次数。

```py
if guess == number:
    guesses = str(guesses)
    print "You guess it in : ", guesses + " guesses" 
```

如果用户 5 次都猜不到正确的数字，打印出
密码是什么。

```py
if guess != number:
    number = str(number)
    print "The secret number was",  number 
```

我希望你喜欢这个猜谜游戏。