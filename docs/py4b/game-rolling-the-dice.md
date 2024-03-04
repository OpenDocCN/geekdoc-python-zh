# Python 游戏:掷色子

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/game-rolling-the-dice>

又到了写游戏脚本的时候了。

### 它是如何工作的

这是一个经典的“掷骰子”程序。

为此，我们将使用 random 模块，因为我们想随机化从骰子中得到的数字。

我们设置了两个变量(最小和最大)，骰子的最低和最高数量。

然后我们使用 while 循环，这样用户可以再次掷骰子。

roll_again 可以设置为任何值，但这里设置为“yes”或“y”，
但您也可以添加其他变量。

### 掷色子

```py
import random
min = 1
max = 6

roll_again = "yes"

while roll_again == "yes" or roll_again == "y":
    print "Rolling the dices..."
    print "The values are...."
    print random.randint(min, max)
    print random.randint(min, max)

    roll_again = raw_input("Roll the dices again?") 
```