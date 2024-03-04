# 用 Python 写的魔术 8 球

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/magic-8-ball-written-in-python>

## 概观

魔术 8 球是一种用于算命或寻求建议的玩具。

## 用 Python 写的魔术 8 球

在这个脚本中，我使用了 8 个可能的答案，但请随意添加更多。魔法 8 球里面有 20 个答案，你可以在这里找到全部

```py
# Import the modules
import sys
import random

ans = True

while ans:
    question = raw_input("Ask the magic 8 ball a question: (press enter to quit) ")

    answers = random.randint(1,8)

    if question == "":
        sys.exit()

    elif answers == 1:
        print "It is certain"

    elif answers == 2:
        print "Outlook good"

    elif answers == 3:
        print "You may rely on it"

    elif answers == 4:
        print "Ask again later"

    elif answers == 5:
        print "Concentrate and ask again"

    elif answers == 6:
        print "Reply hazy, try again"

    elif answers == 7:
        print "My reply is no"

    elif answers == 8:
        print "My sources say no" 
```

练习:有没有办法把所有的 elif 答案都换成别的？