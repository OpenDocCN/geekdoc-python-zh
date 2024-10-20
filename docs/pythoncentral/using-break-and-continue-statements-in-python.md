# 在 Python 中使用 Break 和 Continue 语句

> 原文：<https://www.pythoncentral.io/using-break-and-continue-statements-in-python/>

在 Python 中，break 语句用于退出(或“中断”)使用“for”或“while”的条件循环。循环结束后，代码将从紧跟 break 语句的那一行开始。这里有一个例子:

```py
even_nums = (2, 4, 6)
  num_sum = 0
  count = 0
  for x in even_nums:
    num_sum = num_sum + x
    count = count + 1
    if count == 4
       break
```

在上面的示例中，当 count 变量等于 4 时，代码将中断。

continue 语句用于跳过循环的某些部分。与 break 不同，它不会导致循环结束或退出，而是允许忽略循环的某些迭代，如下所示:

```py
for y in range(7)
   if (y==5):
      continue
   print(y)
```

在这个例子中，除了数字 5 的之外，循环的所有迭代(数字 0-7)都将被打印*，因为通过使用 continue 语句，循环被指示在 y 等于 5 时跳过 y。*

自己练习一下，看看如何使用 break 和 continue 语句。理解这两个语句的区别和用途将使您能够编写更加简洁高效的代码。