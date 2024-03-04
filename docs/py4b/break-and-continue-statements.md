# 中断和继续语句

> 原文：<https://www.pythonforbeginners.com/basics/break-and-continue-statements>

Break 语句用于退出或“中断”一个 [python for 循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)或 while 条件循环。当循环结束时，代码从中断的循环开始执行下一行。

```py
 numbers = (1, 2, 3)
num_sum = 0
count = 0
for x in numbers:
        num_sum = num_sum + x
        count = count + 1
        print(count)
        if count == 2:
                break 
```

在本例中，循环将在计数等于 2 后中断。

continue 语句用于在循环的某些迭代中跳过循环中的代码。跳过代码后，循环从停止的地方继续。

```py
 for x in range(4):
   if (x==2):
      continue
   print(x) 
```

此示例将打印除 2 之外的从 0 到 4 的所有数字。