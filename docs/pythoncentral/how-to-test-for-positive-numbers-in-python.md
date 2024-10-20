# 如何在 Python 中测试正数

> 原文：<https://www.pythoncentral.io/how-to-test-for-positive-numbers-in-python/>

以下 Python 片段演示了如何测试输入的数字是正数、负数还是零。代码简单明了，适用于任何数字或整数。该代码使用 if... 否则如果...else 语句确定*如果*一个数大于 0(在这种情况下它必须是正数)，则 *elif* 一个数等于 0(在这种情况下它既不是正数也不是负数...简直是零)，或者 *else* 小于零(负)。

```py
num = 3
if num > 0:
   print("Positive number")
elif num == 0:
   print("Zero")
else:
   print("Negative number")
```

在上面的例子中，数字等于 3，所以输出将是“正数”，因为它大于零。如果 num 等于-19，那么输出将是“负数”代码非常简单明了。

使用这段代码的一种方法是让用户输入一个数字，并向他们显示输出。您还可以使用该代码根据所讨论的数字是正数还是负数来执行函数。为此，您需要调整代码以插入函数来代替打印命令。