# 使用 Python 找到三个数字中的最大值

> 原文：<https://www.pythoncentral.io/using-python-to-find-the-largest-number-out-of-three/>

这里有一个有用的片段，它将向您展示如何使用 Python 从任意三个给定的数字中找出最大的数字。基本上，该代码片段通过使用 if... 否则如果...else 语句将这三个数字相互比较，并确定哪个数字最大。查看下面的代码片段，了解它是如何工作的:

```py
number1 = 33
number2 = 67
number3 = 51

if (number1 > number2) and (number1 > number3):
 biggest = num1
elif (number2 > number1) and (number2 > number3):
 biggest = number2
else:
 biggest = number3

print("The biggest number between",number1,",",number2,"and",number3,"is",biggest)
```

在上面的例子中，我们知道 num2 (67)是最大的数字。所以第二个陈述(elif)是正确的，因为 num2 大于 num1 和 num3。因此，执行代码的结果将是下面的输出:“33、67 和 51 之间的最大数是 67。”这是可行的，因为 67 被设置为 elif 语句之后的“最大”变量。

这个代码应该与任何三个数字一起工作，以确定哪个是最大的。如果您希望您的用户能够自己输入数字，只需在代码中插入这三行代码来代替其他 num1、num2 和 num3 声明:

```py
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))
num3 = float(input("Enter third number: "))
```

摆弄代码，看看你是否能让它与任何不同数字的组合一起工作(你应该能！).