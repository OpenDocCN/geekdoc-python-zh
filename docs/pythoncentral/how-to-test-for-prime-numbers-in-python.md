# 如何在 Python 中测试质数

> 原文：<https://www.pythoncentral.io/how-to-test-for-prime-numbers-in-python/>

与任何 OOP 语言一样，您可以使用 Python 进行计算并收集关于数字的信息。用 Python 可以做的一件很酷的事情是测试一个数是否是质数。你可能记得很久以前数学课上讲过，质数是任何整数(它必须大于 1)，它的唯一因子是 1 和它自己，这意味着它不能被任何数整除(当然，除了 1 和它自己)。质数包括 2、3、5、7、11、13 等等，直到无穷大。

在 Python 中，我们可以使用下面的代码片段很容易地测试素数。

```py
if num > 1:

for i in range(2,num):
 if (num % i) == 0:
 print(num,"is not a prime number")
 print(i,"times",num//i,"is",num)
 break
 else:
 print(num,"is a prime number")

else:
 print(num,"is not a prime number")
```

首先，代码检查以确保数字大于 1(任何小于 1 的数字都不可能是质数，因为它不是整数)。然后它检查这个数字是否能被 2 和你要检查的数字之间的任何数字整除。如果它是可分的，那么你会从输出中看到这个数不是质数。如果它不能被这些数字中的任何一个整除，那么输出的消息将会是“[num]不是一个质数。”