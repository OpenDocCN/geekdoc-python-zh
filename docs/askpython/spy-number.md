# Python:间谍号还是不是？

> 原文：<https://www.askpython.com/python/examples/spy-number>

你好，程序员朋友！今天我们要了解什么是**间谍号**以及如何使用 python 编程语言来判断一个数字是否是间谍号。

***也读作:[Python 中的 Harshad 数——易实现](https://www.askpython.com/python/examples/harshad-number)***

## 什么是间谍号？

如果某个数字的数字之和正好等于其数字之积，则该数字被称为**间谍号**。让我们看一些例子:

**例 1:** 1421
位数之和== > 1+4+2+1 = 8
位数之积== > 1*4*2*1 = 8

由于数字的乘积和总和完全相同，所以该数字是一个间谍号

**例二:** 1342
位数之和== > 1+3+4+2 = 10
位数之积== > 1*3*4*2 =24

显然，乘积和总和不相等，因此，这个数不是间谍数。

## 用 Python 识别一个间谍号

要知道一个号码是否是间谍号码，需要遵循下面描述的一些步骤:

**步骤 1:** 输入数字
**步骤 2:** 创建两个变量，一个存储总和，另一个存储乘积
**步骤 3:** 从右到左一个接一个地迭代数字位数
**步骤 4:** 在每次迭代中，将数字加到总和上，并将相同的数字乘以乘积
**步骤 5:** 在遇到所有数字之后，比较总和与乘积值:如果它们相等=【T11

现在，让我们按照上面提到的步骤来看看代码。

```py
num=int(input("Enter your number "))
sum=0
product=1
num1 = num

while(num>0):
    d=num%10
    sum=sum+d
    product=product*d
    num=num//10

if(sum==product):
    print("{} is a Spy number!".format(num1))
else:
    print("{} is not a Spy number!".format(num1))

```

希望你能按照上面提到的代码中提到的步骤去做。让我们看一些示例输出。

```py
Enter your number 123
123 is a Spy number!

```

```py
Enter your number 234
234 is not a Spy number!

```

您可以看到代码非常准确，并且给出了正确的结果。

## 结论

到本教程结束时，您已经了解了什么是 spy number 以及如何用 python 编程语言实现它。

感谢您的阅读！