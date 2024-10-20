# python 中的 diarium 数

> 原文：<https://www.askpython.com/python/examples/disarium-number>

你好。今天让我们学习一些新的有趣的东西，数字。我们将理解这个数字是什么，然后执行一个程序来检查一个数字是否是一个不一致的数字。

## 什么是失谐数？

如果每个数字的自乘幂的和等于原来的数，那么这个数就是一个二进制数。

**双数定义**为:abcd…(n 位数)= a^1 + b^2 + c^3 + d^4 +。。。。。诸如此类。

***推荐阅读:[用 Python tkinter](https://www.askpython.com/python-modules/tkinter/celsius-to-fahrenheit-converter)*** 实现一个摄氏到华氏的转换器

## diarium 数的例子

### 示例 1: 153

计算(位数)= 1^1 + 5^2 + 3^3 = 1 + 25+ 9= 35

完成的计算不等于原始数字。因此，这个数字不是一个双数。

### 示例 2: 175

计算(位数)= 1^1 + 7^2 + 5^3 = 1 + 49 + 125 = 175

所做的计算直接等于原始数字。因此这个数字是不一致的数字。

## 检查 diarium 数的算法

检查一个数字是否是一个不一致的数字所涉及的所有步骤是:

1.  读取输入的数字并计算其大小
2.  将号码复制一份，以便稍后检查结果。
3.  创建一个结果变量(设置为 0)和一个迭代器(设置为数字的大小)
4.  创建一个 while 循环来逐个数字地遍历数字。
5.  每次迭代时，结果都是数字的迭代器值的幂
6.  每次遍历递减迭代器
7.  用原始数字的副本检查结果值

## 用于识别 diarium 编号的伪代码

下面的代码显示了检查一个数字是否是一个不一致的数字的伪代码:

```py
READ n
CALCULATE SIZE OF NUMBER len_n
MAKE A COPY OF n
result=0
CREATE AN ITERATOR i = len_n
CHECK DIGIT BY DIGIT:
  WHILE n!=0
     GET CURRENT DIGIT : digit = n % 10
     UPDATE RESULT : result = result + digit^(i)
     TRIM THE LAST DIGIT : n = n / 10
     DECREMENT i BY 1
  ENDWHILE

CHECK FOR DISARIUM NUMBER:
   IF result==COPY OF n
      PRINT "DISARIUM NUMBER"
   ELSE
      PRINT "NOT A DISARIUM NUMBER"

```

## 在 Python 中实现对灾难号的检查

现在我们知道了什么是 Disarium 号以及实现它的步骤，让我们逐行实现 Disarium 检查。

### 1.创建初始变量

我们首先获取一个输入`n`并计算数字的长度。我们还存储了一个输入的副本，这样无论我们对原始数字做了多少修改，我们都将结果初始化为 0，并将一个迭代器的初始值设为数字的长度。

相同的代码如下所示:

```py
n = input()
len_n = len(n)
n=int(n)
copy_n=n
result = 0
i = len_n

```

### 2.遍历数字并更新结果

为了访问每个数字，我们采用数字的模数(mod 10)来提取数字的最后一个数字。下一步是将结果更新为前一个结果和提升到数字位置的数字之和。

我们采取的最后一步是将数字除以 10，去掉数字的最后一位。我们还在每次迭代时将迭代器递减 1。重复相同的过程，直到数字中没有剩余的数字。

相同的代码如下所示:

```py
while(n!=0):
    digit = n%10
    result=result+pow(digit,i)
    n=int(n/10)
    i = i - 1

```

### 3.检查该号码是否为不一致的号码

最后一步是检查我们之前创建的数字的副本，并计算结果，以最终判断该数字是否是一个不一致的数字。相同的代码如下所示:

```py
if(result==copy_n):
    print("Disarium Number!")
else:
    print("Not an Disarium Number!")

```

## 代码的输出示例

现在，我测试了程序的两个输入。两者如下所示:

### 数字 1: 153

```py
153
Not a Disarium Number!

```

### 数字 2: 175

```py
121
Disarium Number!

```

## 结论

恭喜你！您已经成功地学习了 Disarium Number 并在 Python 中实现了它！但是不要就此打住！坚持读书学习！

***推荐阅读:[如何用 Python Tkinter 实现一个 GUI 年龄检查器？](https://www.askpython.com/python-modules/tkinter/age-calculator)***