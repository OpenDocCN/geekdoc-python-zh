# Python 中的阿姆斯特朗数——易于实现

> 原文：<https://www.askpython.com/python/examples/armstrong-number>

你好。今天让我们来学习一些有趣的东西，阿姆斯特朗号。我们将理解这个数字是什么，然后执行一个程序来检查一个数字是否是阿姆斯特朗数字。

## 什么是阿姆斯特朗数？

如果每个数字的幂的和等于原来的数，那么一个数字就是一个阿姆斯特朗数。

**阿姆斯壮数定义**为:abcd…(n 位数)= a^n + b^n + c^n + d^n +。。。。。诸如此类。

## 阿姆斯特朗数的例子

### 示例 1 : 153

总位数= 3

计算(位数)= 1^3 + 5^3 + 3^3 = 1 + 125 + 27 = 153

所做的计算直接等于原始数字。因此这个数字是阿姆斯特朗数。

### 示例 2 : 548834

总位数= 6

计算(数字型)= 5^6+4^6 +8^6+8^6+3^6+4^6 = 15625+4096+262144+262144+729+4096 = 548834

所做的计算直接等于原始数字。因此这个数字是阿姆斯特朗数。

## 检验阿姆斯特朗数的算法

要检查一个数字是否是阿姆斯特朗数，需要遵循以下步骤

1.  计算数字的位数。
2.  在模和除法运算的帮助下，每个数字被一个接一个地访问
3.  每一个数字都是数字的幂，结果存储在一个单独的变量中
4.  重复步骤 2 和 3，直到数字用尽。
5.  检查用原始数字计算的结果
    *   如果匹配:阿姆斯特朗号码
    *   否则:不是阿姆斯特朗号码

## 阿姆斯特朗数的伪代码

下面的代码显示了检查一个数字是否是阿姆斯特朗数的伪代码:

```py
READ n
CALCULATE NO OF DIGITS n_digit
MAKE A COPY OF n
result=0

CHECK DIGIT BY DIGIT:
  WHILE n!=0
     GET CURRENT DIGIT : digit = n % 10
     UPDATE RESULT : result = result + digit^(n_digit)
     TRIM THE LAST DIGIT : n = n / 10
  ENDWHILE

CHECK FOR ARMSTRONG NUMBER:
   IF result==COPY OF n
      PRINT "ARMSTRONG NUMBER"
   ELSE
      PRINT "NOT AN ARMSTRONG NUMBER"

```

## 用 Python 实现 Armstrong 数检查

现在我们知道了什么是阿姆斯特朗数以及实现它的步骤，让我们一行一行地实现阿姆斯特朗检查。

### 1.创建初始变量

我们首先获取一个输入`n`，然后[计算输入的长度](https://www.askpython.com/python/list/length-of-a-list-in-python)。我们还存储了一个输入的副本，这样无论我们对原始数字做了多少修改，我们都可以用这个副本来检查 Armstrong 的数字。我们还将结果初始化为 0。

相同的代码如下所示:

```py
n = input()
n_digit = len(n)
n=int(n)
copy_n=n
result = 0

```

### 2.遍历数字并更新结果

为了访问每个数字，我们取数字的模(mod 10)来提取数字的最后一位。下一步是将结果更新为前一个结果和数字的幂。

我们采取的最后一步是将数字除以 10，去掉数字的最后一位。重复相同的过程，直到数字中没有剩余的数字。

相同的代码如下所示:

```py
while(n!=0):
    digit = n%10
    result=result+pow(digit,n_digit)
    n=int(n/10)

```

### 3.检查该号码是否是阿姆斯特朗号码

最后一步是检查我们之前创建的数字的副本，并计算结果以最终判断该数字是否是 Armstrong 数字。相同的代码如下所示:

```py
if(result==copy_n):
    print("Armstrong Number!")
else:
    print("Not an Armstrong Number!")

```

## 代码的输出示例

现在，我测试了程序的四个输入。所有四个模块的输出如下所示:

### 数字 1: 153

```py
153
Armstrong Number!

```

### 数字 2: 121

```py
121
Not an Armstrong Number!

```

### 号码 3: 548834

```py
548834
Armstrong Number!

```

### 号码 4: 9468632

```py
9468632
Not an Armstrong Number!

```

## 结论

恭喜你！你已经成功地学习了阿姆斯特朗数并实现了它！

但是不要就此打住！坚持读书学习！