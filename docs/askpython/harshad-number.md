# Python 中的 Harshad 数–易于实现

> 原文：<https://www.askpython.com/python/examples/harshad-number>

你好。今天让我们学习哈沙德数。我们将了解这个数字是什么，然后执行一个程序来检查一个数字是否是一个 Harshad 数字。

## 什么是 Harshad 数？

如果一个数可以被它的数位之和整除，那么这个数就是 Harshad 数。

**Harshad 数定义**为:abcd 可被(a+b+c+d)整除。

***推荐阅读:[如何在 Python 中检查阿姆斯特朗数？](https://www.askpython.com/python/examples/armstrong-number)***

## 哈沙德数的例子

### 示例 1: 155

数字之和= 1 + 5 + 5 = 11

但是 155 不能被 11 整除。因此，这个数字不是一个苛刻的数字。

### 示例 2: 156

数字之和= 1 + 5 + 6 = 12

但是 156 能被 12 整除。因此，这个数字是一个苛刻的数字。

## 检查 Harshad 数的算法

检查一个号码是否为 Harshad 号码的所有步骤如下:

1.  读取输入的数字
2.  将号码复制一份，以便稍后检查结果。
3.  创建一个结果变量(设置为 0)
4.  创建一个 while 循环来逐个数字地遍历数字。
5.  在每一次迭代中，结果逐位递增
6.  将数字的副本除以得到的结果。
7.  如果一个数被整除，那么它就是一个 Harshad 数，否则它就不是。

## Harshad 号的伪代码

下面的代码显示了检查数字是否为 Harshad 数字的伪代码:

```py
READ n
MAKE A COPY OF n
result=0
CHECK DIGIT BY DIGIT:
  WHILE n!=0
     GET CURRENT DIGIT : digit = n % 10
     UPDATE RESULT : result = result + digit
     TRIM THE LAST DIGIT : n = n / 10
  ENDWHILE

CHECK FOR HARSHAD NUMBER:
   IF COPY OF n % result == 0
      PRINT "HARSHAD NUMBER"
   ELSE
      PRINT "NOT A HARSHAD NUMBER"

```

## Python 中检查 Harshad 数的代码

现在我们知道了什么是 Harshad 数以及实现它的步骤，让我们一行一行地实现 Harshad 检查。

### 创建初始变量

我们首先获取一个输入`n`并存储该输入的一个副本，这样无论我们如何改变原始数字，我们也将结果初始化为 0。

相同的代码如下所示:

```py
n = input()
n=int(n)
copy_n=n
result = 0

```

### 遍历数字并更新结果

为了访问每个数字，我们用数字的`modulus`( mod 10)来提取数字的最后一个数字。下一步是将结果更新为前一个结果和当前数字的和。

我们采取的最后一步是将数字除以 10，去掉数字的最后一位。重复相同的过程，直到数字中没有剩余的数字。

相同的代码如下所示:

```py
while(n!=0):
    digit = n%10
    result=result + digit
    n=int(n/10)

```

### 检查该号码是否是 Harshad 号码

最后一步是检查我们之前创建的数字的副本是否能被计算的结果整除。相同的代码如下所示:

```py
if(copy_n % result == 0):
    print("Harshad Number!")
else:
    print("Not an Harshad Number!")

```

## 代码的输出示例

现在，我测试了程序的两个输入。两个输出如下所示:

### 数字 1: 156

```py
156
Harshad Number!

```

### 数字 2: 121

```py
121
Not a Harshad Number!

```

## 结论

恭喜你！您已经成功地学习了 Harshad Number 并实现了它！

但是不要就此打住！坚持读书学习！