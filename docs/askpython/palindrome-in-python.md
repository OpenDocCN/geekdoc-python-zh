# Python 中的回文

> 原文：<https://www.askpython.com/python/examples/palindrome-in-python>

今天我们要学习回文系列，以及如何在 Python 中实现和识别回文。所以让我们开始吧！

## 什么是回文？

一个数被定义为一个回文数,如果它从向前和向后读完全相同。疯狂的是，它不仅仅对数字有效。即使一个字符串前后读起来一样，它也是一个回文！

让我们看一些例子来更好地理解它。

## 什么是回文系列？

### 1.回文数字

让我们考虑两个数字:123321 和 1234561。

第一个数字 **123321** ，向前和向后读都是同一个数字。因此它是一个回文数。

另一方面， **1234561** ，倒着读的时候是 1654321 肯定和原数不一样。因此，它不是一个回文数。

### 2.回文字符串

为回文数字解释的逻辑也适用于字符串。让我们考虑两个基本字符串:aba 和 abc。

字符串 **aba** 无论怎么读都是一样的(向后或向前)。但是另一方面，向后读取字符串 **abc** 会导致 **cba** 与原始字符串不同。

因此 aba 是一个回文，而 abc 不是。

## 如何验证为回文？

### 1.回文数字

为了检查一个数是否是回文数，我们首先获取该数的输入，并创建一个作为输入的该数的副本。

然后，我们创建一个新变量来存储反转后的数字，并用 0 初始化它。

使用 mod 10 和除以 10 运算遍历数字，并在每个循环中确保将数字添加到反转的数字变量*10 中。

### 2.回文字符串

为了检查字符串，我们将一个字符串作为输入，[计算它的长度](https://www.askpython.com/python/string/find-string-length-in-python)。我们还初始化一个空字符串来存储字符串的反码。

我们创建一个递减循环，从最后一个索引开始，到第一个索引，每次都将当前反转的字符串与获得的新字母连接起来。

## 用 Python 实现回文的伪代码

### 1.回文数字

```py
READ n
CREATE A COPY OF n as c_n
CREATE r_v = 0 ( to store reversed number)
WHILE n!=0:
d=n%10
r_v=r_v*10+d
n=n/10
if(c_n == r_v):
print "PALINDROME"
else:
print "NOT PALINDROME"

```

### 2.回文字符串

```py
READ s
CALCULATE length of s l_s
CREATE r_s = "" ( to store reversed string)
FOR i: l_s-1 -> 0
r_s + = s[i]

if(r_s == s):
PRINT "PALINDROME"
else:
PRINT "NOT PALINDROME"

```

## 用 Python 实现回文检查的代码

现在你知道了什么是回文，以及在字符串和数字的情况下如何处理它们，让我给你看两者的代码。

### 1.回文实现:数字

让我们使用 Python 来检查回文数字。

```py
n = input()
n = int(n)
copy_n=n
result = 0

while(n!=0):
    digit = n%10
    result = result*10 + digit
    n=int(n/10)

print("Result is: ", result)
if(result==copy_n):
    print("Palindrome!")
else:
    print("Not a Palindrome!")

```

### 2.回文实现:字符串

现在让我们检查 Python 中的回文字符串

```py
s = input()
l_s=len(s)
r_s=""

for i in range(l_s-1,-1,-1):
    r_s+=s[i]

print("Reuslt is: ",r_s)
if(r_s==s):
    print("PALINDROME")
else:
    print("NOT PALINDROME")

```

### 回文数字

```py
123321
Result is:  123321
Palindrome!

```

### 回文字符串

```py
aibohphobia
Reuslt is:  aibohphobia
PALINDROME

```

## 结论

恭喜你！今天在本教程中，你学习了回文以及如何实现它们！希望你学到了一些东西！感谢您的阅读！