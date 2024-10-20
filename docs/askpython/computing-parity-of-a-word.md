# 在 Python 中计算单词的奇偶性

> 原文：<https://www.askpython.com/python/examples/computing-parity-of-a-word>

你好编码器！因此，在本教程中，我们将学习什么是单词的奇偶性，以及如何在 Python 编程中计算单词的奇偶性。我们先来了解一下一个词的奇偶性是如何计算的。

**二进制字的奇偶性为:**

*   如果单词包含奇数个 1 和，
*   如果它包含偶数个 1。

一些例子如下:

1.  字 1:1011
    1 的个数= 3
    0 的个数= 1
    奇偶校验= 1 因为 1 是奇数。

2.  word 2:10001000
    1 的个数= 2
    0 的个数= 6
    奇偶校验= 0 因为 1 是偶数。

问题陈述清楚地说我们需要计算一个词的宇称。简单来说，如果设置位(为 1 的位)的总数为奇数，则奇偶校验为`1`，否则为`0`。

## 使用 XOR 运算在 Python 中计算单词的奇偶性

方法 2 将利用右移位和异或运算。下面实现了这种方法，并添加了一些注释供您理解。

```py
# 1\. Taking Input of the word
n=int(input())
print("The word given by user is: ",n)

# parity variable initally set to 0
parity = 0

# 2\.  Go through all the bits in the while loop one by one
while(n!=0):

    # Check if the current LSB is 1 or 0
    # if the bit is 1 then 1 and 1 => 0 otherwise 1 and 0 ==> 0

    if((n&1)==1):
        # XOR previous parity with 1
        # It will change parity from 0 to 1 and vice versa alternately
        parity^=1

    # Right shift the number
    n>>=1

print("Parity is: ", parity)

```

## 输出

```py
The word given by user is:  1011
Parity is  1

```

```py
The word given by user is:  10001000
Parity is  0

```

## 结论

我希望你很好地理解了问题陈述和解决方案。您可以尝试在您的代码编辑器上实现相同的代码，并更多地了解单词的奇偶性。

感谢您的阅读！编码快乐！