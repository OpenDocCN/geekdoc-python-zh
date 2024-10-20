# 求最长公共子序列的长度

> 原文：<https://www.askpython.com/python/examples/length-of-longest-common-subsequence>

在本教程中，我们将首先简要解释什么是子序列和最长的公共子序列，然后再深入代码。在代码部分，我们将学习如何使用递归和动态编程来发现最长公共子序列的长度。

让我们马上开始吧。

* * *

## 什么是后续？

字符串子序列是通过从先前的字符串中删除部分字符，同时保持字符的相对位置不变而创建的新字符串。

举个例子——
原始字符串= " abcdwxyz "
有效子序列= "ACDW "、" BYZ "、" ACWXYZ"
无效子序列= "VAYZ "、" DYAZ "、" XBACW "

* * *

## 最长公共子序列(LCS)是什么？

给定一组序列，最大的共同子序列挑战是识别所有序列共有的最长子序列。最长公共子序列问题的答案并不总是唯一的。可能有许多具有最长可行长度的公共子序列。

举个例子——
sequence 1 = " BAHJDGSTAH "
sequence 2 = " HDSABTGHD "
sequence 3 = " ABTH "
LCS 的长度= 3
LCS = "ATH "，" BTH "

* * *

## 方法 1:递归

我们从末尾开始比较字符串，在递归中一次比较一个字符。设 LCS 是确定两个字符串共享的最长子序列长度的函数。

有两种可能的情况:

1.  字符是相同的——在 LCS 上加 1，通过删除最后一个字符——LCS(str 1，str2，m-1，n-1 ),使用更新后的字符串递归执行该过程。
2.  字符是不同的——不超过(用删除最后一个字符的 sring 1 进行递归调用，用删除最后一个字符的 string 2 进行递归调用)。

```py
def lcs(str1, str2, m, n):
    if m==0 or n==0:
        return 0 
    elif str1[m-1] == str2[n-1]: 
        return 1+lcs(str1, str2, m-1, n-1) 
    else: 
        return max(lcs(str1, str2, m-1, n),lcs(str1, str2, m,n-1))
str1 = input("Enter first string: ")
str2 = input("Enter second string: ")
lcs_length = lcs(str1, str2, len(str1), len(str2))
print("length of LCS is : {}".format(lcs_length))

```

```py
Enter first string: BAHJDGSTAH
Enter second string: BAHJDGSTAH
length of LCS is : 5

```

* * *

## 方法 2:动态规划方法

这种技术采用自底向上的策略。子问题的解决方案保存在矩阵中以备将来使用。这被称为记忆化。如果两个字符串的长度分别为 m 和 n，则动态规划的时间复杂度为 O(mn)，这大大小于递归的时间复杂度。矩阵的最后一项表示 LCS 的长度。

```py
def lcs(str1 , str2):
    m = len(str1)
    n = len(str2)
    matrix = [[0]*(n+1) for i in range(m+1)] 
    for i in range(m+1):
        for j in range(n+1):
            if i==0 or j==0:
                matrix[i][j] = 0
            elif str1[i-1] == str2[j-1]:
                matrix[i][j] = 1 + matrix[i-1][j-1]
            else:
                matrix[i][j] = max(matrix[i-1][j] , matrix[i][j-1])
    return matrix[-1][-1]
str1 = input("Enter first string: ")
str2 = input("Enter second string: ")
lcs_length = lcs(str1, str2)
print("Length of LCS is : {}".format(lcs_length))

```

```py
Enter first string: BAHJDGSTAH
Enter second string: BAHJDGSTAH
length of LCS is : 5

```

* * *

## 结论

恭喜你！您刚刚学习了如何显示最长公共子序列的长度。

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [用 Python 打印所有可能的子序列/子集](https://www.askpython.com/python/examples/possible-subsequences-subsets)
2.  [Python 随机模块–生成随机数/序列](https://www.askpython.com/python-modules/python-random-module-generate-random-numbers-sequences)
3.  [使用 Keras TensorFlow 预测莎士比亚文本](https://www.askpython.com/python/examples/predict-shakespearean-text)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *