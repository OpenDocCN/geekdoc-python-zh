# 反转列表和字符串

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/reverse-loop-on-a-list>

这篇短文将展示如何在 Python 中进行反向循环。

第一个例子将展示如何在一个列表中这样做，另一个例子将展示如何反转一个字符串。

只需打开一个 Python 解释器并进行测试。

创建一个包含一些值的列表，如下所示:

```py
L1 = ["One", "two", "three", "four", "five"]

#To print the list as it is, simply do:
print L1

#To print a reverse list, do:
for i in L1[::-1]:
    print i

```

```py
 >>Output:
five
four
three
two
One 
```

我们也可以反转字符串，就像这样:

```py
string = "Hello World"

print ' '.join(reversed(string))

```

```py
 >>Output:

d l r o W   o l l e H 
```

## 相关帖子:

[用 Python 列出理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)