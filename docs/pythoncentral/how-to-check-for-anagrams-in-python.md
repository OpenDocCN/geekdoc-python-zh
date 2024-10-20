# 如何在 Python 中检查字谜

> 原文：<https://www.pythoncentral.io/how-to-check-for-anagrams-in-python/>

在 Python 中，有一种相当简单的方法来创建一个方法，该方法可用于检查字符串是否是变位词。看看下面的函数，看看这个方法是如何工作的——本质上，它是通过使用“==”比较操作符来查看操作符两边的字符串在通过计数器时是否相等，这将确定两个字符串之间是否有任何共享字符。如果字符串是字谜，函数将返回 true，如果不是，将返回 false。

```py
from collections import Counter
def is_anagram(str1, str2):
     return Counter(str1) == Counter(str2)
>>> is_anagram('cram','carm')
True
>>> is_anagram('card','cart')
False
```

正如你在上面的例子中所看到的，这个方法是通过传递两个参数来使用的，在那里它们被相互比较。当试图查看特定字符串是否只是重复相同的内容或保存相同的值时，这种方法特别有用——尤其是当这些字符串很长并且很难跟踪自己时。