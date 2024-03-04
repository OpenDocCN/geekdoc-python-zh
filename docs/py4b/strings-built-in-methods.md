# 字符串内置方法

> 原文：<https://www.pythonforbeginners.com/basics/strings-built-in-methods>

## 字符串内置方法

在这篇文章中，我想展示如何使用 python 字符串的内置方法。让我们首先创建一个值为“Hello World”的字符串

```py
string = "Hello World" 
```

为了操作字符串，我们可以使用 Pythons 的一些内置方法

```py
string.upper()   # get all-letters in uppercase

string.lower()    # get all-letters in lowercase

string.capitalize() # capitalize the first letter

string.title()    # capitalize the first letter of words

string.swapcase() # converts uppercase and lowercase

string.strip()    # remove all white spaces

string.lstrip()   # removes white space from left

string.rstrip()   # removes white space from right

string.split()    # splitting words

string.split(',') # split words by comma

string.count('l')   # count how many times l is in the string

string.find('Wo') # find the word Wo in the string

string.index("Wo")  # find the letters Wo in the string

":".join(string)  # add a : between every char

" ".join(string)  # add a white space between every char

len(string)   # find the length of the string 
```