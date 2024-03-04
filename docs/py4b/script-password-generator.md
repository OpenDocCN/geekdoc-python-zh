# Python 密码生成器

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/script-password-generator>

## 密码生成器

您可以使用 Pythons 字符串和随机模块来生成密码。

字符串模块包含了许多有用的常量和类。

其中一些我们将在这个脚本中使用。

**string . ascii _ letters**
ascii(大写和小写)字母的串联

**string.digits**
字符串‘0123456789’。

**string.punctuation**
在 C
语言环境中被视为标点符号的 ASCII 字符串。

打印字符串. ascii _ 字母

打印字符串.数字

打印字符串。标点符号

### 输出

abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
0123456789
！" #$% & '()*+，-。/:;< = > [【邮件保护】](/cdn-cgi/l/email-protection)【]^_`{|}~】

## 密码生成器脚本

因此，为了将这些放在一起，我们可以使用一个非常好的密码生成器脚本。

```py
import string
from random import *
characters = string.ascii_letters + string.punctuation  + string.digits
password =  "".join(choice(characters) for x in range(randint(8, 16)))
print password 
```

## 更多阅读

更多脚本，请参见[代码示例](https://www.pythonforbeginners.com/code-snippets-source-code/python-code-examples "code-snippets")页面。