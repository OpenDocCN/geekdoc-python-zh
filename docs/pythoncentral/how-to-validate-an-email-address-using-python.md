# 如何使用 Python 验证电子邮件地址

> 原文：<https://www.pythoncentral.io/how-to-validate-an-email-address-using-python/>

这个片段向您展示了如何使用 Python 的 awesome isValidEmail()函数轻松验证任何电子邮件。验证电子邮件意味着您正在测试它是否是一封格式正确的电子邮件。虽然这段代码实际上并不测试电子邮件是真是假，但它确实确保了用户输入的电子邮件是有效的电子邮件格式(包含字符、后跟@符号、后跟字符和“.”的特定长度的字符串)其次是更多的字符。

用于验证的代码实际上非常简单，非常类似于使用任何其他 OOP 语言执行这个特定任务的方式。看看下面的片段，自己看看。

```py
import re
def isValidEmail(email):
 if len(email) > 7:
 if re.match("^.+@([?)[a-zA-Z0-9-.]+.([a-zA-Z]{2,3}|[0-9]{1,3})(]?)$", email) != None:
 return True
 return False
if isValidEmail("my.email@gmail.com") == True :
 print "This is a valid email address"
else:
 print "This is not a valid email address"
```

这段代码可以在任何 Python 项目中用来检查电子邮件的格式是否正确。它需要被修改以用作表单验证器，但是如果您希望验证任何表单提交的电子邮件地址，您绝对可以将它作为一个起点。