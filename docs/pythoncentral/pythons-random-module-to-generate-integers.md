# 使用 Python 的随机模块生成整数

> 原文：<https://www.pythoncentral.io/pythons-random-module-to-generate-integers/>

在 Python 中，random 模块允许您生成随机整数。当你需要随机选择一个数字或者从列表中随机选择一个元素时，经常会用到它。使用它实际上非常简单。假设您想要打印一个给定范围内的随机整数，比如 1-100。您应该这样编写代码:

```py
import random
print random.randint(1, 100)
```

上面的代码将返回一个从 0 到 100 的随机数。要编写代码，只需通过圆括号参数传递想要从中抽取随机整数的数字范围。因此，如果您希望范围是 1-100，第一个参数是 1，最后一个参数是 100——非常简单！