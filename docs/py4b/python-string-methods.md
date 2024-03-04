# Python:字符串方法

> 原文：<https://www.pythonforbeginners.com/basics/python-string-methods>

字符串方法作用于调用它的字符串，如果你有一个名为 string = "Hello World "的字符串，那么字符串方法是这样调用的:string.string_method()。关于字符串方法的列表，请查看我的一篇文章，内容是关于字符串中内置方法的[。](https://www.pythonforbeginners.com/basics/strings-built-in-methods)

现在让我们找点乐子，展示一些例子:

```py
string = "Hello World"

print string.isupper()
print string.upper()
print string.islower()
print string.isupper()
print string.capitalize()
print string.split()
print string.split(',')
print string.title()
print string.strip() 
```

如果运行该程序，它应该会给出如下输出:

```py
False
HELLO WORLD
False
False
Hello world
['Hello', 'World']
['Hello World']
Hello World
Hello World 
```