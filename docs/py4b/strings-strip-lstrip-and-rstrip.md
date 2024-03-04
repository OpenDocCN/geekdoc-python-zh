# Python 中的剥离字符

> 原文：<https://www.pythonforbeginners.com/basics/strings-strip-lstrip-and-rstrip>

## 剥离方法

Python 字符串具有 strip()、lstrip()、rstrip()方法，用于移除字符串两端的任何字符。如果没有指定要删除的字符，则空白将被删除

strip()#从两端移除

lstrip()#删除前导字符(左条)

rst rip()#删除尾部字符(右移)

## 例子

```py
 spacious = "   xyz   "
print spacious.strip()

spacious = "   xyz   "
print spacious.lstrip()

spacious =  "xyz   "
print spacious.rstrip()
```

## 更多阅读

*   [Python 函数](https://www.pythonforbeginners.com/basics/python-functions-cheat-sheet)
*   [Python 基础知识](/basics/)