# 如何在 Python 中处理错误和异常

> 原文：<https://www.pythonforbeginners.com/error-handling/how-to-handle-errors-and-exceptions-in-python>

### 错误和异常

如果您(并且您将会)编写不工作的代码，您将会得到一个错误消息。

### 什么是例外？

异常是你第一次运行程序后得到的结果。

### 不同的误差

Python 中有不同种类的错误，这里有一些:
ValueError，TypeError，NameError，IOError，EOError，SyntaxError

此输出显示一个名称错误:

```py
>>> print 10 * ten
Traceback (most recent call last):
  File "", line 1, in 
NameError: name 'ten' is not defined

and this output show it's a TypeError
>>> print 1 + 'ten'
Traceback (most recent call last):
  File "", line 1, in 
TypeError: unsupported operand type(s) for +: 'int' and 'str' 
```

### 试着除了

Python 中有一种方法可以帮助你解决这个问题:尝试 except

```py
#Put the code that may be wrong in a try block, like this:

try:
    fh = open("non_existing_file")

#Put the code that should run if the code inside the try block fails, like this:
except IOError:
    print "The file does not exist, exiting gracefully"

#Putting it together, it will look like this:
try:
    fh = open("non_existing_file")
except IOError:
    print "The file does not exist, exiting gracefully"
    print "This line will always print" 
```

### 搬运 EOFErrors

```py
import sys
try:
    name = raw_input("what is your name?")
except EOFError:
    print "
You did an EOF... "
    sys.exit()

If you do an ctrl+d, you will get an output like this:
>>what is your name?
>>You did an EOF... 
```

### 处理键盘中断

```py
try:
    name = raw_input("Enter your name: ")
    print "You entered: " + name
except KeyboardInterrupt:
    print "You hit control-c"

If you press ctrl+c, you will get an output like this:
>>Enter your name: ^C
>>You hit control-c 
```

### 处理值错误

```py
while True:
try:
    x = int(raw_input("Please enter a number: "))
    break
except ValueError:
    print "Oops!  That was no valid number.  Try again..." 
```

关于所有 Python 内置异常的完整列表，请参见这篇[帖子](https://www.pythonforbeginners.com/error-handling/pythons-built-in-exceptions "built-in-exceptions")