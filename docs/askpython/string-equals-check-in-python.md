# Python 中的字符串等于检查——4 种简单方法

> 原文：<https://www.askpython.com/python/string/string-equals-check-in-python>

在本文中，我们将看看在 Python 中执行字符串等于检查的不同方法。

[字符串](https://www.askpython.com/python/string)比较基本上是两个字符串的比较，即我们通过逐字符比较来检查字符串是否相等的过程。

* * *

## 技巧 1: Python '== '操作符检查两个字符串是否相等

Python 比较运算符可用于比较两个字符串，并检查它们在`case-sensitive manner`中的相等性，即大写字母和小写字母将被区别对待**。**

**Python `'==' operator`以逐字符的方式比较字符串，如果两个字符串相等则返回**真**，否则返回**假**。**

****语法:****

```py
string1 == string2 
```

****举例:****

```py
str1 = "Python"

str2 = "Python"

str3 = "Java"

print(str1 == str2)

print(str1 == str3) 
```

****输出:****

```py
True
False 
```

* * *

## **技巧二:Python '！用于字符串比较的“=”运算符**

**Python！= '运算符也可用于在 python 中执行字符串等于检查。**

**`'!=' operator`比较两个字符串，如果不相等则返回**真**，否则返回**假**。**

****语法:****

```py
string1 != string2 
```

****举例:****

```py
str1 = "Python"

str2 = "Python"

str3 = "Java"

if(str1 != str3):
 print("str1 is not equal to str3")

if(str1 != str2):
 print("str1 is not equal to str2")
else:
 print("str1 is equal to str2") 
```

****输出:****

```py
str1 is not equal to str3
str1 is equal to str2 
```

* * *

## **技巧 3: Python 中执行字符串等于检查的 Python“is”操作符**

****Python“is”操作符**可用于有效检查两个字符串对象的相等性。如果两个变量指向同一个数据对象，则`is operator`返回**真**，否则返回**假**。**

****语法:****

```py
variable1 is variable2 
```

****举例:****

```py
str1 = "Python"

str2 = "Python"

str3 = "Java"

if(str1 is str3):
 print("str1 is equal to str3")
else:
 print("str1 is not equal to str3")

if(str1 is str2):
 print("str1 is equal to str2")
else:
 print("str1 is not equal to str2") 
```

****输出:****

```py
str1 is not equal to str3
str1 is equal to str2 
```

* * *

## **技巧 4:在 python 中执行字符串等于检查的 __eq__()函数**

**Python 内置 __eq__()方法可以用来比较两个 string 对象。 `__eq__()`方法基本上是比较两个对象，如果相等则返回**真**，否则返回**假**。**

****语法:****

```py
string1.__eq__(string2) 
```

****举例:****

```py
str1 = "Python"

str2 = "Python"

str3 = "Java"

if(str1.__eq__(str3)):
 print("str1 is equal to str3")
else:
 print("str1 is not equal to str3")

if(str1.__eq__(str2)):
 print("str1 is equal to str2")
else:
 print("str1 is not equal to str2") 
```

****输出:****

```py
str1 is not equal to str3
str1 is equal to str2 
```

* * *

## **Python 中的字符串等于检查:无案例比较**

```py
str1 = "Python"

str2 = "PYTHON"

if(str1.__eq__(str2)):
 print("str1 is equal to str2")
else:
 print("str1 is not equal to str2") 
```

****输出:****

```py
str1 is not equal to str2 
```

**如上面的例子所示，结果是**假**，因为比较是`Case-sensitive`。**

**为了有一个**无案例的字符串比较**，即以`case-insensitive`的方式，那么我们可以使用 [Python string.casefold()函数](https://www.askpython.com/python/string/python-string-casefold)来达到目的。**

**方法**将字符串立即转换成小写字母**。**

**在字符串比较的场景中，我们可以将两个输入字符串都传递给 casefold()函数。因此，两个字符串都将被转换成小写，这样，我们就可以进行**无案例比较**。**

****语法:****

```py
string.casefold() 
```

****例 2:****

```py
str1 = "Python"

str2 = "PYTHON"

str3 = "PYthoN" 

if((str1.casefold()).__eq__(str2.casefold())):
 print("str1 is equal to str2")
else:
 print("str1 is not equal to str2")

if((str1.casefold()) == (str3.casefold())):
 print("str1 is equal to str3")
else:
 print("str1 is not equal to str3") 
```

****输出:****

```py
str1 is equal to str2
str1 is equal to str3 
```

* * *

## **结论**

**因此，在本文中，我们已经了解了 Python 中大小写字符串比较的方法和技巧。**

* * *

## **参考**

*   **Python 字符串比较–journal dev**
*   **[Python 是操作符——stack overflow](https://stackoverflow.com/questions/13650293/understanding-the-is-operator)**
*   **Python 字符串等于 JournalDev**