# Python 中的字符串比较

> 原文：<https://www.askpython.com/python/string/string-comparison-in-python>

以下是在 Python 中比较两个字符串的方法:

1.  通过使用`== (equal to)`运算符
2.  通过使用`!= (not equal to)`运算符
3.  通过使用`sorted()`方法
4.  通过使用`is`运算符
5.  通过使用`Comparison`运算符

* * *

## 1.使用==(等于)运算符比较两个字符串

```py
str1 = input("Enter the first String: ")

str2 = input("Enter the second String: ")

if str1 == str2:

    print ("First and second strings are equal and same")

else:

    print ("First and second strings are not same")

```

**输出**:

输入第一串:AA
输入第二串:AA
第一串和第二串相等且相同

* * *

## 2.使用！比较两个字符串！=(不等于)运算符

```py
str1 = input("Enter the first String: ")

str2 = input("Enter the second String: ")

if str1 != str2:

    print ("First and second strings are not equal.")

else:

    print ("First and second strings are the same.")

```

**输出**:

输入第一串:ab
输入第二串:ba
第一串和第二串不相等。

* * *

## 3.使用 sorted()方法比较两个字符串

如果我们希望比较两个字符串并检查它们是否相等，即使字符/单词的顺序不同，那么我们首先需要使用 sorted()方法，然后比较两个字符串。

```py
str1 = input("Enter the first String: ")

str2 = input("Enter the second String: ")

if sorted(str1) == sorted(str2):

    print ("First and second strings are equal.")

else:

    print ("First and second strings are not the same.")

```

**输出**:

输入第一串:工程学科
输入第二串:学科工程
第一串和第二串相等。

## 4.使用“is”运算符比较两个字符串

如果两个变量引用同一个对象实例，Python is 运算符将返回 True。

```py
str1 = "DEED"

str2 = "DEED"

str3 = ''.join(['D', 'E', 'E', 'D'])

print(str1 is str2)

print("Comparision result = ", str1 is str3)

```

**输出**:

真
比较结果=假

在上面的示例中，str1 is str3 返回 False，因为对象 str3 是以不同方式创建的。

* * *

## 5.使用比较运算符比较两个字符串

```py
input = 'Engineering'

print(input < 'Engineering')
print(input > 'Engineering')
print(input <= 'Engineering')
print(input >= 'Engineering')

```

**输出**:

假
假
真
真

字符串按字典顺序进行比较。如果左操作数字符串在右字符串之前，则返回 True。

* * *

## 参考

*   Python 字符串比较