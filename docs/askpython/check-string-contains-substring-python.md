# 如何在 Python 中检查一个字符串是否包含子串？

> 原文：<https://www.askpython.com/python/string/check-string-contains-substring-python>

子串是字符串中的字符序列。以下是 Python 中检查一个字符串是否包含另一个字符串(即 substring)的方法。

1.  通过使用 `find()`方法
2.  通过使用`in`运算符
3.  通过使用`count()`方法
4.  通过使用`str.index()` 方法
5.  通过使用`operator.contains()`方法

* * *

## 方法 1:使用 find()方法

find()方法检查一个字符串是否包含特定的子字符串。如果字符串包含特定的子字符串，则该方法返回子字符串的起始索引，否则返回-1。

**语法** : `string.find(substring)`

### 示例:使用 find()方法检查字符串中是否存在子字符串

```py
str="Safa Mulani is a student of Engineering discipline." 
sub1="Safa" 
sub2="Engineering" 

print(str.find(substring1)) 

print(str.find(substring2))

```

**输出**:

0
28

* * *

## 方法 2:通过使用 in 运算符

`in`操作符检查字符串中是否存在子串，如果存在子串，则返回**真**，否则返回**假**。

**语法**:string _ object 中的子串

### 示例:使用 in 运算符检查字符串中是否存在子字符串

```py
str="Safa Mulani is a student of Engineering discipline." 
sub1="Safa" 
sub2="Done" 

print(sub1 in str) 

print(sub2 in str)

```

**输出**:

真
假

* * *

## 方法 3:使用 count()方法

count()方法检查字符串中 substring 的出现。如果在字符串中找不到子字符串，则返回 0。

**语法** : string.count(子串)

### 示例:使用 count()方法检查字符串中是否存在子字符串

```py
str="Safa Mulani is a student of Engineering discipline." 
sub1="Safa" 
sub2="student" 
sub3="Done"
print(str.count(sub1)) 

print(str.count(sub2))

print(str.count(sub3))

```

**输出**:

1
1
0

* * *

## 方法 4:使用 index()方法

方法检查字符串中是否存在子字符串。如果子串不在字符串中，那么它不返回任何值，而是生成一个 **ValueError** 。

**语法** : string.index(子串)

### 示例:使用 index()方法检查字符串中是否存在子字符串

```py
str = "Safa is a Student."
try :  
    result = str.index("Safa") 
    print ("Safa is present in the string.") 
except : 
    print ("Safa is not present in the string.") 

```

**输出**:

Safa 出现在字符串中。

* * *

## 方法 5:使用 operator.contains()方法

**语法** : operator.contains(string，substring)

### 示例:使用 operator.contains()方法检查字符串中是否存在子字符串

```py
import operator 

str = "Safa is a Student."

if operator.contains(str, "Student"): 
    print ("Student is present in the string.") 
else : 
    print ("Student is not present in the string.")  

```

**输出**:学生出现在字符串中。

* * *

## 参考

*   Python 字符串