# Python 字符串 ljust()和 rjust()函数

> 原文：<https://www.askpython.com/python/string/python-string-ljust-rjust-functions>

Python String 包含两个提供左右对齐的内置方法。

* * *

## 1.Python 字符串 ljust()函数

Python String ljust()函数接受用于填充输入字符串的字符。然后，它替换该字符，并在输入字符串的右侧进行填充。

**语法:**

```py
String.ljust(length,fill_char)
```

**参数:**

*   **length** :提供的 length 的值基本上用来对齐给定长度的字符串。
*   **fill_char** :可选参数。这些是需要在字符串周围填充的字符。

**ljust()函数返回的值:**

ljust()函数**返回一个新的字符串，并将给定的 fill_char 替换到原始字符串**的右边。

**例 1:**

```py
input_string = 'Python'
size = 9
fill_char = '@'

print(input_string.ljust(size, fill_char)) 

```

**输出:**

```py
[[email protected]](/cdn-cgi/l/email-protection)@@
```

**例 2:**

```py
inp_str = "Engineering Discipline"

print ("Input string: \n",inp_str) 

print ("Left justified string: \n") 
print (inp_str.ljust(30, '*')) 

```

**输出:**

```py
Input string: 
Engineering Discipline
Left justified string: 
Engineering Discipline********
```

### numpy string ljust()函数

**语法:**

```py
NumPy_array.char.ljust(input_array, length, fill_char) 
```

**参数:**

*   **输入数组**
*   **length** :提供的 length 的值基本上用来对齐给定长度的字符串。
*   **fill_char** :可选参数。这些是需要在字符串周围填充的字符。

**举例:**

```py
import numpy as np

input_arr = np.array(['Safa', 'Aman']) 
print ("Input array : ", input_arr) 

len1 = 10

output_arr = np.char.ljust(input_arr, len1, fillchar ='%') 
print ("Output array: ", output_arr) 

```

**输出:**

```py
Input array :  ['Safa' 'Aman']
Output array:  ['Safa%%%%%%' 'Aman%%%%%%']
```

* * *

## 2.Python 字符串 rjust()函数

Python 的 rjust()函数接受用于填充输入字符串的字符。

然后，它替换该字符，并在输入字符串的左侧进行填充。

**语法:**

```py
String.rjust(length,fill_char)
```

**参数:**

*   **length** :提供的 length 的值基本上用来对齐给定长度的字符串。
*   **fill_char** :可选参数。这些是需要在字符串周围填充的字符。

**从 rjust()函数返回的值:**

rjust()函数**返回一个新字符串，并将给定的 fill_char 替换到原始字符串**的左侧。

**例 1:**

```py
input_string = 'Mary'
size = 7
fill_char = '@'

print(input_string.rjust(size, fill_char)) 

```

**输出:**

```py
@@@Mary
```

**例 2:**

```py
inp_str = "Engineering Discipline"

print ("Input string: \n",inp_str) 

print ("Left justified string: \n") 
print (inp_str.rjust(30, '*')) 

```

**输出:**

```py
Input string: 
 Engineering Discipline
Left justified string: 

********Engineering Discipline
```

### NumPy 字符串 rjust()函数

**语法**:

```py
NumPy_array.char.rjust(input_array, length, fill_char) 
```

**参数:**

*   **输入数组**
*   **length** :提供的 length 的值基本上用来对齐给定长度的字符串。
*   **fill_char** :可选参数。这些是需要在字符串周围填充的字符。

**举例:**

```py
import numpy as np

input_arr = np.array(['Safa', 'Aman']) 
print ("Input array : ", input_arr) 

len1 = 10

output_arr = np.char.rjust(input_arr, len1, fillchar ='%') 
print ("Output array: ", output_arr) 

```

**输出:**

```py
Input array :  ['Safa' 'Aman']
Output array:  ['%%%%%%Safa' '%%%%%%Aman']
```

* * *

## 结论

因此，在本文中，我们已经理解了 Python 的字符串 ljust()和 rjust()函数的功能。

* * *

## 参考

字符串 ljust()和 rjust()函数