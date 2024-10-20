# 如何进行 Python 除法运算？

> 原文：<https://www.askpython.com/python/examples/python-division-operation>

嘿，伙计们！在本文中，我们将关注一个算术运算——Python**除法运算**。

* * *

## Python 除法运算入门

Python 有各种内置的操作符和函数来执行算术运算。

`'/' operator`用于对两种数据类型的数据值进行除法运算，即:**浮点型**和**整型**。

Python“/”操作符的美妙之处在于，该操作符可以分别处理小数和负值。

**语法:**

```py
number1 / number2

```

运算符对数值进行运算，并返回一个浮点值作为结果。除法运算的结果是所执行运算的**商**，表示为**浮点值**。

**例 1:**

```py
a = input("Enter the value for a:")
b = input("Enter the value of b:")
res = int(a)/int(b)
print(res)

```

**输出:**

```py
Enter the value for a:10
Enter the value of b:2
5.0

```

**例 2:**

```py
a = -10
b = 20
res = a/b
print(res)

```

**输出:**

```py
-0.5

```

* * *

## 元组上的 Python 除法运算

Python `floordiv() method`和`map() function`可以用来对存储在[元组](https://www.askpython.com/python/tuple/python-tuple)数据结构中的各种数据值进行除法运算。

Python `floordiv() method`用于对数据结构中存在的所有元素执行除法运算，即它执行**元素式除法**运算。此外，`Python map() function`对一组可重复项(如元组、列表等)应用任何传递/给定的函数或操作。

**语法:**

```py
tuple(map(floordiv, tuple1, tuple2))

```

`floordiv() method`执行整数除法，即将元素相除，只返回商的整数部分，跳过小数部分。

**举例:**

```py
from operator import floordiv 

inp_tup1 = (10,16,9,-4) 
inp_tup2 = (2,-8,4,4) 

tup_div = tuple(map(floordiv, inp_tup1, inp_tup2)) 

print("Resultant tuple after performing division operation : " + str(tup_div)) 

```

**输出:**

```py
Resultant tuple after performing division operation : (5, -2, 2, -1)

```

* * *

## Dict 上的 Python 除法运算

可以使用 Counter()函数和“//”运算符对出现在[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)中的元素执行 Python 除法运算。

`Counter() function`将**字典键值数据存储为字典键值**，并将字典元素的**计数存储为关联值**。

“//”运算符对数据元素执行整数级除法。

**语法:**

```py
Counter({key : dict1[key] // dict2[key] for key in dict1})

```

**举例:**

```py
from collections import Counter

inp_dict1 = {'Python': 100, 'Java': 40, 'C': 36}
inp_dict2 = {'Python': 20, 'Java': -10, 'C': 8}

inp_dict1 = Counter(inp_dict1) 
inp_dict2 = Counter(inp_dict2) 
dict_div = Counter({key : inp_dict1[key] // inp_dict2[key] for key in inp_dict1}) 

print("Resultant dict after performing division operation : " + str(dict(dict_div))) 

```

在上面的例子中，我们使用 Counter()函数存储了输入 dict 的键-值对，输入 dict 现在包含作为 dict 元素的键和作为 dict 中存在的元素计数的值。

此外，我们已经将键传递给了“//”运算符来执行除法运算。

**输出:**

```py
Resultant dict after performing division operation : {'Python': 5, 'Java': -4, 'C': 4}

```

* * *

## Python“/”和 Python“//”除法运算符之间的区别

“/”和“//”除法运算符之间的基本区别，也可能是唯一的区别在于， `'/' operator`返回浮点值作为除法的结果，即它返回整个商(整数和小数部分)。

另一方面， `'//' division operator`返回整数值作为除法的结果，即只返回商值的整数部分。

**举例:**

```py
print(10/3)
print(10//3)

```

**输出:**

```py
3.3333333333333335
3

```

## 结论

因此，在本文中，我们已经了解了在 Python 中执行除法运算的方法。

* * *

## 参考

*   Python 除法运算