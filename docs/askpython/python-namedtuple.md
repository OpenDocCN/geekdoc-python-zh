# Python 命名元组

> 原文：<https://www.askpython.com/python/python-namedtuple>

不可变对象的序列是一个`**[Tuple](https://www.askpython.com/python/tuple/python-tuple)**`。

`**Namedtuple**`允许用户为元组中的元素提供名称。因此，它为用户提供了通过索引或指定名称访问元素的选项。

因此，它增加了代码的可读性。

**语法:**

```py
collections.namedtuple(typename, field_names, *, rename=False, defaults=None, module=None)
```

**typename** :描述分配给 nametuple 对象的名称。

**字段名**:用于定义命名元组的字段名。

**rename** :通过将 rename 变量值设置为 True，用户可以将无效字段重命名为其对应的索引名。

**默认值**:用户可以定义可选参数的默认值。

**模块** : `__module__`命名元组的属性被设置为特定值，只要定义了模块。

**举例:**

```py
from collections import namedtuple

Information = namedtuple('Information', ['city', 'state', 'pin_code'])

# below are also valid ways to namedtuple
# Employee = namedtuple('Employee', 'name age role')
# Employee = namedtuple('Employee', 'name,age,role')

info1 = Information('Pune', 'Maharashtra', 411027)
info2 = Information('Satara', 'Maharashtra', 411587)

for x in [info1, info2]:
    print(x)
print('\n')
print("Accessing the attributes via index values....")
# accessing via index value
for x in [info1, info2]:
    print(x)
print("Accessing the attributes via field name....")
# accessing via name of the field
for x in [info1, info2]:
    print(x.city, 'is in',x.state, 'with', x.pin_code)

```

**输出:**

```py
Information(city='Pune', state='Maharashtra', pin_code=411027)
Information(city='Satara', state='Maharashtra', pin_code=411587)

Accessing the attributes via index values....
Information(city='Pune', state='Maharashtra', pin_code=411027)
Information(city='Satara', state='Maharashtra', pin_code=411587)
Accessing the attributes via field name....
Pune is in Maharashtra with 411027
Satara is in Maharashtra with 411587 
```

* * *

## Python 命名的双重功能

*   **T0**
*   **T2`Python Namedtuple with invalid keys`**
*   **T2`rename variable`**
*   **T2`Namedtuple module`**
*   **T2`_make(iterable) function`**
*   **T2`_asdict() function`**
*   **T2`_replace(**kwargs) function`**
*   **T2`Namedtuple attributes`**
*   **T2 `“**” (double star) operator`**

* * *

### 1.使用 getattr()函数访问属性

`**getattr()**`函数用于访问命名元组的属性。

```py
from collections import namedtuple

Information = namedtuple('Information', ['city', 'state', 'pin_code'])

info1 = Information('Pune', 'Maharashtra', 411027)
info2 = Information('Satara', 'Maharashtra', 411587)

print (getattr(info1,'city'))

```

**输出:**

```py
Pune
```

* * *

### 2.Python 命名了带有无效密钥的元组

如果用户使用无效的名称作为字段值/键，则生成`**ValueError**`。

```py
from collections import namedtuple

Information = namedtuple('Information', ['city', 'state', 'pin_code'])

try:
    Information = namedtuple('Information', ['city', 'state', 'if '])

except ValueError as error:
    print(error)

```

**输出:**

```py
Type names and field names must be valid identifiers: 'if '
```

* * *

### 3.重命名变量

如果用户使用了无效的键，我们可以将**重命名**变量设置为**真**。

这样，键就被它们的索引值代替了。

```py
from collections import namedtuple

Information = namedtuple('Information', 'if',  rename=True)

try:
    info1 = Information('Pune' )
    for x in [info1]:
        print(x)
except ValueError as error:
    print(error)

```

**输出:**

```py
Information(_0='Pune')
```

* * *

### 4.命名的双重模块

```py
from collections import namedtuple

Information = namedtuple('Information', 'city',  rename=True, module='Simple1')
print(Information.__module__)

```

**输出:**

```py
Simple1
```

* * *

### 5._make(iterable)函数

```py
from collections import namedtuple

Information = namedtuple('Information', ['city', 'state', 'pin_code'])

x = ('Pune', 'Maharashtra', 411027)
info1 = Information._make(x)
print(info1)

```

**输出:**

```py
Information(city='Pune', state='Maharashtra', pin_code=411027)
```

* * *

### 6. **_asdict()函数**

`**_asdict()**`函数帮助从 Namedtuple 创建 OrderedDict 的实例。

```py
from collections import namedtuple

Information = namedtuple('Information', ['city', 'state', 'pin_code'])

x = ('Pune', 'Maharashtra', 411027)
info1 = Information._make(x)

ordered_output = info1._asdict()
print(ordered_output)

```

**输出:**

```py
{'city': 'Pune', 'state': 'Maharashtra', 'pin_code': 411027}
```

* * *

### 7.**_ 替换(**kwargs)功能**

因为 namedtuple 是不可变的，所以值不能改变。它通过用一组新值替换相应的键来返回一个新实例。

```py
from collections import namedtuple

Information = namedtuple('Information', ['city', 'state', 'pin_code'])

x = ('Pune', 'Maharashtra', 411027)
info1 = Information._make(x)

info2 = info1._replace(city='Satara', state='Maharashtra', pin_code=411031)
print(info2)

```

**输出:**

```py
Information(city='Satara', state='Maharashtra', pin_code=411031)
```

* * *

### 8.命名的双重属性

*   **_fields** :提供字段的信息。
*   **_fields_defaults** :提供用户设置的字段的默认值信息。

**例 1: _fields 属性**

```py
from collections import namedtuple

Information = namedtuple('Information', ['city', 'state', 'pin_code'])

print("Fields: ")
print (Information._fields)

```

**输出:**

```py
Fields: 
('city', 'state', 'pin_code')
```

**例 2: _fields_defaults 属性**

```py
from collections import namedtuple

Information = namedtuple('Information', ['city', 'state', 'pin_code'], defaults=['Pune', 'Maharashtra'])

print("Default Fields: ")
print (Information._fields_defaults)

```

**输出:'**

```py
Default Fields: 
{'state': 'Pune', 'pin_code': 'Maharashtra'} 
```

* * *

### **9。“**”(双星)符**

该运算符用于将字典转换为命名元组。

```py
from collections import namedtuple

Information = namedtuple('Information', ['city', 'state', 'pin_code'])

dict = { 'city' : "Pune", 'state':'Gujarat', 'pin_code' : '411027' }
print(Information(**dict))

```

**输出:**

```py
Information(city='Pune', state='Gujarat', pin_code='411027')
```

* * *

## 结论

因此，在本文中，我们已经理解了 Python 的集合 Namedtuple object 提供的功能。

* * *

## 参考

*   Python 命名元组
*   [已命名的双重文档](https://docs.python.org/3/library/collections.html#collections.namedtuple)