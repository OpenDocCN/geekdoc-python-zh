

# python-cheatsheet 文档

版本 0.1.0

crazyguitar

2023年1月4日

### 目录

- 1. Python 3 中的新特性
- 2. 备忘录
- 3. 高级备忘录
- 4. 附录

欢迎使用 pysheet。该项目旨在收集有用的 Python 代码片段，以提升 Python 开发者的编码体验。如果您有任何改进代码片段、解释等方面的精彩想法，欢迎随时贡献。

我们欢迎任何代码片段。如果您想参与贡献，可以在 [GitHub 上 fork pysheet](https://github.com/gto76/python-cheatsheet)。如果有任何问题或建议，请在 [GitHub Issues](https://github.com/gto76/python-cheatsheet/issues) 上创建 issue。

# python-cheatsheet 文档，版本 0.1.0

## Python 3 中的新特性

官方文档《[Python 中的新特性](What's New In Python)》展示了所有最重要的变更。但是，如果您太忙而没有时间阅读完整的变更列表，本部分将简要介绍 Python 3 的新特性。

### 1.1 Python 3 中的新特性

- 目录
- *print* 是一个函数
- 字符串即 Unicode
- 除法运算符
- 新的字典实现
- 仅限关键字参数
- 新的 Super
- 移除 <>
- BDFL 退休
- 禁止在函数内使用 *from module import \**
- 新增 nonlocal 关键字
- 可迭代对象解包扩展
- 通用解包
- 函数注解
- 变量注解
- 对 typing 模块和泛型的核心支持
- 格式化字节字符串
- f-string
- 异常抑制
- 生成器委托
- *async* 和 *await* 语法
- 异步生成器
- 异步推导式
- 矩阵乘法
- 数据类
- 内置的 breakpoint()
- 海象运算符（:=）
- 仅限位置参数
- 字典合并

#### 1.1.1 print 是一个函数

Python 3.0 中新增

- PEP 3105 - 让 print 成为函数

Python 2

```
>>> print "print is a statement"
print is a statement
>>> for x in range(3):
...     print x,
...
0 1 2
```

Python 3

```
>>> print("print is a function")
print is a function
>>> print()
>>> for x in range(3):
...     print(x, end=' ')
... else:
...     print()
...
0 1 2
```

#### 1.1.2 字符串即 Unicode

Python 3.0 中新增

- PEP 3138 - Python 3000 中的字符串表示
- PEP 3120 - 使用 UTF-8 作为默认源文件编码
- PEP 3131 - 支持非 ASCII 标识符

Python 2

```
>>> s = 'Café'  # 字节字符串
>>> s
'Caf\xc3\xa9'
>>> type(s)
<type 'str'>
>>> u = u'Café'  # Unicode 字符串
>>> u
u'Caf\xe9'
>>> type(u)
<type 'unicode'>
>>> len([_c for _c in 'Café'])
5
```

Python 3

```
>>> s = 'Café'
>>> s
'Café'
>>> type(s)
<class 'str'>
>>> s.encode('utf-8')
b'Caf\xc3\xa9'
>>> s.encode('utf-8').decode('utf-8')
'Café'
>>> len([_c for _c in 'Café'])
4
```

#### 1.1.3 除法运算符

Python 3.0 中新增

- PEP 238 - 更改除法运算符

Python2

```
>>> 1 / 2
0
>>> 1 // 2
0
>>> 1. / 2
0.5

#### 将“真除法”反向移植到 python2

>>> from __future__ import division
>>> 1 / 2
0.5
>>> 1 // 2
0
```

Python3

```
>>> 1 / 2
0.5
>>> 1 // 2
0
```

#### 1.1.4 新的字典实现

Python 3.6 中新增

- PEP 468 - 在函数中保留 **kwargs 的顺序
- PEP 520 - 保留类属性定义顺序
- bpo 27350 - 更紧凑且迭代更快的字典

Python 3.5 之前

```
>>> import sys
>>> sys.getsizeof({str(i):i for i in range(1000)})
49248

>>> d = {'timmy': 'red', 'barry': 'green', 'guido': 'blue'}
>>> d  # 未保留顺序
{'barry': 'green', 'timmy': 'red', 'guido': 'blue'}
```

Python 3.6

- 内存使用量小于 Python 3.5
- 保留插入顺序

```
>>> import sys
>>> sys.getsizeof({str(i):i for i in range(1000)})
36968

>>> d = {'timmy': 'red', 'barry': 'green', 'guido': 'blue'}
>>> d  # 保留插入顺序
{'timmy': 'red', 'barry': 'green', 'guido': 'blue'}
```

#### 1.1.5 仅限关键字参数

Python 3.0 中新增

- PEP 3102 - 仅限关键字参数

```
>>> def f(a, b, *, kw):
...     print(a, b, kw)
...
>>> f(1, 2, 3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: f() takes 2 positional arguments but 3 were given
>>> f(1, 2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: f() missing 1 required keyword-only argument: 'kw'
>>> f(1, 2, kw=3)
1 2 3
```

#### 1.1.6 新的 Super

Python 3.0 中新增

- PEP 3135 - 新的 Super

Python 2

```
>>> class ParentCls(object):
...     def foo(self):
...         print "call parent"
...
>>> class ChildCls(ParentCls):
...     def foo(self):
...         super(ChildCls, self).foo()
...         print "call child"
...
>>> p = ParentCls()
>>> c = ChildCls()
>>> p.foo()
call parent
>>> c.foo()
call parent
call child
```

Python 3

```
>>> class ParentCls(object):
...     def foo(self):
...         print("call parent")
...
>>> class ChildCls(ParentCls):
...     def foo(self):
...         super().foo()
...         print("call child")
...
>>> p = ParentCls()
>>> c = ChildCls()
>>> p.foo()
call parent
>>> c.foo()
call parent
call child
```

#### 1.1.7 移除 <>

Python 3.0 中新增

Python 2

```
>>> a = "Python2"
>>> a <> "Python3"
True

#### 等同于 !=
>>> a != "Python3"
True
```

Python 3

```
>>> a = "Python3"
>>> a != "Python2"
True
```

#### 1.1.8 BDFL 退休

Python 3.1 中新增

- PEP 401 - BDFL 退休

```
>>> from __future__ import barry_as_FLUFL
>>> 1 != 2
  File "<stdin>", line 1
    1 != 2
    ^
SyntaxError: with Barry as BDFL, use '<>' instead of '!='
>>> 1 <> 2
True
```

#### 1.1.9 禁止在函数内使用 from module import *

Python 3.0 中新增

```
>>> def f():
...     from os import *
...
  File "<stdin>", line 1
SyntaxError: import * only allowed at module level
```

#### 1.1.10 新增 nonlocal 关键字

Python 3.0 中新增

PEP 3104 - 访问外部作用域中的名称

> 注意：nonlocal 允许直接为外部（但非全局）作用域中的变量赋值

```
python
>>> def outf():
...     o = "out"
...     def inf():
...         nonlocal o
...         o = "change out"
...     inf()
...     print(o)
...
>>> outf()
change out
```

#### 1.1.11 可迭代对象解包扩展

Python 3.0 中新增

- PEP 3132 - 可迭代对象解包扩展

```
python
>>> a, *b, c = range(5)
>>> a, b, c
(0, [1, 2, 3], 4)
>>> for a, *b in [(1, 2, 3), (4, 5, 6, 7)]:
...     print(a, b)
...
1 [2, 3]
4 [5, 6, 7]
```

#### 1.1.12 通用解包

Python 3.5 中新增

- PEP 448 - 附加解包泛化

Python 2

```
python
>>> def func(*a, **k):
...     print(a)
...     print(k)
...
>>> func(*[1,2,3,4,5], **{"foo": "bar"})
(1, 2, 3, 4, 5)
{'foo': 'bar'}
```

Python 3

```
>>> print(*[1, 2, 3], 4, *[5, 6])
1 2 3 4 5 6
>>> [*range(4), 4]
[0, 1, 2, 3, 4]
>>> {"foo": "Foo", "bar": "Bar", **{"baz": "baz"}}
{'foo': 'Foo', 'bar': 'Bar', 'baz': 'baz'}
>>> def func(*a, **k):
...     print(a)
...     print(k)
...
>>> func(*[1], *[4,5], **{"foo": "FOO"}, **{"bar": "BAR"})
(1, 4, 5)
{'foo': 'FOO', 'bar': 'BAR'}
```

#### 1.1.13 函数注解

Python 3.0 中新增

- PEP 3107 - 函数注解
- PEP 484 - 类型提示
- PEP 483 - 类型提示理论

```
>>> import types
>>> generator = types.GeneratorType
>>> def fib(n: int) -> generator:
...     a, b = 0, 1
...     for _ in range(n):
...         yield a
...         b, a = a + b, b
...
>>> [f for f in fib(10)]
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

#### 1.1.14 变量注解

Python 3.6 中新增

- PEP 526 - 变量注解的语法

```
>>> from typing import List
>>> x: List[int] = [1, 2, 3]
>>> x
[1, 2, 3]
```

```
>>> from typing import List, Dict
>>> class Cls(object):
...     x: List[int] = [1, 2, 3]
...     y: Dict[str, str] = {"foo": "bar"}
...
>>> o = Cls()
```

#### 1.1.15 对类型模块和泛型的核心支持

Python 3.7 新增

-   PEP 560 - 对类型模块和泛型的核心支持

Python 3.7 之前

```
>>> from typing import Generic, TypeVar
>>> from typing import Iterable
>>> T = TypeVar('T')
>>> class C(Generic[T]): ...
...
>>> def func(l: Iterable[C[int]]) -> None:
...     for i in l:
...         print(i)
...
>>> func([1,2,3])
1
2
3
```

Python 3.7 及以上

```
>>> from typing import Iterable
>>> class C:
...     def __class_getitem__(cls, item):
...         return f"{cls.__name__}[{item.__name__}]"
...
>>> def func(l: Iterable[C[int]]) -> None:
...     for i in l:
...         print(i)
...
>>> func([1,2,3])
1
2
3
```

#### 1.1.16 格式化字节字符串

Python 3.5 新增

-   PEP 461 - 为 bytes 和 bytearray 添加 % 格式化

```
python
>>> b'abc %b %b' % (b'foo', b'bar')
b'abc foo bar'
>>> b'%d %f' % (1, 3.14)
b'1 3.140000'
>>> class Cls(object):
...     def __repr__(self):
...         return "repr"
...     def __str__(self):
...         return "str"
...
'repr'
>>> b'%a' % Cls()
b'repr'
```

#### 1.1.17 f-string

Python 3.6 新增

-   PEP 498 - 字面字符串插值

```
python
>>> py = "Python3"
>>> f'Awesome {py}'
'Awesome Python3'
>>> x = [1, 2, 3, 4, 5]
>>> f'{x}'
'[1, 2, 3, 4, 5]'
>>> def foo(x:int) -> int:
...     return x + 1
...
>>> f'{foo(0)}'
'1'
>>> f'{123.567:1.3}'
'1.24e+02'
```

#### 1.1.18 抑制异常

Python 3.3 新增

-   PEP 409 - 抑制异常上下文

不使用 raise Exception from None

```
python
>>> def func():
...     try:
...         1 / 0
...     except ZeroDivisionError:
...         raise ArithmeticError
...
>>> func()
Traceback (most recent call last):
  File "<stdin>", line 3, in func
ZeroDivisionError: division by zero

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 5, in func
ArithmeticError
```

使用 raise Exception from None

```
>>> def func():
...     try:
...         1 / 0
...     except ZeroDivisionError:
...         raise ArithmeticError from None
...
>>> func()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 5, in func
ArithmeticError
```

#### 调试

```
>>> try:
...     func()
... except ArithmeticError as e:
...     print(e.__context__)
...
division by zero
```

#### 1.1.19 生成器委托

Python 3.3 新增

-   PEP 380 - 委托给子生成器的语法

```
>>> def fib(n: int):
...     a, b = 0, 1
...     for _ in range(n):
...         yield a
...         b, a = a + b, b
...
>>> def delegate(n: int):
...     yield from fib(n)
...
>>> list(delegate(10))
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

#### 1.1.20 async 和 await 语法

Python 3.5 新增

-   PEP 492 - 使用 async 和 await 语法的协程

Python 3.5 之前

```
>>> import asyncio
>>> @asyncio.coroutine
... def fib(n: int):
...     a, b = 0, 1
...     for _ in range(n):
...         b, a = a + b, b
...     return a
...
>>> @asyncio.coroutine
... def coro(n: int):
...     for x in range(n):
...         yield from asyncio.sleep(1)
...         f = yield from fib(x)
...         print(f)
...
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(coro(3))
0
1
1
```

Python 3.5 及以上

```
>>> import asyncio
>>> async def fib(n: int):
...     a, b = 0, 1
...     for _ in range(n):
...         b, a = a + b, b
...     return a
...
>>> async def coro(n: int):
...     for x in range(n):
...         await asyncio.sleep(1)
...         f = await fib(x)
...         print(f)
...
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(coro(3))
0
1
1
```

#### 1.1.21 异步生成器

Python 3.6 新增

-   PEP 525 - 异步生成器

```
python
>>> import asyncio
>>> async def fib(n: int):
...     a, b = 0, 1
...     for _ in range(n):
...         await asyncio.sleep(1)
...         yield a
...         b, a = a + b , b
...
>>> async def coro(n: int):
...     ag = fib(n)
...     f = await ag.asend(None)
...     print(f)
...     f = await ag.asend(None)
...     print(f)
...
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(coro(5))
0
1
```

#### 1.1.22 异步推导式

Python 3.6 新增

-   PEP 530 - 异步推导式

```
python
>>> import asyncio
>>> async def fib(n: int):
...     a, b = 0, 1
...     for _ in range(n):
...         await asyncio.sleep(1)
...         yield a
...         b, a = a + b , b
...
#### async for ... else

>>> async def coro(n: int):
...     async for f in fib(n):
...         print(f, end=" ")
...     else:
...         print()
...
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(coro(5))
0 1 1 2 3
```

```
python
#### 列表中的 async for
>>> async def coro(n: int):
...     return [f async for f in fib(n)]
...
>>> loop.run_until_complete(coro(5))
[0, 1, 1, 2, 3]
```

```
python
#### 列表中的 await
>>> async def slowfmt(n: int) -> str:
...     await asyncio.sleep(0.5)
...     return f'{n}'
...
>>> async def coro(n: int):
...     return [await slowfmt(f) async for f in fib(n)]
...
>>> loop.run_until_complete(coro(5))
['0', '1', '1', '2', '3']
```

#### 1.1.23 矩阵乘法

Python 3.5 新增

-   PEP 465 - 专用的矩阵乘法中缀运算符

```
python
>>> # "@" 代表矩阵乘法
>>> class Arr:
...     def __init__(self, *arg):
...         self._arr = arg
...     def __matmul__(self, other):
...         if not isinstance(other, Arr):
...             raise TypeError
...         if len(self) != len(other):
...             raise ValueError
...         return sum([x*y for x, y in zip(self._arr, other._arr)])
...     def __imatmul__(self, other):
...         if not isinstance(other, Arr):
...             raise TypeError
...         if len(self) != len(other):
...             raise ValueError
...         res = sum([x*y for x, y in zip(self._arr, other._arr)])
...         self._arr = [res]
...         return self
...     def __len__(self):
...         return len(self._arr)
...     def __str__(self):
...         return self.__repr__()
...     def __repr__(self):
...         return "Arr({})".format(repr(self._arr))
...
>>> a = Arr(9, 5, 2, 7)
>>> b = Arr(5, 5, 6, 6)
>>> a @ b  # __matmul__
124
>>> a @= b  # __imatmul__
>>> a
Arr([124])
```

#### 1.1.24 数据类

Python 3.7 新增

-   PEP 557 - 数据类

可变数据类

```
>>> from dataclasses import dataclass
>>> @dataclass
... class DCls(object):
...     x: str
...     y: str
...
>>> d = DCls("foo", "bar")
>>> d
DCls(x='foo', y='bar')
>>> d = DCls(x="foo", y="baz")
>>> d
DCls(x='foo', y='baz')
>>> d.z = "bar"
```

不可变数据类

```
>>> from dataclasses import dataclass
>>> from dataclasses import FrozenInstanceError
>>> @dataclass(frozen=True)
... class DCls(object):
...     x: str
...     y: str
...
>>> try:
...     d.x = "baz"
... except FrozenInstanceError as e:
...     print(e)
...
cannot assign to field 'x'
>>> try:
...     d.z = "baz"
... except FrozenInstanceError as e:
...     print(e)
...
cannot assign to field 'z'
```

#### 1.1.25 内置的 breakpoint()

Python 3.7 新增

-   PEP 553 - 内置的 breakpoint()

```
>>> for x in range(3):
...     print(x)
...     breakpoint()
...
0
> <stdin>(1)<module>() -> None
(Pdb) c
1
> <stdin>(1)<module>() -> None
(Pdb) c
2
> <stdin>(1)<module>() -> None
(Pdb) c
```

#### 1.1.26 海象运算符

Python 3.8 新增

-   PEP 572 - 赋值表达式

海象运算符的目标是在表达式中赋值变量。在完成 PEP 572 后，被称为 BDFL 的 Guido van Rossum 决定辞去 Python 负责人的职务。

```
>>> f = (0, 1)
>>> [(f := (f[1], sum(f))) [0] for i in range(10)]
[1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
```

#### 1.1.27 仅限位置参数

Python 3.8 新增

-   PEP 570 - Python 仅限位置参数

```
>>> def f(a, b, /, c, d):
...     print(a, b, c, d)
...
>>> f(1, 2, 3, 4)
1 2 3 4
>>> f(1, 2, c=3, d=4)
1 2 3 4
>>> f(1, b=2, c=3, d=4)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: f() got some positional-only arguments passed as keyword arguments: 'b'
```

#### 1.1.28 字典合并

Python 3.9 中新增

- PEP 584 - 为字典添加并集运算符

```
>>> a = {"foo": "Foo"}
>>> b = {"bar": "Bar"}

#### old way
>>> {**a, **b}
{'foo': 'Foo', 'bar': 'Bar'}
>>> a.update(b)
>>> a
{'foo': 'Foo', 'bar': 'Bar'}

#### new way
>>> a | b
{'foo': 'Foo', 'bar': 'Bar'}
>>> a |= b
>>> a
{'foo': 'Foo', 'bar': 'Bar'}
```

## 速查表

本部分主要关注 Python 代码中的常见片段。速查表不仅包含 Python 的基本特性，还包括数据结构和算法。

## 2.1 风格

- 目录
  - 风格
    - 命名
      - 类
      - 函数
      - 变量

### 2.1.1 命名

#### 类

不推荐

```python
class fooClass: ...
class foo_class: ...
```

推荐

```python
class FooClass: ...
```

#### 函数

不推荐

```python
def CapCamelCase(*a): ...
def mixCamelCase(*a): ...
```

推荐

```python
def func_separated_by_underscores(*a): ...
```

#### 变量

不推荐

```python
FooVar = "CapWords"
fooVar = "mixedCase"
Foo_Var = "CapWords_With_Underscore"
```

推荐

```python
# 局部变量
var = "lowercase"

# 内部使用
_var = "_single_leading_underscore"

# 避免与 Python 关键字冲突
var_ = "single_trailing_underscore_"

# 类属性（类内私有）
__var = " __double_leading_underscore"

# “魔法”对象或属性，例如：__init__
__name__

# 临时变量，例如：_, v = (1, 2)
_ = "throwaway"
```

## 2.2 从头开始

本速查表的主要目标是收集一些常见且基础的语义或片段。速查表包含一些我们已经知道但在脑海中仍然模糊的语法，或者一些我们一再通过搜索引擎查找的片段。此外，由于 Python 2 的生命周期即将结束，大部分片段主要基于 Python 3 的语法。

- 目录
  - 从头开始
    - Hello world!
    - Python 版本
    - 省略号
    - if ... elif ... else
    - for 循环
    - for ... else ...
    - 使用 range
    - while ... else ...
    - do while 语句
    - try ... except ... else ...
    - 字符串
    - 列表
    - 字典
    - 函数
    - 函数注解
    - 生成器
    - 生成器委托
    - 类
    - async / await
    - 避免使用 exec 和 eval

### 2.2.1 Hello world!

当我们开始学习一门新语言时，通常从打印 **Hello world!** 开始。在 Python 中，我们可以通过导入 `__hello__` 模块来使用另一种方式打印信息。源代码可在 `frozen.c` 中找到。

```
>>> print("Hello world!")
Hello world!
>>> import __hello__
Hello world!
>>> import __phello__
Hello world!
>>> import __phello__.spam
Hello world!
```

### 2.2.2 Python 版本

了解当前的 Python 版本对程序员来说很重要，因为并非所有语法都能在当前版本中工作。在这种情况下，我们可以通过 `python -V` 或使用 `sys` 模块获取 Python 版本。

```
>>> import sys
>>> print(sys.version)
3.7.1 (default, Nov 6 2018, 18:46:03)
[Clang 10.0.0 (clang-1000.11.45.5)]
```

我们也可以使用 `platform.python_version` 来获取 Python 版本。

```
>>> import platform
>>> platform.python_version()
'3.7.1'
```

有时，检查当前 Python 版本很重要，因为我们可能希望在特定版本中启用某些功能。`sys.version_info` 提供了关于解释器的更详细信息。我们可以用它来与我们期望的版本进行比较。

```
>>> import sys
>>> sys.version_info >= (3, 6)
True
>>> sys.version_info >= (3, 7)
False
```

### 2.2.3 省略号

`Ellipsis` 是一个内置常量。Python 3.0 之后，我们可以使用 `...` 作为 `Ellipsis`。它可能是 Python 中最神秘的常量。根据官方文档，我们可以用它来扩展切片语法。尽管如此，它在类型提示、存根文件或函数表达式中还有一些其他惯例。

```
>>> ...
Ellipsis
>>> ... == Ellipsis
True
>>> type(...)
<class 'ellipsis'>
```

下面的片段展示了我们可以使用省略号来表示尚未实现的函数或类。

```
>>> class Foo: ...
...
>>> def foo(): ...
...
```

### 2.2.4 if ... elif ... else

**if 语句**用于控制代码流程。Python 使用 `if ... elif ... else` 序列来控制代码逻辑，而不是使用 `switch` 或 `case` 语句。尽管有人提出可以使用 `dict` 来实现 `switch` 语句，但这种解决方案可能会引入不必要的开销，例如创建一次性字典，并损害代码的可读性。因此，不推荐此方案。

```
>>> import random
>>> num = random.randint(0, 10)
>>> if num < 3:
...     print("less than 3")
... elif num < 5:
...     print("less than 5")
... else:
...     print(num)
...
less than 3
```

### 2.2.5 for 循环

在 Python 中，我们可以直接通过 **for 语句**访问可迭代对象的元素。如果我们需要同时获取可迭代对象（如列表或元组）的索引和元素，使用 `enumerate` 比 `range(len(iterable))` 更好。更多信息可以在 `循环技巧` 中找到。

```
>>> for val in ["foo", "bar"]:
...     print(val)
...
foo
bar
>>> for idx, val in enumerate(["foo", "bar", "baz"]):
...     print(idx, val)
...
(0, 'foo')
(1, 'bar')
(2, 'baz')
```

### 2.2.6 for ... else ...

第一次看到 `else` 属于 `for` 循环时，可能感觉有点奇怪。`else` 子句可以帮助我们在循环中避免使用标志变量。循环的 `else` 子句在没有发生 `break` 时运行。

```
>>> for _ in range(5):
...     pass
... else:
...     print("no break")
...
no break
```

下面的片段展示了使用标志变量和 `else` 子句控制循环的区别。我们可以看到，当循环中发生 `break` 时，`else` 不会运行。

```
>>> is_break = False
>>> for x in range(5):
...     if x % 2 == 0:
...         is_break = True
...         break
...
>>> if is_break:
...     print("break")
...
break

>>> for x in range(5):
...     if x % 2 == 0:
...         print("break")
...         break
...     else:
...         print("no break")
...
break
```

### 2.2.7 使用 range

Python 2 中 range 的问题是，如果我们需要在一个循环中迭代很多次，range 可能会占用大量内存。因此，在 Python 2 中推荐使用 xrange。

```
>>> import platform
>>> import sys
>>> platform.python_version()
'2.7.15'
>>> sys.getsizeof(range(100000000))
800000072
>>> sys.getsizeof(xrange(100000000))
40
```

在 Python 3 中，内置函数 range 返回一个可迭代的 range 对象，而不是一个列表。range 的行为与 Python 2 中的 xrange 相同。因此，如果想在循环中多次运行代码块，使用 range 不再占用巨大内存。更多信息可以在 PEP 3100 中找到。

```
>>> import platform
>>> import sys
>>> platform.python_version()
'3.7.1'
>>> sys.getsizeof(range(100000000))
48
```

### 2.2.8 while ... else ...

属于 while 循环的 else 子句与 for 循环中的 else 子句具有相同的目的。我们可以观察到，当 while 循环中发生 break 时，else 不会运行。

```python
>>> n = 0
>>> while n < 5:
...     if n == 3:
...         break
...     n += 1
... else:
...     print("no break")
...
```

### 2.2.9 do while 语句

许多编程语言，如 C/C++、Ruby 或 Javascript，都提供了 do while 语句。在 Python 中，没有 do while 语句。但是，我们可以将条件和 break 放在 while 循环的末尾来实现相同的功能。

```python
>>> n = 0
>>> while True:
...     n += 1
...     if n == 5:
...         break
...
>>> n
5
```

### 2.2.10 try ... except ... else ...

大多数时候，我们在 except 子句中处理错误，在 finally 子句中清理资源。有趣的是，try 语句也提供了一个 else 子句，以便我们避免捕获到不应由 try ... except 保护的代码引发的异常。当 try 和 except 之间没有发生异常时，else 子句会运行。

```python
>>> try:
...     print("No exception")
... except:
...     pass
... else:
...     print("Success")
...
No exception
Success
```## 2.2.11 字符串

与其他编程语言不同，Python 不直接支持字符串的项赋值。因此，如果需要操作字符串的元素，例如交换元素，我们必须先将字符串转换为列表，完成一系列项赋值操作后再进行连接操作。

```
>>> a = "Hello Python"
>>> l = list(a)
>>> l[0], l[6] = 'h', 'p'
>>> ''.join(l)
'hello python'
```

### 2.2.12 列表

列表是功能多样的容器。Python 提供了多种操作列表的方式，例如 **负数索引**、**切片语句** 或 **列表推导式**。以下代码片段展示了一些常见的列表操作。

```
>>> a = [1, 2, 3, 4, 5]
>>> a[-1]                    # 负数索引
5
>>> a[1:]                    # 切片
[2, 3, 4, 5]
>>> a[1:-1]
[2, 3, 4]
>>> a[1:-1:2]
[2, 4]
>>> a[::-1]                  # 反转
[5, 4, 3, 2, 1]
>>> a[0] = 0                 # 设置项
>>> a
[0, 2, 3, 4, 5]
>>> a.append(6)              # 追加一项
>>> a
[0, 2, 3, 4, 5, 6]
>>> del a[-1]                # 删除一项
>>> a
[0, 2, 3, 4, 5]
>>> b = [x for x in range(3)] # 列表推导式
>>> b
[0, 1, 2]
>>> a + b                    # 两个列表相加
[0, 2, 3, 4, 5, 0, 1, 2]
```

### 2.2.13 字典

字典是键值对容器。与列表类似，Python 支持多种操作字典的方式，例如 **字典推导式**。在 Python 3.6 之后，字典会保持键的插入顺序。以下代码片段展示了一些常见的字典操作。

```
>>> d = {'timmy': 'red', 'barry': 'green', 'guido': 'blue'}
>>> d
{'timmy': 'red', 'barry': 'green', 'guido': 'blue'}
>>> d['timmy'] = "yellow"          # 设置数据
>>> d
{'timmy': 'yellow', 'barry': 'green', 'guido': 'blue'}
>>> del d['guido']                  # 删除数据
>>> d
{'timmy': 'yellow', 'barry': 'green'}
>>> 'guido' in d                    # 是否包含数据
False
{'timmy': 'yellow', 'barry': 'green'}
>>> {k: v for k, v in d.items()} # 字典推导式
{'timmy': 'yellow', 'barry': 'green'}
>>> d.keys()                        # 列出所有键
dict_keys(['timmy', 'barry'])
>>> d.values()                      # 列出所有值
dict_values(['yellow', 'green'])
```

### 2.2.14 函数

在 Python 中定义函数非常灵活。我们可以使用 **函数文档字符串**、**默认值**、**任意参数**、**关键字参数**、**仅限关键字参数** 等来定义函数。以下代码片段展示了一些定义函数的常见形式。

```
def foo_with_doc():
    """Documentation String."""

def foo_with_arg(arg): ...
def foo_with_args(*arg): ...
def foo_with_kwarg(a, b="foo"): ...
def foo_with_args_kwargs(*args, **kwargs): ...
def foo_with_kwonly(a, b, *, k): ...           # python3
def foo_with_annotations(a: int) -> int: ... # python3
```

### 2.2.15 函数注解

除了在函数中编写字符串文档来提示参数和返回值的类型外，我们还可以使用 **函数注解** 来标注类型。函数注解在 Python 3.0 中引入，其详细信息可参见 PEP 3017 和 PEP 484。它们是 **Python 3** 中的一个 **可选** 特性。使用函数注解会失去对 **Python 2** 的兼容性。我们可以通过存根文件来解决这个问题。此外，可以通过 `mypy` 进行静态类型检查。

```
>>> def fib(n: int) -> int:
...     a, b = 0, 1
...     for _ in range(n):
...         b, a = a + b, b
...
    return a
...
>>> fib(10)
55
```

### 2.2.16 生成器

Python 使用 `yield` 语句定义 **生成器函数**。换句话说，当我们调用一个生成器函数时，生成器函数会返回一个 **生成器** 而不是返回值，用于创建一个 **迭代器**。

```
>>> def fib(n):
...     a, b = 0, 1
...     for _ in range(n):
...         yield a
...         b, a = a + b, b
...
>>> g = fib(10)
>>> g
<generator object fib at 0x10b240c78>
>>> for f in fib(5):
...     print(f)
...
0
1
1
2
3
```

### 2.2.17 生成器委托

Python 3.3 引入了 `yield from` 表达式。它允许一个生成器将部分操作委托给另一个生成器。换句话说，我们可以在当前的 **生成器函数** 中从其他 **生成器** 中 **yield** 一个序列。更多信息可参见 PEP 380。

```
>>> def fib(n):
...     a, b = 0, 1
...     for _ in range(n):
...         yield a
...         b, a = a + b, b
...
>>> def fibonacci(n):
...     yield from fib(n)
...
>>> [f for f in fibonacci(5)]
[0, 1, 1, 2, 3]
```

### 2.2.18 类

Python 支持许多常见特性，例如 **类文档字符串、多重继承、类变量、实例变量、静态方法、类方法** 等。此外，Python 还提供了一些特殊方法供程序员实现 **迭代器、上下文管理器** 等。以下代码片段展示了类的常见定义方式。

```
class A: ...
class B: ...
class Foo(A, B):
    """A class document."""

    foo = "class variable"

    def __init__(self, v):
        self.attr = v
        self.__private = "private var"

    @staticmethod
    def bar_static_method(): ...

    @classmethod
    def bar_class_method(cls): ...

    def bar(self):
        """A method document."""

    def bar_with_arg(self, arg): ...
    def bar_with_args(self, *args): ...
    def bar_with_kwarg(self, kwarg="bar"): ...
    def bar_with_args_kwargs(self, *args, **kwargs): ...
    def bar_with_kwonly(self, *, k): ...
    def bar_with_annotations(self, a: int): ...
```

### 2.2.19 async / await

`async` 和 `await` 语法从 Python 3.5 开始引入。它们被设计用于与事件循环一起使用。一些其他特性，如 **异步生成器**，在后续版本中实现。

**协程函数** (`async def`) 用于为事件循环创建一个 **协程**。Python 提供了一个内置模块 **asyncio**，通过 async/await 语法来编写并发代码。以下代码片段展示了使用 **asyncio** 的一个简单示例。该代码必须在 Python 3.7 或更高版本上运行。

```
import asyncio

async def http_ok(r, w):
    head = b"HTTP/1.1 200 OK\r\n"
    head += b"Content-Type: text/html\r\n"
    head += b"\r\n"

    body = b"<html>"
    body += b"<body><h1>Hello world!</h1></body>"
    body += b"</html>"

    _ = await r.read(1024)
    w.write(head + body)
    await w.drain()
    w.close()

async def main():
    server = await asyncio.start_server(
        http_ok, "127.0.0.1", 8888
    )

    async with server:
        await server.serve_forever()

asyncio.run(main())
```

### 2.2.20 避免使用 exec 和 eval

以下代码片段展示了如何使用内置函数 `exec`。然而，出于安全问题以及生成的代码对人类难以阅读等原因，不推荐使用 `exec` 和 `eval`。更多信息可参见《小心 Python 中的 exec 和 eval》以及《Eval 确实很危险》。

```
>>> py = '''
... def fib(n):
...     a, b = 0, 1
...     for _ in range(n):
...         b, a = b + a, b
...     return a
... print(fib(10))
... '''
>>> exec(py, globals(), locals())
55
```

## 2.3 未来语句

未来语句告知解释器将某些语义编译为将在未来 Python 版本中可用的语义。换句话说，Python 使用 `from __future__ import feature` 将其他更高版本 Python 的特性向后移植到当前解释器。在 Python 3 中，许多特性（如 `print_function`）已经默认启用，但我们仍然保留这些未来语句以保持向后兼容性。

未来语句**不是**导入语句。未来语句改变了 Python 解释代码的方式。它们**必须**位于文件顶部。否则，Python 解释器将引发 `SyntaxError`。

如果你对 future 语句感兴趣并想了解更多解释，更多信息可参见 PEP 236 - 回到 `__future__`。

### 目录

- 未来语句
- 列出所有新特性
- 打印函数
- Unicode
- 除法
- 注解
- BDFL 退休
- 花括号

### 2.3.1 列出所有新特性

`__future__` 是一个 Python 模块。我们可以用它来检查哪些 future 特性可以导入到当前 Python 解释器。有趣的是，`import __future__` **不是**一个未来语句，它是一个导入语句。

```
>>> from pprint import pprint
>>> import __future__
>>> pprint(__future__.all_feature_names)
['nested_scopes',
 'generators',
 'division',
 'absolute_import',
 'with_statement',
 'print_function',
 'unicode_literals',
 'barry_as_FLUFL',
 'generator_stop',
 'annotations']
```

未来语句不仅改变 Python 解释器的行为，还将 `__future__._Feature` 导入到当前程序中。

```
>>> from __future__ import print_function
>>> print_function
_Feature((2, 6, 0, 'alpha', 2), (3, 0, 0, 'alpha', 0), 65536)
```

### 2.3.2 打印函数

将 **打印语句** 替换为 **打印函数** 是 Python 历史上最著名的决定之一。然而，这种改变为扩展 `print` 的能力带来了一些灵活性。更多信息可参见 PEP 3105。

```
>>> print "Hello World"  # print 是一个语句
Hello World
>>> from __future__ import print_function
>>> print "Hello World"
  File "<stdin>", line 1
    print "Hello World"
            ^
SyntaxError: invalid syntax
>>> print("Hello World") # print 变成了一个函数
Hello World
```

### 2.3.3 Unicode

作为 **print 函数** 的一部分，使文本转变为 Unicode 是另一个著名的决定。然而，如今许多现代编程语言的文本默认就是 Unicode。这一变化迫使我们及早解码文本，以防止在程序运行一段时间后出现运行时错误。更多信息请参阅 PEP 3112。

```
>>> type("Guido") # string type is str in python2
<type 'str'>
>>> from __future__ import unicode_literals
>>> type("Guido") # string type become unicode
<type 'unicode'>
```

### 2.3.4 除法

有时，当除法结果是整数或长整数时，这可能违反直觉。在这种情况下，Python 3 默认启用了**真除**。然而，在 Python 2 中，我们必须将除法行为移植到当前的解释器中。更多信息请参阅 PEP 238。

```
>>> 1 / 2
0
>>> from __future__ import division
>>> 1 / 2   # return a float (classic division)
0.5
>>> 1 // 2  # return a int (floor division)
0
```

### 2.3.5 类型注解

在 Python 3.7 之前，如果当前作用域中尚不可用，我们不能在类或函数中为类自身添加类型注解。一种常见的情况是定义容器类。

```
class Tree(object):

    def insert(self, tree: Tree): ...
```

示例

```
$ python3 foo.py
Traceback (most recent call last):
  File "foo.py", line 1, in <module>
    class Tree(object):
  File "foo.py", line 3, in Tree
    def insert(self, tree: Tree): ...
NameError: name 'Tree' is not defined
```

在这种情况下，类的定义尚不可用。Python 解析器在定义时无法解析该注解。为了解决此问题，Python 使用字符串字面量来替代该类。

```
class Tree(object):

    def insert(self, tree: 'Tree'): ...
```

在 3.7 版本之后，Python 引入了 future 语句 `annotations`，以执行延迟求值。这将在 Python 4 中成为默认功能。更多信息请参考 [PEP 563](https://peps.python.org/pep-0563/)。

```
from __future__ import annotations

class Tree(object):

    def insert(self, tree: Tree): ...
```

### 2.3.6 BDFL 退休

**Python 3.1 新增功能**

PEP 401 只是一个彩蛋。此功能将当前解释器带回过去。它在 Python 3 中启用了钻石运算符 `<>`。

```
>>> 1 != 2
True
>>> from __future__ import barry_as_FLUFL
>>> 1 != 2
  File "<stdin>", line 1
    1 != 2
       ^
SyntaxError: with Barry as BDFL, use '<>' instead of '!='
>>> 1 <> 2
True
```

### 2.3.7 大括号

`braces` 是一个彩蛋。其源代码可在 `future.c` 中找到。

```
>>> from __future__ import braces
  File "<stdin>", line 1
SyntaxError: not a chance
```

## 2.4 Unicode

本备忘录的主要目标是收集一些与 Unicode 相关的常用代码片段。在 Python 3 中，字符串使用 Unicode 而非字节表示。更多信息请参阅 [PEP 3100](https://peps.python.org/pep-3100/)。

**ASCII** 码是最著名的字符编码标准，它为字符定义了数字编码。最初的数值仅定义了 128 个字符，因此 ASCII 仅包含控制代码、数字、小写字母、大写字母等。然而，这不足以表示世界各地存在的带重音字符、汉字或表情符号等。因此，**Unicode** 被开发出来以解决此问题。它定义了代码点来表示各种字符，就像 ASCII 一样，但其字符数量最多可达 1,111,998 个。

| 目录 |
| --- |
| Unicode |

- 字符串
- 字符
- 移植 unicode(s, 'utf-8')
- Unicode 代码点
- 编码
- 解码
- Unicode 标准化
- 避免 UnicodeDecodeError
- 长字符串

### 2.4.1 字符串

在 Python 2 中，字符串以*字节*表示，而不是 *Unicode*。Python 提供了不同类型的字符串，例如 Unicode 字符串、原始字符串等。在这种情况下，如果要声明一个 Unicode 字符串，我们在字符串字面量前添加 `u` 前缀。

```
>>> s = 'Café'  # byte string
>>> s
'Caf\xc3\xa9'
>>> type(s)
<type 'str'>
>>> u = u'Café' # unicode string
>>> u
u'Caf\xe9'
>>> type(u)
<type 'unicode'>
```

在 Python 3 中，字符串以 *Unicode* 表示。如果要表示一个字节字符串，我们在字符串字面量前添加 `b` 前缀。请注意，早期的 Python 版本（3.0-3.2）不支持 `u` 前缀。为了简化将支持 Unicode 的应用程序从 Python 2 迁移的过程，Python 3.3 再次支持字符串字面量的 `u` 前缀。更多信息请参阅 PEP 414。

```
>>> s = 'Café'
>>> type(s)
<class 'str'>
>>> s
'Café'
>>> s.encode('utf-8')
b'Caf\xc3\xa9'
>>> s.encode('utf-8').decode('utf-8')
'Café'
```

### 2.4.2 字符

Python 2 将所有字符串字符视为字节。在这种情况下，字符串的长度可能不等于字符的数量。例如，`Café` 的长度是 5，而不是 4，因为 `é` 被编码为 2 个字节的字符。

```
>>> s= 'Café'
>>> print([_c for _c in s])
['C', 'a', 'f', '\xc3', '\xa9']
>>> len(s)
5
>>> s = u'Café'
>>> print([_c for _c in s])
[u'C', u'a', u'f', u'\xe9']
>>> len(s)
4
```

Python 3 将所有字符串字符视为 Unicode 代码点。字符串的长度始终等于字符的数量。

```
>>> s = 'Café'
>>> print([_c for _c in s])
['C', 'a', 'f', 'é']
>>> len(s)
4
>>> bs = bytes(s, encoding='utf-8')
>>> print(bs)
b'Caf\xc3\xa9'
>>> len(bs)
5
```

### 2.4.3 移植 unicode(s, 'utf-8')

`unicode()` 内置函数在 Python 3 中已被移除，那么转换表达式 `unicode(s, 'utf-8')` 的最佳方式是什么，使其在 Python 2 和 3 中都能工作？

在 Python 2 中：

```
>>> s = 'Café'
>>> unicode(s, 'utf-8')
u'Caf\xe9'
>>> s.decode('utf-8')
u'Caf\xe9'
>>> unicode(s, 'utf-8') == s.decode('utf-8')
True
```

在 Python 3 中：

```
>>> s = 'Café'
>>> s.decode('utf-8')
AttributeError: 'str' object has no attribute 'decode'
```

所以，真正的答案是……

### 2.4.4 Unicode 代码点

`ord` 是一个强大的内置函数，用于从给定字符获取 Unicode 代码点。因此，如果要检查字符的 Unicode 代码点，可以使用 `ord`。

```
>>> s = u'Café'
>>> for _c in s: print('U+%04x' % ord(_c))
...
U+0043
U+0061
U+0066
U+00e9
>>> u = '中文'
>>> for _c in u: print('U+%04x' % ord(_c))
...
U+4e2d
U+6587
```

### 2.4.5 编码

将 Unicode 代码点转换为字节字符串的过程称为编码。

```
>>> s = u'Café'
>>> type(s.encode('utf-8'))
<class 'bytes'>
```

### 2.4.6 解码

将字节字符串转换为 Unicode 代码点的过程称为解码。

```
>>> s = bytes('Café', encoding='utf-8')
>>> s.decode('utf-8')
'Café'
```

### 2.4.7 Unicode 标准化

某些字符可以用两种相似的形式表示。例如，字符 `é` 可以写作 `e` (规范分解) 或 `é` (规范组合)。在这种情况下，即使两个字符串看起来相似，我们进行比较时也可能得到意想不到的结果。因此，我们可以通过标准化 Unicode 形式来解决这个问题。

```
#### python 3
>>> u1 = 'Café'        # unicode string
>>> u2 = 'Cafe\u0301'
>>> u1, u2
('Café', 'Café')
>>> len(u1), len(u2)
(4, 5)
>>> u1 == u2
False
>>> u1.encode('utf-8') # get u1 byte string
b'Caf\xc3\xa9'
>>> u2.encode('utf-8') # get u2 byte string
b'Cafe\xcc\x81'
>>> from unicodedata import normalize
>>> s1 = normalize('NFC', u1)  # get u1 NFC format
>>> s2 = normalize('NFC', u2)  # get u2 NFC format
>>> s1 == s2
True
>>> s1.encode('utf-8'), s2.encode('utf-8')
(b'Caf\xc3\xa9', b'Caf\xc3\xa9')
>>> s1 = normalize('NFD', u1)  # get u1 NFD format
>>> s2 = normalize('NFD', u2)  # get u2 NFD format
>>> s1, s2
('Cafe\u0301', 'Cafe\u0301')
>>> s1 == s2
True
>>> s1.encode('utf-8'), s2.encode('utf-8')
(b'Cafe\xcc\x81', b'Cafe\xcc\x81')
```

### 2.4.8 避免 UnicodeDecodeError

当字节字符串无法解码为 Unicode 代码点时，Python 会引发 *UnicodeDecodeError*。如果我们想避免这个异常，可以在 `decode` 的 `errors` 参数中传递 *replace*、*backslashreplace* 或 *ignore*。

```
>>> u = b"\xff"
>>> u.decode('utf-8', 'strict')
    Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
>>> # use U+FFFD, REPLACEMENT CHARACTER
>>> u.decode('utf-8', "replace")
'\ufffd'
>>> # inserts a \xNN escape sequence
>>> u.decode('utf-8', "backslashreplace")
'\xff'
>>> # leave the character out of the Unicode result
>>> u.decode('utf-8', "ignore")
''
```

### 2.4.9 长字符串

以下代码片段展示了在 Python 中声明多行字符串的常见方式。

```
#### original long string
s = 'This is a very very very long python string'

#### Single quote with an escaping backslash
s = "This is a very very very " \
    "long python string"
```## 2.5 列表

列表是一种我们常用来存储对象的常见数据结构。大多数时候，程序员关注的是获取、设置、搜索、过滤和排序。此外，有时我们还会不慎陷入内存管理的常见陷阱。因此，这份速查表的主要目标就是收集一些常见操作和注意事项。

- 列表
  - 从头开始
  - 初始化
  - 复制
  - 使用切片
  - 列表推导式
  - 解包
  - 使用 enumerate
  - 打包列表
  - 过滤项
  - 栈
  - `in` 操作
  - 访问项
  - 委托迭代
  - 排序
  - 已排序的列表
  - 创建一个新列表
- 循环缓冲区
- 分块
- 分组
- 二分查找
- 下界
- 上界
- 字典序
- 前缀树

### 2.5.1 从头开始

在 Python 中，我们可以通过多种方式操作列表。在开始学习这些多样化操作之前，以下代码片段展示了列表最常见的操作。

```python
>>> a = [1, 2, 3, 4, 5]
>>> # 包含
>>> 2 in a
True
>>> # 正索引
>>> a[0]
1
>>> # 负索引
>>> a[-1]
5
>>> # 列表切片 list[start:end:step]
>>> a[1:]
[2, 3, 4, 5]
>>> a[1:-1]
[2, 3, 4]
>>> a[1:-1:2]
[2, 4]
>>> # 反转
>>> a[::-1]
[5, 4, 3, 2, 1]
>>> a[:0:-1]
[5, 4, 3, 2]
>>> # 设置一项
>>> a[0] = 0
>>> a
[0, 2, 3, 4, 5]
>>> # 追加项到列表
>>> a.append(6)
>>> a
[0, 2, 3, 4, 5, 6]
>>> a.extend([7, 8, 9])
>>> a
[0, 2, 3, 4, 5, 6, 7, 8, 9]
>>> # 删除一项
>>> del a[-1]
>>> a
[0, 2, 3, 4, 5, 6, 7, 8]
>>> # 列表推导式
>>> b = [x for x in range(3)]
>>> b
[0, 1, 2]
>>> # 添加两个列表
>>> a + b
[0, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2]
```

### 2.5.2 初始化

通常来说，如果列表表达式中的元素是不可变对象，我们可以通过 * 运算符创建一个列表。

```python
>>> a = [None] * 3
>>> a
[None, None, None]
>>> a[0] = "foo"
>>> a
['foo', None, None]
```

然而，如果列表表达式中的元素是可变对象，* 运算符会将该元素的引用复制 N 次。为了避免这个陷阱，我们应该使用列表推导式来初始化列表。

```python
>>> a = [[]] * 3
>>> b = [[] for _ in range(3)]
>>> a[0].append("Hello")
>>> a
[['Hello'], ['Hello'], ['Hello']]
>>> b[0].append("Python")
>>> b
[['Python'], [], []]
```

### 2.5.3 复制

将列表赋值给一个变量是一个常见的陷阱。这种赋值并不会将列表复制给变量，而只是让变量指向该列表并增加其引用计数。

```python
import sys
>>> a = [1, 2, 3]
>>> sys.getrefcount(a)
2
>>> b = a
>>> sys.getrefcount(a)
3
>>> b[2] = 123456  # a[2] = 123456
>>> b
[1, 2, 123456]
>>> a
[1, 2, 123456]
```

复制有两种类型。第一种称为浅拷贝（非递归拷贝），第二种称为深拷贝（递归拷贝）。大多数情况下，使用浅拷贝复制列表就足够了。但是，如果列表是嵌套的，我们就必须使用深拷贝。

```python
>>> # 浅拷贝
>>> a = [1, 2]
>>> b = list(a)
>>> b[0] = 123
>>> a
[1, 2]
>>> b
[123, 2]
>>> a = [[1], [2]]
>>> b = list(a)
>>> b[0][0] = 123
>>> a
[[123], [2]]
>>> b
[[123], [2]]
>>> # 深拷贝
>>> import copy
>>> a = [[1], [2]]
>>> b = copy.deepcopy(a)
>>> b[0][0] = 123
>>> a
[[1], [2]]
>>> b
[[123], [2]]
```

### 2.5.4 使用切片

有时，我们的数据可能连接成一大段，比如数据包。在这种情况下，我们会使用切片对象作为表示变量来表示数据范围，而不是使用切片表达式。

```python
>>> icmp = (
...     b"080062988e2100005bff49c20005767c"
...     b"08090a0b0c0d0e0f1011121314151617"
...     b"18191a1b1c1d1e1f2021222324252627"
...     b"28292a2b2c2d2e2f3031323334353637"
... )
>>> head = slice(0, 32)
>>> data = slice(32, len(icmp))
>>> icmp[head]
b'080062988e2100005bff49c20005767c'
```

### 2.5.5 列表推导式

列表推导式在 PEP 202 中提出，它提供了一种基于另一个列表、序列或可迭代对象来创建新列表的优雅方式。此外，有时我们还可以用这种表达式来替代 `map` 和 `filter`。

```python
>>> [x for x in range(10)]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> [(lambda x: x**2)(i) for i in range(10)]
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
>>> [x for x in range(10) if x > 5]
[6, 7, 8, 9]
>>> [x if x > 5 else 0 for x in range(10)]
[0, 0, 0, 0, 0, 0, 6, 7, 8, 9]
>>> [x + 1 if x < 5 else x + 2 if x > 5 else x + 5 for x in range(10)]
[1, 2, 3, 4, 5, 10, 8, 9, 10, 11]
>>> [(x, y) for x in range(3) for y in range(2)]
[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
```

### 2.5.6 解包

有时，我们想将列表解包到变量中，以使代码更具可读性。在这种情况下，我们将 N 个元素赋值给 N 个变量，如下例所示。

```python
>>> arr = [1, 2, 3]
>>> a, b, c = arr
>>> a, b, c
(1, 2, 3)
```

根据 PEP 3132，在 Python 3 中，我们可以使用单个星号将 N 个元素解包到少于 N 个数量的变量中。

```python
>>> arr = [1, 2, 3, 4, 5]
>>> a, b, *c, d = arr
>>> a, b, d
(1, 2, 5)
>>> c
[3, 4]
```

### 2.5.7 使用 enumerate

`enumerate` 是一个内置函数。它帮助我们无需使用 `range(len(list))` 就能同时获取索引（或计数）和元素。更多信息可以参考循环技巧。

```python
>>> for i, v in enumerate(range(3)):
...     print(i, v)
...
0 0
1 1
2 2
>>> for i, v in enumerate(range(3), 1): # start = 1
...     print(i, v)
...
1 0
2 1
3 2
```

### 2.5.8 打包列表

`zip` 使我们能够一次迭代多个列表中的项。当任一列表耗尽时，迭代即停止。因此，迭代的长度与最短列表相同。如果这种行为不是我们想要的，我们可以使用 **Python 3** 中的 `itertools.zip_longest` 或 **Python 2** 中的 `itertools.izip_longest`。

```python
>>> a = [1, 2, 3]
>>> b = [4, 5, 6]
>>> list(zip(a, b))
[(1, 4), (2, 5), (3, 6)]
>>> c = [1]
>>> list(zip(a, b, c))
[(1, 4, 1)]
>>> from itertools import zip_longest
>>> list(zip_longest(a, b, c))
[(1, 4, 1), (2, 5, None), (3, 6, None)]
```

### 2.5.9 过滤项

`filter` 是一个内置函数，用于帮助我们移除不必要的项。在 **Python 2** 中，`filter` 返回一个列表。然而，在 **Python 3** 中，`filter` 返回一个 *可迭代对象*。请注意，*列表推导式* 或 *生成器表达式* 提供了一种更简洁的方式来移除项。

```python
>>> [x for x in range(5) if x > 1]
[2, 3, 4]
>>> l = ['1', '2', 3, 'Hello', 4]
>>> f = lambda x: isinstance(x, int)
>>> filter(f, l)
<filter object at 0x10bee2198>
>>> list(filter(f, l))
[3, 4]
>>> list((i for i in l if f(i)))
[3, 4]
```

### 2.5.10 栈

在 Python 中，不需要额外的栈数据结构，因为 `list` 提供了 `append` 和 `pop` 方法，使我们可以将列表用作栈。

```python
>>> stack = []
>>> stack.append(1)
>>> stack.append(2)
>>> stack.append(3)
>>> stack
[1, 2, 3]
>>> stack.pop()
3
>>> stack.pop()
2
>>> stack
[1]
```

### 2.5.11 `in` 操作

我们可以实现 `__contains__` 方法来让一个类支持 `in` 操作。对于程序员来说，这是为自定义类模拟成员测试操作的常用方式。

```python
class Stack:

    def __init__(self):
        self.__list = []

    def push(self, val):
        self.__list.append(val)

    def pop(self):
        return self.__list.pop()

    def __contains__(self, item):
        return True if item in self.__list else False

stack = Stack()
stack.push(1)
print(1 in stack)
print(0 in stack)
```

示例

```
python stack.py
True
False
```

### 2.5.12 访问项

让自定义类像列表一样执行获取和设置操作很简单。我们可以实现 `__getitem__` 方法和 `__setitem__` 方法，使类能够通过索引检索和覆盖数据。此外，如果想使用 `len` 函数计算元素数量，我们可以实现 `__len__` 方法。

```python
class Stack:

    def __init__(self):
        self.__list = []

    def push(self, val):
        self.__list.append(val)

    def pop(self):
        return self.__list.pop()

    def __repr__(self):
        return "{}".format(self.__list)

    def __len__(self):
        return len(self.__list)

    def __getitem__(self, idx):
        return self.__list[idx]

    def __setitem__(self, idx, val):
        self.__list[idx] = val

stack = Stack()
stack.push(1)
stack.push(2)
print("stack:", stack)

stack[0] = 3
print("stack:", stack)
print("num items:", len(stack))
```

示例

```
$ python stack.py
stack: [1, 2]
stack: [3, 2]
num items: 2
```

### 2.5.13 委托迭代

如果一个自定义容器类持有一个列表，并且我们希望迭代操作能作用于该容器，我们可以实现一个 `__iter__` 方法，将迭代委托给该列表。请注意，`__iter__` 方法应返回一个*迭代器对象*，因此我们不能直接返回列表本身；否则，Python 会抛出一个 `TypeError`。

```python
class Stack:

    def __init__(self):
        self.__list = []

    def push(self, val):
        self.__list.append(val)

    def pop(self):
        return self.__list.pop()

    def __iter__(self):
        return iter(self.__list)

stack = Stack()
stack.push(1)
stack.push(2)
for s in stack:
    print(s)
```

示例

```
$ python stack.py
1
2
```

### 2.5.14 排序

Python 列表提供了一个内置的 `list.sort` 方法，用于*原地*排序列表，无需使用额外内存。此外，`list.sort` 的返回值为 `None`，这是为了避免与 `sorted` 混淆，并且该函数只能用于列表。

```python
>>> l = [5, 4, 3, 2, 1]
>>> l.sort()
>>> l
[1, 2, 3, 4, 5]
>>> l.sort(reverse=True)
>>> l
[5, 4, 3, 2, 1]
```

`sorted` 函数不会原地修改任何可迭代对象。相反，它返回一个新的排序后的列表。如果列表的某些元素是只读的或不可变的，使用 `list.sort` 更安全。此外，`list.sort` 和 `sorted` 的另一个区别是，`sorted` 接受任何**可迭代对象**。

```python
>>> l = [5, 4, 3, 2, 1]
>>> new = sorted(l)
>>> new
[1, 2, 3, 4, 5]
>>> l
[5, 4, 3, 2, 1]
>>> d = {3: 'andy', 2: 'david', 1: 'amy'}
>>> sorted(d)  # 排序可迭代对象
[1, 2, 3]
```

要对元素为元组的列表进行排序，使用 `operator.itemgetter` 会很有帮助，因为它将一个键函数分配给排序的 key 参数。请注意，键必须是可比较的；否则会抛出 `TypeError`。

```python
>>> from operator import itemgetter
>>> l = [('andy', 10), ('david', 8), ('amy', 3)]
>>> l.sort(key=itemgetter(1))
>>> l
[('amy', 3), ('david', 8), ('andy', 10)]
```

`operator.itemgetter` 很有用，因为该函数返回一个 getter 方法，该方法可以应用于其他具有 `__getitem__` 方法的对象。例如，由于所有元素都有 `__getitem__`，可以使用 `operator.itemgetter` 对元素为字典的列表进行排序。

```python
>>> from pprint import pprint
>>> from operator import itemgetter
>>> l = [
...     {'name': 'andy', 'age': 10},
...     {'name': 'david', 'age': 8},
...     {'name': 'amy', 'age': 3},
... ]
>>> l.sort(key=itemgetter('age'))
>>> pprint(l)
[{'age': 3, 'name': 'amy'},
 {'age': 8, 'name': 'david'},
 {'age': 10, 'name': 'andy'}]
```

如果需要对一个元素既不可比较也没有 `__getitem__` 方法的列表进行排序，分配一个自定义键函数是可行的。

```python
>>> class Node(object):
...     def __init__(self, val):
...         self.val = val
...     def __repr__(self):
...         return f"Node({self.val})"
...
>>> nodes = [Node(3), Node(2), Node(1)]
>>> nodes.sort(key=lambda x: x.val)
>>> nodes
[Node(1), Node(2), Node(3)]
>>> nodes.sort(key=lambda x: x.val, reverse=True)
>>> nodes
[Node(3), Node(2), Node(1)]
```

上面的代码片段可以通过使用 `operator.attrgetter` 来简化。该函数返回一个基于属性名称的属性 getter。请注意，该属性必须是可比较的；否则，`sorted` 或 `list.sort` 将抛出 `TypeError`。

```python
>>> from operator import attrgetter
>>> class Node(object):
...     def __init__(self, val):
...         self.val = val
...     def __repr__(self):
...         return f"Node({self.val})"
...
>>> nodes = [Node(3), Node(2), Node(1)]
>>> nodes.sort(key=attrgetter('val'))
>>> nodes
[Node(1), Node(2), Node(3)]
```

如果一个对象有 `__lt__` 方法，这意味着该对象是可比较的，`sorted` 或 `list.sort` 不需要为其 key 参数输入键函数。一个列表或可迭代序列可以直接排序。

```python
>>> class Node(object):
...     def __init__(self, val):
...         self.val = val
...     def __repr__(self):
...         return f"Node({self.val})"
...     def __lt__(self, other):
...         return self.val - other.val < 0
...
>>> nodes = [Node(3), Node(2), Node(1)]
>>> nodes.sort()
>>> nodes
[Node(1), Node(2), Node(3)]
```

如果一个对象没有 `__lt__` 方法，很可能在对象类声明后为该对象打补丁。换句话说，打补丁之后，该对象就变得可比较了。

```python
>>> class Node(object):
...     def __init__(self, val):
...         self.val = val
...     def __repr__(self):
...         return f"Node({self.val})"
...
>>> Node.__lt__ = lambda s, o: s.val < o.val
>>> nodes = [Node(3), Node(2), Node(1)]
>>> nodes.sort()
>>> nodes
[Node(1), Node(2), Node(3)]
```

请注意，Python3 中的 `sorted` 或 `list.sort` 不支持 `cmp` 参数，该参数在 Python2 中是**唯一**有效的参数。如果需要使用旧的比较函数，例如一些遗留代码，`functools.cmp_to_key` 很有用，因为它将比较函数转换为键函数。

```python
>>> from functools import cmp_to_key
>>> class Node(object):
...     def __init__(self, val):
...         self.val = val
...     def __repr__(self):
...         return f"Node({self.val})"
...
>>> nodes = [Node(3), Node(2), Node(1)]
>>> nodes.sort(key=cmp_to_key(lambda x,y: x.val - y.val))
>>> nodes
[Node(1), Node(2), Node(3)]
```

### 2.5.15 已排序列表

```python
import bisect

class Foo(object):
    def __init__(self, k):
        self.k = k

    def __eq__(self, rhs):
        return self.k == rhs.k

    def __ne__(self, rhs):
        return self.k != rhs.k

    def __lt__(self, rhs):
        return self.k < rhs.k

    def __gt__(self, rhs):
        return self.k > rhs.k

    def __le__(self, rhs):
        return self.k <= rhs.k

    def __ge__(self, rhs):
        return self.k >= rhs.k

    def __repr__(self):
        return f"Foo({self.k})"

    def __str__(self):
        return self.__repr__()

foo = [Foo(1), Foo(3), Foo(2), Foo(0)]
bar = []
for x in foo:
    bisect.insort(bar, x)

print(bar) # [Foo(0), Foo(1), Foo(2), Foo(3)]
```

### 2.5.16 新建列表

```python
#### 新建一个大小为 3 的列表

>>> [0] * 3
[0, 0, 0]

#### 新建一个大小为 3x3 的二维列表

>>> [[0] * 3 for _ in range(3)]
[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
```

请注意，我们应避免通过以下代码片段创建多维列表，因为列表中的所有对象都指向相同的地址。

```python
>>> a = [[0] * 3] * 3
>>> a
[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
>>> a[1][1] = 2
>>> a
[[0, 2, 0], [0, 2, 0], [0, 2, 0]]
```

### 2.5.17 循环缓冲区

```python
>>> from collections import deque
>>> d = deque(maxlen=8)
>>> for x in range(9):
...     d.append(x)
...
>>> d
deque([1, 2, 3, 4, 5, 6, 7, 8], maxlen=8)
```

```python
>>> from collections import deque
>>> def tail(path, n=10):
...     with open(path) as f:
...         return deque(f, n)
...
>>> tail("/etc/hosts")
```

### 2.5.18 分块

```python
>>> def chunk(lst, n):
...     for i in range(0, len(lst), n):
...         yield lst[i:i+n]
...
>>> a = [1, 2, 3, 4, 5, 6, 7, 8]
>>> list(chunk(a, 3))
[[1, 2, 3], [4, 5, 6], [7, 8]]
```

### 2.5.19 分组

```python
>>> import itertools
>>> s = "AAABBCCCCC"
>>> for k, v in itertools.groupby(s):
...     print(k, list(v))
...
A ['A', 'A', 'A']
B ['B', 'B']
C ['C', 'C', 'C', 'C', 'C']

#### 按键分组

>>> x = [('gp1', 'a'), ('gp2', 'b'), ('gp2', 'c')]
>>> for k, v in itertools.groupby(x, lambda x: x[0]):
...     print(k, list(v))
...
gp1 [('gp1', 'a')]
gp2 [('gp2', 'b'), ('gp2', 'c')]
```

### 2.5.20 二分查找

```python
>>> def binary_search(arr, x, lo=0, hi=None):
...     if not hi: hi = len(arr)
...     pos = bisect_left(arr, x, lo, hi)
...     return pos if pos != hi and arr[pos] == x else -1
...
>>> a = [1, 1, 1, 2, 3]
>>> binary_search(a, 1)
0
>>> binary_search(a, 2)
3
```

### 2.5.21 下界

```python
>>> import bisect
>>> a = [1,2,3,3,4,5]
>>> bisect.bisect_left(a, 3)
2
>>> bisect.bisect_left(a, 3.5)
4
```## 2.5.22 上界

```python
>>> import bisect
>>> a = [1,2,3,3,4,5]
>>> bisect.bisect_right(a, 3)
4
>>> bisect.bisect_right(a, 3.5)
4
```

### 2.5.23 字典序排列

```python
#### python compare lists lexicographically

>>> a = [(1,2), (1,1), (1,0), (2,1)]
>>> a.sort()
>>> a
[(1, 0), (1, 1), (1, 2), (2, 1)]
```

### 2.5.24 前缀树

```python
>>> from functools import reduce
>>> from collections import defaultdict
>>> Trie = lambda: defaultdict(Trie)
>>> prefixes = ['abc', 'de', 'g']
>>> trie = Trie()
>>> end = True
>>> for p in prefixes:
...     reduce(dict.__getitem__, p, trie)[end] = p
...

#### search prefix

>>> def find(trie, word):
...     curr = trie
...     for c in word:
...         if c not in curr:
...             return False
...         curr = curr[c]
...     return True
...
>>> find(trie, "abcdef")
False
>>> find(trie, "abc")
True
>>> find(trie, "ab")
True

#### search word

>>> def find(trie, p):
...     curr = trie
...     for c in p:
...         if c not in curr or True in curr:
...             break
...         curr = curr[c]
...     return True if True in curr else False
...
>>> find(trie, "abcdef")
True
>>> find(trie, "abc")
True
>>> find(trie, "ab")
False
```

## 2.6 集合

### 目录

- 集合
- 集合推导式
- 列表去重
- 两个集合的并集
- 向集合添加元素
- 两个集合的交集
- 集合中的共同元素
- 包含关系
- 集合差集
- 对称差集

### 2.6.1 集合推导式

```python
>>> a = [1, 2, 5, 6, 6, 6, 7]
>>> s = {x for x in a}
>>> s
set([1, 2, 5, 6, 7])
>>> s = {x for x in a if x > 3}
>>> s
set([5, 6, 7])
>>> s = {x if x > 3 else -1 for x in a}
>>> s
set([6, 5, -1, 7])
```

### 2.6.2 列表去重

```python
>>> a = [1, 2, 2, 2, 3, 4, 5, 5]
>>> a
[1, 2, 2, 2, 3, 4, 5, 5]
>>> ua = list(set(a))
>>> ua
[1, 2, 3, 4, 5]
```

### 2.6.3 两个集合的并集

```python
>>> a = set([1, 2, 2, 2, 3])
>>> b = set([5, 5, 6, 6, 7])
>>> a | b
set([1, 2, 3, 5, 6, 7])
>>> # 或
>>> a = [1, 2, 2, 2, 3]
>>> b = [5, 5, 6, 6, 7]
>>> set(a + b)
set([1, 2, 3, 5, 6, 7])
```

### 2.6.4 向集合添加元素

```python
>>> a = set([1, 2, 3, 3, 3])
>>> a.add(5)
>>> a
set([1, 2, 3, 5])
>>> # 或
>>> a = set([1, 2, 3, 3, 3])
>>> a |= set([1, 2, 3, 4, 5, 6])
>>> a
set([1, 2, 3, 4, 5, 6])
```

### 2.6.5 两个集合的交集

```python
>>> a = set([1, 2, 2, 2, 3])
>>> b = set([1, 5, 5, 6, 6, 7])
>>> a & b
set([1])
```

### 2.6.6 集合中的共同元素

```python
>>> a = [1, 1, 2, 3]
>>> b = [1, 3, 5, 5, 6, 6]
>>> com = list(set(a) & set(b))
>>> com
[1, 3]
```

### 2.6.7 包含关系

b 包含 a

```python
>>> a = set([1, 2])
>>> b = set([1, 2, 5, 6])
>>> a <= b
True
```

a 包含 b

```python
>>> a = set([1, 2, 5, 6])
>>> b = set([1, 5, 6])
>>> a >= b
True
```

### 2.6.8 集合差集

```python
>>> a = set([1, 2, 3])
>>> b = set([1, 5, 6, 7, 7])
>>> a - b
set([2, 3])
```

### 2.6.9 对称差集

```python
>>> a = set([1,2,3])
>>> b = set([1, 5, 6, 7, 7])
>>> a ^ b
set([2, 3, 5, 6, 7])
```

## 2.7 字典

### 目录

- 获取所有键
- 获取键和值
- 查找相同的键
- 设置默认值
- 更新字典
- 合并两个字典
- 模拟字典
- LRU 缓存

### 2.7.1 获取所有键

```python
>>> a = {"1":1, "2":2, "3":3}
>>> b = {"2":2, "3":3, "4":4}
>>> a.keys()
['1', '3', '2']
```

### 2.7.2 获取键和值

```python
>>> a = {"1":1, "2":2, "3":3}
>>> a.items()
```

### 2.7.3 查找相同的键

```python
>>> a = {"1":1, "2":2, "3":3}
>>> b = {"2":2, "3":3, "4":4}
>>> [_ for _ in a.keys() if _ in b.keys()]
['3', '2']
>>> # 更好的方式
>>> c = set(a).intersection(set(b))
>>> list(c)
['3', '2']
>>> # 或
>>> [_ for _ in a if _ in b]
['3', '2']
[('1', 1), ('3', 3), ('2', 2)]
```

### 2.7.4 设置默认值

```python
#### 直观但不推荐
>>> d = {}
>>> key = "foo"
>>> if key not in d:
...     d[key] = []
...

#### 使用 d.setdefault(key[, default])
>>> d = {}
>>> key = "foo"
>>> d.setdefault(key, [])
[]
>>> d[key] = 'bar'
>>> d
{'foo': 'bar'}

#### 使用 collections.defaultdict
>>> from collections import defaultdict
>>> d = defaultdict(list)
>>> d["key"]
[]
>>> d["foo"]
[]
>>> d["foo"].append("bar")
>>> d
defaultdict(<class 'list'>, {'key': [], 'foo': ['bar']})
```

dict.setdefault(key[, default]) 如果键不在字典中，则返回其默认值。然而，如果键存在于字典中，该函数将返回其对应的值。

```python
>>> d = {}
>>> d.setdefault("key", [])
[]
>>> d["key"] = "bar"
>>> d.setdefault("key", [])
'bar'
```

### 2.7.5 更新字典

```python
>>> a = {"1":1, "2":2, "3":3}
>>> b = {"2":2, "3":3, "4":4}
>>> a.update(b)
>>> a
{'1': 1, '3': 3, '2': 2, '4': 4}
```

### 2.7.6 合并两个字典

Python 3.4 或更低版本

```python
>>> a = {"x": 55, "y": 66}
>>> b = {"a": "foo", "b": "bar"}
>>> c = a.copy()
>>> c.update(b)
>>> c
{'y': 66, 'x': 55, 'b': 'bar', 'a': 'foo'}
```

Python 3.5 或更高版本

```python
>>> a = {"x": 55, "y": 66}
>>> b = {"a": "foo", "b": "bar"}
>>> c = {**a, **b}
>>> c
{'x': 55, 'y': 66, 'a': 'foo', 'b': 'bar'}
```

### 2.7.7 模拟字典

```python
>>> class EmuDict(object):
...     def __init__(self, dict_):
...         self._dict = dict_
...     def __repr__(self):
...         return "EmuDict: " + repr(self._dict)
...     def __getitem__(self, key):
...         return self._dict[key]
...     def __setitem__(self, key, val):
...         self._dict[key] = val
...     def __delitem__(self, key):
...         del self._dict[key]
...     def __contains__(self, key):
...         return key in self._dict
...     def __iter__(self):
...         return iter(self._dict.keys())
...
>>> _ = {"1":1, "2":2, "3":3}
>>> emud = EmuDict(_)
>>> emud  # __repr__
EmuDict: {'1': 1, '2': 2, '3': 3}
>>> emud['1']  # __getitem__
1
>>> emud['5'] = 5  # __setitem__
>>> emud
EmuDict: {'1': 1, '2': 2, '3': 3, '5': 5}
>>> del emud['2']  # __delitem__
>>> emud
EmuDict: {'1': 1, '3': 3, '5': 5}
>>> for _ in emud:
...     print(emud[_], end=' ')  # __iter__
... else:
...     print()
...
1 3 5
>>> '1' in emud  # __contains__
True
```

### 2.7.8 LRU 缓存

```python
from collections import OrderedDict

class LRU(object):
    def __init__(self, maxsize=128):
        self._maxsize = maxsize
        self._cache = OrderedDict()

    def get(self, k):
        if k not in self._cache:
            return None

        self._cache.move_to_end(k)
        return self._cache[k]

    def put(self, k, v):
        if k in self._cache:
            self._cache.move_to_end(k)
        self._cache[k] = v
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def __str__(self):
        return str(self._cache)

    def __repr__(self):
        return self.__str__()
```

注意，从 Python 3.7 开始，字典保留了插入顺序。此外，更新键不会影响顺序。因此，字典也可以模拟 LRU 缓存，这与使用 OrderedDict 类似。

```python
class LRU(object):
    def __init__(self, maxsize=128):
        self._maxsize = maxsize
        self._cache = {}

    def get(self, k):
        if k not in self._cache:
            return None

        self.move_to_end(k)
        return self._cache[k]

    def put(self, k, v):
        if k in self._cache:
            self.move_to_end(k)
        self._cache[k] = v
        if len(self._cache) > self._maxsize:
            self.pop()

    def pop(self):
        it = iter(self._cache.keys())
        del self._cache[next(it)]

    def move_to_end(self, k):
        if k not in self._cache:
            return
        v = self._cache[k]
        del self._cache[k]
        self._cache[k] = v

    def __str__(self):
        return str(self._cache)

    def __repr__(self):
        return self.__str__()
```

## 2.8 堆

### 目录

- 堆
- 堆排序
- 优先队列

### 2.8.1 堆排序

```python
>>> import heapq
>>> a = [5, 1, 3, 2, 6]
>>> h = []
>>> for x in a:
...     heapq.heappush(h, x)
...
>>> x = [heapq.heappop(h) for _ in range(len(a))]
>>> x
[1, 2, 3, 5, 6]
```

### 2.8.2 优先队列

```python
import heapq

h = []
heapq.heappush(h, (1, "1")) # (priority, value)
heapq.heappush(h, (5, "5"))
heapq.heappush(h, (3, "3"))
heapq.heappush(h, (2, "2"))
x = [heapq.heappop(h) for _ in range(len(h))]
print(x)
```

```python
import heapq

class Num(object):
    def __init__(self, n):
        self._n = n

    def __lt__(self, other):
        return self._n < other._n

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Num({self._n})"

h = []
heapq.heappush(h, Num(5))
heapq.heappush(h, Num(2))
heapq.heappush(h, Num(1))
x = [heapq.heappop(h) for _ in range(len(h))]
print(x)
#### output: [Num(1), Num(2), Num(5)]
```

## 2.9 函数

函数可以帮助程序员将其逻辑封装成一个任务，以避免重复代码。在Python中，函数的定义非常灵活，我们可以使用许多特性，如装饰器、注解、文档字符串、默认参数等来定义函数。本速查表收集了许多定义函数的方法，并揭开了一些函数语法的神秘面纱。

### 目录

* 函数
  - 文档函数
  - 默认参数
  - 可选参数
  - 解包参数
  - 仅关键字参数
  - 注解
  - 可调用对象
  - 获取函数名
  - 匿名函数
  - 生成器
  - 装饰器
  - 带参数的装饰器
  - 缓存

### 2.9.1 文档函数

文档为程序员提供关于函数应如何使用的提示。文档字符串提供了一种编写可读函数文档的便捷方式。PEP 257 定义了一些文档字符串的约定。为了避免违反约定，有多种工具，如 doctest 或 pydocstyle，可以帮助我们检查文档字符串的格式。

```python
>>> def example():
...     """This is an example function."""
...     print("Example function")
...
>>> example.__doc__
'This is an example function.'
>>> help(example)
```

### 2.9.2 默认参数

在Python中定义一个参数可选且具有默认值的函数非常简单。我们只需在定义中赋值，并确保默认参数出现在末尾即可。

```python
>>> def add(a, b=0):
...     return a + b
...
>>> add(1)
1
>>> add(1, 2)
3
>>> add(1, b=2)
3
```

### 2.9.3 可选参数

```python
>>> def example(a, b=None, *args, **kwargs):
...     print(a, b)
...     print(args)
...     print(kwargs)
...
>>> example(1, "var", 2, 3, word="hello")
1 var
(2, 3)
{'word': 'hello'}
```

### 2.9.4 解包参数

```python
>>> def foo(a, b, c='BAZ'):
...     print(a, b, c)
...
>>> foo(*("FOO", "BAR"), **{"c": "baz"})
FOO BAR baz
```

### 2.9.5 仅关键字参数

**Python 3.0 新增**

```python
>>> def f(a, b, *, kw):
...     print(a, b, kw)
...
>>> f(1, 2, kw=3)
1 2 3
>>> f(1, 2, 3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: f() takes 2 positional arguments but 3 were given
```

### 2.9.6 注解

**Python 3.0 新增**

注解是向程序员提示参数类型的一种有用方式。此特性的规范在 PEP 3107 中。Python 3.5 引入了 typing 模块来扩展类型提示的概念。此外，从 3.6 版本开始，Python 开始提供一种通用的方式来定义带有注解的变量。更多信息可在 PEP 483、PEP 484 和 PEP 526 中找到。

```python
>>> def fib(n: int) -> int:
...     a, b = 0, 1
...     for _ in range(n):
...         b, a = a + b, b
...     return a
...
>>> fib(10)
55
>>> fib.__annotations__
{'n': <class 'int'>, 'return': <class 'int'>}
```

### 2.9.7 可调用对象

在某些情况下，例如传递回调函数时，我们需要检查一个对象是否是可调用的。内置函数 `callable` 可以帮助我们，避免在对象不可调用时引发 `TypeError`。

```python
>>> a = 10
>>> def fun():
...     print('I am callable')
...
>>> callable(a)
False
>>> callable(fun)
True
```

### 2.9.8 获取函数名

```python
>>> def example_function():
...     pass
...
>>> example_function.__name__
'example_function'
```

### 2.9.9 匿名函数

有时，我们不想使用 `def` 语句来定义一个简短的回调函数。我们可以使用 `lambda` 表达式作为快捷方式来定义匿名或内联函数。然而，`lambda` 中只能指定一个单一表达式。也就是说，不能包含其他特性，如多行语句、条件判断或异常处理。

```python
>>> fn = lambda x: x**2
>>> fn(3)
9
>>> (lambda x: x**2)(3)
9
>>> (lambda x: [x*_ for _ in range(5)])(2)
[0, 2, 4, 6, 8]
>>> (lambda x: x if x>3 else 3)(5)
5
```

### 2.9.10 生成器

```python
>>> def fib(n):
...     a, b = 0, 1
...     for _ in range(n):
...         yield a
...         b, a = a + b, b
...
>>> [f for f in fib(10)]
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### 2.9.11 装饰器

Python 2.4 新增

- PEP 318 - 为函数和方法添加装饰器

```python
>>> from functools import wraps
>>> def decorator(func):
...     @wraps(func)
...     def wrapper(*args, **kwargs):
...         print("Before calling {}".format(func.__name__))
...         ret = func(*args, **kwargs)
...         print("After calling {}".format(func.__name__))
...         return ret
...     return wrapper
...
>>> @decorator
... def example():
...     print("Inside example function.")
...
>>> example()
Before calling example.
Inside example function.
After calling example.
```

等同于

```python
>>> def example():
...     print("Inside example function.")
...
>>> example = decorator(example)
>>> example()
Before calling example.
Inside example function.
After calling example.
```

### 2.9.12 带参数的装饰器

```python
>>> from functools import wraps
>>> def decorator_with_argument(val):
...     def decorator(func):
...         @wraps(func)
...         def wrapper(*args, **kwargs):
...             print("Val is {0}".format(val))
...             return func(*args, **kwargs)
...         return wrapper
...     return decorator
...
>>> @decorator_with_argument(10)
... def example():
...     print("This is example function.")
...
>>> example()
Val is 10
This is example function.
```

等同于

```python
>>> def example():
...     print("This is example function.")
...
>>> example = decorator_with_argument(10)(example)
>>> example()
Val is 10
This is example function.
```

### 2.9.13 缓存

Python 3.2 新增

无缓存

```python
>>> import time
>>> def fib(n):
...     if n < 2:
...         return n
...     return fib(n - 1) + fib(n - 2)
...
>>> s = time.time(); _ = fib(32); e = time.time(); e - s
1.1562161445617676
```

有缓存（动态规划）

```python
>>> from functools import lru_cache
>>> @lru_cache(maxsize=None)
... def fib(n):
...     if n < 2:
...         return n
...     return fib(n - 1) + fib(n - 2)
...
>>> s = time.time(); _ = fib(32); e = time.time(); e - s
2.9087066650390625e-05
>>> fib.cache_info()
CacheInfo(hits=30, misses=33, maxsize=None, currsize=33)
```

## 2.10 类与对象

### 目录

* 类与对象
    - 列出属性
    - 获取实例类型
    - 声明一个类
    - 检查/获取/设置属性
    - 检查继承关系
    - 获取类名
    - __new__ 和 __init__
    - 菱形继承问题
    - 类的表示
    - 可调用对象
    - 上下文管理器
    - 使用 contextlib
    - 属性
    - 计算属性
    - 描述符
    - 单例装饰器
    - 静态方法和类方法
    - 抽象方法
    - 使用 __slot__ 节省内存
    - 常用魔法方法

### 2.10.1 列出属性

```python
>>> dir(list)  # check all attr of list
['__add__', '__class__', ...]
```

### 2.10.2 获取实例类型

```python
>>> ex = 10
>>> isinstance(ex, int)
True
```

### 2.10.3 声明一个类

```python
>>> def fib(self, n):
...     if n <= 2:
...         return 1
...     return fib(self, n-1) + fib(self, n-2)
...
>>> Fib = type('Fib', (object,), {'val': 10,
...                                'fib': fib})
>>> f = Fib()
>>> f.val
10
>>> f.fib(f.val)
55
```

等同于

```python
>>> class Fib(object):
...     val = 10
...     def fib(self, n):
...         if n <=2:
...             return 1
...         return self.fib(n-1)+self.fib(n-2)
...
>>> f = Fib()
>>> f.val
10
>>> f.fib(f.val)
55
```

### 2.10.4 Has / Get / Set 属性

```python
>>> class Example(object):
...     def __init__(self):
...         self.name = "ex"
...     def printex(self):
...         print("This is an example")
...
>>> ex = Example()
>>> hasattr(ex,"name")
True
>>> hasattr(ex,"printex")
True
>>> hasattr(ex,"print")
False
>>> getattr(ex,'name')
'ex'
>>> setattr(ex,'name','example')
>>> ex.name
'example'
```

### 2.10.5 检查继承

```python
>>> class Example(object):
...     def __init__(self):
...         self.name = "ex"
...     def printex(self):
...         print("This is an Example")
...
>>> issubclass(Example, object)
True
```

### 2.10.6 获取类名

```python
>>> class ExampleClass(object):
...     pass
...
>>> ex = ExampleClass()
>>> ex.__class__.__name__
'ExampleClass'
```

### 2.10.7 New 和 Init

`__init__` 会被调用

```python
>>> class ClassA(object):
...     def __new__(cls, arg):
...         print('__new__ ' + arg)
...         return object.__new__(cls, arg)
...     def __init__(self, arg):
...         print('__init__ ' + arg)
...
>>> o = ClassA("Hello")
__new__ Hello
__init__ Hello
```

`__init__` 不会被调用

```python
>>> class ClassB(object):
...     def __new__(cls, arg):
...         print('__new__ ' + arg)
...         return object
...     def __init__(self, arg):
...         print('__init__ ' + arg)
...
>>> o = ClassB("Hello")
__new__ Hello
```

### 2.10.8 菱形问题

多重继承中方法搜索路径的问题

```python
>>> def foo_a(self):
...     print("This is ClsA")
...
>>> def foo_b(self):
...     print("This is ClsB")
...
>>> def foo_c(self):
...     print("This is ClsC")
...
>>> class Type(type):
...     def __repr__(cls):
...         return cls.__name__
...
>>> ClsA = Type("ClsA", (object,), {'foo': foo_a})
>>> ClsB = Type("ClsB", (ClsA,), {'foo': foo_b})
>>> ClsC = Type("ClsC", (ClsA,), {'foo': foo_c})
>>> ClsD = Type("ClsD", (ClsB, ClsC), {})
>>> ClsD.mro()
[ClsD, ClsB, ClsC, ClsA, <type 'object'>]
>>> ClsD().foo()
This is ClsB
```

### 2.10.9 类的表示

```python
>>> class Example(object):
...     def __str__(self):
...         return "Example __str__"
...     def __repr__(self):
...         return "Example __repr__"
...
>>> print(str(Example()))
Example __str__
>>> Example()
Example __repr__
```

### 2.10.10 可调用对象

```python
>>> class CallableObject(object):
...     def example(self, *args, **kwargs):
...         print("I am callable!")
...     def __call__(self, *args, **kwargs):
...         self.example(*args, **kwargs)
...
>>> ex = CallableObject()
>>> ex()
I am callable!
```

### 2.10.11 上下文管理器

```python
#### 替代 try: ... finally: ...
#### 参见: PEP343
#### 常用于打开和关闭操作

import socket

class Socket(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def __enter__(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self.host,self.port))
        sock.listen(5)
        self.sock = sock
        return self.sock

    def __exit__(self,*exc_info):
        if exc_info[0] is not None:
            import traceback
            traceback.print_exception(*exc_info)
        self.sock.close()

if __name__=="__main__":
    host = 'localhost'
    port = 5566
    with Socket(host, port) as s:
        while True:
            conn, addr = s.accept()
            msg = conn.recv(1024)
            print(msg)
            conn.send(msg)
            conn.close()
```

### 2.10.12 使用 contextlib

```python
from contextlib import contextmanager

@contextmanager
def opening(filename, mode='r'):
    f = open(filename, mode)
    try:
        yield f
    finally:
        f.close()

with opening('example.txt') as fd:
    fd.read()
```

### 2.10.13 属性 (Property)

```python
>>> class Example(object):
...     def __init__(self, value):
...         self._val = value
...     @property
...     def val(self):
...         return self._val
...     @val.setter
...     def val(self, value):
...         if not isinstance(value, int):
...             raise TypeError("Expected int")
...         self._val = value
...     @val.deleter
...     def val(self):
...         del self._val
...
>>> ex = Example(123)
>>> ex.val = "str"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "test.py", line 12, in val
    raise TypeError("Expected int")
TypeError: Expected int
```

等同于

```python
>>> class Example(object):
...     def __init__(self, value):
...         self._val = value
...
...     def _val_getter(self):
...         return self._val
...
...     def _val_setter(self, value):
...         if not isinstance(value, int):
...             raise TypeError("Expected int")
...         self._val = value
...
...     def _val_deleter(self):
...         del self._val
...
...     val = property(fget=_val_getter, fset=_val_setter, fdel=_val_deleter, doc=None)
...
```

### 2.10.14 计算属性

`@property` 仅在需要时计算属性的值，不会预先存储在内存中。

```python
>>> class Example(object):
...     @property
...     def square3(self):
...         return 2**3
...
>>> ex = Example()
>>> ex.square3
8
```

### 2.10.15 描述符

```python
>>> class Integer(object):
...     def __init__(self, name):
...         self._name = name
...     def __get__(self, inst, cls):
...         if inst is None:
...             return self
...         else:
...             return inst.__dict__[self._name]
...     def __set__(self, inst, value):
...         if not isinstance(value, int):
...             raise TypeError("Expected int")
...         inst.__dict__[self._name] = value
...     def __delete__(self,inst):
...         del inst.__dict__[self._name]
...
>>> class Example(object):
...     x = Integer('x')
...     def __init__(self, val):
...         self.x = val
...
>>> ex1 = Example(1)
>>> ex1.x
1
>>> ex2 = Example("str")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 4, in __init__
  File "<stdin>", line 11, in __set__
TypeError: Expected an int
>>> ex3 = Example(3)
>>> hasattr(ex3, 'x')
True
>>> del ex3.x
>>> hasattr(ex3, 'x')
False
```

### 2.10.16 单例装饰器

单例是一种设计模式，它限制了类的实例化过程，确保该类只创建一个实例。

```python
#!/usr/bin/env python3
"""Singleton decorator class."""

class Singleton(object):

    def __init__(self, cls):
        self.__cls = cls
        self.__obj = None

    def __call__(self, *args, **kwargs):
        if not self.__obj:
            self.__obj = self.__cls(*args, **kwargs)
        return self.__obj

if __name__ == "__main__":
    # 测试 ...

    @Singleton
    class Test(object):

        def __init__(self, text):
            self.text = text

    a = Test("Hello")
    b = Test("World")

    print("id(a):", id(a), "id(b):", id(b), "Diff:", id(a)-id(b))
```

### 2.10.17 静态方法和类方法

`@classmethod` 绑定到类。`@staticmethod` 类似于一个 Python 函数，但定义在类中。

```python
>>> class example(object):
...     @classmethod
...     def clsmethod(cls):
...         print("I am classmethod")
...     @staticmethod
...     def stmethod():
...         print("I am staticmethod")
...     def instmethod(self):
...         print("I am instancemethod")
...
>>> ex = example()
>>> ex.clsmethod()
I am classmethod
>>> ex.stmethod()
I am staticmethod
>>> ex.instmethod()
I am instancemethod
>>> example.clsmethod()
I am classmethod
>>> example.stmethod()
I am staticmethod
>>> example.instmethod()
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
TypeError: unbound method instmethod() ...
```

### 2.10.18 抽象方法

abc 用于定义方法但不实现它

```python
>>> from abc import ABCMeta, abstractmethod
>>> class base(object):
...     __metaclass__ = ABCMeta
...     @abstractmethod
...     def absmethod(self):
...         """ Abstract method """
...
>>> class example(base):
...     def absmethod(self):
```

# Python速查文档，版本 0.1.0

```python
print("abstract")
...
>>> ex = example()
>>> ex.absmethod()
abstract
```

另一种常见方式是抛出NotImplementedError

```python
>>> class base(object):
...     def absmethod(self):
...         raise NotImplementedError
...
>>> class example(base):
...     def absmethod(self):
...         print("abstract")
...
>>> ex = example()
>>> ex.absmethod()
abstract
```

### 2.10.19 使用`slot`节省内存

```python
#!/usr/bin/env python3

import resource
import platform
import functools


def profile_mem(func):
    @functools.wraps(func)
    def wrapper(*a, **k):
        s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ret = func(*a, **k)
        e = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        uname = platform.system()
        if uname == "Linux":
            print(f"mem usage: {e - s} kByte")
        elif uname == "Darwin":
            print(f"mem usage: {e - s} Byte")
        else:
            raise Exception("not support")
        return ret
    return wrapper


class S(object):
    __slots__ = ['attr1', 'attr2', 'attr3']

    def __init__(self):
        self.attr1 = "Foo"
        self.attr2 = "Bar"
        self.attr3 = "Baz"

class D(object):
    def __init__(self):
        self.attr1 = "Foo"
        self.attr2 = "Bar"
        self.attr3 = "Baz"

@profile_mem
def alloc(cls):
    _ = [cls() for _ in range(1000000)]

alloc(S)
alloc(D)
```

输出：

```bash
$ python3.6 s.py
mem usage: 70922240 Byte
mem usage: 100659200 Byte
```

### 2.10.20 常用魔术方法

```python
#### 参见Python文档：数据模型
#### 命令行类相关
__main__
__name__
__file__
__module__
__all__
__dict__
__class__
__doc__
__init__(self, [...])
__str__(self)
__repr__(self)
__del__(self)

#### 描述符相关
__get__(self, instance, owner)
__set__(self, instance, value)
__delete__(self, instance)

#### 上下文管理器相关
__enter__(self)
__exit__(self, exc_ty, exc_val, tb)

#### 模拟容器类型
__len__(self)
__getitem__(self, key)
__setitem__(self, key, value)
__delitem__(self, key)
__iter__(self)
__contains__(self, value)

#### 控制属性访问
__getattr__(self, name)
__setattr__(self, name, value)
__delattr__(self, name)
__getattribute__(self, name)

#### 可调用对象
__call__(self, [args...])

#### 比较相关
__cmp__(self, other)
__eq__(self, other)
__ne__(self, other)
__lt__(self, other)
__gt__(self, other)
__le__(self, other)
__ge__(self, other)

#### 算术运算相关
__add__(self, other)
__sub__(self, other)
__mul__(self, other)
__div__(self, other)
__mod__(self, other)
__and__(self, other)
__or__(self, other)
__xor__(self, other)
```

## 2.11 生成器

- 生成器
- 生成器术语表
- 通过生成器生成值
- 解包生成器
- 通过生成器实现可迭代对象
- 向生成器发送消息
- yield from 表达式
- yield (from) EXPR 返回 RES
- 生成序列
- `RES = yield from EXP` 实际做了什么？
- `for _ in gen()` 模拟 yield from
- 检查生成器类型
- 检查生成器状态
- 简易编译器
- 上下文管理器与生成器
- `@contextmanager` 实际做了什么？
- 分析代码块
- `yield from` 与 `__iter__`
- `yield from` == `await` 表达式
- Python中的闭包 - 使用生成器
- 实现简易调度器
- 简单阻塞轮询
- 简单阻塞与非阻塞轮询
- 异步生成器
- 异步生成器可以包含 `try..finally` 块
- 向异步生成器发送值并抛出异常
- 简单异步轮询
- 异步生成器性能优于异步迭代器
- 异步推导式

### 2.11.1 生成器术语表

```python
#### 生成器函数

>>> def gen_func():
...     yield 5566
...
>>> gen_func
<function gen_func at 0x1019273a>

#### 生成器
#### 调用生成器函数会返回一个生成器

>>> g = gen_func()
>>> g
<generator object gen_func at 0x101238fd>
>>> next(g)
5566
>>> next(g)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration

#### 生成器表达式
#### 生成器表达式直接求值为生成器

>>> g = (x for x in range(2))
>>> g
<generator object <genexpr> at 0x10a9c191>
>>> next(g)
0
>>> next(g)
1
>>> next(g)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

### 2.11.2 通过生成器生成值

```python
>>> from __future__ import print_function
>>> def prime(n):
...     p = 2
...     while n > 0:
...         for x in range(2, p):
...             if p % x == 0:
...                 break
...         else:
...             yield p
...             n -= 1
...         p += 1
...
>>> p = prime(3)
>>> next(p)
2
>>> next(p)
3
>>> next(p)
5
>>> next(p)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
>>> for x in prime(5):
...     print(x, end=" ")
...
2 3 5 7 11 >>>
```

### 2.11.3 解包生成器

```python
#### PEP 448

#### 在列表内解包

>>> g1 = (x for x in range(3))
>>> g2 = (x**2 for x in range(2))
>>> [1, *g1, 2, *g2]
[1, 0, 1, 2, 2, 0, 1]
>>> # 等同于
>>> g1 = (x for x in range(3))
>>> g2 = (x**2 for x in range(2))
>>> [1] + list(g1) + [2] + list(g2)
[1, 0, 1, 2, 2, 0, 1]

#### 在集合内解包

>>> g = (x for x in [5, 5, 6, 6])
>>> {*g}
{5, 6}

#### 解包赋值给变量

>>> g = (x for x in range(3))
>>> a, b, c = g
>>> print(a, b, c)
0 1 2
>>> g = (x for x in range(6))
>>> a, b, *c, d = g
>>> print(a, b, d)
0 1 5
>>> print(c)
[2, 3, 4]

#### 在函数调用中解包

>>> print(*(x for x in range(3)))
0 1 2
```

### 2.11.4 通过生成器实现可迭代对象

```python
>>> from __future__ import print_function
>>> class Count(object):
...    def __init__(self, n):
...        self._n = n
...    def __iter__(self):
...        n = self._n
...        while n > 0:
...            yield n
...            n -= 1
...    def __reversed__(self):
...        n = 1
...        while n <= self._n:
...            yield n
...            n += 1
...
>>> for x in Count(5):
...    print(x, end=" ")
...
5 4 3 2 1 >>>
>>> for x in reversed(Count(5)):
...    print(x, end=" ")
...
1 2 3 4 5 >>>
```

### 2.11.5 向生成器发送消息

```python
>>> def spam():
...    msg = yield
...    print("Message:", msg)
...
>>> try:
...    g = spam()
...    # 启动生成器
...    next(g)
...    # 向生成器发送消息
...    g.send("Hello World!")
... except StopIteration:
...    pass
...
Message: Hello World!
```

### 2.11.6 yield from 表达式

```python
#### 委托生成器不做任何事（管道）
>>> def subgen():
...     try:
...         yield 9527
...     except ValueError:
...         print("get value error")
...
>>> def delegating_gen():
...     yield from subgen()
...
>>> g = delegating_gen()
>>> try:
...     next(g)
...     g.throw(ValueError)
... except StopIteration:
...     print("gen stop")
...
9527
get value error
gen stop

#### yield from + yield from
>>> import inspect
>>> def subgen():
...     yield from range(5)
...
>>> def delegating_gen():
...     yield from subgen()
...
>>> g = delegating_gen()
>>> inspect.getgeneratorstate(g)
'GEN_CREATED'
>>> next(g)
0
>>> inspect.getgeneratorstate(g)
'GEN_SUSPENDED'
>>> g.close()
>>> inspect.getgeneratorstate(g)
'GEN_CLOSED'
```

### 2.11.7 yield (from) EXPR 返回 RES

```python
>>> def average():
...     total = .0
...     count = 0
...     avg = None
...     while True:
...         val = yield
...         if not val:
...             break
...         total += val
...         count += 1
...         avg = total / count
...     return avg
...
>>> g = average()
>>> next(g) # 启动生成器
>>> g.send(3)
>>> g.send(5)
>>> try:
...     g.send(None)
... except StopIteration as e:
...     ret = e.value
...
>>> ret
4.0

#### yield from EXP 返回 RES
>>> def subgen():
...     yield 9527
...
>>> def delegating_gen():
...     yield from subgen()
...     return 5566
...
>>> try:
...     g = delegating_gen()
...     next(g)
...     next(g)
... except StopIteration as _e:
...     print(_e.value)
...
9527
5566
```

### 2.11.8 生成序列

```python
#### 通过生成器获取列表

>>> def chain():
...     for x in 'ab':
...         yield x
...     for x in range(3):
...         yield x
...
>>> a = list(chain())
>>> a
['a', 'b', 0, 1, 2]
```

### 2.11.9 `RES = yield from EXP` 实际做了什么？

```python
#### 参考：pep380
>>> def subgen():
...     for x in range(3):
...         yield x
...
>>> EXP = subgen()
>>> def delegating_gen():
...     _i = iter(EXP)
...     try:
...         _y = next(_i)
...     except StopIteration as _e:
...         RES = _e.value
...     else:
...         while True:
...             _s = yield _y
...             try:
...                 _y = _i.send(_s)
...             except StopIteration as _e:
...                 RES = _e.value
...                 break
...
>>> g = delegating_gen()
>>> next(g)
0
>>> next(g)
1
>>> next(g)
2

#### 等价于
>>> EXP = subgen()
>>> def delegating_gen():
...     RES = yield from EXP
...
>>> g = delegating_gen()
>>> next(g)
0
>>> next(g)
1
```

### 2.11.10 `for _ in gen()` 模拟 `yield from`

```python
>>> def subgen(n):
...     for x in range(n): yield x
...
>>> def gen(n):
...     yield from subgen(n)
...
>>> g = gen(3)
>>> next(g)
0
>>> next(g)
1

#### 等同于

>>> def gen(n):
...     for x in subgen(n): yield x
...
>>> g = gen(3)
>>> next(g)
0
>>> next(g)
1
```

### 2.11.11 检查生成器类型

```python
>>> from types import GeneratorType
>>> def gen_func():
...     yield 5566
...
>>> g = gen_func()
>>> isinstance(g, GeneratorType)
True
>>> isinstance(123, GeneratorType)
False
```

### 2.11.12 检查生成器状态

```python
>>> import inspect
>>> def gen_func():
...     yield 9527
...
>>> g = gen_func()
>>> inspect.getgeneratorstate(g)
'GEN_CREATED'
>>> next(g)
9527
>>> inspect.getgeneratorstate(g)
'GEN_SUSPENDED'
>>> g.close()
>>> inspect.getgeneratorstate(g)
'GEN_CLOSED'
```

### 2.11.13 简单编译器

```python
#### David Beazley - 生成器：最终前沿

import re
import types
from collections import namedtuple

tokens = [
    r'(?P<NUMBER>\d+)',
    r'(?P<PLUS>\+)',
    r'(?P<MINUS>-)',
    r'(?P<TIMES>\*)',
    r'(?P<DIVIDE>/)',
    r'(?P<WS>\s+)']

Token = namedtuple('Token', ['type', 'value'])
lex = re.compile('|'.join(tokens))

def tokenize(text):
    scan = lex.scanner(text)
    gen = (Token(m.lastgroup, m.group())
           for m in iter(scan.match, None) if m.lastgroup != 'WS')
    return gen

class Node:
    _fields = []
    def __init__(self, *args):
        for attr, value in zip(self._fields, args):
            setattr(self, attr, value)

class Number(Node):
    _fields = ['value']

class BinOp(Node):
    _fields = ['op', 'left', 'right']

def parse(toks):
    lookahead, current = next(toks, None), None

    def accept(*toktypes):
        nonlocal lookahead, current
        if lookahead and lookahead.type in toktypes:
            current, lookahead = lookahead, next(toks, None)
            return True

    def expr():
        left = term()
        while accept('PLUS', 'MINUS'):
            left = BinOp(current.value, left)
            left.right = term()
        return left

    def term():
        left = factor()
        while accept('TIMES', 'DIVIDE'):
            left = BinOp(current.value, left)
            left.right = factor()
        return left

    def factor():
        if accept('NUMBER'):
            return Number(int(current.value))
        else:
            raise SyntaxError()
    return expr()

class NodeVisitor:
    def visit(self, node):
        stack = [self.genvisit(node)]
        ret = None
        while stack:
            try:
                node = stack[-1].send(ret)
                stack.append(self.genvisit(node))
                ret = None
            except StopIteration as e:
                stack.pop()
                ret = e.value
        return ret

    def genvisit(self, node):
        ret = getattr(self, 'visit_' + type(node).__name__)(node)
        if isinstance(ret, types.GeneratorType):
            ret = yield from ret
        return ret

class Evaluator(NodeVisitor):
    def visit_Number(self, node):
        return node.value

    def visit_BinOp(self, node):
        leftval = yield node.left
        rightval = yield node.right
        if node.op == '+':
            return leftval + rightval
        elif node.op == '-':
            return leftval - rightval
        elif node.op == '*':
            return leftval * rightval
        elif node.op == '/':
            return leftval / rightval

def evaluate(exp):
    toks = tokenize(exp)
    tree = parse(toks)
    return Evaluator().visit(tree)

exp = '2 * 3 + 5 / 2'
print(evaluate(exp))
exp = '+'.join([str(x) for x in range(10000)])
print(evaluate(exp))
```

输出：

```
python3 compiler.py
8.5
49995000
```

### 2.11.14 上下文管理器与生成器

```python
>>> import contextlib
>>> @contextlib.contextmanager
... def mylist():
...     try:
...         l = [1, 2, 3, 4, 5]
...         yield l
...     finally:
...         print("exit scope")
...
>>> with mylist() as l:
...     print(l)
...
[1, 2, 3, 4, 5]
exit scope
```

### 2.11.15 `@contextmanager` 实际做了什么？

```python
#### 参考：PyCon 2014 - David Beazley
#### 定义一个上下文管理器类

class GeneratorCM(object):
    def __init__(self, gen):
        self._gen = gen

    def __enter__(self):
        return next(self._gen)

    def __exit__(self, *exc_info):
        try:
            if exc_info[0] is None:
                next(self._gen)
            else:
                self._gen.throw(*exc_info)
            raise RuntimeError
        except StopIteration:
            return True
        except:
            raise

#### 定义一个装饰器
def contextmanager(func):
    def run(*a, **k):
        return GeneratorCM(func(*a, **k))
    return run

#### 上下文管理器示例
@contextmanager
def mylist():
    try:
        l = [1, 2, 3, 4, 5]
        yield l
    finally:
        print("exit scope")

with mylist() as l:
    print(l)
```

```
$ python ctx.py
[1, 2, 3, 4, 5]
exit scope
```

### 2.11.16 分析代码块

```python
>>> import time
>>> @contextmanager
... def profile(msg):
...     try:
...         s = time.time()
...         yield
...     finally:
...         e = time.time()
...         print('{} cost time: {}'.format(msg, e - s))
...
>>> with profile('block1'):
...     time.sleep(1)
...
block1 cost time: 1.00105595589
>>> with profile('block2'):
...     time.sleep(3)
...
block2 cost time: 3.00104284286
```

### 2.11.17 `yield from` 与 `__iter__`

```python
>>> class FakeGen:
...     def __iter__(self):
...         n = 0
...         while True:
...             yield n
...             n += 1
...     def __reversed__(self):
...         n = 9527
...         while True:
...             yield n
...             n -= 1
...
>>> def spam():
...     yield from FakeGen()
...
>>> s = spam()
>>> next(s)
0
>>> next(s)
1
>>> next(s)
2
>>> next(s)
3
>>> def reversed_spam():
...     yield from reversed(FakeGen())
...
>>> g = reversed_spam()
>>> next(g)
9527
>>> next(g)
9526
>>> next(g)
9525
```

### 2.11.18 `yield from` == `await` 表达式

```python
#### "await" 包含在 python3.5 中
import asyncio
import socket

#### 设置套接字和事件循环
loop = asyncio.get_event_loop()
host = 'localhost'
port = 5566
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setblocking(False)
sock.bind((host, port))
sock.listen(10)

@asyncio.coroutine
def echo_server():
    while True:
        conn, addr = yield from loop.sock_accept(sock)
        loop.create_task(handler(conn))

@asyncio.coroutine
def handler(conn):
    while True:
        msg = yield from loop.sock_recv(conn, 1024)
        if not msg:
            break
        yield from loop.sock_sendall(conn, msg)
    conn.close()

#### 等同于
async def echo_server():
    while True:
        conn, addr = await loop.sock_accept(sock)
        loop.create_task(handler(conn))

async def handler(conn):
    while True:
        msg = await loop.sock_recv(conn, 1024)
        if not msg:
            break
        await loop.sock_sendall(conn, msg)
    conn.close()

loop.create_task(echo_server())
loop.run_forever()
```

输出：（bash 1）

```
$ nc localhost 5566
Hello
Hello
```

输出：（bash 2）

```
$ nc localhost 5566
World
World
```

### 2.11.19 Python 中的闭包 - 使用生成器

```python
#### nonlocal 版本
>>> def closure():
...     x = 5566
...     def inner_func():
...         nonlocal x
...         x += 1
...         return x
...     return inner_func
...
>>> c = closure()
>>> c()
5567
>>> c()
5568
>>> c()
5569

#### 类版本
>>> class Closure:
...     def __init__(self):
...         self._x = 5566
...     def __call__(self):
...         self._x += 1
...         return self._x
...
>>> c = Closure()
>>> c()
5567
>>> c()
5568
>>> c()
5569
```## 2.11.20 实现一个简单的调度器

```python
#### 思路：编写一个事件循环（调度器）
>>> def fib(n):
...     if n <= 2:
...         return 1
...     return fib(n-1) + fib(n-2)
...
>>> def g_fib(n):
...     for x in range(1, n + 1):
...         yield fib(x)
...
>>> from collections import deque
>>> t = [g_fib(3), g_fib(5)]
>>> q = deque()
>>> q.extend(t)
>>> def run():
...     while q:
...         try:
...             t = q.popleft()
...             print(next(t))
...             q.append(t)
...         except StopIteration:
...             print("任务完成")
...
>>> run()
1
1
1
1
2
2
任务完成
3
5
任务完成
```

### 2.11.21 带阻塞的简单轮询调度

```python
#### 参考：PyCon 2015 - David Beazley
#### 技巧：使用任务队列和等待队列

from collections import deque
from select import select
import socket

tasks = deque()
w_read = {}
w_send = {}

def run():
    while any([tasks, w_read, w_send]):
        while not tasks:
            # 轮询任务
            can_r, can_s,_ = select(w_read, w_send, [])
            for _r in can_r:
                tasks.append(w_read.pop(_r))
            for _w in can_s:
                tasks.append(w_send.pop(_w))
        try:
            task = tasks.popleft()
            why,what = next(task)
            if why == 'recv':
                w_read[what] = task
            elif why == 'send':
                w_send[what] = task
            else:
                raise RuntimeError
        except StopIteration:
            pass

def server():
    host = ('localhost', 5566)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(host)
    sock.listen(5)
    while True:
        # 告诉调度器希望阻塞
        yield 'recv', sock
        conn,addr = sock.accept()
        tasks.append(client_handler(conn))

def client_handler(conn):
    while True:
        # 告诉调度器希望阻塞
        yield 'recv', conn
        msg = conn.recv(1024)
        if not msg:
            break
        # 告诉调度器希望阻塞
        yield 'send', conn
        conn.send(msg)
    conn.close()

tasks.append(server())
run()
```

### 2.11.22 带阻塞与非阻塞的简单轮询调度

```python
#### 这种方法会导致阻塞饥饿
from collections import deque
from select import select
import socket

tasks = deque()
w_read = {}
w_send = {}

def run():
    while any([tasks, w_read, w_send]):
        while not tasks:
            # 轮询任务
            can_r,can_s,_ = select(w_read, w_send, [])
            for _r in can_r:
                tasks.append(w_read.pop(_r))
            for _w in can_s:
                tasks.append(w_send.pop(_w))
        try:
            task = tasks.popleft()
            why,what = next(task)
            if why == 'recv':
                w_read[what] = task
            elif why == 'send':
                w_send[what] = task
            elif why == 'continue':
                print(what)
                tasks.append(task)
            else:
                raise RuntimeError
        except StopIteration:
            pass

def fib(n):
    if n <= 2:
        return 1
    return fib(n-1) + fib(n-2)

def g_fib(n):
    for x in range(1, n + 1):
        yield 'continue', fib(x)

tasks.append(g_fib(15))

def server():
    host = ('localhost', 5566)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(host)
    sock.listen(5)
    while True:
        yield 'recv', sock
        conn,addr = sock.accept()
        tasks.append(client_handler(conn))

def client_handler(conn):
    while True:
        yield 'recv', conn
        msg = conn.recv(1024)
        if not msg:
            break
        yield 'send', conn
        conn.send(msg)
    conn.close()

tasks.append(server())
run()
```

### 2.11.23 异步生成器

```python
#### PEP 525
#
#### 需要 Python 3.6 或更高版本

>>> import asyncio
>>> async def slow_gen(n, t):
...     for x in range(n):
...         await asyncio.sleep(t)
...         yield x
...
>>> async def task(n):
...     async for x in slow_gen(n, 0.1):
...         print(x)
...
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(task(3))
0
1
2
```

### 2.11.24 异步生成器可以包含 try..finally 块

```python
#### 需要 Python 3.6 或更高版本

>>> import asyncio
>>> async def agen(t):
...     try:
...         await asyncio.sleep(t)
...         yield 1 / 0
...     finally:
...         print("finally 部分")
...
>>> async def main(t=1):
...     try:
...         g = agen(t)
...         await g.__anext__()
...     except Exception as e:
...         print(repr(e))
...
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(main(1))
finally 部分
ZeroDivisionError('division by zero',)
```

### 2.11.25 向异步生成器发送值和抛出异常

```python
#### 需要 Python 3.6 或更高版本

>>> import asyncio
>>> async def agen(n, t=0.1):
...     try:
...         for x in range(n):
...             await asyncio.sleep(t)
...             val = yield x
...             print(f'获取的值: {val}')
...     except RuntimeError as e:
...         await asyncio.sleep(t)
...         yield repr(e)
...
>>> async def main(n):
...     g = agen(n)
...     ret = await g.asend(None) + await g.asend('foo')
...     print(ret)
...     ret = await g.athrow(RuntimeError('触发 RuntimeError'))
...     print(ret)
...
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(main(5))
获取的值: foo
1
RuntimeError('触发 RuntimeError',)
```

## 2.11.26 简单的异步轮询调度

```python
#### 需要 Python 3.6 或更高版本

>>> import asyncio
>>> from collections import deque
>>> async def agen(n, t=0.1):
...     for x in range(n):
...         await asyncio.sleep(t)
...         yield x
...
>>> async def main():
...     q = deque([agen(3), agen(5)])
...     while q:
...         try:
...             g = q.popleft()
...             ret = await g.__anext__()
...             print(ret)
...             q.append(g)
...         except StopAsyncIteration:
...             pass
...
>>> loop.run_until_complete(main())
0
0
1
1
2
2
3
4
```

## 2.11.27 异步生成器比异步迭代器性能更好

```python
#### 需要 Python 3.6 或更高版本
>>> import time
>>> import asyncio
>>> class AsyncIter:
...     def __init__(self, n):
...         self._n = n
...     def __aiter__(self):
...         return self
...     async def __anext__(self):
...         ret = self._n
...         if self._n == 0:
...             raise StopAsyncIteration
...         self._n -= 1
...         return ret
...
>>> async def agen(n):
...     for i in range(n):
...         yield i
...
>>> async def task_agen(n):
...     s = time.time()
...     async for _ in agen(n): pass
...     cost = time.time() - s
...     print(f"agen 耗时: {cost}")
...
>>> async def task_aiter(n):
...     s = time.time()
...     async for _ in AsyncIter(n): pass
...     cost = time.time() - s
...     print(f"aiter 耗时: {cost}")
...
>>> n = 10 ** 7
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(task_agen(n))
agen 耗时: 1.2698817253112793
>>> loop.run_until_complete(task_aiter(n))
aiter 耗时: 4.168368101119995
```

## 2.11.28 异步推导式

```python
# PEP 530
#
#### 需要 Python 3.6 或更高版本
>>> import asyncio
>>> async def agen(n, t):
...     for x in range(n):
...         await asyncio.sleep(t)
...         yield x
>>> async def main():
...     ret = [x  async for x in agen(5, 0.1)]
...     print(*ret)
...     ret = [x async for x in agen(5, 0.1) if x < 3]
...     print(*ret)
...     ret = [x if x < 3 else -1 async for x in agen(5, 0.1)]
...     print(*ret)
...     ret = {f'{x}': x async for x in agen(5, 0.1)}
...     print(ret)

>>> loop.run_until_complete(main())
0 1 2 3 4
0 1 2
0 1 2 -1 -1
{'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
```

## 推导式中的 await

```python
>>> async def foo(t):
...     await asyncio.sleep(t)
...     return "foo"
...
>>> async def bar(t):
...     await asyncio.sleep(t)
...     return "bar"
...
>>> async def baz(t):
...     await asyncio.sleep(t)
...     return "baz"
...
>>> async def gen(*f, t=0.1):
...     for x in f:
...         await asyncio.sleep(t)
...         yield x
...
>>> async def await_simple_task():
...     ret = [await f(0.1) for f in [foo, bar]]
...     print(ret)
...     ret = {await f(0.1) for f in [foo, bar]}
...     print(ret)
...     ret = {f.__name__: await f(0.1) for f in [foo, bar]}
...     print(ret)
...
>>> async def await_other_task():
...     ret = [await f(0.1) for f in [foo, bar] if await baz(1)]
...     print(ret)
...     ret = {await f(0.1) for f in [foo, bar] if await baz(1)}
...     print(ret)
...     ret = {f.__name__: await f(0.1) for f in [foo, bar] if await baz(1)}
...     print(ret)
...
```

## 2.12 类型提示

PEP 484 提供了 Python3 中类型系统应遵循的规范，引入了类型提示的概念。此外，为了更好地理解类型提示的设计哲学，阅读 PEP 483 至关重要，它有助于 Python 开发者理解 Python 引入类型系统的原因。本速查表的主要目标是展示 Python3 中类型提示的一些常见用法。

- 目录
  - 类型提示
    - 无类型检查
    - 有类型检查
    - 基本类型
    - 函数
    - 类
    - 生成器
    - 异步生成器
    - 上下文管理器
    - 异步上下文管理器
    - 避免 `None` 访问
    - 仅位置参数
    - 多返回值
    - `Union[Any, None] == Optional[Any]`
    - 注意 `Optional` 的用法
    - 注意 `cast` 的用法
    - 前向引用
    - 延迟注解求值
    - 类型别名
    - 定义 `NewType`
    - 使用 `TypeVar` 作为模板
    - 使用 `TypeVar` 和 `Generic` 作为类模板
    - `TypeVar` 的作用域规则
    - 限制为固定集合的可能类型
    - 带上界的 `TypeVar`
    - `@overload`
    - 存根文件

## 2.12.1 无类型检查

```python
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        b, a = a + b, b

print([n for n in fib(3.6)])
```

输出：

```
# 错误直到运行时才会被检测到

$ python fib.py
Traceback (most recent call last):
  File "fib.py", line 8, in <module>
    print([n for n in fib(3.5)])
  File "fib.py", line 8, in <listcomp>
    print([n for n in fib(3.5)])
File "fib.py", line 3, in fib
    for _ in range(n):
TypeError: 'float' object cannot be interpreted as an integer
```

## 2.12.2 有类型检查

```python
# 添加类型提示
from typing import Generator

def fib(n: int) -> Generator:
    a: int = 0
    b: int = 1
    for _ in range(n):
        yield a
        b, a = a + b, b

print([n for n in fib(3.6)])
```

输出：

```
# 错误将在运行前被检测到

$ mypy --strict fib.py
fib.py:12: error: Argument 1 to "fib" has incompatible type "float"; expected "int"
```

## 2.12.3 基本类型

```python
import io
import re

from collections import deque, namedtuple
from typing import (
    Dict,
    List,
    Tuple,
    Set,
    Deque,
    NamedTuple,
    IO,
    Pattern,
    Match,
    Text,
    Optional,
    Sequence,
    Iterable,
    Mapping,
    MutableMapping,
    Any,
)

# 未初始化
x: int

# 任意类型
y: Any
y = 1
y = "1"

# 内置类型
var_int: int = 1
var_str: str = "Hello Typing"
var_byte: bytes = b"Hello Typing"
var_bool: bool = True
var_float: float = 1.
var_unicode: Text = u'\u2713'

# 可能为 None
var_could_be_none: Optional[int] = None
var_could_be_none = 1

# 集合
var_set: Set[int] = {i for i in range(3)}
var_dict: Dict[str, str] = {"foo": "Foo"}
var_list: List[int] = [i for i in range(3)]
var_static_length_Tuple: Tuple[int, int, int] = (1, 2, 3)
var_dynamic_length_Tuple: Tuple[int, ...] = (i for i in range(10, 3))
var_deque: Deque = deque([1, 2, 3])
var_nametuple: NamedTuple = namedtuple('P', ['x', 'y'])

# IO
var_io_str: IO[str] = io.StringIO("Hello String")
var_io_byte: IO[bytes] = io.BytesIO(b"Hello Bytes")
var_io_file_str: IO[str] = open(__file__)
var_io_file_byte: IO[bytes] = open(__file__, 'rb')

# 正则表达式
p: Pattern = re.compile("(https?):://([^/\r\n]+)(/[^\r\n]*)?")
m: Optional[Match] = p.match("https://www.python.org/")

# 鸭子类型：列表类
var_seq_list: Sequence[int] = [1, 2, 3]
var_seq_tuple: Sequence[int] = (1, 2, 3)
var_iter_list: Iterable[int] = [1, 2, 3]
var_iter_tuple: Iterable[int] = (1, 2, 3)

# 鸭子类型：字典类
var_map_dict: Mapping[str, str] = {"foo": "Foo"}
var_mutable_dict: MutableMapping[str, str] = {"bar": "Bar"}
```

## 2.12.4 函数

```python
from typing import Generator, Callable

# 函数
def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

# 回调
def fun(cb: Callable[[int, int], int]) -> int:
    return cb(55, 66)

# Lambda 表达式
f: Callable[[int], int] = lambda x: x * 2
```

## 2.12.5 类

```python
from typing import ClassVar, Dict, List

class Foo:

    x: int = 1  # 实例变量，默认值 = 1
    y: ClassVar[str] = "class var"  # 类变量

    def __init__(self) -> None:
        self.i: List[int] = [0]

    def foo(self, a: int, b: str) -> Dict[int, str]:
        return {a: b}

foo = Foo()
foo.x = 123

print(foo.x)
print(foo.i)
print(Foo.y)
print(foo.foo(1, "abc"))
```

## 2.12.6 生成器

```python
from typing import Generator

# Generator[YieldType, SendType, ReturnType]
def fib(n: int) -> Generator[int, None, None]:
    a: int = 0
    b: int = 1
    while n > 0:
        yield a
        b, a = a + b, b
        n -= 1

g: Generator = fib(10)
i: Iterator[int] = (x for x in range(3))
```

## 2.12.7 异步生成器

```python
import asyncio
from typing import AsyncGenerator, AsyncIterator

async def fib(n: int) -> AsyncGenerator:
    a: int = 0
    b: int = 1
    while n > 0:
        await asyncio.sleep(0.1)
        yield a
        b, a = a + b, b
        n -= 1

async def main() -> None:
    async for f in fib(10):
        print(f)
    ag: AsyncIterator = (f async for f in fib(10))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

## 2.12.8 上下文管理器

```python
from typing import ContextManager, Generator, IO
from contextlib import contextmanager

@contextmanager
def open_file(name: str) -> Generator:
    f = open(name)
    yield f
    f.close()

cm: ContextManager[IO] = open_file(__file__)
with cm as f:
    print(f.read())
```

## 2.12.9 异步上下文管理器

```python
import asyncio

from typing import AsyncContextManager, AsyncGenerator, IO
from contextlib import asynccontextmanager

# 需要 Python 3.7 或更高版本
@asynccontextmanager
async def open_file(name: str) -> AsyncGenerator:
    await asyncio.sleep(0.1)
    f = open(name)
    yield f
    await asyncio.sleep(0.1)
    f.close()

async def main() -> None:
    acm: AsyncContextManager[IO] = open_file(__file__)
    async with acm as f:
        print(f.read())

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

## 2.12.10 避免 `None` 访问

```python
import re

from typing import Pattern, Dict, Optional

# 类似 C++
# std::regex url("(https?)://([^\r\n]+)(/[^\r\n]*)?"\);
# std::regex color("^#?([a-f0-9]{6}|[a-f0-9]{3})$");

url: Pattern = re.compile("(https?)://([^\r\n]+)(/[^\r\n]*)?")
color: Pattern = re.compile("^#?([a-f0-9]{6}|[a-f0-9]{3})$")

x: Dict[str, Pattern] = {"url": url, "color": color}
y: Optional[Pattern] = x.get("baz", None)

print(y.match("https://www.python.org/"))
```

输出：

```
$ mypy --strict foo.py
foo.py:15: error: Item "None" of "Optional[Pattern[Any]]" has no attribute "match"
```

## 2.12.11 仅位置参数

```python
# 定义以 __ 开头的参数

def fib(__n: int) -> int:  # 仅位置参数
    a, b = 0, 1
    for _ in range(__n):
        b, a = a + b, b
    return a

def gcd(*, a: int, b: int) -> int:  # 仅关键字参数
    while b:
        a, b = b, a % b
    return a

print(fib(__n=10))  # 错误
print(gcd(10, 5))  # 错误
```

输出：

```
mypy --strict foo.py
foo.py:1: note: "fib" defined here
foo.py:14: error: Unexpected keyword argument "__n" for "fib"
foo.py:15: error: Too many positional arguments for "gcd"
```

## 2.12.12 多返回值

```python
from typing import Tuple, Iterable, Union

def foo(x: int, y: int) -> Tuple[int, int]:
    return x, y

# 或者

def bar(x: int, y: str) -> Iterable[Union[int, str]]:
    # XXX: 不推荐这样声明
    return x, y

a: int
b: int
a, b = foo(1, 2)     # 正确
c, d = bar(3, "bar")  # 正确
```## 2.12.13 Union[Any, None] == Optional[Any]

```python
from typing import List, Union

def first(l: List[Union[int, None]]) -> Union[int, None]:
    return None if len(l) == 0 else l[0]

first([None])

#### 等价于

from typing import List, Optional

def first(l: List[Optional[int]]) -> Optional[int]:
    return None if len(l) == 0 else l[0]

first([None])
```

## 2.12.14 小心 Optional

```python
from typing import cast, Optional

def fib(n):
    a, b = 0, 1
    for _ in range(n):
        b, a = a + b, b
    return a

def cal(n: Optional[int]) -> None:
    print(fib(n))

cal(None)
```

输出：

```bash
# mypy 不会检测到错误
$ mypy foo.py
```

显式声明

```python
from typing import Optional

def fib(n: int) -> int:  # 声明 n 为 int
    a, b = 0, 1
    for _ in range(n):
        b, a = a + b, b
    return a

def cal(n: Optional[int]) -> None:
    print(fib(n))
```

输出：

```
# mypy 可以检测到错误，即使我们没有显式检查 None
$ mypy --strict foo.py
foo.py:11: error: Argument 1 to "fib" has incompatible type "Optional[int]"; expected "int"
```

## 2.12.15 小心 casting

```python
from typing import cast, Optional

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def cal(a: Optional[int], b: Optional[int]) -> None:
    # XXX：避免使用 casting
    ca, cb = cast(int, a), cast(int, b)
    print(gcd(ca, cb))

cal(None, None)
```

输出：

```
# mypy 不会检测到类型错误
$ mypy --strict foo.py
```

## 2.12.16 前向引用

根据 PEP 484，如果我们想在类型被声明之前引用它，必须使用**字符串字面量**来暗示该名称的类型将在文件的后面出现。

```python
from typing import Optional


class Tree:
    def __init__(
        self, data: int,
        left: Optional["Tree"],  # 前向引用。
        right: Optional["Tree"]
    ) -> None:
        self.data = data
        self.left = left
        self.right = right
```

> **注意：** mypy 对前向引用的处理有一些问题不会报警。更多信息请参考 Issue#948。

```python
class A:
    def __init__(self, a: A) -> None:  # 应当失败
        self.a = a
```

输出：

```
$ mypy --strict type.py
$ echo $?
0
$ python type.py   # 运行时失败
Traceback (most recent call last):
  File "type.py", line 1, in <module>
    class A:
  File "type.py", line 2, in A
    def __init__(self, a: A) -> None:  # 应当失败
NameError: name 'A' is not defined
```

## 2.12.17 注解的延迟求值

Python 3.7 中的新特性

- PEP 563 - 注解的延迟求值

Python 3.7 之前

```
>>> class A:
...     def __init__(self, a: A) -> None:
...         self._a = a
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in A
NameError: name 'A' is not defined
```

Python 3.7 之后（包含 3.7）

```
>>> from __future__ import annotations
>>> class A:
...     def __init__(self, a: A) -> None:
...         self._a = a
...
```

> 注意：注解只能在名称已存在的作用域内使用。因此，前向引用不适用于当前作用域中名称不可用的情况。注解的延迟求值将在 Python 4.0 中成为默认行为。

## 2.12.18 类型别名

类似于 C/C++ 中的 `typedef` 或 `using`

```cpp
#include <iostream>
#include <string>
#include <regex>
#include <vector>

typedef std::string Url;
template<typename T> using Vector = std::vector<T>;

int main(int argc, char *argv[])
{
    Url url = "https://python.org";
    std::regex p("(https?)://([^/\r\n]+)(/[^\r\n]*)?");
    bool m = std::regex_match(url, p);
    Vector<int> v = {1, 2};

    std::cout << m << std::endl;
    for (auto it : v) std::cout << it << std::endl;
    return 0;
}
```

类型别名通过简单的变量赋值来定义

```python
import re

from typing import Pattern, List

# 类似于 C/C++ 中的 typedef, using

# PEP 484 建议将别名名称大写
Url = str

url: Url = "https://www.python.org/"

p: Pattern = re.compile("(https?)://([^/\r\n]+)(/[^\r\n]*)?")
m = p.match(url)

Vector = List[int]
v: Vector = [1., 2.]
```

## 2.12.19 定义 NewType

与别名不同，NewType 会返回一个独立的类型，但在运行时与原始类型是完全相同的。

```python
from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from typing import NewType, Any

# 检查 mypy #2477
Base: Any = declarative_base()
```

```python
# 创建一个新类型
Id = NewType('Id', int) # 不等同于别名，它是一个‘新类型’

class User(Base):
    __tablename__ = 'User'
    id = Column(Integer, primary_key=True)
    age = Column(Integer, nullable=False)
    name = Column(String, nullable=False)

    def __init__(self, id: Id, age: int, name: str) -> None:
        self.id = id
        self.age = age
        self.name = name

# 创建用户
user1 = User(Id(1), 62, "Guido van Rossum") # 正确
user2 = User(2, 48, "David M. Beazley")    # 错误
```

输出：

```
$ python foo.py
$ mypy --ignore-missing-imports foo.py
foo.py:24: error: Argument 1 to "User" has incompatible type "int"; expected "Id"
```

延伸阅读：

- Issue#1284

## 2.12.20 使用 TypeVar 作为模板

类似于 C++ 的 `template <typename T>`

```cpp
#include <iostream>

template <typename T>
T add(T x, T y) {
    return x + y;
}

int main(int argc, char *argv[])
{
    std::cout << add(1, 2) << std::endl;
    std::cout << add(1., 2.) << std::endl;
    return 0;
}
```

Python 使用 TypeVar

```python
from typing import TypeVar

T = TypeVar("T")

def add(x: T, y: T) -> T:
    return x + y

add(1, 2)
add(1., 2.)
```

## 2.12.21 使用 TypeVar 和 Generic 作为类模板

类似于 C++ 的 `template <typename T> class`

```cpp
#include <iostream>

template<typename T>
class Foo {
public:
    Foo(T foo) {
        foo_ = foo;
    }
    T Get() {
        return foo_;
    }
private:
    T foo_;
};

int main(int argc, char *argv[])
{
    Foo<int> f(123);
    std::cout << f.Get() << std::endl;
    return 0;
}
```

在 Python 中定义泛型类

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class Foo(Generic[T]):
    def __init__(self, foo: T) -> None:
        self.foo = foo

    def get(self) -> T:
        return self.foo

f: Foo[str] = Foo("Foo")
v: int = f.get()
```

输出：

```
$ mypy --strict foo.py
foo.py:13: error: Incompatible types in assignment (expression has type "str", variable has type "int")
```

## 2.12.22 TypeVar 的作用域规则

- 在不同泛型函数中使用的 TypeVar 将被推断为不同的类型。

```python
from typing import TypeVar

T = TypeVar("T")

def foo(x: T) -> T:
    return x

def bar(y: T) -> T:
    return y

a: int = foo(1)    # 正确：T 被推断为 int
b: int = bar("2")  # 错误：T 被推断为 str
```

输出：

```
$ mypy --strict foo.py
foo.py:12: error: Incompatible types in assignment (expression has type "str", variable has type "int")
```

- 在泛型类中使用的 TypeVar 将被推断为相同的类型。

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Foo(Generic[T]):

    def foo(self, x: T) -> T:
        return x

    def bar(self, y: T) -> T:
        return y

f: Foo[int] = Foo()
a: int = f.foo(1)    # 正确：T 被推断为 int
b: str = f.bar("2")  # 错误：T 应为 int
```

输出：

```
$ mypy --strict foo.py
foo.py:15: error: Incompatible types in assignment (expression has type "int", variable has type "str")
foo.py:15: error: Argument 1 to "bar" of "Foo" has incompatible type "str"; expected "int"
```

- 在方法中使用但没有匹配 Generic 中声明的任何参数的 TypeVar 可以被推断为不同的类型。

```python
from typing import TypeVar, Generic

T = TypeVar("T")
S = TypeVar("S")

class Foo(Generic[T]):    # S 没有匹配参数

    def foo(self, x: T, y: S) -> S:
        return y

    def bar(self, z: S) -> S:
        return z

f: Foo[int] = Foo()
a: str = f.foo(1, "foo")  # S 被推断为 str
b: int = f.bar(12345678)  # S 被推断为 int
```

输出：

```
$ mypy --strict foo.py
```

- 如果 TypeVar 是未绑定类型，则不应出现在方法/函数体中。

```python
from typing import TypeVar, Generic

T = TypeVar("T")
S = TypeVar("S")

def foo(x: T) -> None:
    a: T = x    # 正确
    b: S = 123  # 错误：无效类型
```

输出：

```
$ mypy --strict foo.py
foo.py:8: error: Invalid type "foo.S"
```

## 2.12.23 限制为一组固定的可能类型

`T = TypeVar('T', ClassA, ...)` 意味着我们创建了一个**带值约束的类型变量**。

```python
from typing import TypeVar

# 限制 T = int 或 T = float
T = TypeVar("T", int, float)

def add(x: T, y: T) -> T:
    return x + y

add(1, 2)
```add(1., 2.)
add("1", 2)
add("hello", "world")

输出：

# mypy 可以检测错误类型
$ mypy --strict foo.py
foo.py:10: error: Value of type variable "T" of "add" cannot be "object"
foo.py:11: error: Value of type variable "T" of "add" cannot be "str"

# 2.12.24 具有上界（Upper Bound）的 TypeVar

`T = TypeVar('T', bound=BaseClass)` 意味着我们创建了一个**具有上界的类型变量**。这个概念类似于 C++ 中的**多态**。

```cpp
#include <iostream>

class Shape {
public:
    Shape(double width, double height) {
        width_ = width;
        height_ = height;
    };
    virtual double Area() = 0;
protected:
    double width_;
    double height_;
};

class Rectangle: public Shape {
public:
    Rectangle(double width, double height)
    :Shape(width, height)
    {};

    double Area() {
        return width_ * height_;
    };
};

class Triangle: public Shape {
public:
    Triangle(double width, double height)
    :Shape(width, height)
    {};

    double Area() {
        return width_ * height_ / 2;
    };
};

double Area(Shape &s) {
    return s.Area();
}

int main(int argc, char *argv[])
{
    Rectangle r(1., 2.);
    Triangle t(3., 4.);

    std::cout << Area(r) << std::endl;
    std::cout << Area(t) << std::endl;
    return 0;
}
```

与 C++ 类似，创建一个基类和一个绑定到该基类的 TypeVar。然后，静态类型检查器会将每个子类视为基类的类型。

```python
from typing import TypeVar


class Shape:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    def area(self) -> float:
        return 0


class Rectangle(Shape):
    def area(self) -> float:
        width: float = self.width
        height: float = self.height
        return width * height


class Triangle(Shape):
    def area(self) -> float:
        width: float = self.width
        height: float = self.height
        return width * height / 2


S = TypeVar("S", bound=Shape)


def area(s: S) -> float:
    return s.area()


r: Rectangle = Rectangle(1, 2)
t: Triangle = Triangle(3, 4)
i: int = 5566

print(area(r))
print(area(t))
print(area(i))
```

输出：

$ mypy --strict foo.py
foo.py:40: error: Value of type variable "S" of "area" cannot be "int"

# 2.12.25 @overload

有时，我们使用 Union 来推断函数的返回值可以有多种不同类型。然而，类型检查器无法区分我们想要的是哪种类型。因此，下面的代码片段展示了类型检查器无法确定哪种类型是正确的。

```python
from typing import List, Union

class Array(object):
    def __init__(self, arr: List[int]) -> None:
        self.arr = arr

    def __getitem__(self, i: Union[int, str]) -> Union[int, str]:
        if isinstance(i, int):
            return self.arr[i]
        if isinstance(i, str):
            return str(self.arr[int(i)])

arr = Array([1, 2, 3, 4, 5])
x:int = arr[1]
y:str = arr["2"]
```

输出：

$ mypy --strict foo.py
foo.py:16: error: Incompatible types in assignment (expression has type "Union[int, str]", variable has type "int")
foo.py:17: error: Incompatible types in assignment (expression has type "Union[int, str]", variable has type "str")

虽然我们可以使用 `cast` 来解决这个问题，但这无法避免打字错误，并且 `cast` 本身并不安全。

```python
from typing import List, Union, cast

class Array(object):
    def __init__(self, arr: List[int]) -> None:
        self.arr = arr

    def __getitem__(self, i: Union[int, str]) -> Union[int, str]:
        if isinstance(i, int):
            return self.arr[i]
        if isinstance(i, str):
            return str(self.arr[int(i)])

arr = Array([1, 2, 3, 4, 5])
x: int = cast(int, arr[1])
y: str = cast(str, arr[2])  # typo. we want to assign arr["2"]
```

输出：

$ mypy --strict foo.py
$ echo $?
0

使用 `@overload` 可以解决这个问题。我们可以显式地声明返回类型。

```python
from typing import Generic, List, Union, overload

class Array(object):
    def __init__(self, arr: List[int]) -> None:
        self.arr = arr

    @overload
    def __getitem__(self, i: str) -> str:
        ...

    @overload
    def __getitem__(self, i: int) -> int:
        ...

    def __getitem__(self, i: Union[int, str]) -> Union[int, str]:
        if isinstance(i, int):
            return self.arr[i]
        if isinstance(i, str):
            return str(self.arr[int(i)])

arr = Array([1, 2, 3, 4, 5])
x: int = arr[1]
y: str = arr["2"]
```

输出：

$ mypy --strict foo.py
$ echo $?
0

> **警告：** 根据 PEP 484，`@overload` 装饰器**仅供类型检查器使用**，它不像 C++/Java 那样实现真正的重载。因此，我们必须实现一个精确的非 `@overload` 函数。在运行时，调用 `@overload` 函数会抛出 `NotImplementedError`。

```python
from typing import List, Union, overload

class Array(object):
    def __init__(self, arr: List[int]) -> None:
        self.arr = arr

    @overload
    def __getitem__(self, i: Union[int, str]) -> Union[int, str]:
        if isinstance(i, int):
            return self.arr[i]
        if isinstance(i, str):
            return str(self.arr[int(i)])

arr = Array([1, 2, 3, 4, 5])
try:
    x: int = arr[1]
except NotImplementedError as e:
    print("NotImplementedError")
```

输出：

$ python foo.py
NotImplementedError

# 2.12.26 存根文件（Stub Files）

存根文件就像 C/C++ 中常用的头文件，我们通常用它们来定义接口。在 Python 中，我们可以在同一个模块目录中定义接口，或者导出 `MYPYPATH=${stubs}`。

首先，我们需要为模块创建一个存根文件（接口文件）。

$ mkdir fib
$ touch fib/__init__.py fib/__init__.pyi

然后，在 `__init__.pyi` 中定义函数的接口，并实现该模块。

```python
# fib/__init__.pyi
def fib(n: int) -> int: ...

# fib/__init__.py

def fib(n):
    a, b = 0, 1
    for _ in range(n):
        b, a = a + b, b
    return a
```

接着，编写一个 `test.py` 来测试 `fib` 模块。

```python
# touch test.py
import sys
from pathlib import Path
p = Path(__file__).parent / "fib"
sys.path.append(str(p))
from fib import fib
print(fib(10.0))
```

输出：

$ mypy --strict test.py
test.py:10: error: Argument 1 to "fib" has incompatible type "float"; expected "int"

# 2.13 日期时间（Datetime）

### 目录

- Datetime
- Timestamp
- Date
- Format
- 将日期转换为日期时间

# 2.13.1 时间戳（Timestamp）

```
>>> import time
>>> time.time()
1613526236.395773
>>> datetime.utcnow()
datetime.datetime(2021, 2, 17, 1, 45, 19, 312513)
>>> t = time.time()
>>> datetime.fromtimestamp(t)
datetime.datetime(2021, 2, 17, 9, 45, 41, 95756)
>>> d = datetime.fromtimestamp(t)
>>> d.timestamp()
1613526341.095756
```

# 2.13.2 日期（Date）

```
>>> from datetime import date
>>> date.today()
datetime.date(2021, 2, 17)
```

# 2.13.3 格式化（Format）

```
>>> from datetime import datetime
>>> d = datetime.utcnow()
>>> d.isoformat()
'2021-02-17T02:26:59.584044'
```

# 2.13.4 将日期转换为日期时间

```
>>> from datetime import datetime, date
>>> today = date.today()
>>> d = datetime.combine(today, datetime.min.time())
>>> d
datetime.datetime(2021, 2, 17, 0, 0)
```

# 2.14 文件与 I/O

### 目录

- 文件与 I/O
  - 读取文件
  - 逐行读取
  - 读取文件块
  - 写入文件
  - 创建符号链接
  - 复制文件
  - 移动文件
  - 列出目录
  - 创建目录
  - 复制目录
  - 删除目录
  - 路径拼接
  - 获取绝对路径
  - 获取主目录
- 获取当前目录
- 获取路径属性
- 读取 gzip CSV 文件
- Linux Inotify

## 2.14.1 读取文件

在 Python 2 中，从文件系统读取的文件内容不会被解码。也就是说，文件的内容是字节字符串，而不是 Unicode 字符串。

```
>>> with open("/etc/passwd") as f:
...     content = f.read()
>>> print(type(content))
<type 'str'>
>>> print(type(content.decode("utf-8")))
<type 'unicode'>
```

在 Python 3 中，`open` 提供了 `encoding` 选项。如果文件不是以二进制模式打开，编码将由 `locale.getpreferredencoding(False)` 或用户的输入决定。

```
>>> with open("/etc/hosts", encoding="utf-8") as f:
...     content = f.read()
...
>>> print(type(content))
<class 'str'>
```

二进制模式

```
>>> with open("/etc/hosts", "rb") as f:
...     content = f.read()
...
>>> print(type(content))
<class 'bytes'>
```

## 2.14.2 逐行读取（Readline）

```
>>> with open("/etc/hosts") as f:
...     for line in f:
...         print(line, end=' ')
...
127.0.0.1       localhost
255.255.255.255 broadcasthost
::1             localhost
```

## 2.14.3 读取文件块

```
>>> chunk_size = 16
>>> content = ''
>>> with open('/etc/hosts') as f:
...     for c in iter(lambda: f.read(chunk_size), ''):
...         content += c
...
>>> print(content)
127.0.0.1       localhost
255.255.255.255 broadcasthost
::1             localhost
```

## 2.14.4 写入文件

```
>>> content = "Awesome Python!"
>>> with open("foo.txt", "w") as f:
...     f.write(content)
```

## 2.14.5 创建符号链接

```
>>> import os
>>> os.symlink("foo", "bar")
>>> os.readlink("bar")
'foo'
```

## 2.14.6 复制文件

```
>>> from distutils.file_util import copy_file
>>> copy_file("foo", "bar")
('bar', 1)
```

## 2.14.7 移动文件

```
>>> from distutils.file_util import move_file
>>> move_file("./foo", "./bar")
'./bar'
```

## 2.14.8 列出目录内容

```
>>> import os
>>> dirs = os.listdir(".")
```

在 Python 3.6 之后，我们可以使用 `os.scandir` 来列出目录内容。它更加方便，因为 `os.scandir` 返回一个 `os.DirEntry` 对象的迭代器。在这种情况下，我们可以通过访问 `os.DirEntry` 的属性来获取文件信息。更多信息可以在文档中找到。

```
>>> with os.scandir("foo") as it:
...     for entry in it:
...         st = entry.stat()
...
```

## 2.14.9 创建目录

类似于 `mkdir -p /path/to/dest`

```
>>> from distutils.dir_util import mkpath
>>> mkpath("foo/bar/baz")
['foo', 'foo/bar', 'foo/bar/baz']
```

## 2.14.10 复制目录

```
>>> from distutils.dir_util import copy_tree
>>> copy_tree("foo", "bar")
['bar/baz']
```

## 2.14.11 删除目录

```
>>> from distutils.dir_util import remove_tree
>>> remove_tree("dir")
```

## 2.14.12 路径拼接

```
>>> from pathlib import Path
>>> p = Path("/Users")
>>> p = p / "Guido" / "pysheeet"
>>> p
PosixPath('/Users/Guido/pysheeet')
```

## 2.14.13 获取绝对路径

```
>>> from pathlib import Path
>>> p = Path("README.rst")
PosixPath('/Users/Guido/pysheeet/README.rst')
```

## 2.14.14 获取主目录

```
>>> from pathlib import Path
>>> Path.home()
PosixPath('/Users/Guido')
```

## 2.14.15 获取当前目录

```
>>> from pathlib import Path
>>> p = Path("README.rst")
>>> p.cwd()
PosixPath('/Users/Guido/pysheeet')
```

## 2.14.16 获取路径属性

```
>>> from pathlib import Path
>>> p = Path("README.rst").absolute()
>>> p.root
'/'
>>> p.anchor
'/'
>>> p.parent
PosixPath('/Users/Guido/pysheeet')
>>> p.parent.parent
PosixPath('/Users/Guido')
>>> p.name
'README.rst'
>>> p.suffix
'.rst'
>>> p.stem
'README'
>>> p.as_uri()
'file:///Users/Guido/pysheeet/README.rst'
```

## 2.14.17 读取 gzip 压缩的 CSV 文件

```
import gzip
import csv

f = "example.gz"
with gzip.open(f, 'rt', newline='') as gz:
    reader = csv.DictReader(gz)
    for row in reader:
        print(row)
```

## 2.14.18 Linux Inotify

```
import selectors
import struct
import ctypes
import sys
import os

from pathlib import Path
from ctypes.util import find_library

# ref: <sys/inotify.h>
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200

INOTIFY_EVENT = "iIII"
INOTIFY_EVENT_LEN = struct.calcsize(INOTIFY_EVENT)

lib = find_library("c")
assert lib

libc = ctypes.CDLL(lib)

class Inotify(object):
    def __init__(self, path):
        self._path = path
        self._fd = None
        self._wd = None
        self._buf = b""
        self._sel = selectors.DefaultSelector()

    def init(self):
        fd = libc.inotify_init()
        if fd < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"{os.strerror(errno)}")
        return fd

    def watch(self, fd, path):
        p = str(path).encode("utf8")
        wd = libc.inotify_add_watch(fd, p, IN_CREATE | IN_DELETE)
        if wd < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"{os.strerror(errno)}")
        return wd

    def remove(self, fd, wd):
        libc.inotify_rm_watch(self._fd, self._wd)

    def handle(self, fd, *a):
        b = os.read(fd, 1024)
        if not b:
            return
        yield from self.parse(b);

    def parse(self, buf):
        self._buf += buf
        while True:
            l = len(self._buf)
            if l < INOTIFY_EVENT_LEN:
                break

            hd = self._buf[:INOTIFY_EVENT_LEN]
            wd, mask, cookie, length = struct.unpack(INOTIFY_EVENT, hd)
            event_length = INOTIFY_EVENT_LEN + length
            if l < event_length:
                break

            filename = self._buf[INOTIFY_EVENT_LEN:event_length]
            self._buf = self._buf[event_length:]
            yield mask, filename.rstrip(b"\0").decode("utf8")

    def __enter__(self):
        self._fd = self.init()
        self._wd = self.watch(self._fd, self._path)
        self._sel.register(self._fd, selectors.EVENT_READ, self.handle)
        return self

    def __exit__(self, *e):
        self.remove(self._fd, self._wd)
        if len(e) > 0 and e[0]:
            print(e, file=sys.stderr)

    def run(self):
        while True:
            events = self._sel.select()
            for k, mask in events:
                cb = k.data
                yield from cb(k.fileobj, mask)
```

## 2.15 操作系统

**目录**

- 操作系统
  - 获取 CPU 数量
  - 设置亲和性

## 2.15.1 获取 CPU 数量

```
>>> import os
>>> os.cpu_count()
```

## 2.15.2 设置亲和性

```
# 在 Linux 上运行
import os

pid = os.getpid()
affinity = {1}
os.sched_setaffinity(pid, affinity)
```

本部分的目标是提供包含内置模块和第三方模块用法的常用代码片段。

## 3.1 正则表达式

**目录**

- 正则表达式
  - 比较 HTML 标签
  - re.findall() 匹配字符串
  - 分组比较
  - 非捕获分组
  - 反向引用
  - 命名分组 (?P<name>)
  - 替换字符串
  - 前瞻与后顾
  - 匹配常见用户名或密码
  - 匹配十六进制颜色值
  - 匹配电子邮件
  - 匹配 URL
  - 匹配 IP 地址
  - 匹配 Mac 地址
  - 词法分析器

## 3.1.1 比较 HTML 标签

| 标签类型 | 格式 | 示例 |
|----------|--------|---------|
| 所有标签 | <[^>]+> | <br />, <a> |
| 开始标签 | <[^/>][^>]*> | <a>, <table> |
| 结束标签 | </[^>]+> | </p>, </a> |
| 自闭合标签 | <[^/>]+/> | <br /> |

```
# 开始标签
>>> re.search('<[^/>][^>]*>', '<table>') != None
True
>>> re.search('<[^/>][^>]*>', '<a href="#label">') != None
True
>>> re.search('<[^/>][^>]*>', '<img src="/img">') != None
True
>>> re.search('<[^/>][^>]*>', '</table>') != None
False

# 结束标签
>>> re.search('</[^>]+>', '</table>') != None
True

# 自闭合标签
>>> re.search('<[^/>]+/>', '<br />') != None
True
```

## 3.1.2 re.findall() 匹配字符串

```
# 分割所有字符串
>>> source = "Hello World Ker HAHA"
>>> re.findall('[\w]+', source)
['Hello', 'World', 'Ker', 'HAHA']

# 解析 python.org 网站
>>> import urllib
>>> import re
>>> s = urllib.urlopen('https://www.python.org')
>>> html = s.read()
>>> s.close()
>>> print("open tags")
open tags
>>> re.findall('<[^/>][^>]*>', html)[0:2]
['<!doctype html>', '<!--[if lt IE 7]>']
>>> print("close tags")
close tags
>>> re.findall('</[^>]+>', html)[0:2]
['</script>', '</title>']
>>> print("self-closing tags")
self-closing tags
>>> re.findall('<[^/>]+/>', html)[0:2]
[]
```

## 3.1.3 分组比较

```
# (...) 将正则表达式分组
>>> m = re.search(r'(\d{4})-(\d{2})-(\d{2})', '2016-01-01')
>>> m
<_sre.SRE_Match object; span=(0, 10), match='2016-01-01'>
>>> m.groups()
('2016', '01', '01')
>>> m.group()
'2016-01-01'
>>> m.group(1)
'2016'
>>> m.group(2)
'01'
>>> m.group(3)
'01'

# 嵌套分组
>>> m = re.search(r'(((\d{4})-\d{2})-\d{2})', '2016-01-01')
>>> m.groups()
('2016-01-01', '2016-01', '2016')
>>> m.group()
'2016-01-01'
>>> m.group(1)
'2016-01-01'
>>> m.group(2)
'2016-01'
>>> m.group(3)
'2016'
```

## 3.1.4 非捕获分组

```
# 非捕获分组
>>> url = 'http://stackoverflow.com/'
>>> m = re.search('(?:http|ftp)://([^/\r\n]+)(/[^\r\n]*)?', url)
>>> m.groups()
('stackoverflow.com', '/')

# 捕获分组
>>> m = re.search('(http|ftp)://([^/\r\n]+)(/[^\r\n]*)?', url)
>>> m.groups()
('http', 'stackoverflow.com', '/')
```## 3.1.5 反向引用

```
# 比较 'aa', 'bb'
>>> re.search(r'([a-z])\1$', 'aa') != None
True
>>> re.search(r'([a-z])\1$', 'bb') != None
True
>>> re.search(r'([a-z])\1$', 'ab') != None
False

# 比较开始标签和结束标签
>>> pattern = r'<([^>]+)>[\s\S]*?</\1>'
>>> re.search(pattern, '<bold> test </bold>') != None
True
>>> re.search(pattern, '<h1> test </h1>') != None
True
>>> re.search(pattern, '<bold> test </h1>') != None
False
```

## 3.1.6 命名分组 `(?P<name>)`

```
# 分组引用 ``(?P<name>...)``
>>> pattern = '(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
>>> m = re.search(pattern, '2016-01-01')
>>> m.group('year')
'2016'
>>> m.group('month')
'01'
>>> m.group('day')
'01'

# 反向引用 ``(?P=name)``
>>> re.search('^(?P<char>[a-z])(?P=char)', 'aa')
<_sre.SRE_Match object at 0x10ae0f288>
```

## 3.1.7 替换字符串

```
# 基本替换
>>> res = "1a2b3c"
>>> re.sub(r'[a-z]', ' ', res)
'1 2 3 '

# 带分组引用的替换
>>> date = r'2016-01-01'
>>> re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\2/\3/\1/', date)
'01/01/2016/'

# 驼峰式命名转下划线命名
>>> def convert(s):
...     res = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)
...     return re.sub(r'([a-z])([A-Z])',r'\1_\2', res).lower()
...
>>> convert('CamelCase')
'camel_case'
>>> convert('CamelCamelCase')
'camel_camel_case'
>>> convert('SimpleHTTPServer')
'simple_http_server'
```

## 3.1.8 环视

| 符号 | 比较方向 |
| :--- | :--- |
| `(?=...)` | 从左向右 |
| `(?!...)` | 从左向右 |
| `(?<=...)` | 从右向左 |
| `(?<!...)` | 从右向左 |

```
python
# 基本用法
>>> re.sub('(?=\d{3})', ' ', '12345')
' 1 2 345'
>>> re.sub('(?!\d{3})', ' ', '12345')
'123 4 5 '
>>> re.sub('(?<=\d{3})', ' ', '12345')
'123 4 5 '
>>> re.sub('(?<!\d{3})', ' ', '12345')
' 1 2 345'
```

## 3.1.9 匹配常见用户名或密码

```
python
>>> re.match('^[a-zA-Z0-9-_]{3,16}$', 'Foo') is not None
True
>>> re.match('^\w|[-_]{3,16}$', 'Foo') is not None
True
```

## 3.1.10 匹配十六进制颜色值

```
python
>>> re.match('^#?([a-f0-9]{6}|[a-f0-9]{3})$', '#ffffff')
<_sre.SRE_Match object at 0x10886f6c0>
>>> re.match('^#?([a-f0-9]{6}|[a-f0-9]{3})$', '#fffffh')
<_sre.SRE_Match object at 0x10886f288>
```

## 3.1.11 匹配电子邮件

```
python
>>> re.match('^([a-z0-9_\-\.]+)@([\da-z\.-]+)\.([a-z\.]{2,6})$', 'hello.world@example.com')
...         'hello.world@example.com')
<_sre.SRE_Match object at 0x1087a4d40>

# 或者

>>> exp = re.compile(r'''^([a-zA-Z0-9._%-]+@
                   [a-zA-Z0-9.-]+
                  \.[a-zA-Z]{2,4})*$''', re.X)
>>> exp.match('hello.world@example.hello.com')
<_sre.SRE_Match object at 0x1083efd50>
>>> exp.match('hello%world@example.hello.com')
<_sre.SRE_Match object at 0x1083efeb8>
```

## 3.1.12 匹配 URL

```
python
>>> exp = re.compile(r'''^(https?://)? # 匹配 http 或 https
                   ([\da-z\.-]+)          # 匹配域名
                   \.([a-z\.-]{2,6})       # 匹配顶级域名
                   ([/\w \.-]*)\/?$''', re.X)
>>> exp.match('www.google.com')
<_sre.SRE_Match object at 0x10f01ddf8>
>>> exp.match('http://www.example')
<_sre.SRE_Match object at 0x10f01dd50>
>>> exp.match('http://www.example/file.html')
<_sre.SRE_Match object at 0x10f01ddf8>
>>> exp.match('http://www.example/file!.html')
```

## 3.1.13 匹配 IP 地址

| 符号 | 描述 |
|----------|-------------|
| (?:...) | 不捕获分组 |
| 25[0-5] | 匹配 251-255 模式 |
| 2[0-4][0-9] | 匹配 200-249 模式 |
| [1]?[0-9][0-9] | 匹配 0-199 模式 |

```
python
>>> exp = re.compile(r'''^(?:(?:25[0-5]
                   |2[0-4][0-9]
                   |[1]?[0-9][0-9]?)\.){3}
                   (?:25[0-5]
                   |2[0-4][0-9]
                   |[1]?[0-9][0-9]?)$''', re.X)
>>> exp.match('192.168.1.1')
<_sre.SRE_Match object at 0x108f47ac0>
>>> exp.match('255.255.255.0')
<_sre.SRE_Match object at 0x108f47b28>
>>> exp.match('172.17.0.5')
<_sre.SRE_Match object at 0x108f47ac0>
>>> exp.match('256.0.0.0') is None
True
```

## 3.1.14 匹配 MAC 地址

```
>>> import random
>>> mac = [random.randint(0x00, 0x7f),
...        random.randint(0x00, 0x7f),
...        random.randint(0x00, 0x7f),
...        random.randint(0x00, 0x7f),
...        random.randint(0x00, 0x7f),
...        random.randint(0x00, 0x7f)]
>>> mac = ':'.join(map(lambda x: "%02x" % x, mac))
>>> mac
'3c:38:51:05:03:1e'
>>> exp = re.compile(r'''[0-9a-f]{2}([:])
...                     [0-9a-f]{2}(\1[0-9a-f]{2}){4}$''', re.X)
>>> exp.match(mac) is not None
True
```

## 3.1.15 词法分析器

```
>>> import re
>>> from collections import namedtuple
>>> tokens = [r'(?P<NUMBER>\d+)',
...            r'(?P<PLUS>\+)',
...            r'(?P<MINUS>-)',
...            r'(?P<TIMES>\*)',
...            r'(?P<DIVIDE>/)',
...            r'(?P<WS>\s+)']
>>> lex = re.compile('|'.join(tokens))
>>> Token = namedtuple('Token', ['type', 'value'])
>>> def tokenize(text):
...     scan = lex.scanner(text)
...     return (Token(m.lastgroup, m.group())
...             for m in iter(scan.match, None) if m.lastgroup != 'WS')
...
>>> for _t in tokenize('9 + 5 * 2 - 7'):
...     print(_t)
...
Token(type='NUMBER', value='9')
Token(type='PLUS', value='+')
Token(type='NUMBER', value='5')
Token(type='TIMES', value='*')
Token(type='NUMBER', value='2')
Token(type='MINUS', value='-')
Token(type='NUMBER', value='7')
Token(type='NUMBER', value='2')
Token(type='MINUS', value='-')
Token(type='NUMBER', value='7')
```

## 3.2 Socket

对于大多数程序员来说，即使 Python 提供了诸如 httplib、urllib、imaplib、telnetlib 等高级网络接口，Socket 编程也是不可避免的。一些类 Unix 系统的接口（例如 Netlink、内核加密）是通过 socket 接口调用的。为了减轻阅读冗长文档或源代码的痛苦，本速查表试图收集一些与底层 socket 编程相关的常见或不常见的代码片段。

+   **目录**
- Socket
  - 获取主机名
  - 从字符串获取地址族和套接字地址
  - 转换主机序与网络序字节序
  - IP 点分十进制字符串与字节格式互转
  - MAC 地址与字节格式互转
  - 简单的 TCP 回显服务器
  - 通过 IPv6 的简单 TCP 回显服务器
  - 禁用 IPv6 Only
  - 通过 SocketServer 的简单 TCP 回显服务器
  - 简单的 TLS/SSL TCP 回显服务器
  - 在 TLS/SSL TCP 回显服务器上设置加密算法
  - 简单的 UDP 回显服务器
  - 通过 SocketServer 的简单 UDP 回显服务器
  - 简单的 UDP 客户端 - 发送方
  - 广播 UDP 数据包
  - 简单的 UNIX 域套接字
  - 简单的双向进程通信
  - 简单的异步 TCP 服务器 - 线程
  - 简单的异步 TCP 服务器 - select
  - 简单的异步 TCP 服务器 - poll
  - 简单的异步 TCP 服务器 - epoll
  - 简单的异步 TCP 服务器 - kqueue
  - 高级 API - selectors
- 通过 selectors 的简单非阻塞 TLS/SSL 套接字
- "socketpair" - 类似于 PIPE
- 使用 sendfile 复制数据
- 通过 sendfile 发送文件
- Linux 内核加密 API - AF_ALG
- 通过 AF_ALG 进行 AES-CBC 加密/解密
- 通过 AF_ALG 进行 AES-GCM 加密/解密
- 使用 sendfile 对 AES-GCM 加密/解密文件
- 比较 AF_ALG 与 cryptography 模块的性能
- 嗅探 IP 数据包
- 嗅探 TCP 数据包
- 嗅探 ARP 数据包

## 3.2.1 获取主机名

```
>>> import socket
>>> socket.gethostname()
'MacBookPro-4380.local'
>>> hostname = socket.gethostname()
>>> socket.gethostbyname(hostname)
'172.20.10.4'
>>> socket.gethostbyname('localhost')
'127.0.0.1'
```

## 3.2.2 从字符串获取地址族和套接字地址

```
import socket
import sys

try:
    for res in socket.getaddrinfo(sys.argv[1], None,
                                proto=socket.IPPROTO_TCP):
        family = res[0]
        sockaddr = res[4]
        print(family, sockaddr)
except socket.gaierror:
    print("Invalid")

输出:
$ gai.py 192.0.2.244
AddressFamily.AF_INET ('192.0.2.244', 0)
$ gai.py 2001:db8:f00d::1:d
```

## 3.2.3 转换主机字节序与网络字节序

```python
# little-endian machine
>>> import socket
>>> a = 1 # host endian
>>> socket.htons(a) # network endian
256
>>> socket.htonl(a) # network endian
16777216
>>> socket.ntohs(256) # host endian
1
>>> socket.ntohl(16777216) # host endian
1

# big-endian machine
>>> import socket
>>> a = 1 # host endian
>>> socket.htons(a) # network endian
1
>>> socket.htonl(a) # network endian
1L
>>> socket.ntohs(1) # host endian
1
>>> socket.ntohl(1) # host endian
1L
```

## 3.2.4 IP 点分十进制字符串与字节格式转换

```python
>>> import socket
>>> addr = socket.inet_aton('127.0.0.1')
>>> addr
'\x7f\x00\x00\x01'
>>> socket.inet_ntoa(addr)
'127.0.0.1'
```

## 3.2.5 MAC 地址与字节格式转换

```python
>>> import binascii
>>> mac = '00:11:32:3c:c3:0b'
>>> byte = binascii.unhexlify(mac.replace(':', ''))
>>> byte
'\x00\x112<\xc3\x0b'
>>> binascii.hexlify(byte)
'0011323cc30b'
```

## 3.2.6 简单的 TCP 回显服务器

```python
import socket

class Server(object):
    def __init__(self, host, port):
        self._host = host
        self._port = port
    def __enter__(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.listen(10)
        self._sock = sock
        return self._sock
    def __exit__(self, *exc_info):
        if exc_info[0]:
            import traceback
            traceback.print_exception(*exc_info)
        self._sock.close()

if __name__ == '__main__':
    host = 'localhost'
    port = 5566
    with Server(host, 5566) as s:
        while True:
            conn, addr = s.accept()
            msg = conn.recv(1024)
            conn.send(msg)
            conn.close()
```

输出：

```
$ nc localhost 5566
Hello World
Hello World
```

## 3.2.7 通过 IPv6 的简单 TCP 回显服务器

```python
import contextlib
import socket

host = "::1"
port = 5566

@contextlib.contextmanager
def server(host, port):
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(10)
        yield s
    finally:
        s.close()

with server(host, port) as s:
    try:
        while True:
            conn, addr = s.accept()
            msg = conn.recv(1024)

            if msg:
                conn.send(msg)

            conn.close()
    except KeyboardInterrupt:
        pass
```

输出：

```
$ python3 ipv6.py &
[1] 25752
$ nc -6 ::1 5566
Hello IPv6
Hello IPv6
```

## 3.2.8 禁用仅 IPv6 模式

```python
#!/usr/bin/env python3

import contextlib
import socket

host = "::"
port = 5566

@contextlib.contextmanager
def server(host: str, port: int):
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        s.bind((host, port))
        s.listen(10)
        yield s
    finally:
        s.close()

with server(host, port) as s:
    try:
        while True:
            conn, addr = s.accept()
            remote = conn.getpeername()
            print(remote)
            msg = conn.recv(1024)

            if msg:
                conn.send(msg)

            conn.close()
    except KeyboardInterrupt:
        pass
```

输出：

```
$ python3 ipv6.py &
[1] 23914
$ nc -4 127.0.0.1 5566
('::ffff:127.0.0.1', 42604, 0, 0)
Hello IPv4
Hello IPv4
$ nc -6 ::1 5566
('::1', 50882, 0, 0)
Hello IPv6
Hello IPv6
$ nc -6 fe80::a00:27ff:fe9b:50ee%enp0s3 5566
('fe80::a00:27ff:fe9b:50ee%enp0s3', 42042, 0, 2)
Hello IPv6
```

## 3.2.9 通过 SocketServer 实现的简单 TCP 回显服务器

```python
>>> import SocketServer
>>> bh = SocketServer.BaseRequestHandler
>>> class handler(bh):
...     def handle(self):
...         data = self.request.recv(1024)
...         print(self.client_address)
...         self.request.sendall(data)
...
>>> host = ('localhost', 5566)
>>> s = SocketServer.TCPServer(
...     host, handler)
>>> s.serve_forever()
```

输出：

```
$ nc localhost 5566
Hello World
Hello World
```

## 3.2.10 简单的 TLS/SSL TCP 回显服务器

```python
import socket
import ssl

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('localhost', 5566))
sock.listen(10)

sslctx = ssl.SSLContext(ssl.PROTOCOL_TLSv1)
sslctx.load_cert_chain(certfile='./root-ca.crt',
                       keyfile='./root-ca.key')

try:
    while True:
        conn, addr = sock.accept()
        sslconn = sslctx.wrap_socket(conn, server_side=True)
        msg = sslconn.recv(1024)
        if msg:
            sslconn.send(msg)
        sslconn.close()
finally:
    sock.close()
```

输出：

```
# console 1
$ openssl genrsa -out root-ca.key 2048
$ openssl req -x509 -new -nodes -key root-ca.key -days 365 -out root-ca.crt
$ python3 ssl_tcp_server.py

# console 2
$ openssl s_client -connect localhost:5566
...
Hello SSL
Hello SSL
read:errno=0
```

## 3.2.11 在 TLS/SSL TCP 回显服务器上设置密码套件

```python
import socket
import json
import ssl

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('localhost', 5566))
sock.listen(10)

sslctx = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
sslctx.load_cert_chain(certfile='cert.pem',
                       keyfile='key.pem')
# set ssl ciphers
sslctx.set_ciphers('ECDH-ECDSA-AES128-GCM-SHA256')
print(json.dumps(sslctx.get_ciphers(), indent=2))

try:
    while True:
        conn, addr = sock.accept()
        sslconn = sslctx.wrap_socket(conn, server_side=True)
        msg = sslconn.recv(1024)
        if msg:
            sslconn.send(msg)
        sslconn.close()
finally:
    sock.close()
```

输出：

```
$ openssl ecparam -out key.pem -genkey -name prime256v1
$ openssl req -x509 -new -key key.pem -out cert.pem
$ python3 tls.py&
[2] 64565
[
  {
    "id": 50380845,
    "name": "ECDH-ECDSA-AES128-GCM-SHA256",
    "protocol": "TLSv1/SSLv3",
    "description": "ECDH-ECDSA-AES128-GCM-SHA256 TLSv1.2 Kx=ECDH/ECDSA Au=ECDH_ Enc=AESGCM(128) Mac=AEAD",
    "strength_bits": 128,
    "alg_bits": 128
  }
]

$ openssl s_client -connect localhost:5566 -cipher "ECDH-ECDSA-AES128-GCM-SHA256"
---
Hello ECDH-ECDSA-AES128-GCM-SHA256
Hello ECDH-ECDSA-AES128-GCM-SHA256
read:errno=0
```

## 3.2.12 简单的 UDP 回显服务器

```python
import socket

class UDPServer(object):
    def __init__(self, host, port):
        self._host = host
        self._port = port

    def __enter__(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self._host, self._port))
        self._sock = sock
        return sock
    def __exit__(self, *exc_info):
        if exc_info[0]:
            import traceback
            traceback.print_exception(*exc_info)
        self._sock.close()

if __name__ == '__main__':
    host = 'localhost'
    port = 5566
    with UDPServer(host, port) as s:
        while True:
            msg, addr = s.recvfrom(1024)
            s.sendto(msg, addr)
```

输出：

```
$ nc -u localhost 5566
Hello World
Hello World
```

## 3.2.13 通过SocketServer实现简单的UDP回显服务器

```python
>>> import SocketServer
>>> bh = SocketServer.BaseRequestHandler
>>> class handler(bh):
...     def handle(self):
...         m,s = self.request
...         s.sendto(m,self.client_address)
...         print(self.client_address)
...
>>> host = ('localhost', 5566)
>>> s = SocketServer.UDPServer(
...     host, handler)
>>> s.serve_forever()
```

输出：

```shell
$ nc -u localhost 5566
Hello World
Hello World
```

## 3.2.14 简单的UDP客户端 - 发送端

```python
>>> import socket
>>> import time
>>> sock = socket.socket(
...     socket.AF_INET,
...     socket.SOCK_DGRAM)
>>> host = ('localhost', 5566)
>>> while True:
...     sock.sendto("Hello\n", host)
...     time.sleep(5)
...
```

输出：

```shell
$ nc -lu localhost 5566
Hello
Hello
```

## 3.2.15 广播UDP数据包

```python
>>> import socket
>>> import time
>>> sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
>>> sock.bind(('', 0))
>>> sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST,1)
>>> while True:
...     m = '{0}\n'.format(time.time())
...     sock.sendto(m, ('<broadcast>', 5566))
...     time.sleep(5)
...
```

输出：

```bash
$ nc -k -w 1 -ul 5566
1431473025.72
```

## 3.2.16 简单的UNIX域套接字

```python
import socket
import contextlib
import os

@contextlib.contextmanager
def DomainServer(addr):
    try:
        if os.path.exists(addr):
            os.unlink(addr)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(addr)
        sock.listen(10)
        yield sock
    finally:
        sock.close()
        if os.path.exists(addr):
            os.unlink(addr)

addr = "./domain.sock"
with DomainServer(addr) as sock:
    while True:
        conn, _ = sock.accept()
        msg = conn.recv(1024)
        conn.send(msg)
        conn.close()
```

输出：

```bash
$ nc -U ./domain.sock
Hello
Hello
```

## 3.2.17 简单的双工进程通信

```python
import os
import socket

child, parent = socket.socketpair()
pid = os.fork()
try:
    if pid == 0:
        print('child pid: {}'.format(os.getpid()))

        child.send(b'Hello Parent')
        msg = child.recv(1024)
        print('p[{}] --> c[{}]: {}'.format(
            os.getppid(), os.getpid(), msg))
    else:
        print('parent pid: {}'.format(os.getpid()))

        # simple echo server (parent)
        msg = parent.recv(1024)
        print('c[{}] --> p[{}]: {}'.format(
            pid, os.getpid(), msg))
        parent.send(msg)
except KeyboardInterrupt:
    pass
finally:
    child.close()
    parent.close()
```

输出：

```bash
$ python3 socketpair_demo.py
parent pid: 9497
child pid: 9498
c[9498] --> p[9497]: b'Hello Parent'
p[9497] --> c[9498]: b'Hello Parent'
```

## 3.2.18 简单的异步TCP服务器 - 线程

```python
>>> from threading import Thread
>>> import socket
>>> def work(conn):
...     while True:
...         msg = conn.recv(1024)
...         conn.send(msg)
...
>>> sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
>>> sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
>>> sock.bind(('localhost', 5566))
>>> sock.listen(5)
>>> while True:
...     conn,addr = sock.accept()
...     t=Thread(target=work, args=(conn,))
...     t.daemon=True
...     t.start()
...
```

输出：（bash 1）

```bash
$ nc localhost 5566
Hello
Hello
```

输出：（bash 2）

```bash
$ nc localhost 5566
Ker Ker
Ker Ker
```

## 3.2.19 简单的异步TCP服务器 - select

```python
from select import select
import socket

host = ('localhost', 5566)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(host)
sock.listen(5)
rl = [sock]
wl = []
ml = {}
try:
    while True:
        r, w, _ = select(rl, wl, [])
        # process ready to read
        for _ in r:
            if _ == sock:
                conn, addr = sock.accept()
                rl.append(conn)
            else:
                msg = _.recv(1024)
                ml[_.fileno()] = msg
                wl.append(_)
        # process ready to write
        for _ in w:
            msg = ml[_.fileno()]
            _.send(msg)
            wl.remove(_)
            del ml[_.fileno()]
except:
    sock.close()
```

输出：（bash 1）

```bash
$ nc localhost 5566
Hello
Hello
```

输出：（bash 2）

```bash
$ nc localhost 5566
Ker Ker
Ker Ker
```

## 3.2.20 简单的异步TCP服务器 - poll

```python
from __future__ import print_function, unicode_literals

import socket
import select
import contextlib

host = 'localhost'
port = 5566

con = {}
req = {}
resp = {}

@contextlib.contextmanager
def Server(host,port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setblocking(False)
        s.bind((host,port))
        s.listen(10)
        yield s
    except socket.error:
        print("Get socket error")
        raise
    finally:
        if s: s.close()

@contextlib.contextmanager
def Poll():
    try:
        e = select.poll()
        yield e
    finally:
        for fd, c in con.items():
            e.unregister(fd)
            c.close()

def accept(server, poll):
    conn, addr = server.accept()
    conn.setblocking(False)
    fd = conn.fileno()
    poll.register(fd, select.POLLIN)
    req[fd] = conn
    con[fd] = conn

def recv(fd, poll):
    if fd not in req:
        return

    conn = req[fd]
    msg = conn.recv(1024)
    if msg:
        resp[fd] = msg
        poll.modify(fd, select.POLLOUT)
    else:
        conn.close()
        del con[fd]

    del req[fd]

def send(fd, poll):
    if fd not in resp:
        return

    conn = con[fd]
    msg = resp[fd]
    b = 0
    total = len(msg)
    while total > b:
        l = conn.send(msg)
        msg = msg[l:]
        b += l

    del resp[fd]
    req[fd] = conn
    poll.modify(fd, select.POLLIN)

try:
    with Server(host, port) as server, Poll() as poll:
        poll.register(server.fileno())
        while True:
            events = poll.poll(1)
            for fd, e in events:
                if fd == server.fileno():
                    accept(server, poll)
                elif e & (select.POLLIN | select.POLLPRI):
                    recv(fd, poll)
                elif e & select.POLLOUT:
                    send(fd, poll)
except KeyboardInterrupt:
    pass
```

输出：（bash 1）

```bash
$ python3 poll.py &
[1] 3036
$ nc localhost 5566
Hello poll
Hello poll
Hello Python Socket Programming
Hello Python Socket Programming
```

输出：（bash 2）

```bash
$ nc localhost 5566
Hello Python
Hello Python
Hello Awesome Python
Hello Awesome Python
```

## 3.2.21 简单的异步TCP服务器 - epoll

```python
from __future__ import print_function, unicode_literals

import socket
import select
import contextlib

host = 'localhost'
port = 5566

con = {}
req = {}
resp = {}

@contextlib.contextmanager
def Server(host,port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setblocking(False)
        s.bind((host, port))
        s.listen(10)
        yield s
    except socket.error:
        print("Get socket error")
        raise
    finally:
        if s: s.close()

@contextlib.contextmanager
def Epoll():
    try:
        e = select.epoll()
        yield e
    finally:
        for fd in con: e.unregister(fd)
        e.close()

def accept(server, epoll):
    conn, addr = server.accept()
    conn.setblocking(0)
    fd = conn.fileno()
    epoll.register(fd, select.EPOLLIN)
    req[fd] = conn
    con[fd] = conn

def recv(fd, epoll):
    if fd not in req:
        return

    conn = req[fd]
    msg = conn.recv(1024)
    if msg:
        resp[fd] = msg
        epoll.modify(fd, select.EPOLLOUT)
    else:
        conn.close()
        del con[fd]

    del req[fd]

def send(fd, epoll):
    if fd not in resp:
        return

    conn = con[fd]
``````python
msg = resp[fd]
b = 0
total = len(msg)
while total > b:
    l = conn.send(msg)
    msg = msg[l:]
    b += l

del resp[fd]
req[fd] = conn
epoll.modify(fd, select.EPOLLIN)

try:
    with Server(host, port) as server, Epoll() as epoll:

        epoll.register(server.fileno())

        while True:
            events = epoll.poll(1)
            for fd, e in events:
                if fd == server.fileno():
                    accept(server, epoll)
                elif e & select.EPOLLIN:
                    recv(fd, epoll)
                elif e & select.EPOLLOUT:
                    send(fd, epoll)
except KeyboardInterrupt:
    pass
```

输出：(bash 1)

```
$ python3 epoll.py &
[1] 3036
$ nc localhost 5566
Hello epoll
Hello epoll
Hello Python Socket Programming
Hello Python Socket Programming
```

输出：(bash 2)

```
$ nc localhost 5566
Hello Python
Hello Python
Hello Awesome Python
Hello Awesome Python
```

## 3.2.22 简易异步 TCP 服务器 - kqueue

```python
from __future__ import print_function, unicode_literals

import socket
import select
import contextlib

if not hasattr(select, 'kqueue'):
    print("Not support kqueue")
    exit(1)

host = 'localhost'
port = 5566

con = {}
req = {}
resp = {}

@contextlib.contextmanager
def Server(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setblocking(False)
        s.bind((host, port))
        s.listen(10)
        yield s
    except socket.error:
        print("Get socket error")
        raise
    finally:
        if s: s.close()

@contextlib.contextmanager
def Kqueue():
    try:
        kq = select.kqueue()
        yield kq
    finally:
        kq.close()
        for fd, c in con.items(): c.close()

def accept(server, kq):
    conn, addr = server.accept()
    conn.setblocking(False)
    fd = conn.fileno()
    ke = select.kevent(conn.fileno(),
                       select.KQ_FILTER_READ,
                       select.KQ_EV_ADD)
    kq.control([ke], 0)
    req[fd] = conn
    con[fd] = conn

def recv(fd, kq):
    if fd not in req:
        return

    conn = req[fd]
    msg = conn.recv(1024)
    if msg:
        resp[fd] = msg
        # 移除读事件
        ke = select.kevent(fd,
                           select.KQ_FILTER_READ,
                           select.KQ_EV_DELETE)
        kq.control([ke], 0)
        # 添加写事件
        ke = select.kevent(fd,
                           select.KQ_FILTER_WRITE,
                           select.KQ_EV_ADD)
        kq.control([ke], 0)
        req[fd] = conn
        con[fd] = conn
    else:
        conn.close()
        del con[fd]

    del req[fd]

def send(fd, kq):
    if fd not in resp:
        return

    conn = con[fd]
    msg = resp[fd]
    b = 0
    total = len(msg)
    while total > b:
        l = conn.send(msg)
        msg = msg[l:]
        b += l

    del resp[fd]
    req[fd] = conn
    # 移除写事件
    ke = select.kevent(fd,
                       select.KQ_FILTER_WRITE,
                       select.KQ_EV_DELETE)
    kq.control([ke], 0)
    # 添加读事件
    ke = select.kevent(fd,
                       select.KQ_FILTER_READ,
                       select.KQ_EV_ADD)
    kq.control([ke], 0)

try:
    with Server(host, port) as server, Kqueue() as kq:

        max_events = 1024
        timeout = 1

        ke = select.kevent(server.fileno(),
                           select.KQ_FILTER_READ,
                           select.KQ_EV_ADD)

        kq.control([ke], 0)
        while True:
            events = kq.control(None, max_events, timeout)
            for e in events:
                fd = e.ident
                if fd == server.fileno():
                    accept(server, kq)
                elif e.filter == select.KQ_FILTER_READ:
                    recv(fd, kq)
                elif e.filter == select.KQ_FILTER_WRITE:
                    send(fd, kq)
except KeyboardInterrupt:
    pass
```

输出：(bash 1)

```
$ python3 kqueue.py &
[1] 3036
$ nc localhost 5566
Hello kqueue
Hello kqueue
Hello Python Socket Programming
Hello Python Socket Programming
```

输出：(bash 2)

```
$ nc localhost 5566
Hello Python
Hello Python
Hello Awesome Python
Hello Awesome Python
```

## 3.2.23 高级 API - selectors

```python
# 仅适用于 Python 3.4+
# 参考：selectors
import selectors
import socket
import contextlib

@contextlib.contextmanager
def Server(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(10)
        sel = selectors.DefaultSelector()
        yield s, sel
    except socket.error:
        print("Get socket error")
        raise
    finally:
        if s:
            s.close()

def read_handler(conn, sel):
    msg = conn.recv(1024)
    if msg:
        conn.send(msg)
    else:
        sel.unregister(conn)
        conn.close()

def accept_handler(s, sel):
    conn, _ = s.accept()
    sel.register(conn, selectors.EVENT_READ, read_handler)

host = 'localhost'
port = 5566
with Server(host, port) as (s,sel):
    sel.register(s, selectors.EVENT_READ, accept_handler)
    while True:
        events = sel.select()
        for sel_key, m in events:
            handler = sel_key.data
            handler(sel_key.fileobj, sel)
```

输出：(bash 1)

```
$ nc localhost 5566
Hello
Hello
```

输出：(bash 2)

```
$ nc localhost 5566
Hi
Hi
```

## 3.2.24 基于 selectors 的简易非阻塞 TLS/SSL 套接字

```python
import socket
import selectors
import contextlib
import ssl

from functools import partial

sslctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
sslctx.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

@contextlib.contextmanager
def Server(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(10)
        sel = selectors.DefaultSelector()
        yield s, sel
    except socket.error:
        print("Get socket error")
        raise
    finally:
        if s: s.close()
        if sel: sel.close()

def accept(s, sel):
    conn, _ = s.accept()
    sslconn = sslctx.wrap_socket(conn,
                                 server_side=True,
                                 do_handshake_on_connect=False)
    sel.register(sslconn, selectors.EVENT_READ, do_handshake)

def do_handshake(sslconn, sel):
    sslconn.do_handshake()
    sel.modify(sslconn, selectors.EVENT_READ, read)

def read(sslconn, sel):
    msg = sslconn.recv(1024)
    if msg:
        sel.modify(sslconn,
                   selectors.EVENT_WRITE,
                   partial(write, msg=msg))
    else:
        sel.unregister(sslconn)
        sslconn.close()

def write(sslconn, sel, msg=None):
    if msg:
        sslconn.send(msg)
    sel.modify(sslconn, selectors.EVENT_READ, read)

host = 'localhost'
port = 5566
try:
    with Server(host, port) as (s,sel):
        sel.register(s, selectors.EVENT_READ, accept)
        while True:
            events = sel.select()
            for sel_key, m in events:
                handler = sel_key.data
                handler(sel_key.fileobj, sel)
except KeyboardInterrupt:
    pass
```

输出：

```
# 控制台 1
$ openssl genrsa -out key.pem 2048
$ openssl req -x509 -new -nodes -key key.pem -days 365 -out cert.pem
$ python3 ssl_tcp_server.py &
$ openssl s_client -connect localhost:5566
...
---
Hello TLS
Hello TLS

# 控制台 2
$ openssl s_client -connect localhost:5566
...
---
Hello SSL
Hello SSL
```

## 3.2.25 “socketpair” - 类似于管道

```python
import socket
import os
import time

c_s, p_s = socket.socketpair()
try:
    pid = os.fork()
except OSError:
    print("Fork Error")
    raise

if pid:
    # 父进程
    c_s.close()
    while True:
        p_s.sendall("Hi! Child!")
        msg = p_s.recv(1024)
        print(msg)
        time.sleep(3)
    os.wait()
else:
    # 子进程
    p_s.close()
    while True:
        msg = c_s.recv(1024)
        print(msg)
        c_s.sendall("Hi! Parent!")
```

输出：

```
$ python ex.py
Hi! Child!
Hi! Parent!
Hi! Child!
Hi! Parent!
...
```

## 3.2.26 使用 sendfile 进行复制

```python
# 需要 Python 3.3 或更高版本
from __future__ import print_function, unicode_literals

import os
import sys

if len(sys.argv) != 3:
    print("Usage: cmd src dst")
    exit(1)
```

src = sys.argv[1]
dst = sys.argv[2]

with open(src, 'r') as s, open(dst, 'w') as d:
    st = os.fstat(s.fileno())

    offset = 0
    count = 4096
    s_len = st.st_size

    sfd = s.fileno()
    dfd = d.fileno()

    while s_len > 0:
        ret = os.sendfile(dfd, sfd, offset, count)
        offset += ret
        s_len -= ret

输出：

```
$ dd if=/dev/urandom of=dd.in bs=1M count=1024
1024+0 records in
1024+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 108.02 s, 9.9 MB/s
$ python3 sendfile.py dd.in dd.out
$ md5sum dd.in
e79afdd6aba71b7174142c0bbc289674  dd.in
$ md5sum dd.out
e79afdd6aba71b7174142c0bbc289674  dd.out
```

## 3.2.27 通过sendfile发送文件

```python
# 需要 python 3.5 或以上版本
from __future__ import print_function, unicode_literals

import os
import sys
import time
import socket
import contextlib

@contextlib.contextmanager
def server(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(10)
        yield s
    finally:
        s.close()

@contextlib.contextmanager
def client(host, port):
    try:
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.connect((host, port))
        yield c
    finally:
        c.close()

def do_sendfile(fout, fin, count, fin_len):
    l = fin_len
    offset = 0
    while l > 0:
        ret = fout.sendfile(fin, offset, count)
        offset += ret
        l -= ret

def do_recv(fout, fin):
    while True:
        data = fin.recv(4096)
        if not data: break
        fout.write(data)

host = 'localhost'
port = 5566

if len(sys.argv) != 3:
    print("usage: cmd src dst")
    exit(1)

src = sys.argv[1]
dst = sys.argv[2]
offset = 0

pid = os.fork()

if pid == 0:
    # 客户端
    time.sleep(3)
    with client(host, port) as c, open(src, 'rb') as f:
        fd = f.fileno()
        st = os.fstat(fd)
        count = 4096
        flen = st.st_size
        do_sendfile(c, f, count, flen)

else:
    # 服务器端
    with server(host, port) as s, open(dst, 'wb') as f:
        conn, addr = s.accept()
        do_recv(f, conn)
```

输出：

```bash
$ dd if=/dev/urandom of=dd.in bs=1M count=512
512+0 records in
512+0 records out
536870912 bytes (537 MB, 512 MiB) copied, 3.17787 s, 169 MB/s
$ python3 sendfile.py dd.in dd.out
$ md5sum dd.in
eadfd96c85976b1f46385e89dfd9c4a8  dd.in
$ md5sum dd.out
eadfd96c85976b1f46385e89dfd9c4a8  dd.out
```

## 3.2.28 Linux内核加密API - AF_ALG

```python
# 需要 python 3.6 或以上版本以及 Linux >=2.6.38
import socket
import hashlib
import contextlib

@contextlib.contextmanager
def create_alg(typ, name):
    s = socket.socket(socket.AF_ALG, socket.SOCK_SEQPACKET, 0)
    try:
        s.bind((typ, name))
        yield s
    finally:
        s.close()

msg = b'Python is awesome!'

with create_alg('hash', 'sha256') as algo:
    op, _ = algo.accept()
    with op:
        op.sendall(msg)
        data = op.recv(512)
        print(data.hex())

        # 检查数据
        h = hashlib.sha256(msg).digest()
        if h != data:
            raise Exception(f"sha256({h}) != af_alg({data})")
```

输出：

```
$ python3 af_alg.py
9d50bcac2d5e33f936ec2db7dc7b6579cba8e1b099d77c31d8564df46f66bdf5
```

## 3.2.29 通过AF_ALG进行AES-CBC加密/解密

```python
# 需要 python 3.6 或以上版本以及 Linux >=4.3
import contextlib
import socket
import os

BS = 16 # 字节
pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS).encode('utf-8')

upad = lambda s : s[0:-s[-1]]


@contextlib.contextmanager
def create_alg(typ, name):
    s = socket.socket(socket.AF_ALG, socket.SOCK_SEQPACKET, 0)
    try:
        s.bind((typ, name))
        yield s
    finally:
        s.close()


def encrypt(plaintext, key, iv):
    ciphertext = None
    with create_alg('skcipher', 'cbc(aes)') as algo:
        algo.setsockopt(socket.SOL_ALG, socket.ALG_SET_KEY, key)
        op, _ = algo.accept()
        with op:
            plaintext = pad(plaintext)
            op.sendmsg_afalg([plaintext],
                             op=socket.ALG_OP_ENCRYPT,
                             iv=iv)
            ciphertext = op.recv(len(plaintext))

    return ciphertext

def decrypt(ciphertext, key, iv):
    plaintext = None
    with create_alg('skcipher', 'cbc(aes)') as algo:
        algo.setsockopt(socket.SOL_ALG, socket.ALG_SET_KEY, key)
        op, _ = algo.accept()
        with op:
            op.sendmsg_afalg([ciphertext],
                             op=socket.ALG_OP_DECRYPT,
                             iv=iv)
            plaintext = op.recv(len(ciphertext))

    return upad(plaintext)


key = os.urandom(32)
iv = os.urandom(16)

plaintext = b"Demo AF_ALG"
ciphertext = encrypt(plaintext, key, iv)
plaintext = decrypt(ciphertext, key, iv)

print(ciphertext.hex())
print(plaintext)
```

输出：

```
$ python3 aes_cbc.py
01910e4bd6932674dba9bebd4fdf6cf2
b'Demo AF_ALG'
```

## 3.2.30 通过AF_ALG进行AES-GCM加密/解密

```python
# 需要 python 3.6 或以上版本以及 Linux >=4.9
import contextlib
import socket
import os

@contextlib.contextmanager
def create_alg(typ, name):
    s = socket.socket(socket.AF_ALG, socket.SOCK_SEQPACKET, 0)
    try:
        s.bind((typ, name))
        yield s
    finally:
        s.close()

def encrypt(key, iv, assoc, taglen, plaintext):
    """ 执行aes-gcm加密

    :param key: aes对称密钥
    :param iv: 初始向量
    :param assoc: 关联数据（用于完整性保护）
    :param taglen: 认证标签长度
    :param plaintext: 明文数据
    """

    assoclen = len(assoc)
    ciphertext = None
    tag = None

    with create_alg('aead', 'gcm(aes)') as algo:
        algo.setsockopt(socket.SOL_ALG,
                        socket.ALG_SET_KEY, key)
        algo.setsockopt(socket.SOL_ALG,
                        socket.ALG_SET_AEAD_AUTHSIZE,
                        None,
                        assoclen)

        op, _ = algo.accept()
        with op:
            msg = assoc + plaintext
            op.sendmsg_afalg([msg],
                             op=socket.ALG_OP_ENCRYPT,
                             iv=iv,
                             assoclen=assoclen)

            res = op.recv(assoclen + len(plaintext) + taglen)
            ciphertext = res[assoclen:-taglen]
            tag = res[-taglen:]

    return ciphertext, tag

def decrypt(key, iv, assoc, tag, ciphertext):
    """ 执行aes-gcm解密

    :param key: AES对称密钥
    :param iv: 初始向量
    :param assoc: 关联数据（用于完整性保护）
    :param tag: GCM认证标签
    :param ciphertext: 密文数据
    """
    plaintext = None
    assoclen = len(assoc)

    with create_alg('aead', 'gcm(aes)') as algo:
        algo.setsockopt(socket.SOL_ALG,
                        socket.ALG_SET_KEY, key)
        algo.setsockopt(socket.SOL_ALG,
                        socket.ALG_SET_AEAD_AUTHSIZE,
                        None,
                        assoclen)
        op, _ = algo.accept()
        with op:
            msg = assoc + ciphertext + tag
            op.sendmsg_afalg([msg],
                             op=socket.ALG_OP_DECRYPT, iv=iv,
                             assoclen=assoclen)

            taglen = len(tag)
            res = op.recv(len(msg) - taglen)
            plaintext = res[assoclen:]

    return plaintext

key = os.urandom(16)
iv  = os.urandom(12)
assoc = os.urandom(16)

plaintext = b"Hello AES-GCM"
ciphertext, tag = encrypt(key, iv, assoc, 16, plaintext)
plaintext = decrypt(key, iv, assoc, tag, ciphertext)

print(ciphertext.hex())
print(plaintext)
```

输出：

```
$ python3 aes_gcm.py
2e27b67234e01bcb0ab6b451f4f870ce
b'Hello AES-GCM'
```

## 3.2.31 使用sendfile进行AES-GCM文件加密/解密

```python
# 需要 python 3.6 或以上版本以及 Linux >=4.9
import contextlib
import socket
import sys
import os

@contextlib.contextmanager
def create_alg(typ, name):
    s = socket.socket(socket.AF_ALG, socket.SOCK_SEQPACKET, 0)
    try:
        s.bind((typ, name))
        yield s
    finally:
        s.close()

def encrypt(key, iv, assoc, taglen, pfile):
    assoclen = len(assoc)
    ciphertext = None
    tag = None

    pfd = pfile.fileno()
    offset = 0
    st = os.fstat(pfd)
    totalbytes = st.st_size
```

## 3.2.32 比较 AF_ALG 与 cryptography 的性能

```python
# need python 3.6 or above & Linux >=4.9
import contextlib
import socket
import time
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

@contextlib.contextmanager
def create_alg(typ, name):
    s = socket.socket(socket.AF_ALG, socket.SOCK_SEQPACKET, 0)
    try:
        s.bind((typ, name))
        yield s
    finally:
        s.close()

def encrypt(key, iv, assoc, taglen, op, pfile, psize):
    assoclen = len(assoc)
    ciphertext = None
    tag = None
    offset = 0
    pfd = pfile.fileno()
    totalbytes = psize

    op.sendmsg_afalg(op=socket.ALG_OP_ENCRYPT,
                     iv=iv,
                     assoclen=assoclen,
                     flags=socket.MSG_MORE)

    op.sendall(assoc, socket.MSG_MORE)

    # using sendfile to encrypt file data
    os.sendfile(op.fileno(), pfd, offset, totalbytes)

    res = op.recv(assoclen + totalbytes + taglen)
    ciphertext = res[assoclen:-taglen]
    tag = res[-taglen:]

    return ciphertext, tag

def decrypt(key, iv, assoc, tag, op, ciphertext):
    plaintext = None
    assoclen = len(assoc)

    msg = assoc + ciphertext + tag
    op.sendmsg_afalg([msg],
                     op=socket.ALG_OP_DECRYPT, iv=iv,
                     assoclen=assoclen)

    taglen = len(tag)
    res = op.recv(len(msg) - taglen)
    plaintext = res[assoclen:]

    return plaintext

key = os.urandom(16)
iv = os.urandom(12)
assoc = os.urandom(16)
assoclen = len(assoc)

count = 1000000
plain = "tmp.rand"

# create a tmp file
with open(plain, 'wb') as f:
    f.write(os.urandom(4096))
    f.flush()

# profile AF_ALG with sendfile (zero-copy)
with open(plain, 'rb') as pf, \
create_alg('aead', 'gcm(aes)') as enc_algo, \
create_alg('aead', 'gcm(aes)') as dec_algo:

    enc_algo.setsockopt(socket.SOL_ALG,
                        socket.ALG_SET_KEY, key)
    enc_algo.setsockopt(socket.SOL_ALG,
                        socket.ALG_SET_AEAD_AUTHSIZE,
                        None,
                        assoclen)

    dec_algo.setsockopt(socket.SOL_ALG,
                        socket.ALG_SET_KEY, key)
    dec_algo.setsockopt(socket.SOL_ALG,
                        socket.ALG_SET_AEAD_AUTHSIZE,
                        None,
                        assoclen)

    enc_op, _ = enc_algo.accept()
    dec_op, _ = dec_algo.accept()

    st = os.fstat(pf.fileno())
    psize = st.st_size

    with enc_op, dec_op:

        s = time.time()

        for _ in range(count):
            ciphertext, tag = encrypt(key, iv, assoc, 16, enc_op, pf, psize)
            plaintext = decrypt(key, iv, assoc, tag, dec_op, ciphertext)

        cost = time.time() - s

        print(f"total cost time: {cost}. [AF_ALG]")

# profile cryptography (no zero-copy)
with open(plain, 'rb') as pf:

    aesgcm = AESGCM(key)

    s = time.time()

    for _ in range(count):
        pf.seek(0, 0)
        plaintext = pf.read()
        ciphertext = aesgcm.encrypt(iv, plaintext, assoc)
        plaintext = aesgcm.decrypt(iv, ciphertext, assoc)

    cost = time.time() - s

    print(f"total cost time: {cost}. [cryptography]")

# clean up
os.remove(plain)
```

输出：

```
$ python3 aes-gcm.py
total cost time: 15.317010641098022. [AF_ALG]
total cost time: 50.256704807281494. [cryptography]
```

## 3.2.33 嗅探 IP 数据包

```python
from ctypes import *
import socket
import struct

# ref: IP protocol numbers
PROTO_MAP = {
    1 : "ICMP",
    2 : "IGMP",
    6 : "TCP",
    17 : "UDP",
    27 : "RDP"}

class IP(Structure):
    """ IP header Structure

    In linux api, it define as below:

    struct ip {
        u_char          ip_hl:4;    /* header_len */
        u_char          ip_v:4;     /* version */
        u_char          ip_tos;     /* type of service */
        short           ip_len;     /* total len */
        u_short         ip_id;      /* identification */
        short           ip_off;     /* offset field */
        u_char          ip_ttl;     /* time to live */
        u_char          ip_p;       /* protocol */
        u_short         ip_sum;     /* checksum */
        struct in_addr  ip_src;     /* source */
        struct in_addr  ip_dst;     /* destination */
    };
    """
    _fields_ = [("ip_hl"  , c_ubyte, 4),  # 4 bit
                ("ip_v"   , c_ubyte, 4),  # 1 byte
                ("ip_tos" , c_uint8),      # 2 byte
                ("ip_len" , c_uint16),     # 4 byte
                ("ip_id"  , c_uint16),     # 6 byte
                ("ip_off" , c_uint16),     # 8 byte
                ("ip_ttl" , c_uint8),      # 9 byte
                ("ip_p" , c_uint8),   # 10 byte
                ("ip_sum", c_uint16), # 12 byte
                ("ip_src", c_uint32), # 16 byte
                ("ip_dst", c_uint32)] # 20 byte

    def __new__(cls, buf=None):
        return cls.from_buffer_copy(buf)
    def __init__(self, buf=None):
        src = struct.pack("<L", self.ip_src)
        self.src = socket.inet_ntoa(src)
        dst = struct.pack("<L", self.ip_dst)
        self.dst = socket.inet_ntoa(dst)
        try:
            self.proto = PROTO_MAP[self.ip_p]
        except KeyError:
            print("{} Not in map".format(self.ip_p))
            raise

host = '0.0.0.0'
s = socket.socket(socket.AF_INET,
                   socket.SOCK_RAW,
                   socket.IPPROTO_ICMP)
s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
s.bind((host, 0))

print("Sniffer start...")
try:
    while True:
        buf = s.recvfrom(65535)[0]
        ip_header = IP(buf[:20])
        print('{0}: {1} -> {2}'.format(ip_header.proto,
                                        ip_header.src,
                                        ip_header.dst))
except KeyboardInterrupt:
    s.close()
```

输出：

```
$ python sniffer.py
Sniffer start...
ICMP: 127.0.0.1 -> 127.0.0.1
ICMP: 127.0.0.1 -> 127.0.0.1
ICMP: 127.0.0.1 -> 127.0.0.1
```

```
$ ping -c 3 localhost
PING localhost (127.0.0.1): 56 data bytes
64 bytes from 127.0.0.1: icmp_seq=0 ttl=64 time=0.063 ms
64 bytes from 127.0.0.1: icmp_seq=1 ttl=64 time=0.087 ms
64 bytes from 127.0.0.1: icmp_seq=2 ttl=64 time=0.159 ms
```

--- localhost ping 统计信息 ---
已发送 3 个数据包，已接收 3 个数据包，0.0% 的数据包丢失
往返时间 最小/平均/最大/标准差 = 0.063/0.103/0.159/0.041 毫秒

## 3.2.34 嗅探 TCP 数据包

```python
#!/usr/bin/env python3.6
"""
基于 RFC-793，下图显示了 TCP 头部格式：

0                   1                   2                   3
0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          源端口号          |       目标端口号        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        序列号                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    确认号                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  数据 |           |U|A|P|R|S|F|                               |
| 偏移量| 保留字段  |R|C|S|S|Y|I|            窗口大小             |
|       |           |G|K|H|T|N|N|                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           校验和            |         紧急指针        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    选项                    |    填充字段    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             数据                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

在 Linux API (uapi/linux/tcp.h) 中，定义了 TCP 头部：

struct tcphdr {
    __be16  source;
    __be16  dest;
    __be32  seq;
    __be32  ack_seq;
#if defined(__LITTLE_ENDIAN_BITFIELD)
    __u16   res1:4,
            doff:4,
            fin:1,
            syn:1,
            rst:1,
            psh:1,
            ack:1,
            urg:1,
            ece:1,
            cwr:1;
#elif defined(__BIG_ENDIAN_BITFIELD)
    __u16   doff:4,
            res1:4,
            cwr:1,
            ece:1,
            urg:1,
            ack:1,
            psh:1,
            rst:1,
            syn:1,
            fin:1;
#else
#error    "Adjust your <asm/byteorder.h> defines"
#endif
    __be16 window;
    __sum16 check;
    __be16 urg_ptr;
};
"""
```

```python
import sys
import socket
import platform

from struct import unpack
from contextlib import contextmanager

un = platform.system()
if un != "Linux":
    print(f"{un} is not supported!")
    sys.exit(1)

@contextmanager
def create_socket():
    ''' 创建一个 TCP 原始套接字 '''
    s = socket.socket(socket.AF_INET,
                      socket.SOCK_RAW,
                      socket.IPPROTO_TCP)
    try:
        yield s
    finally:
        s.close()
```

```python
try:
    with create_socket() as s:
        while True:
            pkt, addr = s.recvfrom(65535)

            # 前 20 字节是 IP 头部
            iphdr = unpack('!BBHHHBBH4s4s', pkt[0:20])
            iplen = (iphdr[0] & 0xf) * 4

            # 接下来的 20 字节是 TCP 头部
            tcphdr = unpack('!HHLLBBHHH', pkt[iplen:iplen+20])

            source = tcphdr[0]
            dest = tcphdr[1]
            seq = tcphdr[2]
            ack_seq = tcphdr[3]
            dr = tcphdr[4]
            flags = tcphdr[5]
            window = tcphdr[6]
            check = tcphdr[7]
            urg_ptr = tcphdr[8]

            doff = dr >> 4
            fin = flags & 0x01
            syn = flags & 0x02
            rst = flags & 0x04
            psh = flags & 0x08
            ack = flags & 0x10
            urg = flags & 0x20
            ece = flags & 0x40
            cwr = flags & 0x80

            tcplen = (doff) * 4
            h_size = iplen + tcplen

            #从数据包中获取数据
            data = pkt[h_size:]

            if not data:
                continue

            print("------------- TCP_HEADER ----------------")
            print(f"源端口:            {source}")
            print(f"目标端口:       {dest}")
            print(f"序列号:        {seq}")
            print(f"确认号:  {ack_seq}")
            print(f"数据偏移量:            {doff}")
            print(f"FIN:                    {fin}")
            print(f"SYN:                    {syn}")
            print(f"RST:                    {rst}")
            print(f"PSH:                    {psh}")
            print(f"ACK:                    {ack}")
            print(f"URG:                    {urg}")
            print(f"ECE:                    {ece}")
            print(f"CWR:                    {cwr}")
            print(f"窗口大小:                 {window}")
            print(f"校验和:               {check}")
            print(f"紧急指针:           {urg_ptr}")
            print("------------------- DATA -------------------")
            print(data)

except KeyboardInterrupt:
    pass
```

output:

```
$ python3.6 tcp.py
--------------- TCP_HEADER ---------------
源端口:       38352
目标端口:  8000
序列号:   2907801591
确认号: 398995857
数据偏移量:       8
FIN:               0
SYN:               0
RST:               0
PSH:               8
ACK:               16
URG:               0
ECE:               0
CWR:               0
窗口大小:            342
校验和:          65142
紧急指针:      0
---------------- DATA ----------------
b'GET / HTTP/1.1\r\nHost: localhost:8000\r\nUser-Agent: curl/7.47.0\r\nAccept: */*\r\n\r\n'
```

### 3.2.35 嗅探 ARP 数据包

```python
"""
以太网数据包头部

struct ethhdr {
    unsigned char h_dest[ETH_ALEN];   /* 目标以太网地址 */
    unsigned char h_source[ETH_ALEN]; /* 源以太网地址   */
    __be16        h_proto;            /* 数据包类型ID字段 */
} __attribute__((packed));

ARP 数据包头部

struct arphdr {
    uint16_t htype;    /* 硬件类型            */
    uint16_t ptype;    /* 协议类型            */
    u_char   hlen;     /* 硬件地址长度  */
    u_char   plen;     /* 协议地址长度  */
    uint16_t opcode;   /* 操作码           */
    u_char   sha[6];   /* 发送方硬件地址  */
    u_char   spa[4];   /* 发送方IP地址        */
    u_char   tha[6];   /* 目标硬件地址  */
    u_char   tpa[4];   /* 目标IP地址        */
};
"""

import socket
import struct
import binascii

rawSocket = socket.socket(socket.AF_PACKET,
                       socket.SOCK_RAW,
                       socket.htons(0x0003))

while True:

    packet = rawSocket.recvfrom(2048)
    ethhdr = packet[0][0:14]
    eth = struct.unpack("!6s6s2s", ethhdr)

    arphdr = packet[0][14:42]
    arp = struct.unpack("2s2s1s1s2s6s4s6s4s", arphdr)
    # 跳过非 ARP 数据包
    ethertype = eth[2]
    if ethertype != '\x08\x06': continue

    print("------------------- ETHERNET_FRAME -------------------")
    print("目标MAC:      ", binascii.hexlify(eth[0]))
    print("源MAC:    ", binascii.hexlify(eth[1]))
    print("类型:          ", binascii.hexlify(ethertype))
    print("------------------- ARP_HEADER -----------------------")
    print("硬件类型: ", binascii.hexlify(arp[0]))
    print("协议类型: ", binascii.hexlify(arp[1]))
    print("硬件大小: ", binascii.hexlify(arp[2]))
    print("协议大小: ", binascii.hexlify(arp[3]))
    print("操作码:        ", binascii.hexlify(arp[4]))
    print("源MAC:    ", binascii.hexlify(arp[5]))
    print("源IP:     ", socket.inet_ntoa(arp[6]))
    print("目标MAC:      ", binascii.hexlify(arp[7]))
    print("目标IP:       ", socket.inet_ntoa(arp[8]))
    print("-------------------------------------------------------")
```

output:

```
$ python arp.py
------------------- ETHERNET_FRAME -------------------
目标MAC:          ffffffffffff
源MAC:        f0257252f5ca
类型:              0806
------------------- ARP_HEADER -----------------------
硬件类型:     0001
协议类型:     0800
硬件大小:     06
协议大小:     04
操作码:            0001
源MAC:        f0257252f5ca
源IP:         140.112.91.254
目标MAC:          000000000000
目标IP:           140.112.91.20
-------------------------------------------------------
```

# python-cheatsheet 文档，版本 0.1.0

## 3.3 异步IO

目录

- 异步IO
- asyncio.run
- 类Future对象
- 类Future对象 __await__ 其他任务
- 修补循环运行器 _run_once
- 将阻塞任务放入执行器
- 与 asyncio 结合使用套接字
- 使用轮询的事件循环
- 传输与协议
- 使用 SSL 的传输与协议
- 异步迭代器
- 什么是异步迭代器
- 异步上下文管理器
- 什么是异步上下文管理器
- 装饰器 @asynccontextmanager
- 简单的 asyncio 连接池
- 获取域名名称
- 收集结果
- 简单的 asyncio UDP 回显服务器
- 简单的 asyncio Web 服务器
- 简单的 HTTPS Web 服务器
- 简单的 HTTPS Web 服务器 (低级 API)
- TLS 升级
- 使用 sendfile
- 简单的 asyncio WSGI Web 服务器

## 3.3.1 asyncio.run

Python 3.7中的新功能

```python
>>> import asyncio
>>> from concurrent.futures import ThreadPoolExecutor
>>> e = ThreadPoolExecutor()
>>> async def read_file(file_):
...     loop = asyncio.get_event_loop()
...     with open(file_) as f:
...         return (await loop.run_in_executor(e, f.read))
...
>>> ret = asyncio.run(read_file('/etc/passwd'))
```

## 3.3.2 类似Future的对象

```python
>>> import sys
>>> PY_35 = sys.version_info >= (3, 5)
>>> import asyncio
>>> loop = asyncio.get_event_loop()
>>> class SlowObj:
...     def __init__(self, n):
...         print("__init__")
...         self._n = n
...     if PY_35:
...         def __await__(self):
...             print("__await__ sleep({})".format(self._n))
...             yield from asyncio.sleep(self._n)
...             print("ok")
...             return self
...
>>> async def main():
...     obj = await SlowObj(3)
...
>>> loop.run_until_complete(main())
__init__
__await__ sleep(3)
ok
```

## 3.3.3 类似Future的对象 __await__ 其他任务

```python
>>> import sys
>>> PY_35 = sys.version_info >= (3, 5)
>>> import asyncio
>>> loop = asyncio.get_event_loop()
>>> async def slow_task(n):
...     await asyncio.sleep(n)
...
>>> class SlowObj:
...     def __init__(self, n):
...         print("__init__")
...         self._n = n
...     if PY_35:
...         def __await__(self):
...             print("__await__")
...             yield from slow_task(self._n).__await__()
...             yield from asyncio.sleep(self._n)
...             print("ok")
...             return self
...
>>> async def main():
...     obj = await SlowObj(1)
...
>>> loop.run_until_complete(main())
__init__
__await__
ok
```

## 3.3.4 补丁循环运行器 _run_once

```python
>>> import asyncio
>>> def _run_once(self):
...     num_tasks = len(self._scheduled)
...     print("num tasks in queue: {}".format(num_tasks))
...     super(asyncio.SelectorEventLoop, self)._run_once()
...
>>> EventLoop = asyncio.SelectorEventLoop
>>> EventLoop._run_once = _run_once
>>> loop = EventLoop()
>>> asyncio.set_event_loop(loop)
>>> async def task(n):
...     await asyncio.sleep(n)
...     print("sleep: {} sec".format(n))
...
>>> coro = loop.create_task(task(3))
>>> loop.run_until_complete(coro)
num tasks in queue: 0
num tasks in queue: 1
num tasks in queue: 0
sleep: 3 sec
num tasks in queue: 0
>>> loop.close()
```

## 3.3.5 将阻塞任务放入Executor

```python
>>> import asyncio
>>> from concurrent.futures import ThreadPoolExecutor
>>> e = ThreadPoolExecutor()
>>> loop = asyncio.get_event_loop()
>>> async def read_file(file_):
...     with open(file_) as f:
...         data = await loop.run_in_executor(e, f.read)
...         return data
...
>>> task = loop.create_task(read_file('/etc/passwd'))
>>> ret = loop.run_until_complete(task)
```

## 3.3.6 使用asyncio的Socket

```python
import asyncio
import socket

host = 'localhost'
port = 9527
loop = asyncio.get_event_loop()
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setblocking(False)
s.bind((host, port))
s.listen(10)

async def handler(conn):
    while True:
        msg = await loop.sock_recv(conn, 1024)
        if not msg:
            break
        await loop.sock_sendall(conn, msg)
    conn.close()

async def server():
    while True:
        conn, addr = await loop.sock_accept(s)
        loop.create_task(handler(conn))

loop.create_task(server())
loop.run_forever()
loop.close()
```

输出： (bash 1)

```
$ nc localhost 9527
Hello
Hello
```

输出： (bash 2)

```
$ nc localhost 9527
World
World
```

## 3.3.7 带轮询的事件循环

```python
# 使用选择器
# 参考: PyCon 2015 - David Beazley

import asyncio
import socket
import selectors
from collections import deque

@asyncio.coroutine
def read_wait(s):
    yield 'read_wait', s

@asyncio.coroutine
def write_wait(s):
    yield 'write_wait', s

class Loop:
    """简单的循环原型"""

    def __init__(self):
        self.ready = deque()
        self.selector = selectors.DefaultSelector()

    @asyncio.coroutine
    def sock_accept(self, s):
        yield from read_wait(s)
        return s.accept()

    @asyncio.coroutine
    def sock_recv(self, c, mb):
        yield from read_wait(c)
        return c.recv(mb)

    @asyncio.coroutine
    def sock_sendall(self, c, m):
        while m:
            yield from write_wait(c)
            nsent = c.send(m)
            m = m[nsent:]

    def create_task(self, coro):
        self.ready.append(coro)

    def run_forever(self):
        while True:
            self._run_once()

    def _run_once(self):
        while not self.ready:
            events = self.selector.select()
            for k, _ in events:
                self.ready.append(k.data)
                self.selector.unregister(k.fileobj)

        while self.ready:
            self.cur_t = self.ready.popleft()
            try:
                op, *a = self.cur_t.send(None)
                getattr(self, op)(*a)
            except StopIteration:
                pass

    def read_wait(self, s):
        self.selector.register(s, selectors.EVENT_READ, self.cur_t)

    def write_wait(self, s):
        self.selector.register(s, selectors.EVENT_WRITE, self.cur_t)

loop = Loop()
host = 'localhost'
port = 9527

s = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM, 0)
s.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_REUSEADDR, 1)
s.setblocking(False)
s.bind((host, port))
s.listen(10)

@asyncio.coroutine
def handler(c):
    while True:
        msg = yield from loop.sock_recv(c, 1024)
        if not msg:
            break
        yield from loop.sock_sendall(c, msg)
    c.close()

@asyncio.coroutine
def server():
    while True:
        c, addr = yield from loop.sock_accept(s)
        loop.create_task(handler(c))

loop.create_task(server())
loop.run_forever()
```

## 3.3.8 传输与协议

```python
import asyncio

class EchoProtocol(asyncio.Protocol):
    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        print('Connection from {}'.format(peername))
        self.transport = transport

    def data_received(self, data):
        msg = data.decode()
        self.transport.write(data)

loop = asyncio.get_event_loop()
coro = loop.create_server(EchoProtocol, 'localhost', 5566)
server = loop.run_until_complete(coro)

try:
    loop.run_forever()
except:
    loop.run_until_complete(server.wait_closed())
finally:
    loop.close()
```

输出：

```
# 控制台 1
$ nc localhost 5566
Hello
Hello

# 控制台 2
$ nc localhost 5566
World
World
```

## 3.3.9 使用SSL的传输与协议

```python
import asyncio
import ssl

def make_header():
    head = b"HTTP/1.1 200 OK\r\n"
    head += b"Content-Type: text/html\r\n"
    head += b"\r\n"
    return head

def make_body():
    resp = b"<html>"
    resp += b"<h1>Hello SSL</h1>"
    resp += b"</html>"
    return resp

sslctx = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
sslctx.load_cert_chain(
    certfile="./root-ca.crt", keyfile="./root-ca.key"
)

class Service(asyncio.Protocol):
    def connection_made(self, tr):
        self.tr = tr
        self.total = 0

    def data_received(self, data):
        if data:
            resp = make_header()
            resp += make_body()
            self.tr.write(resp)
        self.tr.close()

async def start():
    server = await loop.create_server(
        Service, "localhost", 4433, ssl=sslctx
    )
    await server.wait_closed()

try:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start())
finally:
    loop.close()
```

输出：

```
$ openssl genrsa -out root-ca.key 2048
$ openssl req -x509 -new -nodes -key root-ca.key -days 365 -out root-ca.crt
$ python3 ssl_web_server.py
# 然后打开浏览器：https://localhost:4433
```

## 3.3.10 异步迭代器

```python
# 参考：PEP-0492
# 需要 Python >= 3.5

>>> class AsyncIter:
...     def __init__(self, it):
...         self._it = iter(it)
...     def __aiter__(self):
...         return self
...     async def __anext__(self):
...         await asyncio.sleep(1)
...         try:
...             val = next(self._it)
...         except StopIteration:
...             raise StopAsyncIteration
...         return val
...
>>> async def foo():
...     it = [1, 2, 3]
...     async for _ in AsyncIter(it):
...         print(_)
...
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(foo())
1
2
3
```

## 3.3.11 什么是异步迭代器

```python
>>> import asyncio
>>> class AsyncIter:
...     def __init__(self, it):
...         self._it = iter(it)
...     def __aiter__(self):
...         return self
...     async def __anext__(self):
...         await asyncio.sleep(1)
...         try:
...             val = next(self._it)
...         except StopIteration:
...             raise StopAsyncIteration
```...     return val
...
>>> async def foo():
...     _ = [1, 2, 3]
...     running = True
...     it = AsyncIter(_)
...     while running:
...         try:
...             res = await it.__anext__()
...             print(res)
...         except StopAsyncIteration:
...             running = False
...
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(loop.create_task(foo()))
1
2
3
```

## 3.3.12 异步上下文管理器

```
# 参考: PEP-0492
# 需要 Python >= 3.5

>>> class AsyncCtxMgr:
...     async def __aenter__(self):
...         await asyncio.sleep(3)
...         print("__aenter__")
...         return self
...     async def __aexit__(self, *exc):
...         await asyncio.sleep(1)
...         print("__aexit__")
...
>>> async def hello():
...     async with AsyncCtxMgr() as m:
...         print("hello block")
...
>>> async def world():
...     print("world block")
...
>>> t = loop.create_task(world())
>>> loop.run_until_complete(hello())
world block
__aenter__
hello block
__aexit__
```

## 3.3.13 什么是异步上下文管理器

```
>>> import asyncio
>>> class AsyncManager:
...     async def __aenter__(self):
...         await asyncio.sleep(5)
...         print("__aenter__")
...     async def __aexit__(self, *exc_info):
...         await asyncio.sleep(3)
...         print("__aexit__")
...
>>> async def foo():
...     import sys
...     mgr = AsyncManager()
...     await mgr.__aenter__()
...     print("body")
...     await mgr.__aexit__(*sys.exc_info())
...
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(loop.create_task(foo()))
__aenter__
body
__aexit__
```

## 3.3.14 装饰器 @asynccontextmanager

Python 3.7 新增

- Issue 29679 - 添加 @contextlib.asynccontextmanager

```
>>> import asyncio
>>> from contextlib import asynccontextmanager
>>> @asynccontextmanager
... async def coro(msg):
...     await asyncio.sleep(1)
...     yield msg
...     await asyncio.sleep(0.5)
...     print('done')
...
>>> async def main():
...     async with coro("Hello") as m:
...         await asyncio.sleep(1)
...         print(m)
...
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(main())
Hello
done
```

## 3.3.15 简单的 asyncio 连接池

```
import asyncio
import socket
import uuid

class Transport:
    def __init__(self, loop, host, port):
        self.used = False
        self._loop = loop
        self._host = host
        self._port = port
        self._sock = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setblocking(False)
        self._uuid = uuid.uuid1()
    
    async def connect(self):
        loop, sock = self._loop, self._sock
        host, port = self._host, self._port
        return (await loop.sock_connect(sock, (host, port)))
    
    async def sendall(self, msg):
        loop, sock = self._loop, self._sock
        return (await loop.sock_sendall(sock, msg))
    
    async def recv(self, buf_size):
        loop, sock = self._loop, self._sock
        return (await loop.sock_recv(sock, buf_size))
    
    def close(self):
        if self._sock: self._sock.close()
    
    @property
    def alive(self):
        ret = True if self._sock else False
        return ret
    
    @property
    def uuid(self):
        return self._uuid

class ConnectionPool:
    def __init__(self, loop, host, port, max_conn=3):
        self._host = host
        self._port = port
        self._max_conn = max_conn
        self._loop = loop
        conns = [Transport(loop, host, port) for _ in range(max_conn)]
        self._conns = conns

    def __await__(self):
        for _c in self._conns:
            yield from _c.connect().__await__()
        return self

    def getconn(self, fut=None):
        if fut is None:
            fut = self._loop.create_future()

        for _c in self._conns:
            if _c.alive and not _c.used:
                _c.used = True
                fut.set_result(_c)
                break
        else:
            loop.call_soon(self.getconn, fut)

        return fut

    def release(self, conn):
        if not conn.used:
            return
        for _c in self._conns:
            if _c.uuid != conn.uuid:
                continue
            _c.used = False
            break

    def close(self):
        for _c in self._conns:
            _c.close()

async def handler(pool, msg):
    conn = await pool.getconn()
    byte = await conn.sendall(msg)
    mesg = await conn.recv(1024)
    pool.release(conn)
    return 'echo: {}'.format(mesg)

async def main(loop, host, port):
    try:
        # 创建连接池
        pool = await ConnectionPool(loop, host, port)

        # 生成消息
        msgs = ['coro_{}'.format(_).encode('utf-8') for _ in range(5)]

        # 创建任务
        fs = [loop.create_task(handler(pool, _m)) for _m in msgs]

        # 等待所有任务完成
        done, pending = await asyncio.wait(fs)
        for _ in done: print(_.result())
    finally:
        pool.close()

loop = asyncio.get_event_loop()
host = '127.0.0.1'
port = 9527

try:
    loop.run_until_complete(main(loop, host, port))
except KeyboardInterrupt:
    pass
finally:
    loop.close()
```

输出:

```
$ ncat -l 9527 --keep-open --exec "/bin/cat" &
$ python3 conn_pool.py
echo: b'coro_1'
echo: b'coro_0'
echo: b'coro_2'
echo: b'coro_3'
echo: b'coro_4'
```

## 3.3.16 获取域名

```
>>> import asyncio
>>> async def getaddrinfo(host, port):
...     loop = asyncio.get_event_loop()
...     return (await loop.getaddrinfo(host, port))
...
>>> addrs = asyncio.run(getaddrinfo('github.com', 443))
>>> for a in addrs:
...     family, typ, proto, name, sockaddr = a
...     print(sockaddr)
...
('192.30.253.113', 443)
('192.30.253.113', 443)
('192.30.253.112', 443)
('192.30.253.112', 443)
```

## 3.3.17 收集结果

```
import asyncio
import ssl

path = ssl.get_default_verify_paths()
sslctx = ssl.SSLContext()
sslctx.verify_mode = ssl.CERT_REQUIRED
sslctx.check_hostname = True
sslctx.load_verify_locations(path.cafile)

async def fetch(host, port):
    r, w = await asyncio.open_connection(host, port, ssl=sslctx)
    req = "GET / HTTP/1.1\r\n"
    req += f"Host: {host}\r\n"
    req += "Connection: close\r\n"
    req += "\r\n"

    # 发送请求
    w.write(req.encode())

    # 接收响应
    resp = ""
    while True:
        line = await r.readline()
        if not line:
            break
        line = line.decode("utf-8")
        resp += line

    # 关闭写入器
    w.close()
    await w.wait_closed()
    return resp

async def main():
    loop = asyncio.get_running_loop()
    url = ["python.org", "github.com", "google.com"]
    fut = [fetch(u, 443) for u in url]
    resps = await asyncio.gather(*fut)
    for r in resps:
        print(r.split("\r\n")[0])

asyncio.run(main())
```

输出:

```
$ python fetch.py
HTTP/1.1 301 Moved Permanently
HTTP/1.1 200 OK
HTTP/1.1 301 Moved Permanently
```

## 3.3.18 简单的 asyncio UDP 回显服务器

```
import asyncio
import socket

loop = asyncio.get_event_loop()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setblocking(False)

host = 'localhost'
port = 3553

sock.bind((host, port))

def recvfrom(loop, sock, n_bytes, fut=None, registed=False):
    fd = sock.fileno()
    if fut is None:
        fut = loop.create_future()
    if registed:
        loop.remove_reader(fd)

    try:
        data, addr = sock.recvfrom(n_bytes)
    except (BlockingIOError, InterruptedError):
        loop.add_reader(fd, recvfrom, loop, sock, n_bytes, fut, True)
    else:
        fut.set_result((data, addr))
    return fut

def sendto(loop, sock, data, addr, fut=None, registed=False):
    fd = sock.fileno()
    if fut is None:
        fut = loop.create_future()
    if registed:
        loop.remove_writer(fd)
    if not data:
        return

    try:
        n = sock.sendto(data, addr)
    except (BlockingIOError, InterruptedError):
        loop.add_writer(fd, sendto, loop, sock, data, addr, fut, True)
    else:
        fut.set_result(n)
    return fut
```## 3.3.19 简单的 asyncio Web 服务器

```python
import asyncio
import socket

host = 'localhost'
port = 9527
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setblocking(False)
s.bind((host, port))
s.listen(10)

loop = asyncio.get_event_loop()

def make_header():
    header = b'HTTP/1.1 200 OK\r\n'
    header += b'Content-Type: text/html\r\n'
    header += b'\r\n'
    return header

def make_body():
    resp = b'<html>'
    resp += b'<body><h3>Hello World</h3></body>'
    resp += b'</html>'
    return resp

async def handler(conn):
    req = await loop.sock_recv(conn, 1024)
    if req:
        resp = make_header()
        resp += make_body()
        await loop.sock_sendall(conn, resp)
        conn.close()

async def server(sock, loop):
    while True:
        conn, addr = await loop.sock_accept(sock)
        loop.create_task(handler(conn))

try:
    loop.run_until_complete(server(s, loop))
except KeyboardInterrupt:
    pass
finally:
    loop.close()
    s.close()
# 然后打开浏览器访问网址：localhost:9527
```

输出：
```
$ python3 udp_server.py
$ nc -u localhost 3553
Hello UDP
Hello UDP
```

## 3.3.20 简单的 HTTPS Web 服务器

```python
import asyncio
import ssl

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain('crt.pem', 'key.pem')

async def conn(reader, writer):
    _ = await reader.read(1024)
    head = b"HTTP/1.1 200 OK\r\n"
    head += b"Content-Type: text/html\r\n"
    head += b"\r\n"

    body = b"<!doctype html>"
    body += b"<html>"
    body += b"<body><h1>Awesome Python</h1></body>"
    body += b"</html>"

    writer.write(head + body)
    writer.close()

async def main(host, port):
    srv = await asyncio.start_server(conn, host, port, ssl=ctx)
    async with srv:
        await srv.serve_forever()

asyncio.run(main('0.0.0.0', 8000))
```

## 3.3.21 简单的 HTTPS Web 服务器（底层 API）

```python
import asyncio
import socket
import ssl

def make_header():
    head = b'HTTP/1.1 200 OK\r\n'
    head += b'Content-type: text/html\r\n'
    head += b'\r\n'
    return head

def make_body():
    resp = b'<html>'
    resp += b'<h1>Hello SSL</h1>'
    resp += b'</html>'
    return resp

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setblocking(False)
sock.bind(('localhost', 4433))
sock.listen(10)

sslctx = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
sslctx.load_cert_chain(certfile='./root-ca.crt',
                      keyfile='./root-ca.key')

def do_handshake(loop, sock, waiter):
    sock_fd = sock.fileno()
    try:
        sock.do_handshake()
    except ssl.SSLWantReadError:
        loop.remove_reader(sock_fd)
        loop.add_reader(sock_fd, do_handshake,
                        loop, sock, waiter)
        return
    except ssl.SSLWantWriteError:
        loop.remove_writer(sock_fd)
        loop.add_writer(sock_fd, do_handshake,
                        loop, sock, waiter)
        return

    loop.remove_reader(sock_fd)
    loop.remove_writer(sock_fd)
    waiter.set_result(None)

def handle_read(loop, conn, waiter):
    try:
        req = conn.recv(1024)
    except ssl.SSLWantReadError:
        loop.remove_reader(conn.fileno())
        loop.add_reader(conn.fileno(), handle_read,
                        loop, conn, waiter)
        return
    loop.remove_reader(conn.fileno())
    waiter.set_result(req)

def handle_write(loop, conn, msg, waiter):
    try:
        resp = make_header()
        resp += make_body()
        ret = conn.send(resp)
    except ssl.SSLWantReadError:
        loop.remove_writer(conn.fileno())
        loop.add_writer(conn.fileno(), handle_write,
                        loop, conn, waiter)
        return
    loop.remove_writer(conn.fileno())
    conn.close()
    waiter.set_result(None)

async def server(loop):
    while True:
        conn, addr = await loop.sock_accept(sock)
        conn.setblocking(False)
        sslconn = sslctx.wrap_socket(conn,
                                    server_side=True,
                                    do_handshake_on_connect=False)
        # 等待 SSL 握手
        waiter = loop.create_future()
        do_handshake(loop, sslconn, waiter)
        await waiter
        # 等待读取请求
        waiter = loop.create_future()
        handle_read(loop, sslconn, waiter)
        msg = await waiter
        # 等待写入响应
        waiter = loop.create_future()
        handle_write(loop, sslconn, msg, waiter)
        await waiter

loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(server(loop))
finally:
    loop.close()
```

输出：
```
# 控制台 1

$ openssl genrsa -out root-ca.key 2048
$ openssl req -x509 -new -nodes -key root-ca.key -days 365 -out root-ca.crt
$ python3 Simple_https_server.py

# 控制台 2

$ curl https://localhost:4433 -v \
    --resolve localhost:4433:127.0.0.1 \
    --cacert ~/test/root-ca.crt
```

## 3.3.22 TLS 升级

Python 3.7 中新增

```python
import asyncio
import ssl

class HttpClient(asyncio.Protocol):
    def __init__(self, on_con_lost):
        self.on_con_lost = on_con_lost
        self.resp = b""

    def data_received(self, data):
        self.resp += data

    def connection_lost(self, exc):
        resp = self.resp.decode()
        print(resp.split("\r\n")[0])
        self.on_con_lost.set_result(True)

async def main():
    paths = ssl.get_default_verify_paths()
    sslctx = ssl.SSLContext()
    sslctx.verify_mode = ssl.CERT_REQUIRED
    sslctx.check_hostname = True
    sslctx.load_verify_locations(paths.cafile)

    loop = asyncio.get_running_loop()
    on_con_lost = loop.create_future()

    tr, proto = await loop.create_connection(
        lambda: HttpClient(on_con_lost), "github.com", 443
    )
    new_tr = await loop.start_tls(tr, proto, sslctx)
    req = f"GET / HTTP/1.1\r\n"
    req += "Host: github.com\r\n"
    req += "Connection: close\r\n"
    req += "\r\n"
    new_tr.write(req.encode())

    await on_con_lost
    new_tr.close()

asyncio.run(main())
```

输出：
```
$ python3 --version
Python 3.7.0
$ python3 https.py
HTTP/1.1 200 OK
```

## 3.3.23 使用 sendfile

Python 3.7 中新增

```python
import asyncio

path = "index.html"

async def conn(reader, writer):

    loop = asyncio.get_event_loop()
    _ = await reader.read(1024)

    with open(path, "rb") as f:
        tr = writer.transport
        head = b"HTTP/1.1 200 OK\r\n"
        head += b"Content-Type: text/html\r\n"
        head += b"\r\n"

        tr.write(head)
        await loop.sendfile(tr, f)
        writer.close()

async def main(host, port):
    # 运行一个简单的 http 服务器
    srv = await asyncio.start_server(conn, host, port)
    async with srv:
        await srv.serve_forever()

asyncio.run(main("0.0.0.0", 8000))
```

输出：
```
$ echo '<!doctype html><h1>Awesome Python</h1>' > index.html
$ python http.py &
[2] 60506
$ curl http://localhost:8000
<!doctype html><h1>Awesome Python</h1>
```

## 3.3.24 简单的 asyncio WSGI Web 服务器

```python
# 参考：PEP333

import asyncio
import socket
import io
import sys

from flask import Flask, Response

host = 'localhost'
port = 9527
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setblocking(False)
s.bind((host, port))
s.listen(10)

loop = asyncio.get_event_loop()

class WSGIServer(object):

    def __init__(self, sock, app):
        self._sock = sock
        self._app = app
        self._header = []

    def parse_request(self, req):
        """ HTTP 请求格式：

            GET /hello.htm HTTP/1.1\r\n
            Accept-Language: en-us\r\n
            ...

            Connection: Keep-Alive\r\n
        """
        # 字节转字符串
        req_info = req.decode('utf-8')
        first_line = req_info.splitlines()[0]
        method, path, ver = first_line.split()
        return method, path, ver

    def get_environ(self, req, method, path):
        env = {}

        # 必需的 WSGI 变量
```## 3.4 并发

### 目录

- 并发
- 执行 shell 命令
- 通过 "threading" 创建线程
- 性能问题 - GIL
- 生产者与消费者
- 线程池模板
- 使用 multiprocessing ThreadPool
- 互斥锁
- 死锁
- 实现 "Monitor"
- 控制原始资源
- 确保任务完成
- 线程安全的优先级队列
- 多进程
- 自定义 multiprocessing map
- 优雅地终止所有子进程
- 简单的轮询调度器
- 带阻塞功能的调度器
- PoolExecutor
- 如何使用 ThreadPoolExecutor？
- “with ThreadPoolExecutor” 是如何工作的？
- Future 对象
- Future 错误处理

### 3.4.1 执行 shell 命令

```python
# 获取标准输出、标准错误、返回码
>>> from subprocess import Popen, PIPE
>>> args = ['time', 'echo', 'hello python']
>>> ret = Popen(args, stdout=PIPE, stderr=PIPE)
>>> out, err = ret.communicate()
>>> out
b'hello python\n'
>>> err
b'        0.00 real         0.00 user         0.00 sys\n'
>>> ret.returncode
0
```

### 3.4.2 通过 “threading” 创建线程

```python
>>> from threading import Thread
>>> class Worker(Thread):
...     def __init__(self, id):
...         super(Worker, self).__init__()
...         self._id = id
...     def run(self):
...         print("I am worker %d" % self._id)
...
>>> t1 = Worker(1)
>>> t2 = Worker(2)
>>> t1.start(); t2.start()
I am worker 1
I am worker 2

# 使用函数可能更灵活
>>> def Worker(worker_id):
...     print("I am worker %d" % worker_id)
...
>>> from threading import Thread
>>> t1 = Thread(target=Worker, args=(1,))
>>> t2 = Thread(target=Worker, args=(2,))
>>> t1.start()
I am worker 1
I am worker 2
```

### 3.4.3 性能问题 - GIL

```python
# GIL - 全局解释器锁
# 参见：Understanding the Python GIL
>>> from threading import Thread
>>> def profile(func):
...     def wrapper(*args, **kwargs):
...         import time
...         start = time.time()
...         func(*args, **kwargs)
...         end = time.time()
...         print(end - start)
...     return wrapper
...
>>> @profile
... def nothread():
...     fib(35)
...     fib(35)
...
>>> @profile
... def hastypead():
...     t1=Thread(target=fib, args=(35,))
...     t2=Thread(target=fib, args=(35,))
...     t1.start(); t2.start()
...     t1.join(); t2.join()
...
>>> nothread()
9.51164007187
>>> hastypead()
11.3131771088
# !线程导致性能下降
# 因为上下文切换的开销
```

### 3.4.4 生产者与消费者

```python
# 这种架构使得并发变得容易
>>> from threading import Thread
>>> from Queue import Queue
>>> from random import random
>>> import time
>>> q = Queue()
>>> def fib(n):
...     if n<=2:
...         return 1
...     return fib(n-1)+fib(n-2)
...
>>> def producer():
...     while True:
...         wt = random()*5
...         time.sleep(wt)
...         q.put((fib,35))
...
>>> def consumer():
...     while True:
...         task,arg = q.get()
...         print(task(arg))
...         q.task_done()
...
>>> t1 = Thread(target=producer)
>>> t2 = Thread(target=consumer)
>>> t1.start();t2.start()
```

### 3.4.5 线程池模板

```python
# 生产者与消费者架构
from Queue import Queue
from threading import Thread

class Worker(Thread):
    def __init__(self,queue):
        super(Worker, self).__init__()
        self._q = queue
        self.daemon = True
        self.start()
    def run(self):
        while True:
            f,args,kwargs = self._q.get()
            try:
                print(f(*args, **kwargs))
            except Exception as e:
                print(e)
            self._q.task_done()

class ThreadPool(object):
    def __init__(self, num_t=5):
        self._q = Queue(num_t)
        # 创建工作线程
        for _ in range(num_t):
            Worker(self._q)
    def add_task(self,f,*args,**kwargs):
        self._q.put((f, args, kwargs))
    def wait_complete(self):
        self._q.join()

def fib(n):
    if n <= 2:
        return 1
    return fib(n-1)+fib(n-2)

if __name__ == '__main__':
    pool = ThreadPool()
    for _ in range(3):
        pool.add_task(fib, 35)
    pool.wait_complete()
```

### 3.4.6 使用 multiprocessing ThreadPool

```python
# ThreadPool 不在 Python 文档中
>>> from multiprocessing.pool import ThreadPool
>>> pool = ThreadPool(5)
>>> pool.map(lambda x: x**2, range(5))
[0, 1, 4, 9, 16]
```

与 “map” 性能比较

```python
# 由于 GIL，pool 会得到较差的结果
import time
from multiprocessing.pool import ThreadPool

pool = ThreadPool(10)
def profile(func):
    def wrapper(*args, **kwargs):
        print(func.__name__)
        s = time.time()
        func(*args, **kwargs)
        e = time.time()
        print("cost: {0}".format(e-s))
    return wrapper

@profile
def pool_map():
    res = pool.map(lambda x:x**2,
                   range(999999))

@profile
def ordinary_map():
    res = map(lambda x:x**2,
              range(999999))

pool_map()
ordinary_map()
```

输出：

```
$ python test_threadpool.py
pool_map
cost: 0.562669038773
ordinary_map
cost: 0.38525390625
```

### 3.4.7 互斥锁

最简单的同步原语锁

```python
>>> from threading import Thread
>>> from threading import Lock
>>> lock = Lock()
>>> def getlock(id):
...     lock.acquire()
...     print("task{0} get".format(id))
...     lock.release()
...
>>> t1=Thread(target=getlock,args=(1,))
>>> t2=Thread(target=getlock,args=(2,))
>>> t1.start();t2.start()
task1 get
task2 get

# 使用锁管理器
>>> def getlock(id):
...     with lock:
...         print("task%d get" % id)
...
>>> t1=Thread(target=getlock,args=(1,))
>>> t2=Thread(target=getlock,args=(2,))
>>> t1.start();t2.start()
task1 get
task2 get
```

### 3.4.8 死锁

当存在多个互斥锁时可能发生。

```python
>>> import threading
>>> import time
>>> lock1 = threading.Lock()
>>> lock2 = threading.Lock()
>>> def task1():
...     with lock1:
...         print("get lock1")
...         time.sleep(3)
...         with lock2:
...             print("No deadlock")
...
>>> def task2():
...     with lock2:
...         print("get lock2")
...         with lock1:
...             print("No deadlock")
...
>>> t1=threading.Thread(target=task1)
>>> t2=threading.Thread(target=task2)
```

## 3.4.9 实现“Monitor”
使用 RLock

```python
# ref: An introduction to Python Concurrency - David Beazley
from threading import Thread
from threading import RLock
import time

class monitor(object):
    lock = RLock()
    def foo(self,tid):
        with monitor.lock:
            print("%d in foo" % tid)
            time.sleep(5)
            self.ker(tid)

    def ker(self,tid):
        with monitor.lock:
            print("%d in ker" % tid)
m = monitor()
def task1(id):
    m.foo(id)

def task2(id):
    m.ker(id)

t1 = Thread(target=task1,args=(1,))
t2 = Thread(target=task2,args=(2,))
t1.start()
t2.start()
t1.join()
t2.join()
```

输出：

```
$ python monitor.py
1 in foo
1 in ker
2 in ker
```

## 3.4.10 控制原始资源
使用 Semaphore

```python
from threading import Thread
from threading import Semaphore
from random    import random
import time

# limit resource to 3
sema = Semaphore(3)
def foo(tid):
    with sema:
        print("%d acquire sema" % tid)
        wt = random()*5
        time.sleep(wt)
    print("%d release sema" % tid)

threads = []
for _t in range(5):
    t = Thread(target=foo,args=(_t,))
    threads.append(t)
    t.start()
for _t in threads:
    _t.join()
```

输出：

```
python semaphore.py
0 acquire sema
1 acquire sema
2 acquire sema
0 release sema
 3 acquire sema
2 release sema
 4 acquire sema
1 release sema
4 release sema
3 release sema
```

## 3.4.11 确保任务已完成
使用 'event'

```python
from threading import Thread
from threading import Event
import time

e = Event()

def worker(id):
    print("%d wait event" % id)
    e.wait()
    print("%d get event set" % id)

t1=Thread(target=worker,args=(1,))
t2=Thread(target=worker,args=(2,))
t3=Thread(target=worker,args=(3,))
t1.start()
t2.start()
t3.start()

# wait sleep task(event) happen
time.sleep(3)
e.set()
```

输出：

```
python event.py
1 wait event
2 wait event
3 wait event
2 get event set
3 get event set
1 get event set
```

## 3.4.12 线程安全的优先级队列
使用 'condition'

```python
import threading
import heapq
import time
import random

class PriorityQueue(object):
    def __init__(self):
        self._q = []
        self._count = 0
        self._cv = threading.Condition()

    def __str__(self):
        return str(self._q)

    def __repr__(self):
        return self._q

    def put(self, item, priority):
        with self._cv:
            heapq.heappush(self._q, (-priority,self._count,item))
            self._count += 1
            self._cv.notify()

    def pop(self):
        with self._cv:
            while len(self._q) == 0:
                print("wait...")
                self._cv.wait()
            ret = heapq.heappop(self._q)[-1]
        return ret

priq = PriorityQueue()
def producer():
    while True:
        print(priq.pop())

def consumer():
    while True:
        time.sleep(3)
        print("consumer put value")
        priority = random.random()
        priq.put(priority,priority*10)

for _ in range(3):
    priority = random.random()
    priq.put(priority,priority*10)

t1=threading.Thread(target=producer)
t2=threading.Thread(target=consumer)
t1.start();t2.start()
t1.join();t2.join()
```

输出：

```
python3 thread_safe.py
0.6657491871045683
0.5278797439991247
0.20990624606296315
wait...
consumer put value
0.09123101305407577
wait...
```

## 3.4.13 多进程
通过进程解决 GIL 问题

```python
>>> from multiprocessing import Pool
>>> def fib(n):
...     if n <= 2:
...         return 1
...     return fib(n-1) + fib(n-2)
...
>>> def profile(func):
...     def wrapper(*args, **kwargs):
...         import time
...         start = time.time()
...         func(*args, **kwargs)
...         end   = time.time()
...         print(end - start)
...     return wrapper
...
>>> @profile
... def nomultiprocess():
...     map(fib,[35]*5)
...
>>> @profile
... def hasmultiprocess():
...     pool = Pool(5)
...     pool.map(fib,[35]*5)
...
>>> nomultiprocess()
23.8454811573
>>> hasmultiprocess()
13.2433719635
```

## 3.4.14 自定义多进程 map

```python
from multiprocessing import Process, Pipe
from itertools import izip

def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),
                  args=(c,x))
          for x,(p,c) in izip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]

print(parmap(lambda x:x**x,range(1,5)))
```

## 3.4.15 优雅地终止所有子进程

```python
from __future__ import print_function

import signal
import os
import time

from multiprocessing import Process, Pipe

NUM_PROCESS = 10

def aurora(n):
    while True:
        time.sleep(n)

if __name__ == "__main__":
    procs = [Process(target=aurora, args=(x,))
             for x in range(NUM_PROCESS)]
    try:
        for p in procs:
            p.daemon = True
            p.start()
        [p.join() for p in procs]
    finally:
        for p in procs:
            if not p.is_alive(): continue
            os.kill(p.pid, signal.SIGKILL)
```

## 3.4.16 简单的轮询调度器

```python
>>> def fib(n):
...     if n <= 2:
...         return 1
...     return fib(n-1)+fib(n-2)
...
>>> def gen_fib(n):
...     for _ in range(1,n+1):
...         yield fib(_)
...
>>> t=[gen_fib(5),gen_fib(3)]
>>> from collections import deque
>>> tasks = deque()
>>> tasks.extend(t)
>>> def run(tasks):
...     while tasks:
...         try:
...             task = tasks.popleft()
...             print(task.next())
...             tasks.append(task)
...         except StopIteration:
...             print("done")
...
>>> run(tasks)
1
1
1
1
2
2
3
done
5
done
```

## 3.4.17 带阻塞函数的调度器

```python
# ref: PyCon 2015 - David Beazley
import socket
from select import select
from collections import deque

tasks = deque()
r_wait = {}
s_wait = {}

def fib(n):
    if n <= 2:
        return 1
    return fib(n-1)+fib(n-2)

def run():
    while any([tasks,r_wait,s_wait]):
        while not tasks:
            # polling
            rr, sr, _ = select(r_wait, s_wait, {})
            for _ in rr:
                tasks.append(r_wait.pop(_))
            for _ in sr:
                tasks.append(s_wait.pop(_))
        try:
            task = tasks.popleft()
            why, what = task.next()
            if why == 'recv':
                r_wait[what] = task
            elif why == 'send':
                s_wait[what] = task
            else:
                raise RuntimeError
        except StopIteration:
            pass

def fib_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('localhost', 5566))
    sock.listen(5)
    while True:
        yield 'recv', sock
        c, a = sock.accept()
        tasks.append(fib_handler(c))

def fib_handler(client):
    while True:
        yield 'recv', client
        req = client.recv(1024)
        if not req:
            break
        resp = fib(int(req))
        yield 'send', client
        client.send(str(resp)+'\n')
    client.close()

tasks.append(fib_server())
run()
```

输出：(bash 1)

```
$ nc localhost 5566
20
6765
```

输出：(bash 2)

```
$ nc localhost 5566
10
55
```

## 3.4.18 PoolExecutor

```python
# python2.x is module futures on PyPI
# new in Python3.2
>>> from concurrent.futures import \
...     ThreadPoolExecutor
>>> def fib(n):
...     if n<=2:
...         return 1
...     return fib(n-1) + fib(n-2)
...
>>> with ThreadPoolExecutor(3) as e:
...     res= e.map(fib,[1,2,3,4,5])
...     for _ in res:
...         print(_, end=' ')
...
1 1 2 3 5 >>> 
# result is generator?!
>>> with ThreadPoolExecutor(3) as e:
...     res = e.map(fib, [1,2,3])
...     inspect.isgenerator(res)
...
True
```

```python
# demo GIL
from concurrent import futures
import time

def fib(n):
    if n <= 2:
        return 1
    return fib(n-1) + fib(n-2)

def thread():
    s = time.time()
    with futures.ThreadPoolExecutor(2) as e:
        res = e.map(fib, [35]*2)
        for _ in res:
            print(_)
    e = time.time()
    print("thread cost: {}".format(e-s))

def process():
    s = time.time()
    with futures.ProcessPoolExecutor(2) as e:
        res = e.map(fib, [35]*2)
        for _ in res:
            print(_)
    e = time.time()
    print("process cost: {}".format(e-s))
```

```
# bash> python3 -i test.py
>>> thread()
9227465
9227465
thread cost: 12.550225019454956
>>> process()
9227465
9227465
process cost: 5.538189888000488
```

### 3.4.19 如何使用 ThreadPoolExecutor？

```python
from concurrent.futures import ThreadPoolExecutor

def fib(n):
    if n <= 2:
        return 1
    return fib(n - 1) + fib(n - 2)

with ThreadPoolExecutor(max_workers=3) as ex:
    futs = []
    for x in range(3):
        futs.append(ex.submit(fib, 30+x))

    res = [fut.result() for fut in futs]

print(res)
```

输出：

```bash
$ python3 thread_pool_ex.py
[832040, 1346269, 2178309]
```

### 3.4.20 “with ThreadPoolExecutor” 是如何工作的？

```python
from concurrent import futures

def fib(n):
    if n <= 2:
        return 1
    return fib(n-1) + fib(n-2)

with futures.ThreadPoolExecutor(3) as e:
    fut = e.submit(fib, 30)
    res = fut.result()
    print(res)

#### 等同于

e = futures.ThreadPoolExecutor(3)
fut = e.submit(fib, 30)
fut.result()
e.shutdown(wait=True)
print(res)
```

输出：

```bash
$ python3 thread_pool_exec.py
832040
832040
```

### 3.4.21 Future 对象

```python
# future: 延迟计算
# add_done_callback
from concurrent import futures

def fib(n):
    if n <= 2:
        return 1
    return fib(n-1) + fib(n-2)

def handler(future):
    res = future.result()
    print("res: {}".format(res))

def thread_v1():
    with futures.ThreadPoolExecutor(3) as e:
        for _ in range(3):
            f = e.submit(fib, 30+_)
            f.add_done_callback(handler)
    print("end")

def thread_v2():
    to_do = []
    with futures.ThreadPoolExecutor(3) as e:
        for _ in range(3):
            fut = e.submit(fib, 30+_)
            to_do.append(fut)
        for _f in futures.as_completed(to_do):
            res = _f.result()
            print("res: {}".format(res))
    print("end")
```

输出：

```bash
$ python3 -i fut.py
>>> thread_v1()
res: 832040
res: 1346269
res: 2178309
end
>>> thread_v2()
res: 832040
res: 1346269
res: 2178309
end
```

### 3.4.22 Future 错误处理

```python
from concurrent import futures

def spam():
    raise RuntimeError

def handler(future):
    print("callback handler")
    try:
        res = future.result()
    except RuntimeError:
        print("get RuntimeError")

def thread_spam():
    with futures.ThreadPoolExecutor(2) as e:
        f = e.submit(spam)
        f.add_done_callback(handler)
```

输出：

```bash
$ python -i fut_err.py
>>> thread_spam()
callback handler
get RuntimeError
```

### 3.5 SQLAlchemy

### 目录

* SQLAlchemy
- 设置数据库 URL
- Sqlalchemy 支持的 DBAPI - PEP249
- 事务与连接对象
- Metadata - 生成数据库模式
- Inspect - 获取数据库信息
- Reflection - 从现有数据库加载表
- 打印带索引的建表语句 (SQL DDL)
- 从 MetaData 中获取表
- 创建存储在 "MetaData" 中的所有表
- 创建特定表
- 创建具有相同列的表
- 删除表
- 一些表对象操作
- SQL 表达式语言
- insert() - 创建一个 “INSERT” 语句
- select() - 创建一个 “SELECT” 语句
- join() - 通过 “JOIN” 语句连接两个表
- 使用 “COPY” 语句在 PostgreSQL 中实现最快的批量插入
- PostgreSQL 批量插入并返回插入的 ID
- 更新多行
- 从表中删除行
- 检查表是否存在
- 一次创建多个表
- 创建具有动态列的表（Table）
- 对象关系型（ORM）添加数据
- 对象关系型（ORM）更新数据
- 对象关系型（ORM）删除行
- 对象关系型（ORM）关系
- 对象关系型（ORM）自关联
- 对象关系型（ORM）基础查询
- mapper：将表映射到类
- 动态获取表
- 对象关系型（ORM）连接两个表
- 基于关系连接和 group_by 计数
- 创建具有动态列的表（ORM）
- 关闭数据库连接
- 关闭会话后不能使用对象
- 钩子 (Hooks)

### 3.5.1 设置数据库 URL

```python
from sqlalchemy.engine.url import URL

postgres_db = {'drivername': 'postgres',
               'username': 'postgres',
               'password': 'postgres',
               'host': '192.168.99.100',
               'port': 5432}
print(URL(**postgres_db))

sqlite_db = {'drivername': 'sqlite', 'database': 'db.sqlite'}
print(URL(**sqlite_db))
```

输出：

```bash
$ python sqlalchemy_url.py
postgres://postgres:postgres@192.168.99.100:5432
sqlite:///db.sqlite
```

### 3.5.2 Sqlalchemy 支持的 DBAPI - PEP249

```python
from sqlalchemy import create_engine

db_uri = "sqlite:///db.sqlite"
engine = create_engine(db_uri)

# DBAPI - PEP249
# 创建表
engine.execute('CREATE TABLE "EX1" ('
                  'id INTEGER NOT NULL,'
                  'name VARCHAR, '
                  'PRIMARY KEY (id));')

# 插入一行
engine.execute('INSERT INTO "EX1" '
                  '(id, name) '
                  'VALUES (1,"raw1")')

# 选择 *
result = engine.execute('SELECT * FROM '
                           '"EX1"')
for _r in result:
    print(_r)

# 删除 *
engine.execute('DELETE from "EX1" where id=1;')
result = engine.execute('SELECT * FROM "EX1"')
print(result.fetchall())
```

### 3.5.3 事务与连接对象

```python
from sqlalchemy import create_engine

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri)

# 创建连接
conn = engine.connect()
# 开始事务
trans = conn.begin()
conn.execute('INSERT INTO "EX1" (name) '
             'VALUES ("Hello")')
trans.commit()
# 关闭连接
conn.close()
```

### 3.5.4 Metadata - 生成数据库模式

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy import Integer, String

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri)

# 创建一个 Metadata 实例
metadata = MetaData(engine)
# 声明一个表
table = Table('Example',metadata,
              Column('id',Integer, primary_key=True),
              Column('name',String))
# 创建所有表
metadata.create_all()
for _t in metadata.tables:
    print("Table: ", _t)
```

### 3.5.5 Inspect - 获取数据库信息

```python
from sqlalchemy import create_engine
from sqlalchemy import inspect

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri)

inspector = inspect(engine)

# 获取表信息
print(inspector.get_table_names())

# 获取列信息
print(inspector.get_columns('EX1'))
```

### 3.5.6 Reflection - 从现有数据库加载表

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri)

# 创建一个 MetaData 实例
metadata = MetaData()
print(metadata.tables)

# 将数据库模式反射到 MetaData
metadata.reflect(bind=engine)
print(metadata.tables)
```

### 3.5.7 打印带索引的建表语句 (SQL DDL)

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

def metadata_dump(sql, *multiparams, **params):
    print(sql.compile(dialect=engine.dialect))

meta = MetaData()
example_table = Table('Example',meta,
                Column('id', Integer, primary_key=True),
                Column('name', String(10), index=True))

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri, strategy='mock', executor=metadata_dump)

meta.create_all(bind=engine, tables=[example_table])
```

输出：

```sql
CREATE TABLE "Example" (
	id INTEGER NOT NULL, 
	name VARCHAR(10), 
	PRIMARY KEY (id)
)

CREATE INDEX "ix_Example_name" ON "Example" (name)
```

### 3.5.8 从 MetaData 中获取表

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri)

# 创建 MetaData 实例
metadata = MetaData(engine).reflect()
print(metadata.tables)

# 获取表
ex_table = metadata.tables['Example']
print(ex_table)
```

### 3.5.9 创建存储在 “MetaData” 中的所有表

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy import Integer, String

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri)
meta = MetaData(engine)

# 将 t1, t2 注册到 metadata
t1 = Table('EX1', meta,
           Column('id',Integer, primary_key=True),
           Column('name',String))

t2 = Table('EX2', meta,
           Column('id',Integer, primary_key=True),
           Column('val',Integer))
# 创建 meta 中的所有表
meta.create_all()
```

### 3.5.10 创建特定表

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy import Integer, String

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri)
``````python
meta = MetaData(engine)
t1 = Table('Table_1', meta,
           Column('id', Integer, primary_key=True),
           Column('name', String))
t2 = Table('Table_2', meta,
           Column('id', Integer, primary_key=True),
           Column('val', Integer))
t1.create()
```

## 3.5.11 创建具有相同列的表

```python
from sqlalchemy import (
    create_engine,
    inspect,
    Column,
    String,
    Integer)

from sqlalchemy.ext.declarative import declarative_base

db_url = "sqlite://"
engine = create_engine(db_url)

Base = declarative_base()

class TemplateTable(object):
    id   = Column(Integer, primary_key=True)
    name = Column(String)
    age  = Column(Integer)

class DowntownAPeople(TemplateTable, Base):
    __tablename__ = "downtown_a_people"

class DowntownBPeople(TemplateTable, Base):
    __tablename__ = "downtown_b_people"

Base.metadata.create_all(bind=engine)

# 检查表是否存在
ins = inspect(engine)
for _t in ins.get_table_names():
    print(_t)
```

## 3.5.12 删除表

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import inspect
from sqlalchemy import Table
from sqlalchemy import Column, Integer, String
from sqlalchemy.engine.url import URL

db_url = {'drivername': 'postgres',
          'username': 'postgres',
          'password': 'postgres',
          'host': '192.168.99.100',
          'port': 5432}
engine = create_engine(URL(**db_url))
m = MetaData()
table = Table('Test', m,
              Column('id', Integer, primary_key=True),
              Column('key', String, nullable=True),
              Column('val', String))

table.create(engine)
inspector = inspect(engine)
print('Test' in inspector.get_table_names())

table.drop(engine)
inspector = inspect(engine)
print('Test' in inspector.get_table_names())
```

输出：
```
$ python sqlalchemy_drop.py
$ True
$ False
```

## 3.5.13 一些表对象操作

```python
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy import Integer, String

meta = MetaData()
t = Table('ex_table', meta,
          Column('id', Integer, primary_key=True),
          Column('key', String),
          Column('val', Integer))
# 获取表名
print(t.name)

# 获取列
print(t.columns.keys())

# 获取列
c = t.c.key
print(c.name)
# 或者
c = t.columns.key
print(c.name)

# 从列获取所属表
print(c.table)
```

## 3.5.14 SQL 表达式语言

```python
# 将列视为“ColumnElement”
# 通过重写特殊函数来实现
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy import Integer, String
from sqlalchemy import or_

meta = MetaData()
table = Table('example', meta,
            Column('id', Integer, primary_key=True),
            Column('l_name', String),
            Column('f_name', String))
# SQL表达式二元对象
print(repr(table.c.l_name == 'ed'))
# 显示SQL表达式
print(str(table.c.l_name == 'ed'))

print(repr(table.c.f_name != 'ed'))

# 比较运算符
print(repr(table.c.id > 3))

# or 表达式
print((table.c.id > 5) | (table.c.id < 2))
#### 等价于
print(or_(table.c.id > 5, table.c.id < 2))

# 与None比较产生 IS NULL
print(table.c.l_name == None)
#### 等价于
print(table.c.l_name.is_(None))

# + 表示“加法”
print(table.c.id + 5)
# 或表示“字符串拼接”
print(table.c.l_name + "some name")

# in 表达式
print(table.c.l_name.in_(['a','b']))
```

## 3.5.15 insert() - 创建“INSERT”语句

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri)

# 创建表
meta = MetaData(engine)
table = Table('user', meta,
    Column('id', Integer, primary_key=True),
    Column('l_name', String),
    Column('f_name', String))
meta.create_all()

# 通过insert()构造插入数据
ins = table.insert().values(
    l_name='Hello',
    f_name='World')
conn = engine.connect()
conn.execute(ins)

# 插入多条数据
conn.execute(table.insert(),[
    {'l_name':'Hi','f_name':'bob'},
    {'l_name':'yo','f_name':'alice'}])
```

## 3.5.16 select() - 创建“SELECT”语句

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import select
from sqlalchemy import or_

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri)
conn = engine.connect()

meta = MetaData(engine).reflect()
table = meta.tables['user']

# select * from 'user'
select_st = select([table]).where(
    table.c.l_name == 'Hello')
res = conn.execute(select_st)
for _row in res:
    print(_row)

# 或等价于
select_st = table.select().where(
    table.c.l_name == 'Hello')
res = conn.execute(select_st)
for _row in res:
    print(_row)

# 结合 "OR"
select_st = select([
    table.c.l_name,
    table.c.f_name]).where(or_(
        table.c.l_name == 'Hello',
        table.c.l_name == 'Hi'))
res = conn.execute(select_st)
for _row in res:
    print(_row)

# 结合 "ORDER BY"
select_st = select([table]).where(or_(
        table.c.l_name == 'Hello',
        table.c.l_name == 'Hi')).order_by(table.c.f_name)
res = conn.execute(select_st)
for _row in res:
    print(_row)
```

## 3.5.17 join() - 通过“JOIN”语句连接两个表

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import select

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri)

meta = MetaData(engine).reflect()
email_t = Table('email_addr', meta,
    Column('id', Integer, primary_key=True),
    Column('email', String),
    Column('name', String))
meta.create_all()

# 获取 user 表
user_t = meta.tables['user']

# 插入
conn = engine.connect()
conn.execute(email_t.insert(),[
  {'email':'ker@test','name':'Hi'},
  {'email':'yo@test','name':'Hello'}])
# 连接语句
join_obj = user_t.join(email_t,
            email_t.c.name == user_t.c.l_name)
# 使用 select_from
sel_st = select(
  [user_t.c.l_name, email_t.c.email]).select_from(join_obj)
res = conn.execute(sel_st)
for _row in res:
    print(_row)
```

## 3.5.18 通过“COPY”语句在PostgreSQL中进行最快的批量插入

```python
# 此方法来源: https://gist.github.com/jsheedy/efa9a69926a754bebf0e9078fd085df6
import io
from datetime import date

from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Date


db_url = {'drivername': 'postgres',
          'username': 'postgres',
          'password': 'postgres',
          'host': '192.168.99.100',
          'port': 5432}
engine = create_engine(URL(**db_url))

# 创建表
meta = MetaData(engine)
table = Table('userinfo', meta,
    Column('id', Integer, primary_key=True),
    Column('first_name', String),
    Column('age', Integer),
    Column('birth_day', Date),
)
meta.create_all()

# 类文件对象（tsv格式）
datafile = io.StringIO()

# 生成行数据
for i in range(100):
    line = '\t'.join(
        [
            f'Name {i}',    # first_name
            str(18 + i),    # age
            str(date.today()),   # birth_day
        ]
    )
    datafile.write(line + '\n')

# 将文件指针重置到开始位置
datafile.seek(0)

# 通过 `COPY` 语句批量插入
conn = engine.raw_connection()
with conn.cursor() as cur:
    # https://www.psycopg.org/docs/cursor.html#cursor.copy_from
    cur.copy_from(
        datafile,
        table.name,  # 表名
        sep='\t',
        columns=('first_name', 'age', 'birth_day'),
    )
conn.commit()
```

## 3.5.19 PostgreSQL批量插入并返回插入的ID

```python
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

db_url = {'drivername': 'postgres',
          'username': 'postgres',
          'password': 'postgres',
          'host': '192.168.99.100',
```

## 3.5.20 更新多行

```python
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.sql.expression import bindparam

db_url = {'drivername': 'postgres',
          'username': 'postgres',
          'password': 'postgres',
          'host': '192.168.99.100',
          'port': 5432}
engine = create_engine(URL(**db_url))

# 创建表
meta = MetaData(engine)
table = Table('userinfo', meta,
    Column('id', Integer, primary_key=True),
    Column('first_name', String),
    Column('birth_year', Integer),
)
meta.create_all()

# 更新数据
data = [
    {'_id': 1, 'first_name': 'Johnny', 'birth_year': 1975},
    {'_id': 2, 'first_name': 'Jim', 'birth_year': 1973},
    {'_id': 3, 'first_name': 'Kaley', 'birth_year': 1985},
    {'_id': 4, 'first_name': 'Simon', 'birth_year': 1980},
    {'_id': 5, 'first_name': 'Kunal', 'birth_year': 1981},
    {'_id': 6, 'first_name': 'Mayim', 'birth_year': 1975},
    {'_id': 7, 'first_name': 'Melissa', 'birth_year': 1980},
]

stmt = table.update().where(table.c.id == bindparam('_id')).\
      values({
          'first_name': bindparam('first_name'),
          'birth_year': bindparam('birth_year'),
      })
# 转换为SQL语句：
# UPDATE userinfo SET first_name=%(first_name)s, birth_year=%(birth_year)s WHERE userinfo.id = %(_id)s

engine.execute(stmt, data)
```

## 3.5.21 从表中删除行

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri)
conn = engine.connect()

meta = MetaData(engine).reflect()
user_t = meta.tables['user']

# select * from user_t
sel_st = user_t.select()
res = conn.execute(sel_st)
for _row in res:
    print(_row)

# 删除 l_name == 'Hello' 的行
del_st = user_t.delete().where(
    user_t.c.l_name == 'Hello')
print('----- delete -----')
res = conn.execute(del_st)

# 检查行是否已被删除
sel_st = user_t.select()
res = conn.execute(sel_st)
for _row in res:
    print(_row)
```

## 3.5.22 检查表是否存在

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Column
from sqlalchemy import Integer, String
from sqlalchemy import inspect
from sqlalchemy.ext.declarative import declarative_base

Modal = declarative_base()
class Example(Modal):
    __tablename__ = "ex_t"
    id = Column(Integer, primary_key=True)
    name = Column(String(20))

db_uri = 'sqlite:///db.sqlite'
engine = create_engine(db_uri)
Modal.metadata.create_all(engine)

# 检查已注册到 Modal 的表是否存在
for _t in Modal.metadata.tables:
    print(_t)

# 检查数据库中的所有表
meta = MetaData(engine).reflect()
for _t in meta.tables:
    print(_t)

# 通过 inspect 检查表名是否存在
ins = inspect(engine)
for _t in ins.get_table_names():
    print(_t)
```

## 3.5.23 一次创建多个表

```python
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import inspect
from sqlalchemy import Column, String, Integer
from sqlalchemy.engine.url import URL

db = {'drivername': 'postgres',
      'username': 'postgres',
      'password': 'postgres',
      'host': '192.168.99.100',
      'port': 5432}

url = URL(**db)
engine = create_engine(url)

metadata = MetaData()
metadata.reflect(bind=engine)

def create_table(name, metadata):
    tables = metadata.tables.keys()
    if name not in tables:
        table = Table(name, metadata,
                      Column('id', Integer, primary_key=True),
                      Column('key', String),
                      Column('val', Integer))
        table.create(engine)

tables = ['table1', 'table2', 'table3']
for _t in tables:
    create_table(_t, metadata)

inspector = inspect(engine)
print(inspector.get_table_names())
```

输出结果：

```
$ python sqlalchemy_create.py
['table1', 'table2', 'table3']
```

## 3.5.24 创建具有动态列的表（Table）

```python
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy import Table
from sqlalchemy import MetaData
from sqlalchemy import inspect
from sqlalchemy.engine.url import URL

db_url = {'drivername': 'postgres',
          'username': 'postgres',
          'password': 'postgres',
          'host': '192.168.99.100',
          'port': 5432}

engine = create_engine(URL(**db_url))

def create_table(name, *cols):
    meta = MetaData()
    meta.reflect(bind=engine)
    if name in meta.tables:
        return
    
    table = Table(name, meta, *cols)
    table.create(engine)

create_table('Table1',
             Column('id', Integer, primary_key=True),
             Column('name', String))
create_table('Table2',
             Column('id', Integer, primary_key=True),
             Column('key', String),
             Column('val', String))

inspector = inspect(engine)
for _t in inspector.get_table_names():
    print(_t)
```

输出结果：

```
$ python sqlalchemy_dynamic.py
Table1
Table2
```

## 3.5.25 对象关系映射添加数据

```python
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL

db_url = {'drivername': 'postgres',
          'username': 'postgres',
          'password': 'postgres',
          'host': '192.168.99.100',
          'port': 5432}
engine = create_engine(URL(**db_url))
Base = declarative_base()

class TestTable(Base):
    __tablename__ = 'Test Table'
    id   = Column(Integer, primary_key=True)
    key  = Column(String, nullable=False)
    val  = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

# 创建表
Base.metadata.create_all(bind=engine)

# 创建会话
Session = sessionmaker()
Session.configure(bind=engine)
session = Session()

data = {'a': 5566, 'b': 9527, 'c': 183}
try:
    for _key, _val in data.items():
        row = TestTable(key=_key, val=_val)
        session.add(row)
    session.commit()
except SQLAlchemyError as e:
    print(e)
finally:
    session.close()
```

## 3.5.26 对象关系映射更新数据

```python
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL

db_url = {'drivername': 'postgres',
          'username': 'postgres',
          'password': 'postgres',
          'host': '192.168.99.100',
          'port': 5432}
engine = create_engine(URL(**db_url))
Base = declarative_base()

class TestTable(Base):
    __tablename__ = 'Test Table'
    id   = Column(Integer, primary_key=True)
    key  = Column(String, nullable=False)
    val  = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

# 创建表
Base.metadata.create_all(bind=engine)

# 创建会话
Session = sessionmaker()
Session.configure(bind=engine)
session = Session()

try:
    # 向数据库添加一行数据
    row = TestTable(key="hello", val="world")
    session.add(row)
    session.commit()

    # 更新数据库中的一行数据
    row = session.query(TestTable).filter(
        TestTable.key == 'hello').first()
    print('original:', row.key, row.val)
    row.key = "Hello"
    row.val = "World"
    session.commit()

    # 检查更新是否正确
    row = session.query(TestTable).filter(
        TestTable.key == 'Hello').first()
    print('update:', row.key, row.val)
except SQLAlchemyError as e:
    print(e)
finally:
    session.close()
```print(e)
finally:
    session.close()
```

输出：

```
$ python sqlalchemy_update.py
original: hello world
update: Hello World
```

## 3.5.27 对象关系删除行

```python
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL

db_url = {'drivername': 'postgres',
           'username': 'postgres',
           'password': 'postgres',
           'host': '192.168.99.100',
           'port': 5432}
engine = create_engine(URL(**db_url))
Base = declarative_base()

class TestTable(Base):
    __tablename__ = 'Test Table'
    id   = Column(Integer, primary_key=True)
    key  = Column(String, nullable=False)
    val  = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

# 创建表
Base.metadata.create_all(bind=engine)

# 创建会话
Session = sessionmaker()
Session.configure(bind=engine)
session = Session()

row = TestTable(key='hello', val='world')
session.add(row)
query = session.query(TestTable).filter(
    TestTable.key=='hello')
print(query.first())
```

```python
query.delete()
query = session.query(TestTable).filter(
    TestTable.key=='hello')
print(query.all())
```

输出：

```
$ python sqlalchemy_delete.py
<__main__.TestTable object at 0x104eb8f50>
[]
```

## 3.5.28 对象关系关系

```python
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    addresses = relationship("Address", backref="user")

class Address(Base):
    __tablename__ = 'address'
    id = Column(Integer, primary_key=True)
    email = Column(String)
    user_id = Column(Integer, ForeignKey('user.id'))

u1 = User()
a1 = Address()
print(u1.addresses)
print(a1.user)

u1.addresses.append(a1)
print(u1.addresses)
print(a1.user)
```

输出：

```
$ python sqlalchemy_relationship.py
[]
None
[<__main__.Address object at 0x10c4edb50>]
<__main__.User object at 0x10c4ed810>
```

## 3.5.29 对象关系自关联

```python
import json

from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    Table)

from sqlalchemy.orm import (
    sessionmaker,
    relationship)

from sqlalchemy.ext.declarative import declarative_base

base = declarative_base()

association = Table("Association", base.metadata,
    Column('left', Integer, ForeignKey('node.id'), primary_key=True),
    Column('right', Integer, ForeignKey('node.id'), primary_key=True))

class Node(base):
    __tablename__ = 'node'
    id = Column(Integer, primary_key=True)
    label = Column(String)
    friends = relationship('Node',
                           secondary=association,
                           primaryjoin=id==association.c.left,
                           secondaryjoin=id==association.c.right,
                           backref='left')
    def to_json(self):
        return dict(id=self.id,
                    friends=[_.label for _ in self.friends])

nodes = [Node(label='node_{}'.format(_)) for _ in range(0, 3)]
nodes[0].friends.extend([nodes[1], nodes[2]])
nodes[1].friends.append(nodes[2])

print('----> right')
print(json.dumps([_.to_json() for _ in nodes], indent=2))

print('----> left')
print(json.dumps([_n.to_json() for _n in nodes[1].left], indent=2))
```

输出：

```
----> right
[
  {
    "friends": [
      "node_1",
      "node_2"
    ],
    "id": null
  },
  {
    "friends": [
      "node_2"
    ],
    "id": null
  },
  {
    "friends": [],
    "id": null
  }
]
----> left
[
  {
    "friends": [
      "node_1",
      "node_2"
    ],
    "id": null
  }
]
```

## 3.5.30 对象关系基本查询

```python
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy import or_
from sqlalchemy import desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL

db_url = {'drivername': 'postgres',
          'username': 'postgres',
          'password': 'postgres',
          'host': '192.168.99.100',
          'port': 5432}

Base = declarative_base()

class User(Base):
    __tablename__ = 'User'
    id      = Column(Integer, primary_key=True)
    name    = Column(String, nullable=False)
    fullname = Column(String, nullable=False)
    birth   = Column(DateTime)

# 创建表
engine = create_engine(URL(**db_url))
Base.metadata.create_all(bind=engine)

users = [
    User(name='ed',
         fullname='Ed Jones',
         birth=datetime(1989,7,1)),
    User(name='wendy',
         fullname='Wendy Williams',
         birth=datetime(1983,4,1)),
    User(name='mary',
         fullname='Mary Contrary',
         birth=datetime(1990,1,30)),
    User(name='fred',
         fullname='Fred Flinstone',
         birth=datetime(1977,3,12)),
    User(name='justin',
         fullname="Justin Bieber")]

# 创建会话
Session = sessionmaker()
Session.configure(bind=engine)
session = Session()

# add_all
session.add_all(users)
session.commit()

print("----> order_by(id):")
query = session.query(User).order_by(User.id)
for _row in query.all():
    print(_row.name, _row.fullname, _row.birth)

print("\n----> order_by(desc(id)):")
query = session.query(User).order_by(desc(User.id))
for _row in query.all():
    print(_row.name, _row.fullname, _row.birth)

print("\n----> order_by(date):")
query = session.query(User).order_by(User.birth)
for _row in query.all():
    print(_row.name, _row.fullname, _row.birth)

print("\n----> EQUAL:")
query = session.query(User).filter(User.id == 2)
_row = query.first()
print(_row.name, _row.fullname, _row.birth)

print("\n----> NOT EQUAL:")
query = session.query(User).filter(User.id != 2)
for _row in query.all():
    print(_row.name, _row.fullname, _row.birth)

print("\n----> IN:")
query = session.query(User).filter(User.name.in_(['ed', 'wendy']))
for _row in query.all():
    print(_row.name, _row.fullname, _row.birth)

print("\n----> NOT IN:")
query = session.query(User).filter(~User.name.in_(['ed', 'wendy']))
for _row in query.all():
    print(_row.name, _row.fullname, _row.birth)

print("\n----> AND:")
query = session.query(User).filter(
        User.name=='ed', User.fullname=='Ed Jones')
_row = query.first()
print(_row.name, _row.fullname, _row.birth)

print("\n----> OR:")
query = session.query(User).filter(
        or_(User.name=='ed', User.name=='wendy'))
for _row in query.all():
    print(_row.name, _row.fullname, _row.birth)

print("\n----> NULL:")
query = session.query(User).filter(User.birth == None)
for _row in query.all():
    print(_row.name, _row.fullname)

print("\n----> NOT NULL:")
query = session.query(User).filter(User.birth != None)
for _row in query.all():
    print(_row.name, _row.fullname)

print("\n----> LIKE")
query = session.query(User).filter(User.name.like('%ed%'))
for _row in query.all():
    print(_row.name, _row.fullname)
```

输出：

```
----> order_by(id):
ed Ed Jones 1989-07-01 00:00:00
wendy Wendy Williams 1983-04-01 00:00:00
mary Mary Contrary 1990-01-30 00:00:00
fred Fred Flinstone 1977-03-12 00:00:00
justin Justin Bieber None
```

```
----> order_by(desc(id)):
justin Justin Bieber None
fred Fred Flinstone 1977-03-12 00:00:00
mary Mary Contrary 1990-01-30 00:00:00
wendy Wendy Williams 1983-04-01 00:00:00
ed Ed Jones 1989-07-01 00:00:00
```

```
----> order_by(date):
fred Fred Flinstone 1977-03-12 00:00:00
wendy Wendy Williams 1983-04-01 00:00:00
ed Ed Jones 1989-07-01 00:00:00
mary Mary Contrary 1990-01-30 00:00:00
justin Justin Bieber None
```

```
----> EQUAL:
wendy Wendy Williams 1983-04-01 00:00:00
```

```
----> NOT EQUAL:
ed Ed Jones 1989-07-01 00:00:00
mary Mary Contrary 1990-01-30 00:00:00
fred Fred Flinstone 1977-03-12 00:00:00
justin Justin Bieber None
```

```
----> IN:
ed Ed Jones 1989-07-01 00:00:00
wendy Wendy Williams 1983-04-01 00:00:00
```

```
----> NOT IN:
mary Mary Contrary 1990-01-30 00:00:00
fred Fred Flinstone 1977-03-12 00:00:00
justin Justin Bieber None
```

```
----> AND:
ed Ed Jones 1989-07-01 00:00:00
```

```
----> OR:
ed Ed Jones 1989-07-01 00:00:00
wendy Wendy Williams 1983-04-01 00:00:00
```

```
----> NULL:
justin Justin Bieber
```

```
----> NOT NULL:
ed Ed Jones
wendy Wendy Williams
mary Mary Contrary
fred Fred Flinstone
```

```
----> LIKE
ed Ed Jones
fred Fred Flinstone
```## 3.5.31 映射器：将表映射到类

```python
from sqlalchemy import (
    create_engine,
    Table,
    MetaData,
    Column,
    Integer,
    String,
    ForeignKey)

from sqlalchemy.orm import (
    mapper,
    relationship,
    sessionmaker)

# 经典映射：将 "table" 映射到 "class"
db_url = 'sqlite://'
engine = create_engine(db_url)

meta = MetaData(bind=engine)

user = Table('User', meta,
             Column('id', Integer, primary_key=True),
             Column('name', String),
             Column('fullname', String),
             Column('password', String))

addr = Table('Address', meta,
             Column('id', Integer, primary_key=True),
             Column('email', String),
             Column('user_id', Integer, ForeignKey('User.id')))

# 将表映射到类
class User(object):
    def __init__(self, name, fullname, password):
        self.name = name
        self.fullname = fullname
        self.password = password

class Address(object):
    def __init__(self, email):
        self.email = email

mapper(User, user, properties={
    'addresses': relationship(Address, backref='user')})
mapper(Address, addr)

# 创建表
meta.create_all()

# 创建会话
Session = sessionmaker()
Session.configure(bind=engine)
session = Session()

u = User(name='Hello', fullname='HelloWorld', password='ker')
a = Address(email='hello@hello.com')
u.addresses.append(a)
try:
    session.add(u)
    session.commit()

    # 查询结果
    u = session.query(User).filter(User.name == 'Hello').first()
    print(u.name, u.fullname, u.password)

finally:
    session.close()
```

输出：

```
$ python map_table_class.py
Hello HelloWorld ker
```

## 3.5.32 动态获取表

```python
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    inspect,
    Column,
    String,
    Integer)

from sqlalchemy.orm import (
    mapper,
    scoped_session,
    sessionmaker)

db_url = "sqlite://"
engine = create_engine(db_url)
metadata = MetaData(engine)

class TableTemp(object):
    def __init__(self, name):
        self.name = name

def get_table(name):
    if name in metadata.tables:
        table = metadata.tables[name]
    else:
        table = Table(name, metadata,
                      Column('id', Integer, primary_key=True),
                      Column('name', String))
        table.create(engine)

    cls = type(name.title(), (TableTemp,), {})
    mapper(cls, table)
    return cls

# 第一次获取表
t = get_table('Hello')

# 第二次获取表
t = get_table('Hello')

Session = scoped_session(sessionmaker(bind=engine))
try:
    Session.add(t(name='foo'))
    Session.add(t(name='bar'))
    for _ in Session.query(t).all():
        print(_.name)
except Exception as e:
    Session.rollback()
finally:
    Session.close()
```

输出：

```
$ python get_table.py
foo
bar
```

## 3.5.33 对象关系连接两个表

```python
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    addresses = relationship("Address", backref="user")

class Address(Base):
    __tablename__ = 'address'
    id = Column(Integer, primary_key=True)
    email = Column(String)
    user_id = Column(Integer, ForeignKey('user.id'))

db_url = {'drivername': 'postgres',
          'username': 'postgres',
          'password': 'postgres',
          'host': '192.168.99.100',
          'port': 5432}

# 创建引擎
engine = create_engine(URL(**db_url))

# 创建表
Base.metadata.create_all(bind=engine)

# 创建会话
Session = sessionmaker()
Session.configure(bind=engine)
session = Session()

user = User(name='user1')
mail1 = Address(email='user1@foo.com')
mail2 = Address(email='user1@bar.com')
user.addresses.extend([mail1, mail2])

session.add(user)
session.add_all([mail1, mail2])
session.commit()

query = session.query(Address, User).join(User)
for _a, _u in query.all():
    print(_u.name, _a.email)
```

输出：

```
$ python sqlalchemy_join.py
user1 user1@foo.com
user1 user1@bar.com
```

## 3.5.34 基于关系进行连接和按组计数

```python
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    ForeignKey,
    func)

from sqlalchemy.orm import (
    relationship,
    sessionmaker,
    scoped_session)

from sqlalchemy.ext.declarative import declarative_base

db_url = 'sqlite://'
engine = create_engine(db_url)

Base = declarative_base()

class Parent(Base):
    __tablename__ = 'parent'
    id       = Column(Integer, primary_key=True)
    name     = Column(String)
    children = relationship('Child', back_populates='parent')

class Child(Base):
    __tablename__ = 'child'
    id         = Column(Integer, primary_key=True)
    name       = Column(String)
    parent_id  = Column(Integer, ForeignKey('parent.id'))
    parent     = relationship('Parent', back_populates='children')

Base.metadata.create_all(bind=engine)
Session = scoped_session(sessionmaker(bind=engine))

p1 = Parent(name="Alice")
p2 = Parent(name="Bob")

c1 = Child(name="foo")
c2 = Child(name="bar")
c3 = Child(name="ker")
c4 = Child(name="cat")

p1.children.extend([c1, c2, c3])
p2.children.append(c4)

try:
    Session.add(p1)
    Session.add(p2)
    Session.commit()

    # 计算子项数量
    q = Session.query(Parent, func.count(Child.id))\
            .join(Child)\
            .group_by(Parent.id)

    # 打印结果
    for _p, _c in q.all():
        print('parent: {}, num_child: {}'.format(_p.name, _c))
finally:
    Session.remove()
```

输出：

```
$ python join_group_by.py
parent: Alice, num_child: 3
parent: Bob, num_child: 1
```

## 3.5.35 创建动态列的表（ORM）

```python
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy import inspect
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base

db_url = {'drivername': 'postgres',
          'username': 'postgres',
          'password': 'postgres',
          'host': '192.168.99.100',
          'port': 5432}

engine = create_engine(URL(**db_url))
Base = declarative_base()

def create_table(name, cols):
    Base.metadata.reflect(engine)
    if name in Base.metadata.tables: return

    table = type(name, (Base,), cols)
    table.__table__.create(bind=engine)

create_table('Table1', {
    '__tablename__': 'Table1',
    'id': Column(Integer, primary_key=True),
    'name': Column(String)})

create_table('Table2', {
    '__tablename__': 'Table2',
    'id': Column(Integer, primary_key=True),
    'key': Column(String),
    'val': Column(String)})

inspector = inspect(engine)
for _t in inspector.get_table_names():
    print(_t)
```

输出：

```
$ python sqlalchemy_dynamic_orm.py
Table1
Table2
```

### 3.5.36 关闭数据库连接

```python
from sqlalchemy import (
    create_engine,
    event,
    Column,
    Integer)

from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite://')
base = declarative_base()

@event.listens_for(engine, 'engine_disposed')
def receive_engine_disposed(engine):
    print("engine dispose")

class Table(base):
    __tablename__ = 'example table'
    id = Column(Integer, primary_key=True)

base.metadata.create_all(bind=engine)
session = sessionmaker(bind=engine)()

try:
    try:
        row = Table()
        session.add(row)
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()
finally:
    engine.dispose()
```

输出：

```
$ python db_dispose.py
engine dispose
```

> **警告：** 注意。关闭 *会话* 并不意味着关闭数据库连接。SQLAlchemy 的 *会话* 通常代表 *事务*，而非连接。

## 3.5.37 关闭会话后不能使用该对象

```python
from __future__ import print_function

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer)

from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

url = 'sqlite://'
engine = create_engine(url)
base = declarative_base()

class Table(base):
    __tablename__ = 'table'
    id  = Column(Integer, primary_key=True)
    key = Column(String)
    val = Column(String)

base.metadata.create_all(bind=engine)
session = sessionmaker(bind=engine)()

try:
    t = Table(key="key", val="val")
    try:
        print(t.key, t.val)
        session.add(t)
        session.commit()
    except Exception as e:
        print(e)
        session.rollback()
    finally:
        session.close()

    print(t.key, t.val) # exception raised from here
except Exception as e:
    print("Cannot use the object after close the session")
finally:
    engine.dispose()
```

输出：

```
$ python sql.py
key val
Cannot use the object after close the session
```

## 3.5.38 钩子函数

```python
from sqlalchemy import Column, String, Integer
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

url = "sqlite:///:memory:"
engine = create_engine(url)
Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)

@event.listens_for(User, "before_insert")
def before_insert(mapper, connection, user):
    print(f"before insert: {user.name}")

@event.listens_for(User, "after_insert")
def after_insert(mapper, connection, user):
    print(f"after insert: {user.name}")

try:
    session = scoped_session(Session)
    user = User(name="bob", age=18)
    session.add(user)
    session.commit()
except SQLAlchemyError as e:
    session.rollback()
finally:
    session.close()
```

## 3.6 安全

**目录**
*   安全
    *   简单的HTTPS服务器
    *   生成SSH密钥对
    *   获取证书信息
    *   生成自签名证书
    *   准备证书签名请求（CSR）
    *   生成无密码的RSA密钥文件
    *   使用给定私钥签名文件
    *   根据签名摘要验证文件
    *   通过PEM文件进行简单的RSA加密
    *   通过RSA模块进行简单的RSA加密
    *   通过PEM文件进行简单的RSA解密
    *   使用OAEP的简单RSA加密
    *   使用OAEP的简单RSA解密
    *   使用DSA进行身份验证
    *   使用AES CBC模式加密文件
    *   使用AES CBC模式解密文件
    *   通过密码进行AES CBC模式加密（使用cryptography库）
    *   通过密码进行AES CBC模式解密（使用cryptography库）
    *   通过密码进行AES CBC模式加密（使用pycrypto库）
    *   通过密码进行AES CBC模式解密（使用pycrypto库）
    *   使用cryptography库进行临时Diffie-Hellman密钥交换
    *   使用cryptography库手动计算DH共享密钥
    *   根据 (p, g, pubkey) 计算DH共享密钥

### 3.6.1 简单的HTTPS服务器

```python
# python2
>>> import BaseHTTPServer, SimpleHTTPServer
>>> import ssl
>>> host, port = 'localhost', 5566
>>> handler = SimpleHTTPServer.SimpleHTTPRequestHandler
>>> httpd = BaseHTTPServer.HTTPServer((host, port), handler)
>>> httpd.socket = ssl.wrap_socket(httpd.socket,
...                     certfile='./cert.crt',
...                     keyfile='./cert.key',
...                     server_side=True)
>>> httpd.serve_forever()

# python3
>>> from http import server
>>> handler = server.SimpleHTTPRequestHandler
>>> import ssl
>>> host, port = 'localhost', 5566
>>> httpd = server.HTTPServer((host, port), handler)
>>> httpd.socket = ssl.wrap_socket(httpd.socket,
...                     certfile='./cert.crt',
...                     keyfile='./cert.key',
...                     server_side=True)
>>> httpd.serve_forever()
```

### 3.6.2 生成SSH密钥对

```python
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

key = rsa.generate_private_key(
    backend=default_backend(),
    public_exponent=65537,
    key_size=2048
)
private_key = key.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
)
public_key = key.public_key().public_bytes(
    serialization.Encoding.OpenSSH,
    serialization.PublicFormat.OpenSSH
)

with open('id_rsa', 'wb') as f, open('id_rsa.pub', 'wb') as g:
    f.write(private_key)
    g.write(public_key)
```

### 3.6.3 获取证书信息

```python
from cryptography import x509
from cryptography.hazmat.backends import default_backend

backend = default_backend()
with open('./cert.crt', 'rb') as f:
    crt_data = f.read()
    cert = x509.load_pem_x509_certificate(crt_data, backend)

class Certificate:

    _fields = ['country_name',
               'state_or_province_name',
               'locality_name',
               'organization_name',
               'organizational_unit_name',
               'common_name',
               'email_address']

    def __init__(self, cert):
        assert isinstance(cert, x509.Certificate)
        self._cert = cert
        for attr in self._fields:
            oid = getattr(x509, 'OID_' + attr.upper())
            subject = cert.subject
            info = subject.get_attributes_for_oid(oid)
            setattr(self, attr, info)

cert = Certificate(cert)
for attr in cert._fields:
    for info in getattr(cert, attr):
        print("{}: {}".format(info._oid._name, info._value))
```

输出：

```
$ genrsa -out cert.key
Generating RSA private key, 1024 bit long modulus
...........++++++
...++++++
e is 65537 (0x10001)
$ openssl req -x509 -new -nodes \
>        -key cert.key -days 365 \
>        -out cert.crt
You are about to be asked to enter information that will be incorporated
into your certificate request.
What you are about to enter is what is called a Distinguished Name or a DN.
There are quite a few fields but you can leave some blank
For some fields there will be a default value,
If you enter '.', the field will be left blank.
-----
Country Name (2 letter code) [AU]:TW
State or Province Name (full name) [Some-State]:Taiwan
Locality Name (eg, city) []:Taipei
Organization Name (eg, company) [Internet Widgits Pty Ltd]:personal
Organizational Unit Name (eg, section) []:personal
Common Name (e.g. server FQDN or YOUR name) []:localhost
Email Address []:test@example.com
$ python3 cert.py
countryName: TW
stateOrProvinceName: Taiwan
localityName: Taipei
organizationName: personal
organizationalUnitName: personal
commonName: localhost
emailAddress: test@example.com
```

### 3.6.4 生成自签名证书

```python
from __future__ import print_function, unicode_literals

from datetime import datetime, timedelta
from OpenSSL import crypto

# load private key
ftype = crypto.FILETYPE_PEM
with open('key.pem', 'rb') as f: k = f.read()
k = crypto.load_privatekey(ftype, k)

now    = datetime.now()
expire = now + timedelta(days=365)

# country (countryName, C)
# state or province name (stateOrProvinceName, ST)
# locality (locality, L)
# organization (organizationName, O)
# organizational unit (organizationalUnitName, OU)
# common name (commonName, CN)

cert = crypto.X509()
cert.get_subject().C  = "TW"
cert.get_subject().ST = "Taiwan"
cert.get_subject().L  = "Taipei"
cert.get_subject().O  = "pysheeet"
cert.get_subject().OU = "cheat sheet"
cert.get_subject().CN = "pythonsheets.com"
cert.set_serial_number(1000)
cert.set_notBefore(now.strftime("%Y%m%d%H%M%SZ").encode())
cert.set_notAfter(expire.strftime("%Y%m%d%H%M%SZ").encode())
cert.set_issuer(cert.get_subject())
cert.set_pubkey(k)
cert.sign(k, 'sha1')

with open('cert.pem', "wb") as f:
    f.write(crypto.dump_certificate(ftype, cert))
```

输出：

```
$ openssl genrsa -out key.pem 2048
Generating RSA private key, 2048 bit long modulus
..............+++
......................................................+++
e is 65537 (0x10001)
$ python3 x509.py
$ openssl x509 -subject -issuer -noout -in cert.pem
subject= /C=TW/ST=Taiwan/L=Taipei/O=pysheeet/OU=cheat sheet/CN=pythonsheets.com
issuer= /C=TW/ST=Taiwan/L=Taipei/O=pysheeet/OU=cheat sheet/CN=pythonsheets.com
```

### 3.6.5 准备证书签名请求（CSR）

```python
from __future__ import print_function, unicode_literals

from OpenSSL import crypto

# load private key
ftype = crypto.FILETYPE_PEM
with open('key.pem', 'rb') as f: key = f.read()
key = crypto.load_privatekey(ftype, key)
req    = crypto.X509Req()

alt_name  = [ b"DNS:www.pythonsheet.com",
              b"DNS:doc.pythonsheet.com" ]
key_usage = [ b"Digital Signature",
              b"Non Repudiation",
              b"Key Encipherment" ]

# country (countryName, C)
# state or province name (stateOrProvinceName, ST)
# locality (locality, L)
# organization (organizationName, O)
# organizational unit (organizationalUnitName, OU)
# common name (commonName, CN)

req.get_subject().C  = "TW"
req.get_subject().ST = "Taiwan"
req.get_subject().L  = "Taipei"
req.get_subject().O  = "pysheeet"
req.get_subject().OU = "cheat sheet"
req.get_subject().CN = "pythonsheets.com"
req.add_extensions([
    crypto.X509Extension( b"basicConstraints",
                         False,
```

## 3.6.6 生成无密码短语的RSA密钥文件

```python
>>> from cryptography.hazmat.backends import default_backend
>>> from cryptography.hazmat.primitives import serialization
>>> from cryptography.hazmat.primitives.asymmetric import rsa
>>> key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend())

>>> with open('cert.key', 'wb') as f:
    f.write(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()))
```

## 3.6.7 使用给定私钥对文件进行签名

```python
from __future__ import print_function, unicode_literals

from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA256


def signer(privkey, data):
    rsakey = RSA.importKey(privkey)
    signer = PKCS1_v1_5.new(rsakey)
    digest = SHA256.new()
    digest.update(data)
    return signer.sign(digest)


with open('private.key', 'rb') as f: key = f.read()
with open('foo.tgz', 'rb') as f: data = f.read()

sign = signer(key, data)
with open('foo.tgz.sha256', 'wb') as f: f.write(sign)
```

```bash
# 生成公钥和私钥
$ openssl genrsa -out private.key 2048
$ openssl rsa -in private.key -pubout -out public.key

$ python3 sign.py
$ openssl dgst -sha256 -verify public.key -signature foo.tgz.sha256 foo.tgz
Verified OK
```

## 3.6.8 从签名摘要验证文件

```python
from __future__ import print_function, unicode_literals

import sys

from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA256

def verifier(pubkey, sig, data):
    rsakey = RSA.importKey(key)
    signer = PKCS1_v1_5.new(rsakey)
    digest = SHA256.new()

    digest.update(data)
    return signer.verify(digest, sig)

with open("public.key", 'rb') as f: key = f.read()
with open("foo.tgz.sha256", 'rb') as f: sig = f.read()
with open("foo.tgz", 'rb') as f: data = f.read()

if verifier(key, sig, data):
    print("Verified OK")
else:
    print("Verification Failure")
```

```bash
# 生成公钥和私钥
$ openssl genrsa -out private.key 2048
$ openssl rsa -in private.key -pubout -out public.key

# 执行验证
$ cat /dev/urandom | head -c 512 | base64 > foo.txt
$ tar -zcf foo.tgz foo.txt
$ openssl dgst -sha256 -sign private.key -out foo.tgz.sha256 foo.tgz
$ python3 verify.py
Verified OK

# 通过openssl执行验证
$ openssl dgst -sha256 -verify public.key -signature foo.tgz.sha256 foo.tgz
Verified OK
```

## 3.6.9 通过PEM文件进行简单的RSA加密

```python
from __future__ import print_function, unicode_literals

import base64
import sys

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5

key_text = sys.stdin.read()

# 通过rsa模块导入密钥
pubkey = RSA.importKey(key_text)

# 通过PKCS1.5创建密码器
cipher = PKCS1_v1_5.new(pubkey)

# 加密
cipher_text = cipher.encrypt(b"Hello RSA!")

# 进行base64编码
cipher_text = base64.b64encode(cipher_text)
print(cipher_text.decode('utf-8'))
```

```bash
$ openssl genrsa -out private.key 2048
$ openssl rsa -in private.key -pubout -out public.key
$ cat public.key | \
> python3 rsa.py | \
> openssl base64 -d -A | \
> openssl rsautl -decrypt -inkey private.key
Hello RSA!
```

## 3.6.10 通过RSA模块进行简单的RSA加密

```python
from __future__ import print_function, unicode_literals

import base64
import sys

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
from Crypto.PublicKey.RSA import construct

# 准备公钥
e = int('10001', 16)
n = int(sys.stdin.read(), 16)
pubkey = construct((n, e))
# 通过PKCS1.5创建密码器
cipher = PKCS1_v1_5.new(pubkey)
# 加密
cipher_text = cipher.encrypt(b"Hello RSA!")
# 进行base64编码
cipher_text = base64.b64encode(cipher_text)
print(cipher_text.decode('utf-8'))
```

```bash
$ openssl genrsa -out private.key 2048
$ openssl rsa -in private.key -pubout -out public.key
$ # 检查 (n, e)
$ openssl rsa -pubin -inform PEM -text -noout < public.key
Public-Key: (2048 bit)
Modulus:
    00:93:d5:58:0c:18:cf:91:f0:74:af:1b:40:09:73:
    0c:d8:13:23:6c:44:60:0d:83:71:e6:f9:61:85:e5:
    b2:d0:8a:73:5c:02:02:51:9a:4f:a7:ab:05:d5:74:
    ff:4d:88:3d:e2:91:b8:b0:9f:7e:a9:a3:b2:3c:99:
    1c:9a:42:4d:ac:2f:6a:e7:eb:0f:a7:e0:a5:81:e5:
    98:49:49:d5:15:3d:53:42:12:08:db:b0:e7:66:2d:
    71:5b:ea:55:4e:2d:9b:40:79:f8:7d:6e:5d:f4:a7:
    d8:13:cb:13:91:c9:ac:5b:55:62:70:44:25:50:ca:
    94:de:78:5d:97:e8:a9:33:66:4f:90:10:00:62:21:
    b6:60:52:65:76:bd:a3:3b:cf:2a:db:3f:66:5f:0d:
    a3:35:ff:29:34:26:6d:63:a2:a6:77:96:5a:84:c7:
    6a:0c:4f:48:52:70:11:8f:85:11:a0:78:f8:60:4b:
    5d:d8:4b:b2:64:e5:ec:99:72:c5:a8:1b:ab:5c:09:
    e1:80:70:91:06:22:ba:97:33:56:0b:65:d8:f3:35:
    66:f8:f9:ea:b9:84:64:8e:3c:14:f7:3d:1f:2c:67:
    ce:64:cf:f9:c5:16:6b:03:a1:7a:c7:fa:4c:38:56:
    ee:e0:4d:5f:ec:46:7e:1f:08:7c:e6:45:a1:fc:17:
    1f:91
Exponent: 65537 (0x10001)
$ openssl rsa -pubin -in public.key -modulus -noout | \
> cut -d'=' -f 2 | \
> python3 rsa.py | \
> openssl base64 -d -A | \
> openssl rsautl -decrypt -inkey private.key
Hello RSA!
```

## 3.6.11 通过PEM文件进行简单的RSA解密

```python
from __future__ import print_function, unicode_literals

import base64
import sys

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5

# 读取密钥文件
with open('private.key') as f: key_text = f.read()

# 创建私钥对象
privkey = RSA.importKey(key_text)

# 创建密码器对象
cipher = PKCS1_v1_5.new(privkey)

# 解码base64
cipher_text = base64.b64decode(sys.stdin.read())

# 解密
plain_text = cipher.decrypt(cipher_text, None)
print(plain_text.decode('utf-8').strip())
```

```bash
$ openssl genrsa -out private.key 2048
$ openssl rsa -in private.key -pubout -out public.key
$ echo "Hello openssl RSA encrypt" | \
> openssl rsautl -encrypt -pubin -inkey public.key | \
> openssl base64 -e -A | \
> python3 rsa.py
Hello openssl RSA encrypt
```

## 3.6.12 使用OAEP进行简单的RSA加密

```python
from __future__ import print_function, unicode_literals

import base64
import sys

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# 读取密钥文件
key_text = sys.stdin.read()

# 创建公钥对象
pubkey = RSA.importKey(key_text)
```## 3.6.13 使用OAEP进行简单的RSA解密

```python
# 创建一个密码对象
cipher = PKCS1_OAEP.new(pubkey)

# 加密明文
cipher_text = cipher.encrypt(b"Hello RSA OAEP!")

# 使用base64编码
cipher_text = base64.b64encode(cipher_text)
print(cipher_text.decode('utf-8'))
```

输出：

```bash
$ openssl genrsa -out private.key 2048
$ openssl rsa -in private.key -pubout -out public.key
$ cat public.key       | \
> python3 rsa.py       | \
> openssl base64 -d -A | \
> openssl rsautl -decrypt -oaep -inkey private.key
Hello RSA OAEP!
```

```python
from __future__ import print_function, unicode_literals

import base64
import sys

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# 读取密钥文件
with open('private.key') as f: key_text = f.read()

# 创建一个私钥对象
privkey = RSA.importKey(key_text)

# 创建一个密码对象
cipher = PKCS1_OAEP.new(privkey)

# 解码base64
cipher_text = base64.b64decode(sys.stdin.read())

# 解密
plain_text = cipher.decrypt(cipher_text)
print(plain_text.decode('utf-8').strip())
```

输出：

```bash
$ openssl genrsa -out private.key 2048
$ openssl rsa -in private.key -pubout -out public.key
$ echo "Hello RSA encrypt via OAEP" | \
> openssl rsautl -encrypt -pubin -oaep -inkey public.key | \
> openssl base64 -e -A | \
> python3 rsa.py
Hello RSA encrypt via OAEP
```

## 3.6.14 使用DSA进行身份验证

```python
import socket

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dsa

alice, bob = socket.socketpair()

def gen_dsa_key():
    private_key = dsa.generate_private_key(
        key_size=2048, backend=default_backend())
    return private_key, private_key.public_key()

def sign_data(data, private_key):
    signature = private_key.sign(data, hashes.SHA256())
    return signature

def verify_data(data, signature, public_key):
    try:
        public_key.verify(signature, data, hashes.SHA256())
    except InvalidSignature:
        print("收到消息：{} 不可信！".format(data))
    else:
        print("验证消息：{} 成功！".format(data))

# 生成Alice的私钥和公钥
alice_private_key, alice_public_key = gen_dsa_key()

# Alice向Bob发送消息，Bob接收
alice_msg = b"Hello Bob"
b = alice.send(alice_msg)
bob_recv_msg = bob.recv(1024)

# Alice向Bob发送签名，Bob接收
signature = sign_data(alice_msg, alice_private_key)
b = alice.send(signature)
bob_recv_signature = bob.recv(1024)

# Bob验证从Alice收到的消息
verify_data(bob_recv_msg, bob_recv_signature, alice_public_key)

# 攻击者修改消息会导致验证失败
verify_data(b"I'm attacker!", bob_recv_signature, alice_public_key)
```

输出：

```text
$ python3 test_dsa.py
check msg: b'Hello Bob' success!
recv msg: b"I'm attacker!" not trust!
```

## 3.6.15 使用AES CBC模式加密文件

```python
from __future__ import print_function, unicode_literals

import struct
import sys
import os

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import (
    Cipher,
    algorithms,
    modes)

backend = default_backend()
key = os.urandom(32)
iv  = os.urandom(16)

def encrypt(ptext):
    pad = padding.PKCS7(128).padder()
    ptext = pad.update(ptext) + pad.finalize()

    alg = algorithms.AES(key)
    mode = modes.CBC(iv)
    cipher = Cipher(alg, mode, backend=backend)
    encryptor = cipher.encryptor()
    ctext = encryptor.update(ptext) + encryptor.finalize()

    return ctext

print("密钥: {}".format(key.hex()))
print("初始向量: {}".format(iv.hex()))

if len(sys.argv) != 3:
    raise Exception("用法: 命令 [文件] [加密文件]")

# 从文件读取明文
with open(sys.argv[1], 'rb') as f:
    plaintext = f.read()

# 加密文件
ciphertext = encrypt(plaintext)
with open(sys.argv[2], 'wb') as f:
    f.write(ciphertext)
```

输出：

```bash
$ echo "Encrypt file via AES-CBC" > test.txt
$ python3 aes.py test.txt test.enc
key: f239d9609e3f318b7afda7e4bb8db5b8734f504cf67f55e45dfe75f90d24fefc
iv: 8d6383b469f100d25293fb244ccb951e
$ openssl aes-256-cbc -d -in test.enc -out secrets.txt.new \
> -K f239d9609e3f318b7afda7e4bb8db5b8734f504cf67f55e45dfe75f90d24fefc \
> -iv 8d6383b469f100d25293fb244ccb951e
$ cat secrets.txt.new
Encrypt file via AES-CBC
```

## 3.6.16 使用AES CBC模式解密文件

```python
from __future__ import print_function, unicode_literals

import struct
import sys
import os

from binascii import unhexlify

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import (
    Cipher,
    algorithms,
    modes)

backend = default_backend()

def decrypt(key, iv, ctext):
    alg = algorithms.AES(key)
    mode = modes.CBC(iv)
    cipher = Cipher(alg, mode, backend=backend)
    decryptor = cipher.decryptor()
    ptext = decryptor.update(ctext) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder() # 128位
    ptext = unpadder.update(ptext) + unpadder.finalize()

    return ptext

if len(sys.argv) != 4:
    raise Exception("用法: 命令 [密钥] [初始向量] [文件]")

# 从文件读取密文
with open(sys.argv[3], 'rb') as f:
    ciphertext = f.read()

# 解密文件
key, iv = unhexlify(sys.argv[1]), unhexlify(sys.argv[2])
plaintext = decrypt(key, iv, ciphertext)
print(plaintext)
```

输出：

```bash
$ echo "Encrypt file via AES-CBC" > test.txt
$ key=`openssl rand -hex 32`
$ iv=`openssl rand -hex 16`
$ openssl enc -aes-256-cbc -in test.txt -out test.enc -K $key -iv $iv
$ python3 aes.py $key $iv test.enc
```

## 3.6.17 使用密码进行AES CBC模式加密（使用cryptography库）

```python
from __future__ import print_function, unicode_literals

import base64
import struct
import sys
import os

from hashlib import md5, sha1

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import (
    Cipher,
    algorithms,
    modes)

backend = default_backend()

def EVP_ByteToKey(pwd, md, salt, key_len, iv_len):
    buf = md(pwd + salt).digest()
    d = buf
    while len(buf) < (iv_len + key_len):
        d = md(d + pwd + salt).digest()
        buf += d
    return buf[:key_len], buf[key_len:key_len + iv_len]

def aes_encrypt(pwd, ptext, md):
    key_len, iv_len = 32, 16

    # 生成盐值
    salt = os.urandom(8)

    # 从密码生成密钥和初始向量
    key, iv = EVP_ByteToKey(pwd, md, salt, key_len, iv_len)

    # 填充明文
    pad = padding.PKCS7(128).padder()
    ptext = pad.update(ptext) + pad.finalize()

    # 创建加密器
    alg = algorithms.AES(key)
    mode = modes.CBC(iv)
    cipher = Cipher(alg, mode, backend=backend)
    encryptor = cipher.encryptor()

    # 加密明文
    ctext = encryptor.update(ptext) + encryptor.finalize()
    ctext = b'Salted__' + salt + ctext

    # 进行base64编码
    ctext = base64.b64encode(ctext)
    return ctext

if len(sys.argv) != 2: raise Exception("用法: 命令 [消息摘要算法]")

md = globals()[sys.argv[1]]

plaintext = sys.stdin.read().encode('utf-8')
pwd = b"password"

print(aes_encrypt(pwd, plaintext, md).decode('utf-8'))
```

输出：

```bash
# 使用md5消息摘要算法
$ echo "Encrypt plaintext via AES-CBC from a given password" | \
> python3 aes.py md5 | \
> openssl base64 -d -A | \
> openssl aes-256-cbc -md md5 -d -k password
Encrypt plaintext via AES-CBC from a given password

# 使用sha1消息摘要算法
$ echo "Encrypt plaintext via AES-CBC from a given password" | \
> python3 aes.py sha1 | \
> openssl base64 -d -A | \
> openssl aes-256-cbc -md sha1 -d -k password
Encrypt plaintext via AES-CBC from a given password
```

## 3.6.18 使用密码进行AES CBC模式解密（使用cryptography库）

```python
from __future__ import print_function, unicode_literals

import base64
import struct
import sys
import os

from hashlib import md5, sha1

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import (
    Cipher,
    algorithms,
    modes)

backend = default_backend()

def EVP_ByteToKey(pwd, md, salt, key_len, iv_len):
    buf = md(pwd + salt).digest()
    d = buf
    while len(buf) < (iv_len + key_len):
        d = md(d + pwd + salt).digest()
        buf += d
    return buf[:key_len], buf[key_len:key_len + iv_len]

def aes_decrypt(pwd, ctext, md):
    ctext = base64.b64decode(ctext)

    # 检查魔数
    if ctext[:8] != b'Salted__':
        raise Exception("错误的魔数")

    # 获取盐值
    salt = ctext[8:16]

    # 从密码生成密钥和初始向量
    key, iv = EVP_ByteToKey(pwd, md, salt, 32, 16)

    # 解密
    alg = algorithms.AES(key)
    mode = modes.CBC(iv)
    cipher = Cipher(alg, mode, backend=backend)
    decryptor = cipher.decryptor()
    ptext = decryptor.update(ctext[16:]) + decryptor.finalize()

    # 移除明文填充
    unpadder = padding.PKCS7(128).unpadder() # 128位
    ptext = unpadder.update(ptext) + unpadder.finalize()
    return ptext.strip()
```

## 3.6.19 使用密码通过 AES CBC 模式加密（使用 pycrypto）

```
from __future__ import print_function, unicode_literals

import struct
import base64
import sys

from hashlib import md5, sha1
from Crypto.Cipher import AES
from Crypto.Random.random import getrandbits

# AES CBC 要求块对齐在 16 字节边界上。
BS = 16

pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS).encode('utf-8')
unpad = lambda s : s[0:-ord(s[-1])]

def EVP_ByteToKey(pwd, md, salt, key_len, iv_len):
    buf = md(pwd + salt).digest()
    d = buf
    while len(buf) < (iv_len + key_len):
        d = md(d + pwd + salt).digest()
        buf += d
    return buf[:key_len], buf[key_len:key_len + iv_len]

def aes_encrypt(pwd, plaintext, md):
    key_len, iv_len = 32, 16

    # 生成盐值
    salt = struct.pack('=Q', getrandbits(64))

    # 从密码生成密钥和初始化向量
    key, iv = EVP_ByteToKey(pwd, md, salt, key_len, iv_len)

    # 填充明文
    plaintext = pad(plaintext)

    # 创建密码对象
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # 参考：openssl/apps/enc.c
    ciphertext = b'Salted__' + salt + cipher.encrypt(plaintext)

    # 进行 base64 编码
    ciphertext = base64.b64encode(ciphertext)
    return ciphertext

if len(sys.argv) != 2: raise Exception("usage: CMD [md]")

md = globals()[sys.argv[1]]

plaintext = sys.stdin.read().encode('utf-8')
pwd = b"password"

print(aes_encrypt(pwd, plaintext, md).decode('utf-8'))
```

输出：

```
# 使用 md5 摘要
$ echo "Encrypt plaintext via AES-CBC from a given password" |\n> python3 aes.py md5                                          |\n> openssl base64 -d -A                                         |\n> openssl aes-256-cbc -md md5 -d -k password
Encrypt plaintext via AES-CBC from a given password

# 使用 sha1 摘要
$ echo "Encrypt plaintext via AES-CBC from a given password" |\n> python3 aes.py sha1                                          |\n> openssl base64 -d -A                                         |\n> openssl aes-256-cbc -md sha1 -d -k password
Encrypt plaintext via AES-CBC from a given password
```

## 3.6.20 使用密码通过 AES CBC 模式解密（使用 pycrypto）

```
from __future__ import print_function, unicode_literals

import struct
import base64
import sys

from hashlib import md5, sha1
from Crypto.Cipher import AES
from Crypto.Random.random import getrandbits

# AES CBC 要求块对齐在 16 字节边界上。
BS = 16

unpad = lambda s : s[0:-s[-1]]

def EVP_ByteToKey(pwd, md, salt, key_len, iv_len):
    buf = md(pwd + salt).digest()
    d = buf
    while len(buf) < (iv_len + key_len):
        d = md(d + pwd + salt).digest()
        buf += d
    return buf[:key_len], buf[key_len:key_len + iv_len]

def aes_decrypt(pwd, ciphertext, md):
    ciphertext = base64.b64decode(ciphertext)

    # 检查魔术数字
    if ciphertext[:8] != b'Salted__':
        raise Exception("bad magic number")

    # 获取盐值
    salt = ciphertext[8:16]

    # 获取密钥和初始化向量
    key, iv = EVP_ByteToKey(pwd, md, salt, 32, 16)

    # 解密
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext[16:])).strip()

if len(sys.argv) != 2: raise Exception("usage: CMD [md]")

md = globals()[sys.argv[1]]

ciphertext = sys.stdin.read().encode('utf-8')
pwd = b"password"

print(aes_decrypt(pwd, ciphertext, md).decode('utf-8'))
```

输出：

```
# 使用 md5 摘要
$ echo "Decrypt ciphertext via AES-CBC from a given password" |\n> openssl aes-256-cbc -e -md md5 -salt -A -k password       |\n> openssl base64 -e -A                                   |\n> python3 aes.py md5
Decrypt ciphertext via AES-CBC from a given password

# 使用 sha1 摘要
$ echo "Decrypt ciphertext via AES-CBC from a given password" |\n> openssl aes-256-cbc -e -md sha1 -salt -A -k password      |\n> openssl base64 -e -A                                  |\n> python3 aes.py sha1
Decrypt ciphertext via AES-CBC from a given password
```

## 3.6.21 通过 cryptography 库实现临时 Diffie Hellman 密钥交换

```python
>>> from cryptography.hazmat.backends import default_backend
>>> from cryptography.hazmat.primitives.asymmetric import dh
>>> params = dh.generate_parameters(2, 512, default_backend())
>>> a_key = params.generate_private_key()  # alice 的私钥
>>> b_key = params.generate_private_key()  # bob 的私钥
>>> a_pub_key = a_key.public_key()
>>> b_pub_key = b_key.public_key()
>>> a_shared_key = a_key.exchange(b_pub_key)
>>> b_shared_key = b_key.exchange(a_pub_key)
>>> a_shared_key == b_shared_key
True
```

## 3.6.22 通过 cryptography 库手动计算 DH 共享密钥

```python
>>> from cryptography.hazmat.backends import default_backend
>>> from cryptography.hazmat.primitives.asymmetric import dh
>>> from cryptography.utils import int_from_bytes
>>> a_key = params.generate_private_key()  # alice 的私钥
>>> b_key = params.generate_private_key()  # bob 的私钥
>>> a_pub_key = a_key.public_key()
>>> b_pub_key = b_key.public_key()
>>> shared_key = int_from_bytes(a_key.exchange(b_pub_key), 'big')
>>> shared_key_manual = pow(a_pub_key.public_numbers().y,
                            b_key.private_numbers().x,
                            params.parameter_numbers().p)
>>> shared_key == shared_key_manual
True
```

## 3.6.23 根据 (p, g, pubkey) 计算 DH 共享密钥

```
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.utils import int_from_bytes

backend = default_backend()

p = int("11859949538425015739337467917303613431031019140213666"
        "12902540730065402658508634532306628480096346320424639"
        "0256567934582260424238844463330887962689642467123")

g = 2

y = int("32155788395534640648739966373159697798396966919821525"
        "72238852825117261342483718574508213761865276905503199"
        "969908098203345481366464874759377454476688391248")

x = int("409364065449673443397833358558926598469347813468816037"
        "268451847116982490733450463194921405069999008617231539"
        "7147035896687401350877308899732826446337707128")

params = dh.DHParameterNumbers(p, g)
public = dh.DHPublicNumbers(y, params)
private = dh.DHPrivateNumbers(x, public)

key = private.private_key(backend)
shared_key = key.exchange(public.public_key(backend))

# 检查共享密钥
shared_key = int_from_bytes(shared_key, 'big')
shared_key_manual = pow(y, x, p)    # y^x mod p

assert shared_key == shared_key_manual
```

## 3.7 安全 Shell

- 登录 ssh

## 3.7.1 登录 ssh

```
# ssh me@localhost "uname"

from paramiko.client import SSHClient
with SSHClient() as ssh:
    ssh.connect("localhost", username="me", password="pwd")
    stdin, stdout, stderr = ssh.exec_command("uname")
    print(stdout.read())
```

```
# ssh -p 2222 me@localhost "uname"

from paramiko.client import SSHClient
with SSHClient() as ssh:
    ssh.connect("localhost", 2222, username="me", password="pwd")
    stdin, stdout, stderr = ssh.exec_command("uname")
    print(stdout.read())
```

```
# 忽略已知主机
# ssh -o StrictHostKeyChecking=no \n#     -o UserKnownHostsFile=/dev/null \n#     me@localhost "uname"

import paramiko
from paramiko.client import SSHClient
with SSHClient() as ssh:
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("localhost", username="me", password="pwd")
    stdin, stdout, stderr = ssh.exec_command("uname")
    print(stdout.read())
```

```
# ssh-keygen -f key -m pem -t rsa
# ssh-copy-id -i key me@localhost
# ssh -i key me@localhost "uname"

with SSHClient() as ssh:
    ssh.connect('localhost', username="me", key_filename="key")
    stdin, stdout, stderr = ssh.exec_command("uname")
    print(stdout.read())
```

```
# ssh-keygen -m pem -f key -t rsa -P passphrase
# eval $(ssh-agent)
# ssh-add key
# ssh -i key me@localhost
```

# 3.8 Boto3

# 3.9 测试

- *测试*
- 一个简单的 Python 单元测试
- Python 单元测试的 setUp 与 tearDown 层级结构
- 不同模块中的 setUp 与 tearDown 层级结构
- 通过 unittest.TextTestRunner 运行测试
- 测试抛出异常
- 向 TestCase 传递参数
- 将多个测试用例组合成一个测试套件
- 从不同的 TestCase 中组合多个测试
- 在 TestCase 中跳过某些测试
- 单体测试
- 跨模块变量用于测试文件
- 当测试被跳过时，跳过 setUp 与 tearDown
- 重用旧的测试代码
- 测试你的文档是否正确
- 将 doctest 重用到 unittest 中
- 自定义测试报告
- Mock - 使用 @patch 替代原始方法
- unittest.mock.patch 的作用是什么？
- Mock - 替代 open

## 3.9.1 一个简单的 Python 单元测试

```
# python unittests only run the function with prefix "test"
>>> from __future__ import print_function
>>> import unittest
>>> class TestFoo(unittest.TestCase):
...     def test_foo(self):
...         self.assertTrue(True)
...     def fun_not_run(self):
...         print("no run")
```

```
>>> unittest.main()
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
>>> import unittest
>>> class TestFail(unittest.TestCase):
...     def test_false(self):
...             self.assertTrue(False)
...
>>> unittest.main()
F
======================================================================
FAIL: test_false (__main__.TestFail)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "<stdin>", line 3, in test_false
AssertionError: False is not true

----------------------------------------------------------------------
Ran 1 test in 0.000s

FAILED (failures=1)
```

## 3.9.2 Python 单元测试的 setUp 与 tearDown 层级结构

```
from __future__ import print_function

import unittest

def fib(n):
    return 1 if n<=2 else fib(n-1)+fib(n-2)

def setUpModule():
    print("setup module")
def tearDownModule():
    print("teardown module")

class TestFib(unittest.TestCase):

    def setUp(self):
        print("setUp")
        self.n = 10
    def tearDown(self):
        print("tearDown")
        del self.n
    @classmethod
    def setUpClass(cls):
        print("setUpClass")
```

```
python
@classmethod
def tearDownClass(cls):
    print("tearDownClass")
def test_fib_assert_equal(self):
    self.assertEqual(fib(self.n), 55)
def test_fib_assert_true(self):
    self.assertTrue(fib(self.n) == 55)

if __name__ == "__main__":
    unittest.main()

```

输出：

```

$ python test.py
setup module
setUpClass
setUp
tearDown
.setUp
tearDown
.tearDownClass
teardown module
---------------------------------------------------------------------
Ran 2 tests in 0.000s
OK

```

## 3.9.3 不同模块中的 setUp 与 tearDown 层级结构

```
python
# test_module.py
from __future__ import print_function

import unittest

class TestFoo(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("foo setUpClass")
    @classmethod
    def tearDownClass(self):
        print("foo tearDownClass")
    def setUp(self):
        print("foo setUp")
    def tearDown(self):
        print("foo tearDown")
    def test_foo(self):
        self.assertTrue(True)

class TestBar(unittest.TestCase):
```

```
python
def setUp(self):
    print("bar setUp")
def tearDown(self):
    print("bar tearDown")
def test_bar(self):
    self.assertTrue(True)

# test.py
from __future__ import print_function

from test_module import TestFoo
from test_module import TestBar
import test_module
import unittest

def setUpModule():
    print("setUpModule")

def tearDownModule():
    print("tearDownModule")

if __name__ == "__main__":
    test_module.setUpModule = setUpModule
    test_module.tearDownModule = tearDownModule
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestFoo)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestBar)
    suite = unittest.TestSuite([suite1, suite2])
    unittest.TextTestRunner().run(suite)

```

输出：

```
bash
$ python test.py
setUpModule
foo setUpClass
foo setUp
foo tearDown
.foo tearDownClass
bar setUp
bar tearDown
.tearDownModule
----------------------------------------------------------------------
Ran 2 tests in 0.000s

OK

```

## 3.9.4 通过 unittest.TextTestRunner 运行测试

```
python
>>> import unittest
>>> class TestFoo(unittest.TestCase):
...     def test_foo(self):
...         self.assertTrue(True)
...     def test_bar(self):
...         self.assertFalse(False)
...
>>> suite = unittest.TestLoader().loadTestsFromTestCase(TestFoo)
>>> unittest.TextTestRunner(verbosity=2).run(suite)
test_bar (__main__.TestFoo) ... ok
test_foo (__main__.TestFoo) ... ok

-----------------------------------------------------------
Ran 2 tests in 0.000s

OK
```

## 3.9.5 测试抛出异常

```
python
>>> import unittest

>>> class TestRaiseException(unittest.TestCase):
...     def test_raise_except(self):
...         with self.assertRaises(SystemError):
...             raise SystemError
>>> suite_loader = unittest.TestLoader()
>>> suite = suite_loader.loadTestsFromTestCase(TestRaiseException)
>>> unittest.TextTestRunner().run(suite)
.
-----------------------------------------------------------
Ran 1 test in 0.000s

OK

>>> class TestRaiseFail(unittest.TestCase):
...     def test_raise_fail(self):
...         with self.assertRaises(SystemError):
...             pass
>>> suite = unittest.TestLoader().loadTestsFromTestCase(TestRaiseFail)
>>> unittest.TextTestRunner(verbosity=2).run(suite)
test_raise_fail (__main__.TestRaiseFail) ... FAIL

======================================================================
FAIL: test_raise_fail (__main__.TestRaiseFail)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "<stdin>", line 4, in test_raise_fail
AssertionError: SystemError not raised

----------------------------------------------------------------------
```

(接上一页)
Ran 1 test in 0.000s

FAILED (failures=1)

## 3.9.6 向 TestCase 传递参数

```
python
>>> from __future__ import print_function
>>> import unittest
>>> class TestArg(unittest.TestCase):
...     def __init__(self, testname, arg):
...         super(TestArg, self).__init__(testname)
...         self._arg = arg
...     def setUp(self):
...         print("setUp:", self._arg)
...     def test_arg(self):
...         print("test_arg:", self._arg)
...         self.assertTrue(True)
...
>>> suite = unittest.TestSuite()
>>> suite.addTest(TestArg('test_arg', 'foo'))
>>> unittest.TextTestRunner(verbosity=2).run(suite)
test_arg (__main__.TestArg) ... setUp: foo
test_arg: foo
ok

----------------------------------------------------------------------
Ran 1 test in 0.000s

OK

```

## 3.9.7 将多个测试用例组合成一个测试套件

```
python
>>> import unittest
>>> class TestFooBar(unittest.TestCase):
...     def test_foo(self):
...         self.assertTrue(True)
...     def test_bar(self):
...         self.assertTrue(True)
...
>>> class TestHelloWorld(unittest.TestCase):
...     def test_hello(self):
...         self.assertEqual("Hello", "Hello")
...     def test_world(self):
...         self.assertEqual("World", "World")
...
>>> suite_loader = unittest.TestLoader()
>>> suite1 = suite_loader.loadTestsFromTestCase(TestFooBar)
>>> suite2 = suite_loader.loadTestsFromTestCase(TestHelloWorld)
>>> suite = unittest.TestSuite([suite1, suite2])
```

```
>>> unittest.TextTestRunner(verbosity=2).run(suite)
test_bar (__main__.TestFooBar) ... ok
test_foo (__main__.TestFooBar) ... ok
test_hello (__main__.TestHelloWorld) ... ok
test_world (__main__.TestHelloWorld) ... ok
-------------------------------------------------------------------------------
Ran 4 tests in 0.000s

OK
```

## 3.9.8 从不同的 TestCase 中组合多个测试

```
>>> import unittest
>>> class TestFoo(unittest.TestCase):
...     def test_foo(self):
...         assert "foo" == "foo"
...
>>> class TestBar(unittest.TestCase):
...     def test_bar(self):
...         assert "bar" == "bar"
...
>>> suite = unittest.TestSuite()
>>> suite.addTest(TestFoo('test_foo'))
>>> suite.addTest(TestBar('test_bar'))
>>> unittest.TextTestRunner(verbosity=2).run(suite)
test_foo (__main__.TestFoo) ... ok
test_bar (__main__.TestBar) ... ok
-------------------------------------------------------------------------------
Ran 2 tests in 0.001s

OK
```

## 3.9.9 在 TestCase 中跳过某些测试

```
>>> import unittest
>>> RUN_FOO = False
>>> DONT_RUN_BAR = False
>>> class TestSkip(unittest.TestCase):
...     def test_always_run(self):
...         self.assertTrue(True)
...     @unittest.skip("always skip this test")
...     def test_always_skip(self):
...         raise RuntimeError
...     @unittest.skipIf(RUN_FOO == False, "demo skipIf")
...     def test_skipif(self):
...         raise RuntimeError
...     @unittest.skipUnless(DONT_RUN_BAR == True, "demo skipUnless")
```...    def test_skipunless(self):
...        raise RuntimeError
...
>>> suite = unittest.TestLoader().loadTestsFromTestCase(TestSkip)
>>> unittest.TextTestRunner(verbosity=2).run(suite)
test_always_run (__main__.TestSkip) ... ok
test_always_skip (__main__.TestSkip) ... skipped 'always skip this test'
test_skipif (__main__.TestSkip) ... skipped 'demo skipIf'
test_skipunless (__main__.TestSkip) ... skipped 'demo skipUnless'

----------------------------------------------------------------------
Ran 4 tests in 0.000s

OK (skipped=3)

## 3.9.10 单体测试

```
>>> from __future__ import print_function
>>> import unittest
>>> class Monolithic(unittest.TestCase):
...    def step1(self):
...        print('step1')
...    def step2(self):
...        print('step2')
...    def step3(self):
...        print('step3')
...    def _steps(self):
...        for attr in sorted(dir(self)):
...            if not attr.startswith('step'):
...                continue
...            yield attr
...    def test_foo(self):
...        for _s in self._steps():
...            try:
...                getattr(self, _s)()
...            except Exception as e:
...                self.fail('{} failed({})'.format(attr, e))
...
>>> suite = unittest.TestLoader().loadTestsFromTestCase(Monolithic)
>>> unittest.TextTestRunner().run(suite)
step1
step2
step3
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
<unittest.runner.TextTestResult run=1 errors=0 failures=0>
```

## 3.9.11 跨模块变量到测试文件

test_foo.py

```
from __future__ import print_function

import unittest

print(conf)

class TestFoo(unittest.TestCase):
    def test_foo(self):
        print(conf)

    @unittest.skipIf(conf.isskip==True, "skip test")
    def test_skip(self):
        raise RuntimeError
```

test_bar.py

```
from __future__ import print_function

import unittest
import __builtin__

if __name__ == "__main__":
    conf = type('TestConf', (object,), {})
    conf.isskip = True

    # make a cross-module variable
    __builtin__.conf = conf
    module = __import__('test_foo')
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(module.TestFoo)
    unittest.TextTestRunner(verbosity=2).run(suite)
```

输出：

```
$ python test_bar.py
<class '__main__.TestConf'>
test_foo (test_foo.TestFoo) ... <class '__main__.TestConf'>
ok
test_skip (test_foo.TestFoo) ... skipped 'skip test'

----------------------------------------------------------------------
Ran 2 tests in 0.000s

OK (skipped=1)
```

## 3.9.12 当测试被跳过时跳过 setup 和 teardown

```
>>> from __future__ import print_function
>>> import unittest
>>> class TestSkip(unittest.TestCase):
...     def setUp(self):
...         print("setUp")
...     def tearDown(self):
...         print("tearDown")
...     @unittest.skip("skip this test")
...     def test_skip(self):
...         raise RuntimeError
...     def test_not_skip(self):
...         self.assertTrue(True)
...
>>> suite = unittest.TestLoader().loadTestsFromTestCase(TestSkip)
>>> unittest.TextTestRunner(verbosity=2).run(suite)
test_not_skip (__main__.TestSkip) ... setUp
tearDown
ok
test_skip (__main__.TestSkip) ... skipped 'skip this test'
----------------------------------------------------------------------
Ran 2 tests in 0.000s

OK (skipped=1)
```

## 3.9.13 重用旧测试代码

```
>>> from __future__ import print_function
>>> import unittest
>>> def old_func_test():
...     assert "Hello" == "Hello"
...
>>> def old_func_setup():
...     print("setup")
...
>>> def old_func_teardown():
...     print("teardown")
...
>>> testcase = unittest.FunctionTestCase(old_func_test,
...                                      setUp=old_func_setup,
...                                      tearDown=old_func_teardown)
>>> suite = unittest.TestSuite([testcase])
>>> unittest.TextTestRunner().run(suite)
setup
teardown
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
<unittest.runner.TextTestResult run=1 errors=0 failures=0>
```

## 3.9.14 测试你的文档是否正确

```
"""
This is an example of doctest

>>> fib(10)
55
"""

def fib(n):
    """ This function calculate fib number.

    Example:

        >>> fib(10)
        55
        >>> fib(-1)
        Traceback (most recent call last):
        ...
        ValueError
    """
    if n < 0:
        raise ValueError('')
    return 1 if n<=2 else fib(n-1) + fib(n-2)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
```

输出：

```
$ python demo_doctest.py -v
Trying:
    fib(10)
Expecting:
    55
ok
Trying:
    fib(10)
Expecting:
    55
ok
Trying:
    fib(-1)
Expecting:
    Traceback (most recent call last):
    ...

ValueError
ok
2 items passed all tests:
1 tests in __main__
2 tests in __main__.fib
3 tests in 2 items.
3 passed and 0 failed.
Test passed.
```

### 3.9.15 将 doctest 重用到 unittest

```
python
import unittest
import doctest

"""
This is an example of doctest

>>> fib(10)
55
"""

def fib(n):
    """ This function calculate fib number.

    Example:

        >>> fib(10)
        55
        >>> fib(-1)
        Traceback (most recent call last):
            ...
        ValueError
    """
    if n < 0:
        raise ValueError('')
    return 1 if n<=2 else fib(n-1) + fib(n-2)

if __name__ == "__main__":
    finder = doctest.DocTestFinder()
    suite = doctest.DocTestSuite(test_finder=finder)
    unittest.TextTestRunner(verbosity=2).run(suite)
```

```

fib (__main__)
Doctest: __main__.fib ... ok

----------------------------------------------------------------------
Ran 1 test in 0.023s

OK
```

## 3.9.16 自定义测试报告

```
from unittest import (
    TestCase,
    TestLoader,
    TextTestResult,
    TextTestRunner)

from pprint import pprint
import unittest
import os

OK = 'ok'
FAIL = 'fail'
ERROR = 'error'
SKIP = 'skip'

class JsonTestResult(TextTestResult):

    def __init__(self, stream, descriptions, verbosity):
        super_class = super(JsonTestResult, self)
        super_class.__init__(stream, descriptions, verbosity)

        # TextTestResult has no successes attr
        self.successes = []

    def addSuccess(self, test):
        # addSuccess do nothing, so we need to overwrite it.
        super(JsonTestResult, self).addSuccess(test)
        self.successes.append(test)

    def json_append(self, test, result, out):
        suite = test.__class__.__name__
        if suite not in out:
            out[suite] = {OK: [], FAIL: [], ERROR:[], SKIP: []}
        if result is OK:
            out[suite][OK].append(test._testMethodName)
        elif result is FAIL:
            out[suite][FAIL].append(test._testMethodName)
        elif result is ERROR:
            out[suite][ERROR].append(test._testMethodName)
        elif result is SKIP:
            out[suite][SKIP].append(test._testMethodName)
        else:
            raise KeyError("No such result: {}".format(result))
        return out

    def jsonify(self):
        json_out = dict()
        for t in self.successes:
            json_out = self.json_append(t, OK, json_out)

        for t, _ in self.failures:
```

```
python
json_out = self.json_append(t, FAIL, json_out)

for t, _ in self.errors:
    json_out = self.json_append(t, ERROR, json_out)

for t, _ in self.skipped:
    json_out = self.json_append(t, SKIP, json_out)

return json_out

class TestSimple(TestCase):
    def test_ok_1(self):
        foo = True
        self.assertTrue(foo)

    def test_ok_2(self):
        bar = True
        self.assertTrue(bar)

    def test_fail(self):
        baz = False
        self.assertTrue(baz)

    def test_raise(self):
        raise RuntimeError

    @unittest.skip("Test skip")
    def test_skip(self):
        raise NotImplementedError

if __name__ == '__main__':
    # redirector default output of unittest to /dev/null
    with open(os.devnull, 'w') as null_stream:
        # new a runner and overwrite resultclass of runner
        runner = TextTestRunner(stream=null_stream)
        runner.resultclass = JsonTestResult

        # create a testsuite
        suite = TestLoader().loadTestsFromTestCase(TestSimple)

        # run the testsuite
        result = runner.run(suite)

        # print json output
        pprint(result.jsonify())
```

输出：

```

$ python test.py
{'TestSimple': {'error': ['test_raise'],
               'fail': ['test_fail'],
```

3.9. 测试

299## 3.9.17 模拟 - 使用 @patch 替换原始方法

```python
# python-3.3 或更高版本

>>> from unittest.mock import patch
>>> import os
>>> def fake_remove(path, *a, **k):
...     print("remove done")
...
>>> @patch('os.remove', fake_remove)
... def test():
...     try:
...         os.remove('%$!?&*') # 伪造的 os.remove
...     except OSError as e:
...         print(e)
...     else:
...         print('test success')
...
>>> test()
remove done
test success
```

**注意：** 如果没有模拟，上述测试将总是失败。

```python
>>> import os
>>> def test():
...     try:
...         os.remove('%$!?&*')
...     except OSError as e:
...         print(e)
...     else:
...         print('test success')
...
>>> test()
[Errno 2] No such file or directory: '%$!?&*'
```

## 3.9.18 unittest.mock.patch 做了什么？

```python
from unittest.mock import patch
import os

PATH = '$@!%?&'

def fake_remove(path):
    print("Fake remove")

class SimplePatch:

    def __init__(self, target, new):
        self._target = target
        self._new = new

    def get_target(self, target):
        target, attr = target.rsplit('.', 1)
        getter = __import__(target)
        return getter, attr

    def __enter__(self):
        orig, attr = self.get_target(self._target)
        self.orig, self.attr = orig, attr
        self.orig_attr = getattr(orig, attr)
        setattr(orig, attr, self._new)
        return self._new

    def __exit__(self, *exc_info):
        setattr(self.orig, self.attr, self.orig_attr)
        del self.orig_attr

print('---> 在 unittest.mock.patch 作用域内')
with patch('os.remove', fake_remove):
    os.remove(PATH)

print('---> 在简易补丁作用域内')
with SimplePatch('os.remove', fake_remove):
    os.remove(PATH)

print('---> 在补丁作用域外')
try:
    os.remove(PATH)
except OSError as e:
    print(e)
```

输出：

```
$ python3 simple_patch.py
---> 在 unittest.mock.patch 作用域内
Fake remove
---> 在简易补丁作用域内
Fake remove
---> 在补丁作用域外
[Errno 2] No such file or directory: '$@!%?&'
```

## 3.9.19 模拟 - 替换 open

```python
>>> import urllib
>>> from unittest.mock import patch, mock_open
>>> def send_req(url):
...     with urllib.request.urlopen(url) as f:
...         if f.status == 200:
...             return f.read()
...         raise urllib.error.URLError
...
>>> fake_html = b'<html><h1>Mock Content</h1></html>'
>>> mock_urlopen = mock_open(read_data=fake_html)
>>> ret = mock_urlopen.return_value
>>> ret.status = 200
>>> @patch('urllib.request.urlopen', mock_urlopen)
... def test_send_req_success():
...     try:
...         ret = send_req('http://www.mockurl.com')
...         assert ret == fake_html
...     except Exception as e:
...         print(e)
...     else:
...         print('test send_req success')
...
>>> test_send_req_success()
test send_req success
>>> ret = mock_urlopen.return_value
>>> ret.status = 404
>>> @patch('urllib.request.urlopen', mock_urlopen)
... def test_send_req_fail():
...     try:
...         ret = send_req('http://www.mockurl.com')
...         assert ret == fake_html
...     except Exception as e:
...         print('test fail success')
...
>>> test_send_req_fail()
test fail success
```

## 3.10 C 扩展

偶尔，Python 开发者不可避免地需要编写 C 扩展。例如，将 C 库或新的系统调用移植到 Python 需要通过 C 扩展来实现新的对象类型。为了简要了解 C 扩展的工作原理，本速查表主要关注编写 Python C 扩展。

请注意，C 扩展接口是特定于官方 CPython 的。扩展模块很可能无法在其他 Python 实现（如 PyPy）上工作。即使是官方 CPython，Python C API 也可能与不同版本（例如 Python2 和 Python3）不兼容。因此，如果考虑让扩展模块在其他 Python 解释器上运行，最好使用 ctypes 模块或 cffi。

### 目录

- C 扩展
    - 简单的 setup.py
    - 自定义 CFLAGS
    - 文档字符串
    - 简单的 C 扩展
    - 释放 GIL
    - 获取 GIL
    - 获取引用计数
    - 解析参数
    - 调用 Python 函数
    - 抛出异常
    - 自定义异常
    - 迭代列表
    - 迭代字典
    - 简单类
    - 包含成员和方法的简单类
    - 包含 Getter 和 Setter 的简单类
    - 从其他类继承
    - 运行 Python 命令
    - 运行 Python 文件
    - 导入 Python 模块
    - 导入模块的所有内容
    - 访问属性
    - C 扩展的性能
    - ctypes 的性能
    - ctypes 错误处理

## 3.10.1 简单的 setup.py

```python
from distutils.core import setup, Extension

ext = Extension('foo', sources=['foo.c'])
setup(name="Foo", version="1.0", ext_modules=[ext])
```

## 3.10.2 自定义 CFLAGS

```python
import sysconfig
from distutils.core import setup, Extension

cflags = sysconfig.get_config_var("CFLAGS")

extra_compile_args = cflags.split()
extra_compile_args += ["-Wextra"]

ext = Extension(
    "foo", ["foo.c"],
    extra_compile_args=extra_compile_args
)

setup(name="foo", version="1.0", ext_modules=[ext])
```

## 3.10.3 文档字符串

```c
PyDoc_STRVAR(doc_mod, "Module document\n");
PyDoc_STRVAR(doc_foo, "foo() -> None\n\nFoo doc");

static PyMethodDef methods[] = {
    {"foo", (PyCFunction)foo, METH_NOARGS, doc_foo},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    .m_base    = PyModuleDef_HEAD_INIT,
    .m_name    = "Foo",
    .m_doc     = doc_mod,
    .m_size    = -1,
    .m_methods = methods
};
```

## 3.10.4 简单的 C 扩展

foo.c

```c
#include <Python.h>

PyDoc_STRVAR(doc_mod, "Module document\n");
PyDoc_STRVAR(doc_foo, "foo() -> None\n\nFoo doc");

static PyObject* foo(PyObject* self)
{
    PyObject* s = PyUnicode_FromString("foo");
    PyObject_Print(s, stdout, 0);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"foo", (PyCFunction)foo, METH_NOARGS, doc_foo},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "Foo", doc_mod, -1, methods
};

PyMODINIT_FUNC PyInit_foo(void)
{
    return PyModule_Create(&module);
}
```

输出：

```
$ python setup.py -q build
$ python setup.py -q install
$ python -c "import foo; foo.foo()"
'foo'
```

## 3.10.5 释放 GIL

```c
#include <Python.h>

static PyObject* foo(PyObject* self)
{
    Py_BEGIN_ALLOW_THREADS
    sleep(3);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"foo", (PyCFunction)foo, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "Foo", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_foo(void)
{
    return PyModule_Create(&module);
}
```

输出：

```
$ python setup.py -q build
$ python setup.py -q install
$ python -c "
> import threading
> import foo
> from datetime import datetime
> def f(n):
>     now = datetime.now()
>     print(f'{now}: thread {n}')
>     foo.foo()
> ts = [threading.Thread(target=f, args=(n,)) for n in range(3)]
> [t.start() for t in ts]
> [t.join() for t in ts]"
2018-11-04 20:15:34.860454: thread 0
2018-11-04 20:15:34.860592: thread 1
2018-11-04 20:15:34.860705: thread 2
```

在 C 扩展中，阻塞式 I/O 操作应插入到由 `Py_BEGIN_ALLOW_THREADS` 和 `Py_END_ALLOW_THREADS` 包裹的代码块中，以临时释放 GIL；否则，阻塞式 I/O 操作将不得不等待之前的（持有 GIL 的）操作完成。例如：

```c
#include <Python.h>

static PyObject* foo(PyObject* self)
{
    sleep(3);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"foo", (PyCFunction)foo, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "Foo", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_foo(void)
{
    return PyModule_Create(&module);
}
```

## 3.10.6 获取 GIL

```c
#include <pthread.h>
#include <Python.h>

typedef struct {
    PyObject *sec;
    PyObject *py_callback;
} foo_args;

void *
foo_thread(void *args)
{
    long n = -1;
    PyObject *rv = NULL, *sec = NULL, *py_callback = NULL;
    foo_args *a = NULL;

    if (!args)
        return NULL;

    a = (foo_args *)args;
    sec = a->sec;
    py_callback = a->py_callback;

    n = PyLong_AsLong(sec);
    if ((n == -1) && PyErr_Occurred()) {
        return NULL;
    }

    sleep(n);  // 耗时任务

    // 获取 GIL
    PyGILState_STATE state = PyGILState_Ensure();
    rv = PyObject_CallFunction(py_callback, "s", "Awesome Python!");
    // 释放 GIL
    PyGILState_Release(state);
    Py_XDECREF(rv);
    return NULL;
}

static PyObject *
foo(PyObject *self, PyObject *args)
{
    long i = 0, n = 0;
    pthread_t *arr = NULL;
    PyObject *py_callback = NULL;
    PyObject *sec = NULL, *num = NULL;
    PyObject *rv = NULL;
    foo_args a = {};

    if (!PyArg_ParseTuple(args, "OOO:callback", &num, &sec, &py_callback))
        return NULL;

    // 允许释放 GIL
    Py_BEGIN_ALLOW_THREADS

    if (!PyLong_Check(sec) || !PyLong_Check(num)) {
        PyErr_SetString(PyExc_TypeError, "should be int");
        goto error;
    }

    if (!PyCallable_Check(py_callback)) {
        PyErr_SetString(PyExc_TypeError, "should be callable");
        goto error;
    }

    n = PyLong_AsLong(num);
    if (n == -1 && PyErr_Occurred())
        goto error;

    arr = (pthread_t *)PyMem_RawCalloc(n, sizeof(pthread_t));
    if (!arr)
        goto error;

    a.sec = sec;
    a.py_callback = py_callback;
    for (i = 0; i < n; i++) {
        if (pthread_create(&arr[i], NULL, foo_thread, &a)) {
            PyErr_SetString(PyExc_TypeError, "create a thread failed");
            goto error;
        }
    }

    for (i = 0; i < n; i++) {
        if (pthread_join(arr[i], NULL)) {
            PyErr_SetString(PyExc_TypeError, "thread join failed");
            goto error;
        }
    }
    Py_XINCREF(Py_None);
    rv = Py_None;
error:
    PyMem_RawFree(arr);
    Py_XDECREF(sec);
    Py_XDECREF(num);
    Py_XDECREF(py_callback);
    // 恢复 GIL
    Py_END_ALLOW_THREADS
    return rv;
}

static PyMethodDef methods[] = {
    {"foo", (PyCFunction)foo, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "foo", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_foo(void)
{
    return PyModule_Create(&module);
}
```

输出：

```bash
$ python setup.py -q build
$ python setup.py -q install
$ python -q
>>> import foo
>>> from datetime import datetime
>>> def cb(s):
...     now = datetime.now()
...     print(f'{now}: {s}')
...
>>> foo.foo(3, 1, cb)
2018-11-05 09:33:50.642543: Awesome Python!
2018-11-05 09:33:50.642634: Awesome Python!
2018-11-05 09:33:50.642672: Awesome Python!
```

如果线程是从 C/C++ 创建的，这些线程不会持有 GIL。在不获取 GIL 的情况下，解释器无法安全地访问 Python 函数。例如：

```c
void *
foo_thread(void *args)
{
    ...
    // 未获取 GIL
    rv = PyObject_CallFunction(py_callback, "s", "Awesome Python!");
    Py_XDECREF(rv);
    return NULL;
}
```

输出：

```bash
>>> import foo
>>> from datetime import datetime
>>> def cb(s):
...     now = datetime.now()
...     print(f"{now}: {s}")
...
>>> foo.foo(1, 1, cb)
[2]    8590 segmentation fault  python -q
```

> **警告：** 为了安全地调用 Python 函数，我们可以简单地在 C 扩展代码中将 **Python 函数** 调用包裹在 `PyGILState_Ensure` 和 `PyGILState_Release` 之间。

```c
PyGILState_STATE state = PyGILState_Ensure();
// 执行 Python 操作
result = PyObject_CallFunction(callback)
// 错误处理
PyGILState_Release(state);
```

## 3.10.7 获取引用计数

```c
#include <Python.h>

static PyObject *
getrefcount(PyObject *self, PyObject *a)
{
    return PyLong_FromSsize_t(Py_REFCNT(a));
}

static PyMethodDef methods[] = {
    {"getrefcount", (PyCFunction)getrefcount, METH_O, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "foo", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_foo(void)
{
    return PyModule_Create(&module);
}
```

输出：

```bash
$ python setup.py -q build
$ python setup.py -q install
$ python -q
>>> import sys
>>> import foo
>>> l = [1, 2, 3]
>>> sys.getrefcount(l[0])
104
>>> foo.getrefcount(l[0])
104
>>> i = l[0]
>>> sys.getrefcount(l[0])
105
>>> foo.getrefcount(l[0])
105
```

## 3.10.8 解析参数

```c
#include <Python.h>

static PyObject *
foo(PyObject *self)
{
    Py_RETURN_NONE;
}

static PyObject *
bar(PyObject *self, PyObject *arg)
{
    return Py_BuildValue("O", arg);
}

static PyObject *
baz(PyObject *self, PyObject *args)
{
    PyObject *x = NULL, *y = NULL;
    if (!PyArg_ParseTuple(args, "OO", &x, &y)) {
        return NULL;
    }
    return Py_BuildValue("OO", x, y);
}

static PyObject *
qux(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *keywords[] = {"x", "y", NULL};
    PyObject *x = NULL, *y = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "O|O", keywords,
                                     &x, &y))
    {
        return NULL;
    }
    if (!y) {
        y = Py_None;
    }
    return Py_BuildValue("OO", x, y);
}

static PyMethodDef methods[] = {
    {"foo", (PyCFunction)foo, METH_NOARGS, NULL},
    {"bar", (PyCFunction)bar, METH_O, NULL},
    {"baz", (PyCFunction)baz, METH_VARARGS, NULL},
    {"qux", (PyCFunction)qux, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "foo", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_foo(void)
{
    return PyModule_Create(&module);
}
```

输出：

```bash
$ python setup.py -q build
$ python setup.py -q install
$ python -q
>>> import foo
>>> foo.foo()
>>> foo.bar(3.7)
3.7
>>> foo.baz(3, 7)
(3, 7)
>>> foo.qux(3, y=7)
(3, 7)
>>> foo.qux(x=3, y=7)
(3, 7)
>>> foo.qux(x=3)
(3, None)
```

## 3.10.9 调用 Python 函数

```c
#include <Python.h>

static PyObject *
foo(PyObject *self, PyObject *args)
{
    PyObject *py_callback = NULL;
    PyObject *rv = NULL;

    if (!PyArg_ParseTuple(args, "O:callback", &py_callback))
        return NULL;

    if (!PyCallable_Check(py_callback)) {
        PyErr_SetString(PyExc_TypeError, "should be callable");
        return NULL;
    }

    // 确保我们拥有 GIL
    PyGILState_STATE state = PyGILState_Ensure();
    // 类似于 py_callback("Awesome Python!")
    rv = PyObject_CallFunction(py_callback, "s", "Awesome Python!");
    // 恢复之前的 GIL 状态
    PyGILState_Release(state);
    return rv;
}

static PyMethodDef methods[] = {
    {"foo", (PyCFunction)foo, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "foo", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_foo(void)
{
    return PyModule_Create(&module);
}
```

输出：

```bash
$ python setup.py -q build
$ python setup.py -q install
$ python -c "import foo; foo.foo(print)"
Awesome Python!
```

## 3.10.10 引发异常

```c
#include <Python.h>

PyDoc_STRVAR(doc_mod, "Module document\n");
PyDoc_STRVAR(doc_foo, "foo() -> None\n\nFoo doc");

static PyObject*
foo(PyObject* self)
{
    // 引发 NotImplementedError
    PyErr_SetString(PyExc_NotImplementedError, "Not implemented");
    return NULL;
}

static PyMethodDef methods[] = {
    {"foo", (PyCFunction)foo, METH_NOARGS, doc_foo},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "Foo", doc_mod, -1, methods
};

PyMODINIT_FUNC PyInit_foo(void)
{
    return PyModule_Create(&module);
}
```

输出：

```bash
$ python setup.py -q build
$ python setup.py -q install
$ python -c "import foo; foo.foo(print)"
$ python -c "import foo; foo.foo()"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
NotImplementedError: Not implemented
```

## 3.10.11 自定义异常

```c
#include <stdio.h>
#include <Python.h>

static PyObject *FooError;

PyDoc_STRVAR(doc_foo, "foo() -> void\n\n"
    "等同于以下示例:\n\n"
    "def foo():\n"
    "    raise FooError(\"在 C 中引发异常\")"
);
```

## 3.10.12 遍历列表

```
static PyObject *
foo(PyObject *self __attribute__((unused)))
{
    PyErr_SetString(FooError, "Raise exception in C");
    return NULL;
}

static PyMethodDef methods[] = {
    {"foo", (PyCFunction)foo, METH_NOARGS, doc_foo},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "foo", "doc", -1, methods
};

PyMODINIT_FUNC PyInit_foo(void)
{
    PyObject *m = NULL;
    m = PyModule_Create(&module);
    if (!m) return NULL;

    FooError = PyErr_NewException("foo.FooError", NULL, NULL);
    Py_INCREF(FooError);
    PyModule_AddObject(m, "FooError", FooError);
    return m;
}
```

```
$ python setup.py -q build
$ python setup.py -q install
$ python -c "import foo; foo.foo()"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
foo.FooError: Raise exception in C
```

```
#include <Python.h>

#define PY_PRINTF(o) \n    PyObject_Print(o, stdout, 0); printf("\n");

static PyObject *
iter_list(PyObject *self, PyObject *args)
{
    PyObject *list = NULL, *item = NULL, *iter = NULL;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "O", &list))
        goto error;

    if (!PyList_Check(list))
        goto error;

    // Get iterator
    iter = PyObject_GetIter(list);
    if (!iter)
        goto error;

    // for i in arr: print(i)
    while ((item = PyIter_Next(iter)) != NULL) {
        PY_PRINTF(item);
        Py_XDECREF(item);
    }

    Py_XINCREF(Py_None);
    result = Py_None;
error:
    Py_XDECREF(iter);
    return result;
}

static PyMethodDef methods[] = {
    {"iter_list", (PyCFunction)iter_list, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "foo", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_foo(void)
{
    return PyModule_Create(&module);
}
```

```
$ python setup.py -q build
$ python setup.py -q install
$ python -c "import foo; foo.iter_list([1,2,3])"


1
2
3
```

## 3.10.13 遍历字典

```
#include <Python.h>

#define PY_PRINTF(o) \n    PyObject_Print(o, stdout, 0); printf("\n");

static PyObject *
iter_dict(PyObject *self, PyObject *args)
{
    PyObject *dict = NULL;
    PyObject *key = NULL, *val = NULL;
    PyObject *o = NULL, *result = NULL;
    Py_ssize_t pos = 0;

    if (!PyArg_ParseTuple(args, "O", &dict))
        goto error;

    // for k, v in d.items(): print(f"({k}, {v})")
    while (PyDict_Next(dict, &pos, &key, &val)) {
        o = PyUnicode_FromFormat("(%S, %S)", key, val);
        if (!o) continue;
        PY_PRINTF(o);
        Py_XDECREF(o);
    }

    Py_INCREF(Py_None);
    result = Py_None;
error:
    return result;
}

static PyMethodDef methods[] = {
    {"iter_dict", (PyCFunction)iter_dict, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "foo", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_foo(void)
{
    return PyModule_Create(&module);
}
```

```
$ python setup.py -q build
$ python setup.py -q install
$ python -c "import foo; foo.iter_dict({'k': 'v'})"
'(k, v)'
```

## 3.10.14 简单类

```
#include <Python.h>

typedef struct {
    PyObject_HEAD
} FooObject;

/* class Foo(object): pass */

static PyTypeObject FooType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "foo.Foo",
    .tp_doc = "Foo objects",
    .tp_basicsize = sizeof(FooObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "foo",
    .m_doc = "module foo",
    .m_size = -1
};

PyMODINIT_FUNC
PyInit_foo(void)
{
    PyObject *m = NULL;
    if (PyType_Ready(&FooType) < 0)
        return NULL;
    if ((m = PyModule_Create(&module)) == NULL)
        return NULL;
    Py_INCREF(&FooType);
    PyModule_AddObject(m, "Foo", (PyObject *) &FooType);
    return m;
}
```

```
$ python setup.py -q build
$ python setup.py -q install
$ python -q
>>> import foo
>>> print(type(foo.Foo))
<class 'type'>
>>> o = foo.Foo()
>>> print(type(o))
<class 'foo.Foo'>
>>> class Foo(object): ...
...
>>> print(type(Foo))
<class 'type'>
>>> o = Foo()
>>> print(type(o))
<class '__main__.Foo'>
```

### 3.10.15 带有成员和方法的简单类

```
#include <Python.h>
#include <structmember.h>

/*
 * class Foo:
 *     def __new__(cls, *a, **kw):
 *         foo_obj = object.__new__(cls)
 *         foo_obj.foo = ""
 *         foo_obj.bar = ""
 *         return foo_obj
 *
 *     def __init__(self, foo, bar):
 *         self.foo = foo
 *         self.bar = bar
 *
 *     def fib(self, n):
 *         if n < 2:
 *             return n
 *         return self.fib(n - 1) + self.fib(n - 2)
 */

typedef struct {
    PyObject_HEAD
    PyObject *foo;
    PyObject *bar;
} FooObject;

static void
Foo_dealloc(FooObject *self)
{
    Py_XDECREF(self->foo);
    Py_XDECREF(self->bar);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
Foo_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    int rc = -1;
    FooObject *self = NULL;
    self = (FooObject *) type->tp_alloc(type, 0);
    if (!self) goto error;

    /* allocate attributes */
    self->foo = PyUnicode_FromString("");
    if (self->foo == NULL) goto error;

    self->bar = PyUnicode_FromString("");
    if (self->bar == NULL) goto error;

    rc = 0;
error:
    if (rc < 0) {
        Py_XDECREF(self->foo);
        Py_XDECREF(self->bar);
        Py_XDECREF(self);
    }
    return (PyObject *) self;
}

static int
Foo_init(FooObject *self, PyObject *args, PyObject *kw)
{
    int rc = -1;
    static char *keywords[] = {"foo", "bar", NULL};
    PyObject *foo = NULL, *bar = NULL, *ptr = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw,
                                    "|OO", keywords,
                                    &foo, &bar))
    {
        goto error;
    }

    if (foo) {
        ptr = self->foo;
        Py_INCREF(foo);
        self->foo = foo;
        Py_XDECREF(ptr);
    }

    if (bar) {
        ptr = self->bar;
        Py_INCREF(bar);
        self->bar = bar;
        Py_XDECREF(ptr);
    }
    rc = 0;
error:
    return rc;
}

static unsigned long
fib(unsigned long n)
{
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
}

static PyObject *
Foo_fib(FooObject *self, PyObject *args)
{
    unsigned long n = 0;
    if (!PyArg_ParseTuple(args, "k", &n)) return NULL;
    return PyLong_FromUnsignedLong(fib(n));
}

static PyMemberDef Foo_members[] = {
    {"foo", T_OBJECT_EX, offsetof(FooObject, foo), 0, NULL},
    {"bar", T_OBJECT_EX, offsetof(FooObject, bar), 0, NULL}
};

static PyMethodDef Foo_methods[] = {
    {"fib", (PyFunction)Foo_fib, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject FooType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "foo.Foo",
    .tp_doc = "Foo objects",
    .tp_basicsize = sizeof(FooObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Foo_new,
    .tp_init = (initproc) Foo_init,
    .tp_dealloc = (destructor) Foo_dealloc,
    .tp_members = Foo_members,
    .tp_methods = Foo_methods
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "foo", NULL, -1, NULL
};

PyMODINIT_FUNC
PyInit_foo(void)
{
    PyObject *m = NULL;
    if (PyType_Ready(&FooType) < 0)
        return NULL;
    if ((m = PyModule_Create(&module)) == NULL)
        return NULL;
    Py_INCREF(&FooType);
    PyModule_AddObject(m, "Foo", (PyObject *) &FooType);
    return m;
}
```

```
$ python setup.py -q build
$ python setup.py -q install
$ python -q
>>> import foo
>>> o = foo.Foo('foo', 'bar')
>>> o.foo
'foo'
>>> o.bar
'bar'
>>> o.fib(10)
55
```

### 3.10.16 带有 getter 和 setter 的简化类

```
#include <Python.h>

/*
 * class Foo:
 *     def __new__(cls, *a, **kw):
 *         foo_obj = object.__new__(cls)
 *         foo_obj._foo = ""
 *         return foo_obj
 *
 *     def __init__(self, foo=None):
 *         if foo and isinstance(foo, 'str'):
 *             self._foo = foo
 *
 *     @property
 *     def foo(self):
 *         return self._foo
 *
 *     @foo.setter
 *     def foo(self, value):
 *         if not value or not isinstance(value, str):
 *             raise TypeError("value should be unicode")
 *         self._foo = value
 */

typedef struct {
    PyObject_HEAD
    PyObject *foo;
} FooObject;

static void
Foo_dealloc(FooObject *self)
```

## 3.10.17 从其他类继承

```c
#include <Python.h>
#include <structmember.h>

/*
 * class Foo:
 *     def __new__(cls, *a, **kw):
 *         foo_obj = object.__new__(cls)
 *         foo_obj.foo = ""
 *         return foo_obj
 *
 *     def __init__(self, foo):
 *         self.foo = foo
 *
 *     def fib(self, n):
 *         if n < 2:
 *             return n
 *         return self.fib(n - 1) + self.fib(n - 2)
 */

/* FooObject */

typedef struct {
    PyObject_HEAD
    PyObject *foo;
} FooObject;

static void
Foo_dealloc(FooObject *self)
{
    Py_XDECREF(self->foo);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
Foo_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    int rc = -1;
    FooObject *self = NULL;
    self = (FooObject *) type->tp_alloc(type, 0);

    if (!self) goto error;

    /* allocate attributes */
    self->foo = PyUnicode_FromString("");
    if (self->foo == NULL) goto error;

    rc = 0;
error:
    if (rc < 0) {
        Py_XDECREF(self->foo);
        Py_XDECREF(self);
    }
    return (PyObject *) self;
}

static int
Foo_init(FooObject *self, PyObject *args, PyObject *kw)
{
    int rc = -1;
    static char *keywords[] = {"foo", NULL};
    PyObject *foo = NULL, *ptr = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "|O", keywords, &foo)) {
        goto error;
    }

    if (foo) {
        ptr = self->foo;
        Py_INCREF(foo);
        self->foo = foo;
        Py_XDECREF(ptr);
    }
    rc = 0;
error:
    return rc;
}

static unsigned long
fib(unsigned long n)
{
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
}

static PyObject *
Foo_fib(FooObject *self, PyObject *args)
{
    unsigned long n = 0;
    if (!PyArg_ParseTuple(args, "k", &n)) return NULL;
    return PyLong_FromUnsignedLong(fib(n));
}

static PyMemberDef Foo_members[] = {
    {"foo", T_OBJECT_EX, offsetof(FooObject, foo), 0, NULL}
};

static PyMethodDef Foo_methods[] = {
    {"fib", (PyCFunction)Foo_fib, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject FooType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "foo.Foo",
    .tp_doc = "Foo objects",
    .tp_basicsize = sizeof(FooObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Foo_new,
    .tp_init = (initproc) Foo_init,
    .tp_dealloc = (destructor) Foo_dealloc,
    .tp_members = Foo_members,
    .tp_methods = Foo_methods
};

/*
 * class Bar(Foo):
 *     def __init__(self, bar):
 *         super().__init__(bar)
 *
 *     def gcd(self, a, b):
 *         while b:
 *             a, b = b, a % b
 *         return a
 */

/* BarObject */

typedef struct {
    FooObject super;
} BarObject;

static unsigned long
gcd(unsigned long a, unsigned long b)
{
    unsigned long t = 0;
    while (b) {
        t = b;
        b = a % b;
        a = t;
    }
    return a;
}

static int
Bar_init(FooObject *self, PyObject *args, PyObject *kw)
{
    return FooType.tp_init((PyObject *) self, args, kw);
}

static PyObject *
Bar_gcd(BarObject *self, PyObject *args)
{
    unsigned long a = 0, b = 0;
    if (!PyArg_ParseTuple(args, "kk", &a, &b)) return NULL;
    return PyLong_FromUnsignedLong(gcd(a, b));
}

static PyMethodDef Bar_methods[] = {
    {"gcd", (PyCFunction)Bar_gcd, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject BarType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "foo.Bar",
    .tp_doc = "Bar objects",
    .tp_basicsize = sizeof(BarObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_base = &FooType,
    .tp_init = (initproc) Bar_init,
    .tp_methods = Bar_methods
};

/* Module */

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "foo", NULL, -1, NULL
};

PyMODINIT_FUNC
PyInit_foo(void)
{
    PyObject *m = NULL;
    if (PyType_Ready(&FooType) < 0)
        return NULL;
    if (PyType_Ready(&BarType) < 0)
        return NULL;
    if ((m = PyModule_Create(&module)) == NULL)
        return NULL;

    Py_XINCREF(&FooType);
    Py_XINCREF(&BarType);
    PyModule_AddObject(m, "Foo", (PyObject *) &FooType);
    PyModule_AddObject(m, "Bar", (PyObject *) &BarType);
    return m;
}
```

输出：

```
$ python setup.py -q build
$ python setup.py -q install
$ python -q
>>> import foo
>>> bar = foo.Bar('bar')
>>> bar.foo
'bar'
>>> bar.fib(10)
55
>>> bar.gcd(3, 7)
1
```

## 3.10.18 运行 Python 命令

```c
#include <stdio.h>
#include <Python.h>

int
main(int argc, char *argv[])
{
    int rc = -1;
    Py_Initialize();
    rc = PyRun_SimpleString(argv[1]);
    Py_Finalize();
    return rc;
}
```

输出：

```
$ clang `python3-config --cflags` -c foo.c -o foo.o
$ clang `python3-config --ldflags` foo.o -o foo
$ ./foo "print('Hello Python')"
Hello Python
```

## 3.10.19 运行 Python 文件

```c
#include <stdio.h>
#include <Python.h>

int
main(int argc, char *argv[])
{
    int rc = -1, i = 0;
    wchar_t **argv_copy = NULL;
    const char *filename = NULL;
    FILE *fp = NULL;
    PyCompilerFlags cf = {.cf_flags = 0};

    filename = argv[1];
    fp = fopen(filename, "r");
    if (!fp)
        goto error;

    // copy argv
    argv_copy = PyMem_RawMalloc(sizeof(wchar_t*) * argc);
    if (!argv_copy)
        goto error;

    for (i = 0; i < argc; i++) {
        argv_copy[i] = Py_DecodeLocale(argv[i], NULL);
        if (argv_copy[i]) continue;
        fprintf(stderr, "Unable to decode the argument");
        goto error;
    }

    Py_Initialize();
    Py_SetProgramName(argv_copy[0]);
    PySys_SetArgv(argc, argv_copy);
    rc = PyRun_AnyFileExFlags(fp, filename, 0, &cf);

error:
    if (argv_copy) {
        for (i = 0; i < argc; i++)
            PyMem_RawFree(argv_copy[i]);
        PyMem_RawFree(argv_copy);
    }
    if (fp) fclose(fp);
    Py_Finalize();
    return rc;
}
```

$ clang `python3-config --cflags` -c foo.c -o foo.o
$ clang `python3-config --ldflags` foo.o -o foo
$ echo "import sys; print(sys.argv)" > foo.py
$ ./foo foo.py arg1 arg2 arg3
['./foo', 'foo.py', 'arg1', 'arg2', 'arg3']

## 3.10.20 导入一个 Python 模块

```c
#include <stdio.h>
#include <Python.h>

#define PYOBJECT_CHECK(obj, label) \n    if (!obj) { \n        PyErr_Print(); \n        goto label; \n    }

int
main(int argc, char *argv[])
{
    int rc = -1;
    wchar_t *program = NULL;
    PyObject *json_module = NULL, *json_dict = NULL;
    PyObject *json_dumps = NULL;
    PyObject *dict = NULL;
    PyObject *result = NULL;

    program = Py_DecodeLocale(argv[0], NULL);
    if (!program) {
        fprintf(stderr, "unable to decode the program name");
        goto error;
    }

    Py_SetProgramName(program);
    Py_Initialize();

    // import json
    json_module = PyImport_ImportModule("json");
    PYOBJECT_CHECK(json_module, error);

    // json_dict = json.__dict__
    json_dict = PyModule_GetDict(json_module);
    PYOBJECT_CHECK(json_dict, error);

    // json_dumps = json.__dict__['dumps']
    json_dumps = PyDict_GetItemString(json_dict, "dumps");
    PYOBJECT_CHECK(json_dumps, error);

    // dict = {'foo': 'Foo', 'bar': 123}
    dict = Py_BuildValue("({sssi})", "foo", "Foo", "bar", 123);
    PYOBJECT_CHECK(dict, error);

    // result = json.dumps(dict)
    result = PyObject_CallObject(json_dumps, dict);
    PYOBJECT_CHECK(result, error);
    PyObject_Print(result, stdout, 0);
    printf("\n");
    rc = 0;

error:
    Py_XDECREF(result);
    Py_XDECREF(dict);
    Py_XDECREF(json_dumps);
    Py_XDECREF(json_dict);
    Py_XDECREF(json_module);

    PyMem_RawFree(program);
    Py_Finalize();
    return rc;
}
```

```
$ clang `python3-config --cflags` -c foo.c -o foo.o
$ clang `python3-config --ldflags` foo.o -o foo
$ ./foo
'{"foo": "Foo", "bar": 123}'
```

## 3.10.21 导入一个模块的所有内容

```c
#include <stdio.h>
#include <Python.h>

#define PYOBJECT_CHECK(obj, label) \n    if (!obj) { \n        PyErr_Print(); \n        goto label; \n    }

int
main(int argc, char *argv[])
{
    int rc = -1;
    wchar_t *program = NULL;
    PyObject *main_module = NULL, *main_dict = NULL;
    PyObject *uname = NULL;
    PyObject *sysname = NULL;
    PyObject *result = NULL;

    program = Py_DecodeLocale(argv[0], NULL);
    if (!program) {
        fprintf(stderr, "unable to decode the program name");
        goto error;
    }

    Py_SetProgramName(program);
    Py_Initialize();

    // import __main__
    main_module = PyImport_ImportModule("__main__");
    PYOBJECT_CHECK(main_module, error);

    // main_dict = __main__.__dict__
    main_dict = PyModule_GetDict(main_module);
    PYOBJECT_CHECK(main_dict, error);

    // from os import *
    result = PyRun_String("from os import *",
                          Py_file_input,
                          main_dict,
                          main_dict);
    PYOBJECT_CHECK(result, error);
    Py_XDECREF(result);
    Py_XDECREF(main_dict);

    // uname = __main__.__dict__['uname']
    main_dict = PyModule_GetDict(main_module);
    PYOBJECT_CHECK(main_dict, error);

    // result = uname()
    uname = PyDict_GetItemString(main_dict, "uname");
    PYOBJECT_CHECK(uname, error);
    result = PyObject_CallObject(uname, NULL);
    PYOBJECT_CHECK(result, error);

    // sysname = result.sysname
    sysname = PyObject_GetAttrString(result, "sysname");
    PYOBJECT_CHECK(sysname, error);
    PyObject_Print(sysname, stdout, 0);
    printf("\n");

    rc = 0;
error:
    Py_XDECREF(sysname);
    Py_XDECREF(result);
    Py_XDECREF(uname);
    Py_XDECREF(main_dict);
    Py_XDECREF(main_module);

    PyMem_RawFree(program);
    Py_Finalize();
    return rc;
}
```

```
$ clang `python3-config --cflags` -c foo.c -o foo.o
$ clang `python3-config --ldflags` foo.o -o foo
$ ./foo
'Darwin'
```

## 3.10.22 访问属性

```c
#include <stdio.h>
#include <Python.h>

#define PYOBJECT_CHECK(obj, label) \
    if (!obj) { \
        PyErr_Print(); \
        goto label; \
    }

int
main(int argc, char *argv[])
{
    int rc = -1;
    wchar_t *program = NULL;
    PyObject *json_module = NULL;
    PyObject *json_dumps = NULL;
    PyObject *dict = NULL;
    PyObject *result = NULL;

    program = Py_DecodeLocale(argv[0], NULL);
    if (!program) {
        fprintf(stderr, "unable to decode the program name");
        goto error;
    }

    Py_SetProgramName(program);
    Py_Initialize();

    // import json
    json_module = PyImport_ImportModule("json");
    PYOBJECT_CHECK(json_module, error);

    // json_dumps = json.dumps
    json_dumps = PyObject_GetAttrString(json_module, "dumps");
    PYOBJECT_CHECK(json_dumps, error);

    // dict = {'foo': 'Foo', 'bar': 123}
    dict = Py_BuildValue("{sssi}", "foo", "Foo", "bar", 123);
    PYOBJECT_CHECK(dict, error);

    // result = json.dumps(dict)
    result = PyObject_CallObject(json_dumps, dict);
    PYOBJECT_CHECK(result, error);
    PyObject_Print(result, stdout, 0);
    printf("\n");
    rc = 0;
error:
    Py_XDECREF(result);
    Py_XDECREF(dict);
    Py_XDECREF(json_dumps);
    Py_XDECREF(json_module);

    PyMem_RawFree(program);
    Py_Finalize();
    return rc;
}
```

```
$ clang `python3-config --cflags` -c foo.c -o foo.o
$ clang `python3-config --ldflags` foo.o -o foo
$ ./foo
'{"foo": "Foo", "bar": 123}'
```

## 3.10.23 C 扩展的性能

```c
#include <Python.h>

static unsigned long
fib(unsigned long n)
{
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
}

static PyObject *
fibonacci(PyObject *self, PyObject *args)
{
    unsigned long n = 0;
    if (!PyArg_ParseTuple(args, "k", &n)) return NULL;
    return PyLong_FromUnsignedLong(fib(n));
}

static PyMethodDef methods[] = {
    {"fib", (PyCFunction)fibonacci, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "foo", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_foo(void)
{
    return PyModule_Create(&module);
}
```

与纯 Python 的性能对比

```
>>> from time import time
>>> import foo
>>> def fib(n):
...     if n < 2: return n
...     return fib(n - 1) + fib(n - 2)
...
>>> s = time(); _ = fib(35); e = time(); e - s
4.953313112258911
>>> s = time(); _ = foo.fib(35); e = time(); e - s
0.04628586769104004
```

## 3.10.24 ctypes 的性能

```c
// Compile (Mac)
// -----------
//
//   $ clang -Wall -Werror -shared -fPIC -o libfib.dylib fib.c
//
unsigned int fib(unsigned int n)
{
    if ( n < 2) {
        return n;
    }
    return fib(n-1) + fib(n-2);
}
```

与纯 Python 的性能对比

```
>>> from time import time
>>> from ctypes import CDLL
>>> def fib(n):
...     if n < 2: return n
...     return fib(n - 1) + fib(n - 2)
...
>>> cfib = CDLL("./libfib.dylib").fib
>>> s = time(); _ = fib(35); e = time(); e - s
4.918856859207153
>>> s = time(); _ = cfib(35); e = time(); e - s
0.07283687591552734
```

## 3.10.25 ctypes 的错误处理

```python
from __future__ import print_function

import os

from ctypes import *
from sys import platform, maxsize

is_64bits = maxsize > 2 ** 32

if is_64bits and platform == "darwin":
    libc = CDLL("libc.dylib", use_errno=True)
else:
    raise RuntimeError("Not support platform: {}".format(platform))

stat = libc.stat

class Stat(Structure):
    """
    From /usr/include/sys/stat.h

    struct stat {
        dev_t       st_dev;
        ino_t       st_ino;
        mode_t      st_mode;
        nlink_t     st_nlink;
        uid_t       st_uid;
        gid_t       st_gid;
        dev_t       st_rdev;
#ifndef _POSIX_SOURCE
        struct timespec st_atimespec;
        struct timespec st_mtimespec;
        struct timespec st_ctimespec;
#else
        time_t      st_atime;
        long        st_atimensec;
        time_t      st_mtime;
        long        st_mtimensec;
        time_t      st_ctime;
        long        st_ctimensec;
#endif
        off_t       st_size;
        int64_t     st_blocks;
        u_int32_t   st_blksize;
        u_int32_t   st_flags;
        u_int32_t   st_gen;
        int32_t     st_lspare;
        int64_t     st_qspare[2];
    };
    """
    _fields_ = [
        ("st_dev", c_ulong),
```

## 第四章

## 附录

本附录主要涵盖了一些速查表中缺失的关键概念。

## 4.1 为什么装饰器需要 @wraps

`@wraps` 用于保留原始函数的属性，否则被装饰函数的属性将被**包装函数**所替换。例如

不使用 @wraps

```
>>> def decorator(func):
...     def wrapper(*args, **kwargs):
...         print('wrap function')
...         return func(*args, **kwargs)
...     return wrapper
...
>>> @decorator
... def example(*a, **kw):
...     pass
...
>>> example.__name__  # 函数属性丢失
'wrapper'
```

使用 @wraps

```
>>> from functools import wraps
>>> def decorator(func):
...     @wraps(func)
...     def wrapper(*args, **kwargs):
...         print('wrap function')
...         return func(*args, **kwargs)
...     return wrapper
...
>>> @decorator
... def example(*a, **kw):
...     pass
...
>>> example.__name__  # 函数属性得以保留
'example'
```

## 4.2 异步编程漫游指南

- 目录
    - 异步编程漫游指南
        - 摘要
        - 引言
        - 回调函数
        - 事件循环
        - 什么是协程？
        - 结论
        - 参考文献

### 4.2.1 摘要

C10k 问题对于程序员来说仍然是一个待解的谜题。通常，开发者通过线程、epoll 或 kqueue 来处理大量的 I/O 操作，以避免其软件在昂贵的任务上等待。然而，由于数据共享和任务依赖，开发可读且无并发错误的代码颇具挑战性。尽管一些强大的工具（如 Valgrind）可以帮助开发者检测死锁或其他异步问题，但当软件规模变大时，解决这些问题可能非常耗时。因此，许多编程语言（如 Python、Javascript 或 C++）致力于开发更好的库、框架或语法，以帮助程序员妥善管理并发任务。本文不侧重于如何使用现代并行 API，而是主要关注异步编程模式背后的设计哲学。

使用线程是开发者在不阻塞主线程的情况下分派任务的一种更自然的方式。然而，线程可能导致性能问题，例如锁定临界区以执行某些原子操作。虽然在某些情况下使用事件循环可以提高性能，但由于回调问题（例如回调地狱），编写可读的代码颇具挑战性。幸运的是，像 Python 这样的编程语言引入了 async/await 概念，帮助开发者编写易于理解且高性能的代码。下图展示了使用 async/await 处理套接字连接的主要目标，类似于使用线程。

```
async def handler(conn):
    while True:
        msg = await loop.sock_recv(conn, 1024)
        if not msg:
            break
        await loop.sock_sendall(conn, msg)
    conn.close()

async def server():
    while True:
        conn, addr = await loop.sock_accept(s)
        loop.create_task(handler(conn))

loop.create_task(server())
loop.run_forever()
```

```
def handler(conn):
    while True:
        msg = conn.recv(1024)
        if not msg:
            break
        conn.send(msg)
    conn.close()

def server():
    while True:
        conn, addr = s.accept()
        t = threading.Thread(target=handler, args=(conn,))
        t.start()

server()
```

事件循环

线程

### 4.2.2 引言

处理 I/O 操作（如网络连接）是程序中最昂贵的任务之一。以一个简单的 TCP 阻塞回显服务器为例（如下代码片段）。如果一个客户端成功连接到服务器但未发送任何请求，它会阻塞其他客户端的连接。即使客户端尽快发送数据，如果没有其他客户端尝试建立连接，服务器也无法处理其他请求。此外，处理多个请求效率低下，因为它浪费了大量时间等待来自硬件（如网络接口）的 I/O 响应。因此，使用并发的套接字编程对于管理大量请求变得不可避免。

```
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 5566))
s.listen(10)

while True:
    conn, addr = s.accept()
    msg = conn.recv(1024)
    conn.send(msg)
```

防止服务器等待 I/O 操作的一个可能解决方案是将任务分派给其他线程。以下示例展示了如何创建一个线程来同时处理连接。然而，创建大量线程可能会耗尽所有计算能力而无法获得高吞吐量。更糟糕的是，应用程序可能会浪费时间等待锁来处理临界区中的任务。尽管使用线程可以解决套接字服务器的阻塞问题，但其他因素（如 CPU 利用率）对于程序员克服 C10k 问题至关重要。因此，在不创建无限线程的情况下，事件循环是管理连接的另一种解决方案。

```
import threading
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 5566))
s.listen(10240)

def handler(conn):
    while True:
        msg = conn.recv(65535)
        conn.send(msg)

while True:
    conn, addr = s.accept()
    t = threading.Thread(target=handler, args=(conn,))
    t.start()
```

一个简单的事件驱动套接字服务器包含三个主要组件：一个 I/O 多路复用模块（例如 `select`）、一个调度器（循环）和回调函数（事件）。例如，以下服务器在循环中使用高级 I/O 多路复用 `selectors` 来检查 I/O 操作是否就绪。如果数据可读/写，循环获取 I/O 事件并执行回调函数 `accept`、`read` 或 `write` 来完成任务。

```
import socket

from selectors import DefaultSelector, EVENT_READ, EVENT_WRITE
from functools import partial

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 5566))
s.listen(10240)
s.setblocking(False)

sel = DefaultSelector()

def accept(s, mask):
    conn, addr = s.accept()
    conn.setblocking(False)
    sel.register(conn, EVENT_READ, read)

def read(conn, mask):
    msg = conn.recv(65535)
    if not msg:
        sel.unregister(conn)
        return conn.close()
    sel.modify(conn, EVENT_WRITE, partial(write, msg=msg))

def write(conn, mask, msg=None):
    if msg:
        conn.send(msg)
    sel.modify(conn, EVENT_READ, read)

sel.register(s, EVENT_READ, accept)
while True:
    events = sel.select()
    for e, m in events:
        cb = e.data
        cb(e.fileobj, m)
```

尽管通过线程管理连接可能效率不高，但利用事件循环调度任务的程序也不易阅读。为了提高代码可读性，包括 Python 在内的许多编程语言引入了协程、future 或 async/await 等抽象概念来处理 I/O 多路复用。为了更好地理解编程术语并正确使用它们，以下章节将讨论这些概念是什么以及它们试图解决什么问题。

### 4.2.3 回调函数

回调函数用于在事件触发时控制运行时的数据流。然而，保留当前回调函数的状态颇具挑战性。例如，如果程序员想在 TCP 服务器上实现握手，他/她可能需要将之前的状态存储在某处。

```
import socket

from selectors import DefaultSelector, EVENT_READ, EVENT_WRITE
from functools import partial
```

```python
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 5566))
s.listen(10240)
s.setblocking(False)

sel = DefaultSelector()
is_hello = {}

def accept(s, mask):
    conn, addr = s.accept()
    conn.setblocking(False)
    is_hello[conn] = False;
    sel.register(conn, EVENT_READ, read)

def read(conn, mask):
    msg = conn.recv(65535)
    if not msg:
        sel.unregister(conn)
        return conn.close()

    # check whether handshake is successful or not
    if is_hello[conn]:
        sel.modify(conn, EVENT_WRITE, partial(write, msg=msg))
        return

    # do a handshake
    if msg.decode("utf-8").strip() != "hello":
        sel.unregister(conn)
        return conn.close()

    is_hello[conn] = True

def write(conn, mask, msg=None):
    if msg:
        conn.send(msg)
    sel.modify(conn, EVENT_READ, read)

sel.register(s, EVENT_READ, accept)
while True:
    events = sel.select()
    for e, m in events:
        cb = e.data
        cb(e.fileobj, m)

```
虽然变量 `is_hello` 有助于存储状态以检查握手是否成功，但这使得代码对程序员来说更难理解。实际上，先前实现的概念很简单，它等同于以下代码片段（阻塞版本）。

```python
def accept(s):
    conn, addr = s.accept()
    success = handshake(conn)
    if not success:
        conn.close()

def handshake(conn):
    data = conn.recv(65535)
    if not data:
        return False
    if data.decode('utf-8').strip() != "hello":
        return False
    conn.send(b"hello")
    return True
```
为了将类似的结构从阻塞迁移到非阻塞，当一个函数（或任务）需要等待 I/O 操作时，它需要**快照**当前状态，包括参数、变量和断点。同时，调度器必须能够在 I/O 操作完成后重新进入该函数并执行剩余的代码。与其他编程语言（如 C++）不同，Python 可以轻松实现上述概念，因为其**生成器**可以通过调用内置函数 `next()` 来保留所有状态并重新进入。通过利用生成器，在事件循环内部处理像前面片段那样的 I/O 操作（但是以非阻塞形式），被称为*内联回调*，是可行的。

### 4.2.4 事件循环

事件循环是一个调度器，用于管理程序内的任务，而不是依赖操作系统。以下代码片段展示了如何用一个简单的事件循环异步处理 socket 连接。实现的概念是：当 I/O 操作未就绪时，将任务追加到一个 FIFO 作业队列，并注册一个*选择器*。同时，一个*生成器*保存了任务的状态，使其能够在 I/O 结果可用时，无需回调函数即可执行剩余的工作。通过观察事件循环的工作方式，将有助于理解 Python 生成器确实是*协程*的一种形式。

```python
# loop.py

from selectors import DefaultSelector, EVENT_READ, EVENT_WRITE

class Loop(object):
    def __init__(self):
        self.sel = DefaultSelector()
        self.queue = []

    def create_task(self, task):
        self.queue.append(task)

    def polling(self):
        for e, m in self.sel.select(0):
            self.queue.append((e.data, None))
            self.sel.unregister(e.fileobj)

    def is_registered(self, fileobj):
        try:
            self.sel.get_key(fileobj)
        except KeyError:
            return False
        return True

    def register(self, t, data):
        if not data:
            return False

        if data[0] == EVENT_READ:
            if self.is_registered(data[1]):
                self.sel.modify(data[1], EVENT_READ, t)
            else:
                self.sel.register(data[1], EVENT_READ, t)
        elif data[0] == EVENT_WRITE:
            if self.is_registered(data[1]):
                self.sel.modify(data[1], EVENT_WRITE, t)
            else:
                self.sel.register(data[1], EVENT_WRITE, t)
        else:
            return False

        return True

    def accept(self, s):
        conn, addr = None, None
        while True:
            try:
                conn, addr = s.accept()
            except BlockingIOError:
                yield (EVENT_READ, s)
            else:
                break
        return conn, addr

    def recv(self, conn, size):
        msg = None
        while True:
            try:
                msg = conn.recv(1024)
            except BlockingIOError:
                yield (EVENT_READ, conn)
            else:
                break
        return msg

    def send(self, conn, msg):
        size = 0
        while True:
            try:
                size = conn.send(msg)
            except BlockingIOError:
                yield (EVENT_WRITE, conn)
            else:
                break
        return size

    def once(self):
        self.polling()
        unfinished = []
        for t, data in self.queue:
            try:
                data = t.send(data)
            except StopIteration:
                continue

            if self.register(t, data):
                unfinished.append((t, None))

        self.queue = unfinished

    def run(self):
        while self.queue or self.sel.get_map():
            self.once()

```
通过将作业分配给事件循环来处理连接，这种编程模式类似于使用线程来管理 I/O 操作，但利用的是用户级调度器。此外，PEP 380 实现了生成器委托，允许一个生成器可以等待其他生成器完成其工作。显然，以下代码片段比使用回调函数处理 I/O 操作更直观、更可读。

```python
# foo.py
# $ python3 foo.py &
# $ nc localhost 5566

import socket

from selectors import EVENT_READ, EVENT_WRITE

# import loop.py
from loop import Loop

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 5566))
s.listen(10240)
s.setblocking(False)

loop = Loop()

def handler(conn):
    while True:
        msg = yield from loop.recv(conn, 1024)
        if not msg:
            conn.close()
            break
        yield from loop.send(conn, msg)

def main():
    while True:
        conn, addr = yield from loop.accept(s)
        conn.setblocking(False)
        loop.create_task((handler(conn), None))
    loop.create_task((main(), None))
    loop.run()

```
使用带有 `yield from` 语法的事件循环，可以在不阻塞主线程的情况下管理连接，这是 Python 3.5 之前 `asyncio` 模块的使用方式。然而，使用 `yield from` 语法存在歧义，因为它可能让程序员感到困惑：为什么添加 `@asyncio.coroutine` 就能让一个生成器变成协程？为了代替 `yield from` 处理异步操作，PEP 492 提出协程应该成为 Python 中一个独立的概念，这就是引入新的 `async/await` 语法以增强异步编程可读性的原因。

### 4.2.5 什么是协程？

Python 文档定义协程是子程序的一种通用形式。然而，这个定义含糊不清，阻碍了开发者理解协程是什么。基于前面的讨论，事件循环负责调度生成器来执行特定任务，这类似于将作业分派给线程。在这种情况下，生成器就像线程一样负责执行“例程工作”。显然，协程是一个术语，表示由程序中的事件循环（而不是操作系统）调度的任务。以下代码片段展示了 `@coroutine` 是什么。这个装饰器主要将函数转换为生成器函数，并使用包装器 `types.coroutine` 来保持向后兼容性。

```python
import asyncio
import inspect
import types

from functools import wraps
from asyncio.futures import Future

def coroutine(func):
    """Simple prototype of coroutine"""
    if inspect.isgeneratorfunction(func):
        return types.coroutine(func)

    @wraps(func)
    def coro(*a, **k):
        res = func(*a, **k)
        if isinstance(res, Future) or inspect.isgenerator(res):
            res = yield from res
        return res
    return types.coroutine(coro)

@coroutine
def foo():
    yield from asyncio.sleep(1)
    print("Hello Foo")

loop = asyncio.get_event_loop()
```## 4.2.6 结论

得益于现代语法和库的支持，通过事件循环进行异步编程如今变得更加直观和易读。大多数编程语言，包括Python，都实现了库来通过与新语法交互来管理任务调度。虽然新语法起初看起来有些神秘，但它们为程序员提供了一种在代码中构建逻辑结构的方式，就像使用线程一样。此外，由于任务完成后无需调用回调函数，程序员不必担心如何将当前任务的状态（如局部变量和参数）传递给其他回调。因此，程序员将能够专注于开发程序，而无需花费大量时间排查并发问题。

## 4.2.7 参考资料

1. asyncio — 异步I/O
2. PEP 342 - 通过增强生成器实现协程
3. PEP 380 - 委托给子生成器的语法
4. PEP 492 - 使用async和await语法的协程

## 4.3 Asyncio 背后的原理

- Asyncio 背后的原理
  - 什么是 Task？
  - 事件循环如何工作？
  - asyncio.wait 如何工作？
  - 简单的 asyncio.run
  - loop.sock_* 如何工作？
  - loop.create_server 如何工作？

### 4.3.1 什么是 Task？

```python
# 目标：监督协程的运行状态
# 参考：asyncio/tasks.py

import asyncio
Future = asyncio.futures.Future

class Task(Future):
    """Task 的简单原型"""

    def __init__(self, gen, *, loop):
        super().__init__(loop=loop)
        self._gen = gen
        self._loop.call_soon(self._step)

    def _step(self, val=None, exc=None):
        try:
            if exc:
                f = self._gen.throw(exc)
            else:
                f = self._gen.send(val)
        except StopIteration as e:
            self.set_result(e.value)
        except Exception as e:
            self.set_exception(e)
        else:
            f.add_done_callback(
                self._wakeup)

    def _wakeup(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            self._step(None, e)
        else:
            self._step(res, None)

@asyncio.coroutine
def foo():
    yield from asyncio.sleep(3)
    print("Hello Foo")

@asyncio.coroutine
def bar():
    yield from asyncio.sleep(1)
    print("Hello Bar")

loop = asyncio.get_event_loop()
tasks = [Task(foo(), loop=loop),
         loop.create_task(bar())]
loop.run_until_complete(
    asyncio.wait(tasks))
loop.close()
```

输出：

```
$ python test.py
Hello Bar
Hello Foo
```

### 4.3.2 事件循环如何工作？

```python
import asyncio
from collections import deque

def done_callback(fut):
    fut._loop.stop()

class Loop:
    """简单的事件循环原型"""

    def __init__(self):
        self._ready = deque()
        self._stopping = False

    def create_task(self, coro):
        Task = asyncio.tasks.Task
        task = Task(coro, loop=self)
        return task

    def run_until_complete(self, fut):
        tasks = asyncio.tasks
        # 获取任务
        fut = tasks.ensure_future(
                fut, loop=self)
        # 将任务添加到就绪队列
        fut.add_done_callback(done_callback)
        # 运行任务
        self.run_forever()
        # 从就绪队列中移除任务
        fut.remove_done_callback(done_callback)

    def run_forever(self):
        """运行任务直到停止"""
        try:
            while True:
                self._run_once()
                if self._stopping:
                    break
        finally:
            self._stopping = False

    def call_soon(self, cb, *args):
        """将任务追加到就绪队列"""
        self._ready.append((cb, args))
    def call_exception_handler(self, c):
        pass

    def _run_once(self):
        """立即运行任务"""
        ntodo = len(self._ready)
        for i in range(ntodo):
            t, a = self._ready.popleft()
            t(*a)

    def stop(self):
        self._stopping = True

    def close(self):
        self._ready.clear()

    def get_debug(self):
        return False

@asyncio.coroutine
def foo():
    print("Foo")

@asyncio.coroutine
def bar():
    print("Bar")

loop = Loop()
tasks = [loop.create_task(foo()),
         loop.create_task(bar())]
loop.run_until_complete(
    asyncio.wait(tasks))
loop.close()
```

输出：

```
$ python test.py
Foo
Bar
```

### 4.3.3 asyncio.wait 如何工作？

```python
import asyncio

async def wait(fs, loop=None):
    fs = {asyncio.ensure_future(_) for _ in set(fs)}
    if loop is None:
        loop = asyncio.get_event_loop()

    waiter = loop.create_future()
    counter = len(fs)

    def _on_complete(f):
        nonlocal counter
        counter -= 1
        if counter <= 0 and not waiter.done():
            waiter.set_result(None)

    for f in fs:
        f.add_done_callback(_on_complete)

    # 等待所有任务完成
    await waiter

    done, pending = set(), set()
    for f in fs:
        f.remove_done_callback(_on_complete)
        if f.done():
            done.add(f)
        else:
            pending.add(f)
    return done, pending

async def slow_task(n):
    await asyncio.sleep(n)
    print('sleep "{}" sec'.format(n))

loop = asyncio.get_event_loop()

try:
    print("---> wait")
    loop.run_until_complete(
        wait([slow_task(_) for _ in range(1, 3)]))
    print("---> asyncio.wait")
    loop.run_until_complete(
        asyncio.wait([slow_task(_) for _ in range(1, 3)]))
finally:
    loop.close()
```

输出：

```
---> wait
sleep "1" sec
sleep "2" sec
---> asyncio.wait
sleep "1" sec
sleep "2" sec
```

### 4.3.4 简单的 asyncio.run

```python
>>> import asyncio
>>> async def getaddrinfo(host, port):
...     loop = asyncio.get_event_loop()
...     return (await loop.getaddrinfo(host, port))
...
>>> def run(main):
...     loop = asyncio.new_event_loop()
...     asyncio.set_event_loop(loop)
...     return loop.run_until_complete(main)
...
>>> ret = run(getaddrinfo('google.com', 443))
>>> ret = asyncio.run(getaddrinfo('google.com', 443))
```

### 4.3.5 loop.sock_* 如何工作？

```python
import asyncio
import socket

def sock_accept(self, sock, fut=None, registed=False):
    fd = sock.fileno()
    if fut is None:
        fut = self.create_future()
    if registed:
        self.remove_reader(fd)
    try:
        conn, addr = sock.accept()
        conn.setblocking(False)
    except (BlockingIOError, InterruptedError):
        self.add_reader(fd, self.sock_accept, sock, fut, True)
    except Exception as e:
        fut.set_exception(e)
    else:
        fut.set_result((conn, addr))
    return fut

def sock_recv(self, sock, n, fut=None, registed=False):
    fd = sock.fileno()
    if fut is None:
        fut = self.create_future()
    if registed:
        self.remove_reader(fd)
    try:
        data = sock.recv(n)
    except (BlockingIOError, InterruptedError):
        self.add_reader(fd, self.sock_recv, sock, n, fut, True)
    except Exception as e:
        fut.set_exception(e)
    else:
        fut.set_result(data)
    return fut

def sock_sendall(self, sock, data, fut=None, registed=False):
    fd = sock.fileno()
    if fut is None:
        fut = self.create_future()
    if registed:
        self.remove_writer(fd)
    try:
        n = sock.send(data)
    except (BlockingIOError, InterruptedError):
        n = 0
    except Exception as e:
        fut.set_exception(e)
        return
    if n == len(data):
        fut.set_result(None)
    else:
        if n:
            data = data[n:]
        self.add_writer(fd, sock, data, fut, True)
    return fut

async def handler(loop, conn):
    while True:
        msg = await loop.sock_recv(conn, 1024)
        if msg: await loop.sock_sendall(conn, msg)
        else: break
    conn.close()

async def server(loop):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setblocking(False)
    sock.bind(('localhost', 9527))
    sock.listen(10)

    while True:
        conn, addr = await loop.sock_accept(sock)
        loop.create_task(handler(loop, conn))

EventLoop = asyncio.SelectorEventLoop
EventLoop.sock_accept = sock_accept
EventLoop.sock_recv = sock_recv
EventLoop.sock_sendall = sock_sendall
loop = EventLoop()

try:
    loop.run_until_complete(server(loop))
except KeyboardInterrupt:
    pass
finally:
    loop.close()
```

输出：

```
# 控制台 1
$ python3 async_sock.py &
$ nc localhost 9527
Hello
Hello
```

## 4.3.6 `loop.create_server` 是如何工作的？

```python
import asyncio
import socket

loop = asyncio.get_event_loop()

async def create_server(loop, protocol_factory, host, port, *args, **kwargs):
    sock = socket.socket(socket.AF_INET,
                         socket.SOCK_STREAM, 0)
    sock.setsockopt(socket.SOL_SOCKET,
                    socket.SO_REUSEADDR, 1)
    sock.setblocking(False)
    sock.bind((host, port))
    sock.listen(10)
    sockets = [sock]
    server = asyncio.base_events.Server(loop, sockets)
    loop._start_serving(protocol_factory, sock, None, server)

    return server

class EchoProtocol(asyncio.Protocol):
    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        print('Connection from {}'.format(peername))
        self.transport = transport

    def data_received(self, data):
        message = data.decode()
        self.transport.write(data)

# Equal to: loop.create_server(EchoProtocol,
#                             'localhost', 5566)
coro = create_server(loop, EchoProtocol, 'localhost', 5566)
server = loop.run_until_complete(coro)

try:
    loop.run_forever()
finally:
    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()
```

输出：
```
# 控制台1
$ nc localhost 5566
Hello
Hello

# 控制台2
$ nc localhost 5566
asyncio
asyncio
```

## 4.4 PEP 572 与海象运算符

**目录**

*   *PEP 572 与海象运算符*
*   摘要
*   引言
*   为什么是 `:=` ？
*   作用域
*   陷阱
*   结论
*   参考文献

## 4.4.1 摘要

PEP 572 是 Python3 历史上最具争议的提案之一，因为在表达式内赋值似乎没有必要。此外，开发者也难以区分 **海象运算符** (`:=`) 和等于运算符 (`=`) 的区别。即使经验丰富的开发者可以熟练使用 `:=`，他们可能也会担心代码的可读性。为了更好地理解 `:=` 的用法，本文将探讨其设计哲学以及它试图解决的问题。

## 4.4.2 引言

对于 C/C++ 开发者来说，将函数返回值赋给变量是常见的做法，这源于其错误代码风格的处理方式。管理函数错误包括两个步骤：一是检查返回值；二是检查 errno。例如，

```c
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

int main(int argc, char *argv[]) {
    int rc = -1;

    // assign access return to rc and check its value
    if ((rc = access("hello_walrus", R_OK)) == -1) {
        fprintf(stderr, "%s", strerror(errno));
        goto end;
    }
    rc = 0;
    end:
        return rc;
}
```

在这个例子中，`access` 会先将其返回值赋给变量 `rc`。然后，程序会将 `rc` 的值与 `-1` 进行比较，以检查 `access` 的执行是否成功。然而，在 3.8 版本之前，Python 不允许在表达式内为变量赋值。为了解决这个问题，PEP 572 引入了海象运算符供开发者使用。以下的 Python 代码片段与前面的 C 示例等效。

```python
>>> import os
>>> from ctypes import *
>>> libc = CDLL("libc.dylib", use_errno=True)
>>> access = libc.access
>>> path = create_string_buffer(b"hello_walrus")
>>> if (rc := access(path, os.R_OK)) == -1:
...     errno = get_errno()
...     print(os.strerror(errno), file=sys.stderr)
...
No such file or directory
```

### 4.4.3 为什么是 `:=` ？

开发者可能会混淆 `:=` 和 `=` 之间的区别。实际上，它们的目的一样，都是将值赋给变量。为什么 Python 引入 `:=` 而不是使用 `=`？使用 `:=` 有什么好处？一个原因是为了增强视觉识别性，避免 C/C++ 开发者常犯的错误。例如，

```c
int rc = access("hello_walrus", R_OK);

// rc is unintentionally assigned to -1
if (rc = -1) {
    fprintf(stderr, "%s", strerror(errno));
    goto end;
}
```

变量 `rc` 被错误地赋值为 `-1`，而不是进行比较。为了防止这种错误，一些人提倡在表达式中使用 Yoda 条件式。

```c
int rc = access("hello_walrus", R_OK);

// -1 = rc will raise a compile error
if (-1 == rc) {
    fprintf(stderr, "%s", strerror(errno));
    goto end;
}
```

然而，Yoda 风格的可读性不够好，就像 Yoda 说不标准英语一样。此外，与 C/C++ 可以通过编译器选项（如 -Wparentheses）在编译时检测赋值错误不同，Python 解释器很难在运行时区分这类错误。因此，PEP 572 的最终结果是采用新语法作为实现 *赋值表达式* 的解决方案。

海象运算符并非 PEP 572 的第一个解决方案。最初的提案使用 `EXPR as NAME` 将值赋给变量。不幸的是，这个解决方案以及其他一些解决方案都被否决了。经过激烈的讨论，最终决定使用 `:=`。

## 4.4.4 作用域

与其他表达式不同，在其他表达式中变量会绑定到作用域，而赋值表达式属于当前作用域。这种设计的目的是允许以紧凑的方式编写代码。

```python
>>> if not (env := os.environ.get("HOME")):
...     raise KeyError("env HOME does not find!")
...
>>> print(env)
/root
```

在 PEP 572 中，另一个好处是方便地为 `any()` 或 `all()` 表达式捕获一个“见证者”。虽然捕获函数输入可以帮助交互式调试器，但其优势并不那么明显，且示例缺乏可读性。因此，此处不讨论此好处。请注意，其他语言（如 C/C++ 或 Go）可能会将赋值绑定到作用域。以 Golang 为例。

```go
package main

import (
    "fmt"
    "os"
)

func main() {
    if env := os.Getenv("HOME"); env == "" {
        panic(fmt.Sprintf("Home does not find"))
    }
    fmt.Print(env) // <--- compile error: undefined: env
}
```

## 4.4.5 陷阱

尽管赋值表达式允许编写紧凑的代码，但当开发者在列表推导式中使用它时，会存在许多陷阱。一个常见的 `SyntaxError` 是重新绑定迭代变量。

```python
>>> [i := i+1 for i in range(5)]  # invalid
```

然而，更新迭代变量会降低可读性并引入错误。即使在 Python 3.8 未实现海象运算符的情况下，程序员也应避免在作用域内重复使用迭代变量。

另一个陷阱是 Python 禁止在类作用域下的推导式中使用赋值表达式。

```python
>>> class Example:
...     [(j := i) for i in range(5)] # invalid
...
```

这个限制源于 bpo-3692。当类声明包含列表推导式时，解释器的行为是不可预测的。为了避免这种极端情况，在类内部赋值表达式是无效的。

```python
>>> class Foo:
...     a = [1, 2, 3]
...     b = [4, 5, 6]
...     c = [i for i in zip(a, b)]  # b is defined
...
>>> class Bar:
...     a = [1,2,3]
...     b = [4,5,6]
...     c = [x * y for x in a for y in b] # b is undefined
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 4, in Bar
  File "<stdin>", line 4, in <listcomp>
NameError: name 'b' is not defined
```

## 4.4.6 结论

海象运算符 (`:=`) 如此具有争议性的原因是它可能会降低代码的可读性。事实上，在讨论邮件列表中，PEP 572 的作者 Christoph Groth 曾考虑使用 `:=` 来实现像 C/C++ 那样的内联赋值。抛开 `:=` 是否丑陋不谈，许多开发者认为区分 `:=` 和 `=` 的功能是困难的，因为它们目的一样，但行为不一致。此外，编写紧凑的代码也不够有说服力，因为“小”并不总是“好”。然而，在某些情况下，海象运算符可以增强可读性（如果你知道如何使用 `:=`）。例如，

```python
buf = b""
while True:
    data = read(1024)
    if not data:
        break
    buf += data
```

通过使用 `:=`，前面的示例可以简化。

```python
buf = b""
while (data := read(1024)):
    buf += data
```

Python 文档和 GitHub issue-8122 提供了许多关于通过 `:=` 提高代码可读性的优秀示例。然而，使用海象运算符需要谨慎。有些情况，比如 `foo(x := 3, cat='vector')`，如果开发者没有意识到作用域问题，可能会引入新的错误。尽管 PEP 572 可能会让开发者更容易写出有 bug 的代码，但深入理解其设计哲学和有用的示例将有助于我们在合适的时候用它编写可读的代码。

## 4.4.7 参考资料

- PEP 572 - 赋值表达式
- Python 3.8 新特性
- PEP 572 与 Python 中的决策制定
- PEP 572 终局之战
- 在标准库中使用赋值表达式（合并后的 PR）
- 在类声明中使用列表推导式时的变量作用域不当问题

## 4.5 在 GNU 调试器中的 Python 解释器

- 目录
  - 在 GNU 调试器中的 Python 解释器
    - 摘要
    - 引言
    - 定义命令
    - 内存转储
    - JSON 转储
    - 语法高亮
    - 跟踪点
    - 性能分析
    - 美化打印
    - 结论
    - 参考资料

### 4.5.1 摘要

GNU 调试器（GDB）是开发者用于调试代码错误的强大工具。然而，它对于初学者来说难以掌握，这也是为什么许多程序员更倾向于插入 `print` 语句来检查运行时状态。幸运的是，`GDB 文本用户界面 (TUI)` 为开发者提供了一种同时查看源代码和进行调试的方式。更令人兴奋的是，在 GDB 7 中，**Python 解释器**被内置到了 GDB 中。这一特性通过 Python 库提供了更直接的方式来定制 GDB 的打印机和命令。本文将通过讨论示例，尝试探索通过 Python 实现的高级调试技术，以为 GDB 开发工具包。

### 4.5.2 引言

排查软件错误对开发者来说是一个巨大的挑战。虽然 GDB 提供了许多“调试命令”来检查程序的运行时状态，但其非直观的用法阻碍了程序员用它来解决问题。的确，精通 GDB 是一个长期的过程。但是，快速入门并不复杂；你必须像尤达大师那样，“忘掉你已学过的东西”。为了更好地理解如何在 GDB 中使用 Python，本文将重点讨论 GDB 中的 Python 解释器。

### 4.5.3 定义命令

GDB 支持使用 `define` 来自定义命令。这对于同时运行一组调试命令很有用。例如，开发者可以通过定义一个 `sf` 命令来显示当前栈帧信息。

```
# 在 .gdbinit 中定义
define sf
    where        # 找出程序当前在哪里
    info args    # 显示参数
    info locals  # 显示局部变量
end
```

然而，由于 API 有限，编写用户自定义命令可能不太方便。幸运的是，通过与 GDB 中的 Python 解释器交互，开发者可以轻松利用 Python 库来构建自己的调试工具包。接下来的部分将展示如何使用 Python 来简化调试过程。

### 4.5.4 内存转储

检查进程的内存信息是排查内存问题的有效方法。开发者可以通过 `info proc mappings` 和 `dump memory` 获取内存内容。为了简化这些步骤，定义一个自定义命令很有帮助。然而，使用纯 GDB 语法来实现并不直接。即使 GDB 支持条件判断，处理输出也不直观。要解决这个问题，在 GDB 中使用 Python API 会很有帮助，因为 Python 包含许多处理字符串的实用操作。

```
# mem.py
import gdb
import time
import re

class DumpMemory(gdb.Command):
    """将内存信息转储到文件。"""

    def __init__(self):
        super().__init__("dm", gdb.COMMAND_USER)

    def get_addr(self, p, tty):
        """获取内存地址。"""
        cmd = "info proc mappings"
        out = gdb.execute(cmd, tty, True)
        addrs = []
        for l in out.split("\n"):
            if re.match(f".*{p}.*", l):
                s, e, *_ = l.split()
                addrs.append((s, e))

    def dump(self, addrs):
        """执行内存转储。"""
        if not addrs:
            return

        for s, e in addrs:
            f = int(time.time() * 1000)
            gdb.execute(f"dump memory {f}.bin {s} {e}")

    def invoke(self, args, tty):
        try:
            # cat /proc/self/maps
            addrs = self.get_addr(args, tty)
            # 转储内存
            self.dump(addrs)
        except Exception as e:
            print("用法: dm [模式]")

DumpMemory()
```

运行 `dm` 命令将调用 `DumpMemory.invoke`。通过在 `.gdbinit` 中加载或实现 Python 脚本，开发者可以利用自定义命令在程序运行时跟踪错误。例如，以下步骤展示了如何在 GDB 中调用 `DumpMemory`。

```
(gdb) start
...
(gdb) source mem.py  # 加载命令
(gdb) dm stack       # 将栈转储到 ${timestamp}.bin
(gdb) shell ls       # 列出当前目录
1577283091687.bin  a.cpp  a.out  mem.py
```

### 4.5.5 JSON 转储

当开发者在检查运行中的程序里的 JSON 字符串时，解析 JSON 很有帮助。GDB 可以通过 `gdb.parse_and_eval` 解析 `std::string`，并将其作为 `gdb.Value` 返回。通过处理 `gdb.Value`，开发者可以将 JSON 字符串传递给 Python 的 `json` API，并以美化格式打印它。

```
# dj.py
import gdb
import re
import json

class DumpJson(gdb.Command):
    """将 std::string 转储为格式化的 JSON。"""

    def __init__(self):
        super().__init__("dj", gdb.COMMAND_USER)

    def get_json(self, args):
        """将 std::string 解析为 JSON 字符串。"""
        ret = gdb.parse_and_eval(args)
        typ = str(ret.type)
        if re.match("^std::.*::string", typ):
            return json.loads(str(ret))
        return None

    def invoke(self, args, tty):
        try:
            # 字符串转 JSON 字符串
            s = self.get_json(args)
            # JSON 字符串转对象
            o = json.loads(s)
            print(json.dumps(o, indent=2))
        except Exception as e:
            print(f"解析 JSON 错误！ {args}")

DumpJson()
```

命令 `dj` 在 GDB 中以更易读的 JSON 格式显示内容。此命令有助于在 JSON 字符串较大时改善视觉识别。同时，通过使用此命令，可以检测或监控一个 `std::string` 是否为 JSON。

```
(gdb) start
(gdb) list
1	#include <string>
2
3	int main(int argc, char *argv[])
4	{
5	    std::string json = R"({"foo": "FOO","bar": "BAR"})";
6	    return 0;
7	}
...
(gdb) ptype json
type = std::string
(gdb) p json
$1 = "{"foo": "FOO","bar": "BAR"}"
(gdb) source dj.py
(gdb) dj json
{
  "foo": "FOO",
  "bar": "BAR"
}
```

### 4.5.6 语法高亮

语法高亮对于开发者跟踪源代码或排查问题很有用。通过使用 Pygments，可以轻松地为源代码着色，而无需手动定义 ANSI 转义码。以下示例展示了如何为 `list` 命令的输出添加颜色。

```
import gdb
from pygments import highlight
from pygments.lexers import CLexer
from pygments.formatters import TerminalFormatter

class PrettyList(gdb.Command):
    """打印带颜色的源代码。"""

    def __init__(self):
        super().__init__("pl", gdb.COMMAND_USER)
        self.lex = CLexer()
        self.fmt = TerminalFormatter()

    def invoke(self, args, tty):
        try:
            out = gdb.execute(f"l {args}", tty, True)
            print(highlight(out, self.lex, self.fmt))
        except Exception as e:
            print(e)

PrettyList()
```

### 4.5.7 跟踪点

虽然开发者可以插入 `printf`、`std::cout` 或 `syslog` 来检查函数，但在大型项目中，打印消息并不是一种有效的调试方式。开发者可能会浪费时间在构建源代码上，而获得的信息却很少。更糟糕的是，输出可能太多而无法检测问题。事实上，检查函数或变量并不需要在代码中嵌入打印函数。通过编写使用 GDB API 的 Python 脚本，开发者可以自定义监视点，在运行时动态跟踪问题。例如，通过实现一个 `gdb.Breakpoint` 和一个 `gdb.Command`，开发者可以获取关键信息，如参数、调用栈或内存使用情况。

```
# tp.py
import gdb

tp = {}

class Tracepoint(gdb.Breakpoint):
    def __init__(self, *args):
        super().__init__(*args)
        self.silent = True
        self.count = 0

    def stop(self):
        self.count += 1
        frame = gdb.newest_frame()
        block = frame.block()
        sym_and_line = frame.find_sal()
        framename = frame.name()
        filename = sym_and_line.symtab.filename
        line = sym_and_line.line
        # 显示跟踪点信息
        print(f"{framename} @ {filename}:{line}")
        # 显示参数和变量
        for s in block:
            if not s.is_argument and not s.is_variable:
                continue
            typ = s.type
            val = s.value(frame)
            size = typ.sizeof
            name = s.name
            print(f"\t{name}({typ}: {val}) [{size}]")
        # 不在跟踪点处停止
        return False

class SetTracepoint(gdb.Command):
    def __init__(self):
        super().__init__("tp", gdb.COMMAND_USER)

    def invoke(self, args, tty):
        try:
            global tp
            tp[args] = Tracepoint(args)
        except Exception as e:
            print(e)

def finish(event):
    for t, p in tp.items():
        c = p.count
        print(f"跟踪点 '{t}' 命中次数: {c}")

gdb.events.exited.connect(finish)
SetTracepoint()
```

与其在函数开头插入 `std::cout`，不如在函数入口点使用跟踪点，可以提供检查参数、变量和栈的有用信息。例如，通过在 `fib` 上设置一个跟踪点，有助于检查内存使用、栈和调用次数。

```c
int fib(int n)
{
    if (n < 2) {
        return 1;
    }
    return fib(n-1) + fib(n-2);
}
```

```c
int main(int argc, char *argv[])
```

```python
# 4.5.8 性能分析

即使不插入时间戳，通过跟踪点（tracepoint）仍然可以进行性能分析。通过在`gdb.Breakpoint`之后使用`gdb.FinishBreakpoint`，GDB会在一个栈帧的返回地址处设置一个临时断点，以便开发者获取当前时间戳并计算时间差。请注意，通过GDB进行的性能分析并不精确。其他工具，如Linux perf或Valgrind，能提供更有用且更准确的信息来追踪性能问题。

```python
import gdb
import time

class EndPoint(gdb.FinishBreakpoint):
    def __init__(self, breakpoint, *a, **kw):
        super().__init__(*a, **kw)
        self.silent = True
        self.breakpoint = breakpoint

    def stop(self):
        # normal finish
        end = time.time()
        start, out = self.breakpoint.stack.pop()
        diff = end - start
        print(out.strip())
        print(f"\tCost: {diff}")
        return False

class StartPoint(gdb.Breakpoint):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.silent = True
        self.stack = []

    def stop(self):
        start = time.time()
        # start, end, diff
        frame = gdb.newest_frame()
        sym_and_line = frame.find_sal()
        func = frame.function().name
        filename = sym_and_line.symtab.filename
        line = sym_and_line.line
        block = frame.block()

        args = []
        for s in block:
            if not s.is_argument:
                continue
            name = s.name
            typ = s.type
            val = s.value(frame)
            args.append(f"{name}: {val} [{typ}]")

        # format
        out = ""
        out += f"{func} @ {filename}:{line}\n"
        for a in args:
            out += f"\t{a}\n"

        # append current status to a breakpoint stack
        self.stack.append((start, out))
        EndPoint(self, internal=True)
        return False

class Profile(gdb.Command):
    def __init__(self):
        super().__init__("prof", gdb.COMMAND_USER)

    def invoke(self, args, tty):
        try:
            StartPoint(args)
        except Exception as e:
            print(e)

Profile()
```

以下输出展示了在函数`fib`上设置跟踪点后的性能分析结果。这便于同时检查函数的性能和栈信息。

```
(gdb) source prof.py
(gdb) prof fib
Breakpoint 1 at 0x606: file a.cpp, line 3.
(gdb) r
Starting program: /root/a.out
fib(int) @ a.cpp:3
    n: 1 [int]
    Cost: 0.0007786750793457031
fib(int) @ a.cpp:3
    n: 0 [int]
    Cost: 0.002572298049926758
fib(int) @ a.cpp:3
    n: 2 [int]
    Cost: 0.008517265319824219
fib(int) @ a.cpp:3
    n: 1 [int]
    Cost: 0.0014069080352783203
fib(int) @ a.cpp:3
    n: 3 [int]
    Cost: 0.01870584487915039
```

# 4.5.9 美化打印

虽然在GDB中开启`set print pretty`可以提供更好的格式来检查变量，但开发者可能需要解析变量的值以提高可读性。以系统调用`stat`为例。虽然它提供了检查文件属性的有用信息，但输出的值（如权限）可能对调试来说不易读。通过实现用户自定义的美化打印，开发者可以解析`struct stat`并以可读的格式输出信息。

```python
import gdb
import pwd
import grp
import stat
import time

from datetime import datetime


class StatPrint:
    def __init__(self, val):
        self.val = val

    def get_filetype(self, st_mode):
        if stat.S_ISDIR(st_mode):
            return "directory"
        if stat.S_ISCHR(st_mode):
            return "character device"
        if stat.S_ISBLK(st_mode):
            return "block device"
        if stat.S_ISREG:
            return "regular file"
        if stat.S_ISFIFO(st_mode):
            return "FIFO"
        if stat.S_ISLNK(st_mode):
            return "symbolic link"
        if stat.S_ISSOCK(st_mode):
            return "socket"
        return "unknown"

    def get_access(self, st_mode):
        out = "-"
        info = ("r", "w", "x")
        perm = [
            (stat.S_IRUSR, stat.S_IWUSR, stat.S_IXUSR),
            (stat.S_IRGRP, stat.S_IRWXG, stat.S_IXGRP),
            (stat.S_IROTH, stat.S_IWOTH, stat.S_IXOTH),
        ]
        for pm in perm:
            for c, p in zip(pm, info):
                out += p if st_mode & c else "-"
        return out

    def get_time(self, st_time):
        tv_sec = int(st_time["tv_sec"])
        return datetime.fromtimestamp(tv_sec).isoformat()

    def to_string(self):
        st = self.val
        st_ino = int(st["st_ino"])
        st_mode = int(st["st_mode"])
        st_uid = int(st["st_uid"])
        st_gid = int(st["st_gid"])
        st_size = int(st["st_size"])
        st_blksize = int(st["st_blksize"])
        st_blocks = int(st["st_blocks"])
        st_atim = st["st_atim"]
        st_mtim = st["st_mtim"]
        st_ctim = st["st_ctim"]

        out = "{\n"
        out += f"Size: {st_size}\n"
        out += f"Blocks: {st_blocks}\n"
        out += f"IO Block: {st_blksize}\n"
        out += f"Inode: {st_ino}\n"
        out += f"Access: {self.get_access(st_mode)}\n"
        out += f"File Type: {self.get_filetype(st_mode)}\n"
        out += f"Uid: ({st_uid}/{pwd.getpwuid(st_uid).pw_name})\n"
        out += f"Gid: ({st_gid}/{grp.getgrgid(st_gid).gr_name})\n"
        out += f"Access: {self.get_time(st_atim)}\n"
        out += f"Modify: {self.get_time(st_mtim)}\n"
        out += f"Change: {self.get_time(st_ctim)}\n"
        out += "}"
        return out

p = gdb.printing.RegexpCollectionPrettyPrinter("sp")
p.add_printer("stat", "^stat$", StatPrint)
o = gdb.current_objfile()
gdb.printing.register_pretty_printer(o, p)
```

通过加载前面的Python脚本，PrettyPrinter可以识别`struct stat`并以可读的格式输出，方便开发者检查文件属性。无需插入函数来解析和打印`struct stat`，这是一种通过Python API获得更好输出的更便捷方式。

```
(gdb) list 15
10          struct stat st;
11          
12          if ((rc = stat("./a.cpp", &st)) < 0) {
13              perror("stat failed.");
14              goto end;
15          }
16          
17          rc = 0;
18      end:
19          return rc;
(gdb) source st.py
(gdb) b 17
Breakpoint 1 at 0x762: file a.cpp, line 17.
(gdb) r
Starting program: /root/a.out

Breakpoint 1, main (argc=1, argv=0x7fffffffe788) at a.cpp:17
17          rc = 0;
(gdb) p st
$1 = {
Size: 298
Blocks: 8
IO Block: 4096
Inode: 1322071
Access: -rw-rw-r--
File Type: regular file
Uid: (0/root)
Gid: (0/root)
Access: 2019-12-28T15:53:17
Modify: 2019-12-28T15:53:01
Change: 2019-12-28T15:53:01
}
```

请注意，开发者可以通过`disable`命令禁用用户自定义的美化打印。例如，前面的Python脚本在全局美化打印器下注册了一个打印器。通过调用`disable pretty-print`，打印器`sp`将被禁用。

```
(gdb) disable pretty-print global sp
1 printer disabled
1 of 2 printers enabled
(gdb) i pretty-print
global pretty-printers:
  builtin
    mpx_bound128
  sp [disabled]
    stat
```

此外，如果不再需要，开发者可以在当前GDB调试会话中排除一个打印器。以下代码片段展示了如何通过`gdb.pretty_printers.remove`删除sp打印器。

```
(gdb) python
>import gdb
>for p in gdb.pretty_printers:
>    if p.name == "sp":
>        gdb.pretty_printers.remove(p)
>end
(gdb) i pretty-print
global pretty-printers:
  builtin
    mpx_bound128
```

# 4.5.10 结论

将Python解释器集成到GDB中，为排查问题提供了许多灵活的方式。虽然许多集成开发环境（IDE）可能嵌入GDB以进行可视化调试，但GDB允许开发者实现自己的命令并在运行时解析变量的输出。通过使用调试脚本，开发者可以在不修改代码的情况下监控和记录必要信息。老实说，插入或启用调试代码块可能会改变程序的行为，开发者应该摒弃这种不良习惯。此外，当问题重现时，GDB可以附加到该进程并检查其状态，而无需停止它。显然，如果出现棘手的问题，通过GDB进行调试是不可避免的。得益于将Python集成到GDB中，开发用于排查问题的脚本变得更加容易，这使得开发者能够多样化地建立自己的调试方法。

# 4.5.11 参考资料

1. 使用Python扩展GDB
2. gcc/gcc/gdbhooks.py
3. gdbinit/Gdbinit
4. cyrus-and/gdb-dashboard
5. hugsy/gef
6. sharkdp/stack-inspector
7. GDB调试完整示例（教程）