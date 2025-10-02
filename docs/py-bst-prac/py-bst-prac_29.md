# 与 C/C++库交互

## C 语言外部函数接口(CFFI)

[CFFI](https://cffi.readthedocs.org/en/latest/) [https://cffi.readthedocs.org/en/latest/] 通过 CPython 和 PyPy 给出了和 C 语言交互的简单使用机制。它支持两种模式：一种是内联的 ABI 兼容模式(示例如下)，它允许你动态加载和运行可执行模块的函数(本质上与 LoadLibrary 和 dlopen 拥有相同的功能)；另一种为 API 模式，它允许你构建 C 语言扩展模块.

### ABI 交互

|  
```py
1
2
3
4
5
6
7
```

 |  
```py
from cffi import FFI
ffi = FFI()
ffi.cdef("size_t strlen(const char*);")
clib = ffi.dlopen(None)
length = clib.strlen("String to be evaluated.")
# prints: 23
print("{}".format(length)) 
```

 |

## ctypes

[ctypes](https://docs.python.org/3/library/ctypes.html) [https://docs.python.org/3/library/ctypes.html] 是 CPython 中与 C/C++交互的事实上的库。它不仅能完全访问大多数主流操作系统(比如：Windows 上的 Kernel32，*nix 上的 libc)的纯 C 接口，并且支持对动态库的加载和交互，如 DLL 和运行时共享对象。它同时涵盖许多可和系统 API 交互的类型，并允许你以相对简单的方式定义自己的复杂类型，如 struct 和 union，并在需要时允许你作出如填充、对齐这样的修改。对它的使用可能稍显复杂，但与 [struct](https://docs.python.org/3.5/library/struct.html) [https://docs.python.org/3.5/library/struct.html] 模块配合使用，可通过纯 C(++)方法让你从根本上控制你的数据类型转换成更有用的东西。

### Struct Equivalents

`MyStruct.h`

|  
```py
1
2
3
4
```

 |  
```py
struct my_struct {
    int a;
    int b;
}; 
```

 |

`MyStruct.py`

|  
```py
1
2
3
4
```

 |  
```py
import ctypes
class my_struct(ctypes.Structure):
    _fields_ = [("a", c_int),
                ("b", c_int)] 
```

 |

## SWIG

[SWIG](http://www.swig.org) [http://www.swig.org] 并不仅仅应用于 Python(它支持多种脚本语言)，它是生成解释性语言和 C/C++头文件绑定的工具。它极易使用：使用者只需简单的定义接口文件(详见相关指南和文档)，包含必要的 C/C++头文件，并对它们运行生成工具。但它也有其局限性，目前，它与 C++部分新特性间仍存在问题，而模板重码的工作多少有些冗繁。只需少量的工作，它便能提供诸多作用，并展现 Python 的许多特性。同时，你可以简单的扩展 SWIG 生成的绑定(在接口文件中)来重载操作符和内建函数，也可以有效的重新转换 C++异常，使其可被 Python 所捕获。

### Example: Overloading __repr__

`MyClass.h`

|  
```py
1
2
3
4
5
6
7
```

 |  
```py
#include <string> class MyClass {
private:
    std::string name;
public:
    std::string getName();
}; 
```

 |

`myclass.i`

|  
```py
 1
 2
 3
 4
 5
 6
 7
 8
 9
10
11
12
13
14
15
16
```

 |  
```py
%include "string.i"

%module myclass
%{
#include <string>
#include "MyClass.h"
%}

%extend MyClass {
    std::string __repr__()
    {
        return $self->getName();
    }
}

%include "MyClass.h" 
```

 |

## Boost.Python

[Boost.Python](http://www.boost.org/doc/libs/1_59_0/libs/python/doc/) [http://www.boost.org/doc/libs/1_59_0/libs/python/doc/] 需要一些手动工作来展现 C++对象的功能，但它可提供 SWIG 拥有的所有特性，同时，它可提供在 C++中访问 Python 对象的封装，也可提取 SWIG 封装的对象，甚至可在 C++代码中嵌入部分 Python。

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.