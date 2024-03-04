# Python 的内置异常

> 原文：<https://www.pythonforbeginners.com/error-handling/pythons-built-in-exceptions>

Python 的内置异常

### 基本异常

所有内置异常的基类。

### 例外

所有内置的、非系统退出的异常都是从这个类派生的。

所有用户定义的异常也应该从此类派生。

### 标准误差

除 StopIteration、GeneratorExit、
KeyboardInterrupt 和 SystemExit 之外的所有内置异常的基类。StandardError 本身是从 Exception 派生的。

### 算术误差

针对各种
算术错误引发的内置异常的基类:OverflowError、ZeroDivisionError、FloatingPointError

### LookupError

当
映射或序列上使用的键或索引无效时引发的异常的基类:IndexError，KeyError。

这可以由 sys.setdefaultencoding()直接引发

### 环境错误

Python 系统之外可能发生的异常的基类:
IOError，OSError。

### 断言错误

assert 语句失败时引发。

### 属性错误

当属性引用或赋值失败时引发。

### 埃费罗尔

当其中一个内置函数(input()或 raw_input())在没有读取任何数据的情况下遇到
文件结束条件(e of)时引发。

### 浮点错误

浮点运算失败时引发。

### 发电机出口

调用生成器的 close()方法时引发。

它直接继承自 Exception 而不是 StandardError，因为从技术上来说它不是一个错误。

### io 错误

当 I/O 操作(如打印语句、内置 open()
函数或 file 对象的方法)由于 I/O 相关的原因而失败时引发，
例如，“找不到文件”或“磁盘已满”。

此类派生自 EnvironmentError。

### 导入错误

当 import 语句找不到模块定义或当
from…import 找不到要导入的名称时引发。

### 索引错误

当序列下标超出范围时引发。

### KeyError

在现有键集中找不到映射(字典)键时引发。

### 键盘中断

当用户按下中断键(通常是 Control-C 或 Delete)时引发。

### 存储器错误

当一个操作耗尽了内存，但这种情况仍可能被
挽救(通过删除一些对象)时引发。

### NameError

找不到本地或全局名称时引发。

这只适用于非限定名。

相关值是一个错误消息，其中包含可能找不到的名称
。

### notimplemontederror

该异常源自 RuntimeError。

在用户定义的基类中，当抽象方法需要派生类来重写方法时，它们应该抛出这个异常。

### OSError

该类派生自 EnvironmentError，主要用作
os 模块的 os.error 异常。

### 溢出误差

当算术运算的结果太大而无法表示时引发。

### 参考错误

当由
weakref.proxy()函数创建的弱引用代理在被垃圾收集后用于访问 referent
的属性时，会引发该异常。

### RuntimeError

当检测到不属于任何其他类别的错误时引发。

### 停止迭代:

由迭代器的 next()方法引发，表示没有其他值。

### 句法误差

当分析器遇到语法错误时引发。

### 系统误差

当解释器发现一个内部错误时引发，但是情况看起来没有严重到使它放弃所有希望。

相关的值是一个字符串，指示出了什么问题(在低级术语中)。

### 系统退出

此异常由 sys.exit()函数引发。

不处理时，Python 解释器退出；不打印堆栈回溯。

如果关联值是一个普通整数，则指定系统退出状态
(传递给 C 的 exit()函数)；如果没有，则退出状态为零；
如果它有另一种类型(比如字符串)，对象的值被打印出来，
退出状态为 1。

### TypeError

当操作或函数应用于不适当的
类型的对象时引发。

关联的值是一个字符串，给出关于类型不匹配的详细信息。

### unboundlocalrerror

当引用函数或方法中的局部变量，但没有值绑定到该变量时引发。

### UnicodeDecodeError

当发生与 Unicode 相关的编码或解码错误时引发。

它是 ValueError 的子类。

### unicode encoded error

当编码过程中出现与 Unicode 相关的错误时引发。

它是 UnicodeError 的子类。

### UnicodeError

当解码过程中出现与 Unicode 相关的错误时引发。

它是 UnicodeError 的子类。

### UnicodeTranslateError

当翻译过程中出现与 Unicode 相关的错误时引发。

它是 UnicodeError 的子类。

### ValueError

当内置操作或函数接收到一个类型为
但值不合适的参数，并且这种情况不是由
更精确的异常(如 IndexError)描述时引发。

### WindowsError

当发生特定于 Windows 的错误或错误号与错误值不对应时引发。

### 零除法错误

当除法或模运算的第二个参数为零时引发。

相关值是一个字符串，指示操作数和
操作的类型。