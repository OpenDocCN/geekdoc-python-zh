# 用 Decorators 验证 Python 函数参数和返回类型

> 原文：<https://www.pythoncentral.io/validate-python-function-parameters-and-return-types-with-decorators/>

## 概观

所以前几天我和 [Python decorators](https://www.pythoncentral.io/python-decorators-overview/ "Python Decorators") 一起玩(和你一样)。我一直想知道是否可以让 Python 验证函数参数类型和/或返回类型，就像静态语言一样。有些人会说这很有用，而其他人会说由于 Python 的动态特性，这从来都不是必需的。我只是把它破解了一下，看看这是否可行，并认为看看你能用 Python 做什么是非常酷的。所以我将把政治排除在外:-)。欢迎在评论中发表你的想法。

本文假设您至少对装饰者及其工作方式有基本的了解。

下面的代码验证了顶级函数参数和返回类型的`type`。然而，它不会查看数据结构下的类型。例如，您可以指定第二个参数必须是一个元组。但是，您不能验证子值，例如:`(<type 'int'>, <type 'str'>)`。如果有人能成功，我倒要看看！

所以让我们开始吧！

## 验证程序异常

在我们开始用实际代码验证函数参数并返回`type`之前，我们将添加一些自定义异常。请注意，这不是必需的，我只是希望它能够抽象出一些消息，并创建更干净的代码。我们将创建以下例外:

*   `ArgumentValidationError`:当一个函数的参数的`type`不是它应该的时候。
*   `InvalidArgumentNumberError`:提供给函数的参数个数不正确。
*   `InvalidReturnType`:返回值错误时`type`。

每个异常都有一个定制的错误消息，所以消息的数据被传递给构造函数方法:`__init__()`。让我们看一下代码:

```py

class ArgumentValidationError(ValueError):

    '''

    Raised when the type of an argument to a function is not what it should be.

    '''

    def __init__(self, arg_num, func_name, accepted_arg_type):

        self.error = 'The {0} argument of {1}() is not a {2}'.format(arg_num,

                                                                     func_name,

                                                                     accepted_arg_type)
def __str__(self): 
返回 self.error
class InvalidArgumentNumberError(value error):
' ' '
当提供给函数的参数数量不正确时引发。
注意，这个检查只从 validate_accept()装饰器中指定的参数数量
开始执行。如果 validate_accept() 
调用不正确，那么这个
可能会报告一个错误的验证。
 ''' 
 def __init__(self，func_name): 
 self.error = '无效的{0}()'参数个数。格式(函数名)
def __str__(self): 
返回 self.error
class invalid returntype(value error):
' ' '
顾名思义，返回值是错误的类型。
 ''' 
 def __init__(self，return_type，func_name): 
 self.error = '对于{1}()'无效的返回类型{0} '。格式(返回类型，
函数名)
def __str__(self): 
返回 self.error 

```

相当直接。注意，`ArgumentValidationError`异常需要参数`arg_num`，以指定第 n 个具有错误`type`的参数。这需要一个序数:如第一、第二、第三等。所以我们的下一步是创建一个简单的函数来将一个`int`转换成一个序数。

## 序数转换器

```py

def ordinal(num):

    '''

    Returns the ordinal number of a given integer, as a string.

    eg. 1 -> 1st, 2 -> 2nd, 3 -> 3rd, etc.

    '''

    if 10 <= num % 100 < 20:
        return '{0}th'.format(num)
    else:
        ord = {1 : 'st', 2 : 'nd', 3 : 'rd'}.get(num % 10, 'th')
        return '{0}{1}'.format(num, ord)
[/python]

We've called it ordinal() just to keep things simple, where it takes one int argument. But how does it work? For numbers 6 - 20, the number suffix (in order from 6 - 20) is always "th". However outside of that range:

```

*   如果最后一个数字以 1 结尾，则后缀为“st”。
*   如果最后一个数字以 2 结尾，则后缀为“nd”。
*   如果最后一个数字以 3 结尾，则后缀为“rd”。
*   否则(当最后一个数字> = 4 时)，后缀将为“th”。

在本例中，我们使用了一个`%`字符，也就是`modulus`操作符。当除以一个数时，它提供给我们余数。在这种情况下，如果我们将一个数除以 10，就会得到最后一位数字。注意，我们也可以使用`str(num)[-1]`。然而，这是低效的，丑陋的。

现在，到实际的验证器了！

## 功能参数验证

我们将从向函数参数验证器(decorator)展示代码开始，并从那里开始。这两个验证函数非常相似，但是有一点不同。

```py

def accepts(*accepted_arg_types):

    '''

    A decorator to validate the parameter types of a given function.

    It is passed a tuple of types. eg. (, <type>)</type>
注意:它不做深度检查，例如检查类型的
元组。传递的参数只能是类型。
' ' '
def accept _ decorator(validate _ function):
#检查验证器
 #函数的参数个数是否与提供给实际函数
 #进行验证的参数个数相同。我们不需要
 #来检查要验证的函数是否有正确的
 #数量的参数，因为 Python 会自动完成这个
 #(也有一个类型错误)。
@ func tools . wraps(validate _ function)
def decorator _ wrapper(* function _ args，**function_args_dict): 
如果 len(accepted_arg_types)不是 len(accepted_arg_types): 
抛出 InvalidArgumentNumberError(validate _ function。__name__)
#我们使用 enumerate 来获取索引，因此我们可以将类型不正确的
 #参数传递给 ArgumentValidationError。
 for arg_num，(actual_arg，accepted _ arg _ type)in enumerate(zip(function _ args，accepted _ arg _ types)):
if not type(actual _ arg)is accepted _ arg _ type:
ord _ num = ordinal(arg _ num+1)
raise ArgumentValidationError(ord _ num，
 validate_function。__name__，
 accepted_arg_type)
return validate _ function(* function _ args)
return decorator _ wrapper
return accept _ decorator

```

好吧！那到底是什么？注意，对于常规装饰器，有一个函数返回一个函数。通常，子函数会在调用函数包装之前和/或之前做一些事情。然而在我们的例子中，我们使用了*元装饰器*。这些将 decorator 模型向前推进了一步，它们本质上是一个带有子 decorator 的 decorator，子 decorator 接受被检查函数的函数参数。相当困惑！但是一旦你明白了，它们就很简单了。

我们使用的是`zip()`函数，它返回一个元组列表。它让我们同时遍历两个列表。在我们的例子中，允许的函数参数和实际的函数参数。然后`zip()`呼叫被`enumerate()`打包。`enumerate()`函数本身返回一个`enumerate`对象。每次调用`enumerate.next()`函数时(例如在我们的`for`语句中)，它都会返回一个`tuple`。这是`loop-index, (list-1-element, list-2-element)`的形式。这样我们就可以得到参数的索引，如果需要的话可以传递给异常。然后使用`isinstance()`函数来比较所需参数和实际参数的类型。你也可以使用`if type(accepted_arg) is type(actual_arg)`，但是我发现这种方式在语法上更简洁。

需要考虑的一点是:在我们的`add_nums()`函数被调用后，从技术上讲，装饰器返回的是值，而不是`add_nums()`。所以如果你看从`add_nums()`返回的对象，它实际上将是装饰者。因此，它将有一个不同的名称，你会失去你的文件字符串。这也许没什么大不了的，但是很容易解决。幸运的是`functools.wraps()`函数来帮忙了，它本身就是一个装饰器。要使用它，您必须导入`functools`模块，并在您的验证装饰器上添加装饰器。

## 返回类型验证

现在进入返回`type`验证。它在这里，在所有的荣耀中。

```py

def returns(*accepted_return_type_tuple):

    '''

    Validates the return type. Since there's only ever one

    return type, this makes life simpler. Along with the

    accepts() decorator, this also only does a check for

    the top argument. For example you couldn't check

    (, <type>, <type>).

    In that case you could only check if it was a tuple.

    '''

    def return_decorator(validate_function):

        # No return type has been specified.

        if len(accepted_return_type_tuple) == 0:

            raise TypeError('You must specify a return type.')</type></type>
@ func tools . wrapps(validate _ function)
def decorator _ wrapper(* function _ args):
#指定了多个返回类型。
if len(accepted _ return _ type _ tuple)>1:
raise type error('您必须指定一个返回类型。')
#因为装饰器接收到一组参数
 #并且只返回一个对象，所以我们只需要
 #获取第一个参数。
accepted _ return _ type = accepted _ return _ type _ tuple[0]
#我们将执行函数，
 #看看返回类型。
return _ value = validate _ function(* function _ args)
return _ value _ type = type(返回值)
如果 return_value_type 不被接受 _return_type: 
抛出 invalid return type(return _ value _ type，
 validate_function。__name__)
return 返回值
return 装饰器 _ 包装器
 return 装饰器

```

可以看到，它类似于参数验证函数(`validate_accept`)。然而，由于它不需要迭代所有的值，我们只需要检查类型是否相同。

## 我们的功能是验证

我决定使用一个非常简单的示例函数。因为最终，不管它有多复杂，它还是会以同样的方式工作。没必要把事情弄得比应该的更复杂！已经实现了两个。一个带有名为`add_nums_correct`的正确`return type`，另一个返回名为`add_nums_incorrect`的`str`。这样我们就可以测试`return`验证器是否工作。

```py

@accepts(int, int)

@returns(int)

def add_nums_correct(a, b):

    '''

    Adds two numbers. It accepts two

    integers, and returns an integer.

    '''
返回 a + b
@accepts(int，int)
@ returns(int)
def add _ nums _ incorrect(a，b): 
 ''' 
将两个数相加。它接受两个
整数，并返回一个整数。
' ' '
返回“Not an int！”

```

所以让我们检查一下，看看它是否有效。

```py

>>> # All good.

>>> print(add_nums_correct(1, 2))

3

>>> # Incorrect argument type (first).

>>> add_nums_correct('foo', 5)

Traceback (most recent call last):

  File "Validate Function Parameter and Return Types with Decorators.py", line 196, in 

    add_nums_correct('foo', 5)

  File "Validate Function Parameter and Return Types with Decorators.py", line 120, in decorator_wrapper

    accepted_arg_type)

__main__.ArgumentValidationError: The 1st argument of add_nums_correct() is not a <type>

>>>

>>> # Incorrect argument type (second).

>>> add_nums_correct(5, 'bar')

Traceback (most recent call last):

  File "Validate Function Parameter and Return Types with Decorators.py", line 200, in <module>

    add_nums_correct(5, 'bar')

  File "Validate Function Parameter and Return Types with Decorators.py", line 120, in decorator_wrapper

    accepted_arg_type)

__main__.ArgumentValidationError: The 2nd argument of add_nums_correct() is not a <type>

>>>

>>> # Incorrect argument number.

>>> add_nums_correct(1, 2, 3, 4)


```