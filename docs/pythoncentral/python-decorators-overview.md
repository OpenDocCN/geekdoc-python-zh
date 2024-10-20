# Python 装饰者概述

> 原文：<https://www.pythoncentral.io/python-decorators-overview/>

Python 中的 Decorators 看起来很复杂，但是它们非常简单。你可能见过他们；它们是以'`@`'开头的函数定义前的奇数位，例如:

```py

def decorator(fn):

def inner(n):

return fn(n) + 1

return inner
@ decorator
def f(n):
return n+1

```

请注意名为 decorator 的函数；它将一个函数作为参数，定义并返回一个新函数，该函数使用传递给它的函数。几乎所有的装饰者都有这种模式。`@decorator`符号只是一种调用现有函数的特殊语法，将新函数作为参数传递，并使用返回值替换新函数。

在上面的例子中，用`f`作为参数调用 decorator，它返回一个新函数来代替`f`。同样的效果可能不那么简洁地写出来:

```py

def decorator(fn):

def inner(n):

return fn(n) + 1

return inner
def f(n): 
返回 n + 1
f =装饰者(f) 

```

所有的`@`符号都是为了使语法更加简洁。

## 查看工作中的 Python 装饰者

下面是一个简单的例子，它的输出说明了装饰器是如何工作的。

```py

def wrap_with_prints(fn):

# This will only happen when a function decorated

# with @wrap_with_prints is defined

print('wrap_with_prints runs only once')
def wrapped(): 
 #每次在
之前都会发生这种情况#被修饰的函数被调用
 print('即将运行%s' % fn。__name__) 
 #这里是包装器调用装饰函数
 fn() 
 #这将在每次
 #装饰函数被调用
 print('Done running %s' % fn。__name__)
包装退货
@ wrap _ with _ prints
def func _ to _ decorate():
print('运行被修饰的函数。')
func _ to _ decoration()

```

运行该示例时，输出将如下所示:

```py

wrap_with_prints runs only once

About to run func_to_decorate

Running the function that was decorated.

Done running func_to_decorate

```

注意，装饰器(`wrap_with_prints`)只在创建装饰函数时运行一次，但是内部函数(`wrapped`)会在每次运行`func_to_decorate`时运行。

# 一个几乎实际的例子

Python 的函数就像任何其他 Python 对象一样；您可以将它们赋给变量，在函数调用中将它们作为参数传递，从其他函数返回它们，将它们放入列表和字典中，等等。(大家说 Python 有一级函数就是这个意思。)装饰器是利用这一事实来提供有用功能的一种简洁方式。

例如，假设我们有以下代码(您会将其识别为 [fizzbuzz](http://www.codinghorror.com/blog/2007/02/why-cant-programmers-program.html "fizzbuzz") ):

```py

def fizz_buzz_or_number(i):

''' Return "fizz" if i is divisible by 3, "buzz" if by

5, and "fizzbuzz" if both; otherwise, return i. '''

if i % 15 == 0:

return 'fizzbuzz'

elif i % 3 == 0:

return 'fizz'

elif i % 5 == 0:

return 'buzz'

else:

return i
对于范围(1，31)中的 I:
print(fizz _ buzz _ or _ number(I))

```

然后，假设我们想要记录函数的参数和返回值以便调试。(有更好的调试方法，您应该使用它们，但这对我们来说是一个有用的例子。)我们可以在函数中添加日志记录语句，但是如果我们有更多的函数，那就太麻烦了，而且更改正在调试的函数很可能会引入其他错误，这也是必须要调试的。我们需要一个保持函数完整的通用方法。

输入装饰师来拯救这一天！我们可以编写一个装饰函数来为我们记录日志:

```py

def log_calls(fn):

''' Wraps fn in a function named "inner" that writes

the arguments and return value to logfile.log '''

def inner(*args, **kwargs):

# Call the function with the received arguments and

# keyword arguments, storing the return value

out = apply(fn, args, kwargs)
#在日志文件
中写入一行函数名、其
 #参数和返回值，并打开(' logfile.log '，' a ')作为日志文件:
 logfile.write( 
')用 args %s 和 kwargs %s 调用%s，返回% s \ n“%
(fn。__name__，args，kwargs，out))
#返回返回值
返回出
返回内

```

然后，我们需要做的就是将我们的装饰器添加到`fizz_buzz_or_number`的定义中:

```py

@log_calls

def fizz_buzz_or_number(i):

# Do something

```

运行该程序将生成如下所示的日志文件:

```py

fizz_buzz_or_number called with args (1,) and kwargs {}, returning 1

fizz_buzz_or_number called with args (2,) and kwargs {}, returning 2

fizz_buzz_or_number called with args (3,) and kwargs {}, returning fizz

fizz_buzz_or_number called with args (4,) and kwargs {}, returning 4

fizz_buzz_or_number called with args (5,) and kwargs {}, returning buzz

# Do something

fizz_buzz_or_number called with args (28,) and kwargs {}, returning 28

fizz_buzz_or_number called with args (29,) and kwargs {}, returning 29

fizz_buzz_or_number called with args (30,) and kwargs {}, returning fizzbuzz

```

同样，有更好的调试方法，但这是装饰器可能用途的一个很好的例子。

# 更高级的用途

上面的装饰者是简单的例子；还有更高级的可能性。

## 多个装饰者

你可以连锁装修。例如，您可以使用 decorators 将函数返回的文本包装在 HTML 标签中——尽管我不建议这样做；请改用模板引擎。在任何情况下:

```py

def b(fn):

return lambda s: '<b>%s</b>' % fn(s)
def em(fn):
return lambda s:'<em>% s</em>' % fn(s)
@ b
@ em
def greet(name):
回‘你好，% s！’% name 

```

然后，调用`greet('world')`将返回:

```py

<b><em>Hello, world!</em></b>

```

请注意，装饰器是按照从下到上的顺序应用的；函数先被`em`包装，然后结果被`b`包装。没有按照正确的顺序放置装饰器会导致混乱的、难以追踪的错误。

## 有争论的装饰者

有时，您可能希望将参数与要装饰的新函数一起传递给装饰函数。例如，您可能想要将前一个示例中的 HTML 标记包装 decorators b 和 em 抽象成一个通用的`tag_wrap` decorator，它将标记作为一个额外的参数，这样我们就不必为每个 HTML 标记都有单独的 decorator。可惜，你做不到。然而，你可以做一些看起来像是给装饰器传递参数的事情，并且有类似的效果:你可以写一个接受参数的函数，然后*返回*一个生成器函数:

```py

def tag_wrap(tag):

def decorator(fn):

def inner(s):

return '<%s>%s' % (fn(s), tag)

return inner

return decorator
@ tag _ wrap(' b ')
@ tag _ wrap(' em ')
def greet(name):
回' Hello，%s！'% name
打印(问候('世界'))

```

这个例子的结果与上一个例子的结果相同。

让我们从外向内回顾一下`tag_wrap`做了什么。`tag_wrap`是一个接受标签并返回接受函数的函数装饰器的函数。当传递一个函数时，decorator 返回一个名为`inner`的函数，该函数接受一个字符串，将该字符串传递给传递给 decorator 的函数，并将结果包装在传递给`tag_wrap`的任何标签中。

这是一系列令人难以理解的定义，但是从一个库作者的角度来考虑它:对于不得不这么做的人来说这很复杂，但是对于包含它的库的用户来说，这非常简单。

# 现实世界中的装饰者

许多库的作者利用 decorators 来为库用户提供简单的方法，将复杂的功能添加到他们的代码中。例如，web 框架 [Django](https://www.djangoproject.com/ "Django Project") 使用装饰器`login_required`，使视图需要用户认证——这无疑是以如此复杂的方式扩展功能的一种便捷方式。

另一个 Python web 框架 [CherryPy](http://www.cherrypy.org/ "CherryPy Project") 更进一步，它的工具处理认证、静态目录、错误处理、缓存等等。 [CherryPy](http://www.cherrypy.org/ "CherryPy") 提供的每一个工具都可以用一个配置文件来配置，通过直接调用工具，或者通过使用 decorators，其中一些“接受参数”，比如:

```py

@tools.staticdir(root='/path/to/app', dir='static')

def page(self):

# Do something

```

它定义了一个名为“static”的静态目录，位于“/path/to/app”。

虽然这些 decorator 的设计是为了简化它们的使用，即使对于那些不完全理解它们如何工作的人来说也是如此，但是对 decorator 的全面理解将允许您更灵活、更智能地使用它们，并在适当的时候创建自己的 decorator，从而节省您的开发时间和精力，并简化维护。