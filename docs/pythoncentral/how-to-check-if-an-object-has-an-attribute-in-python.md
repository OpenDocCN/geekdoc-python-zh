# 如何在 Python 中检查对象是否有属性

> 原文：<https://www.pythoncentral.io/how-to-check-if-an-object-has-an-attribute-in-python/>

因为 Python 中的一切都是对象，而对象都有属性(字段和方法)，所以编写程序来检查对象有什么样的属性是很自然的。例如，一个 Python 程序可以在服务器上打开一个套接字，它接受从客户机通过网络发送的 Python 脚本。在接收到新脚本时，服务器端 Python 程序可以检查或更精确地内省新脚本中的对象、模块和函数，以决定如何执行函数、记录结果和各种有用的任务。

## **hasattr vs. try-except**

有两种方法可以检查 Python 对象是否有属性。第一种方法是调用内置函数`hasattr(object, name)`，如果字符串`name`是`object`的一个属性的名称，则返回`True`，否则返回`False`。第二种方式是通过`try`来访问`object`中的属性，并在`AttributeError`被触发时执行一些其他功能。

```py

>>> hasattr('abc', 'upper')

True

>>> hasattr('abc', 'lower')

True

>>> hasattr('abc', 'convert')

False

```

并且:

```py

>>> try:

... 'abc'.upper()

... except AttributeError:

... print("abc does not have attribute 'upper'")

...

'ABC'

>>> try:

... 'abc'.convert()

... except AttributeError:

... print("abc does not have attribute 'convert'")

...

abc does not have attribute 'convert'

```

这两种风格有什么区别？`hasattr`通常被称为“三思而后行”(LBYL)的 Python 编程风格，因为在访问一个对象之前，你会检查它是否有属性。而`try-except`被称为“请求原谅比请求允许容易”(EAFP)，因为你`try`属性先访问，在`except`块请求原谅，而不是像`hasattr`那样请求允许。

那么，哪种方式更好呢？嗯，这两种学说都有忠实的支持者，而且这两种风格似乎都精通于处理任何现实世界的编程挑战。有时，如果您希望确保某个属性确实存在，并在不存在时停止执行，那么使用 LBYL 是有意义的。例如，在程序的某一点上，您肯定知道传入的`object`应该有一个有效的文件指针属性，下面的代码可以在这个属性上工作。另一方面，如果您知道某个属性在程序执行的某个时候可能不存在，那么使用 EAFP 也是有意义的。例如，音乐播放器不能保证 MP3 文件总是在磁盘上的同一位置，因为它可能会被用户随时删除、修改或移动。在这种情况下，音乐播放器可以先`try`访问 MP3 文件，并通知用户该文件不存在于`except`块中。

## **hasattr vs __dict__**

虽然`hasattr`是一个内置函数，它被设计用来检查一个属性是否存在于一个对象中，但有时检查一个对象的`__dict__`来检查一个属性的存在可能更准确，因为`hasattr`并不关心一个属性为什么被附加到一个对象上，而你可能想知道一个属性为什么被附加到一个对象上。例如，属性可能由于其父类而不是对象本身而被附加到对象。

```py

>>> class A(object):

... foo = 1

...

>>> class B(A):

... pass

...

>>> b = B()

>>> hasattr(b, 'foo')

True

>>> 'foo' in b.__dict__

False

```

在前面的代码中，由于类`B`是类`A`的子类，类`B`也有属性“foo”。但是，因为“foo”是从`A`继承而来的，`B.__dict__`并不包含它。有时，知道一个属性是来自对象本身的类还是来自对象的超类可能是至关重要的。

### **提示和建议**

*   `hasattr` follows the duck-typing principle in Python:

    > 当我看到一只像鸭子一样走路、像鸭子一样游泳、像鸭子一样嘎嘎叫的鸟时，我就把那只鸟叫做鸭子。

    所以，大多数时候你要用`hasattr`来检查一个属性是否存在于一个对象中。

*   `try-except`和`__dict__`有它们自己的用例，与`hasattr`相比，这些用例实际上非常狭窄。因此，将这些特殊的用例记在脑子里是有益的，这样你就可以在编码时识别它们，并相应地使用正确的习惯用法。

一旦你学会了如何在 Python 中检查一个对象是否有属性，看看如何从一个对象中获取属性。