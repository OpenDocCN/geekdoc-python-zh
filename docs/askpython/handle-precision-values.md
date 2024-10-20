# Python 中处理精度值的 5 种方法

> 原文：<https://www.askpython.com/python/examples/handle-precision-values>

读者朋友们，你们好！在本文中，我们将关注 Python 中处理精度值的 **5 种方法。所以，让我们开始吧！**

* * *

## “处理精度值”是什么意思？

无论是哪种编程语言，我们经常会遇到这样的情况:在任何应用程序或操作步骤中处理整数或数字数据。在同样的过程中，我们找到了十进制值的数据。这就是我们需要处理精度值的时候。

Python 为我们提供了各种函数来处理整数数据的精度值。它让我们可以根据十进制值在数字中的位置，选择是排除小数点，还是自定义值。

## Python 中如何处理精度值？

让我们看看 Python 为处理代码输出中的精度数据和小数提供的不同函数。

1.  **Python %运算符**
2.  **Python 格式()函数**
3.  **Python round()函数**
4.  **trunc()函数**
5.  **数学 ceil() & floor()函数**

现在，让我们在接下来的部分中逐一了解一下。

* * *

### 1。Python %运算符

使用 ['% '操作符](https://www.askpython.com/python/python-modulo-operator-math-fmod)，我们可以格式化数字以及设置相同的精度限制。这样，我们可以定制要包含在结果数中的精度点的限制。

看看下面的语法！

**语法:**

```py
'%.point'%number

```

*   点:它是指我们希望在数字的小数点后有多少个点。
*   number:要处理的整数值。

* * *

### 2。Python format()函数

像%操作符一样，我们也可以使用 [format()函数](https://www.askpython.com/python/string/python-format-function)，它帮助我们设置精度值的限制。使用 format()函数，我们将数据格式化为一个字符串，并设置数字小数部分后包含的点数的限制。

**语法:**

```py
print ("{0:.pointf}".format(number)) 

```

* * *

### 3。Python round()函数

使用 Python round()函数，我们可以提取并以自定义格式显示整数值，也就是说，我们可以选择小数点后要显示的位数，作为精度处理的检查。

**语法:**

```py
round(number, point)

```

* * *

## 在 Python 中实现精度处理

在下面的例子中，我们利用了上面解释的 3 个函数来处理 Python 中的精度值。我们在这里试图通过将小数点后显示的数字设置为 4 来处理精度。

```py
num = 12.23456801

# using "%" operator
print ('%.4f'%num) 

# using format() function
print ("{0:.4f}".format(num)) 

# using round() function
print (round(num,4)) 

```

**输出:**

```py
12.2346
12.2346
12.2346

```

* * *

## Python 数学函数处理精度

除了上述函数，python 还为我们提供了[数学模块](https://www.askpython.com/python-modules/python-math-module)，它包含了一组处理精度值的函数。

Python 数学模块具有以下一组处理精度值的函数——

1.  trunc()函数
2.  Python ceil()和 floor()函数

让我们一个一个来看看。

* * *

### 4。Python trunc()函数

使用 trunc()函数，小数点后的所有数字都被终止。也就是说，它只返回小数点前面的数字。

看看下面的语法。

**语法:**

```py
import math
math.trunc(number)

```

**举例:**

```py
import math
num = 12.23456801

# using trunc() function
print (math.trunc(num)) 

```

**输出:**

```py
12

```

* * *

### 5.Python ceil()和 floor()函数

使用 ceil()和 floor()函数，我们可以将十进制数四舍五入到最接近的高值或低值。

ceil()函数将十进制数四舍五入到其后最接近的大值。另一方面，floor()函数将该值舍入到它前面最近的低值。

**举例:**

```py
import math
num = 12.23456801

# using ceil() function
print (math.ceil(num)) 

# using floor() function
print (math.floor(num)) 

```

**输出:**

```py
13
12

```

* * *

## 结论

如果你遇到任何问题，欢迎在下面评论。更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂