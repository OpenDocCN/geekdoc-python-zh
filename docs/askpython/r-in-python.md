# 在 Python 中使用 R

> 原文：<https://www.askpython.com/python/examples/r-in-python>

**让我们学习在 Python 中使用 R**。纵观历史，在数据科学世界中，两种语言总是在数据分析方面竞争，以求超越自己。

这两个都是，R 和 Python。

这两种编程语言都有自己的粉丝群，而且各有各的优势。

R 为统计分析提供了更大的支持，并在其中实现了专门化，而 Python 提供了面向对象的方法以及与其他模块的大量集成。

Python 和 R 的优点和缺点结合在一起会成为一个强大的组合。因为 Python 缺乏的地方，R 压倒了，反之亦然。

因此，开发人员创建了`rpy2`库，这也是我们今天的主题。

这对需要两者结合的开发者意味着什么？*一个机会*。

## 安装 rpy2 模块

**开始的先决条件是`rpy2`模块只有在您已经安装了所需的 R 版本的情况下才能工作。**

和其他模块一样，`rpy2`模块需要通过 Python 发行版的 [pip 包安装程序](https://www.askpython.com/python-modules/python-pip)进行安装。

在 pip 中，安装 *rpy2* 的命令很简单，

```py
pip install rpy2

```

这应该会自动安装所需的模块，我们可以继续在 Python 脚本中使用它！

*如果你想在体验~~脚~~系统之前测试一下 rpy2 的功能，你可以先试着使用 docker 镜像，看看 rpy2 的 [docker hub](https://hub.docker.com/r/rpy2/rpy2) 。*

## 在 Python 中使用 R 和 rpy2 模块

为了在 Python 中使用 R，我们首先将 rpy2 导入到代码中。

```py
import rpy2
from rpy2 import robjects

```

现在，我们可以开始使用 Python 中的 R 了。但是，在您开始使用这两种语言之前，了解一下在本模块中 R 语言的使用上的细微差别会很有帮助。

### 1.通过 rpy2 导入包

使用 R 的大量工作都与导入包进行数据分析有关。而`rpy2`通过`py2.robjects.packages.importr()`函数给我们提供了这个。

这个函数作为一种方法，将为 R 设计的包导入 Python，我们可以使用它们来基本上拥有脚本中存在的两种语言的特性。

```py
from rpy2.robjects.packages import importr
# imports the base module for R.
base = importr("base")

# imports the utils package for R.
utils = importr("utils")

```

我们现在可以使用通过这个方法导入的函数了。

### 2.在 Python 中使用 R

在脚本中使用 R 的方法是使用`robjects.r`实例，它允许我们本质上使用 R 控制台。

如果你想知道这到底是如何工作的，那是因为`rpy2`模块正在后台运行一个嵌入式 R。

```py
# Essentially retrieving the value of pi in the R console
pi = robjects.r['pi']
print(pi[0])

# Output : 3.14159265358979

```

虽然这种方法可能适用于单行代码。值得一提的是，如果我们希望处理大量需要用 r 处理的代码，这不是一个可行的方法。

幸运的是，我们可以在*三个引号*中输入一整块代码。

```py
robjects.r('''
        # create a function `f`
        f <- function(r, verbose=FALSE) {
            if (verbose) {
                cat("I am calling f().\n")
            }
            2 * pi * r
        }
        # call the function `f` with argument value 3
        f(3)
        ''')
# The result of the function is returned to the Python Environment

```

函数本身仍然存在于 R 全局环境中，但是可以用命令`robjects.globalenv['f']`访问，其中 *f* 是我们的 R 环境中的变量。

`rpy2`模块为我们提供了很多功能，虽然一开始看起来有点难，但它主要只是提到 R 环境的语法。

这里有几个使用 R 中不同特性的例子！

```py
# Working with different kinds of vectors
res1 = robjects.StrVector(['abc', 'def'])
res2 = robjects.IntVector([1, 2, 3])
res3 = robjects.FloatVector([1.1, 2.2, 3.3])

print(res1.r_repr())
# Output : c("abc", "def")

print(res2.r_repr())
# Output : 1:3

print(res3.r_repr())
# Output : c(1.1, 2.2, 3.3)

# Working with different functions of R
rsort = robjects.r['sort']
res4 = rsort(robjects.IntVector([1,2,3]), decreasing=True)
print(res4.r_repr())

# Working with matrices in R
v = robjects.FloatVector([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
m = robjects.r['matrix'](v, nrow = 2)
print(m)
# Output :
#       [,1] [,2] [,3]
# [1,]  1.1  3.3  5.5
# [2,]  2.2  4.4  6.6

# Working with Graphics in R
r = robjects.r

x = robjects.IntVector(range(10))
y = r.rnorm(10)

r.X11()

r.layout(r.matrix(robjects.IntVector([1,2,3,2]), nrow=2, ncol=2))
r.plot(r.runif(10), y, xlab="runif", ylab="foo/bar", col="red")

```

### 3.走向

使用两种不同的语言来解决问题的实现打开了许多新发现的大门。

继续使用 Python 中的 R 将会使用 Python 提供的功能来处理各种不同的模块，并扩展数据科学和数学逻辑领域的功能。

将 [Pandas](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial) 、 [OpenCV](https://www.askpython.com/python-modules/read-images-in-python-opencv) 和 *Scikit-Learn* 集成到程序中是值得研究的，以便扩展和测试新的想法，而没有语言提供的任何障碍。

如果您发现自己不清楚某个特性是否可以在`rpy2`模块中使用，请随意浏览他们维护良好的[文档！](https://rpy2.github.io/doc/v3.3.x/html/introduction.html#the-r-instance)

## *rpy2* 的替代品

虽然`rpy2`是一个很棒的模块，但你可能希望看看其他模块，以便找出哪一个最适合你。

因此，这里有一个列表来帮助你确定你需要哪个模块，而不是为`rpy2`中不存在或不适合你的特性寻找变通办法。

*   [字母“T1”](https://www.rdocumentation.org/packages/rJython/versions/0.0-4)
*   大蟒
*   [SnakeCharmR](https://github.com/asieira/SnakeCharmR)
*   [PythonInR](https://cran.r-project.org/web/packages/PythonInR/index.html)
*   [网状](https://rstudio.github.io/reticulate/)

## 结论

现在您已经知道了`rpy2`模块提供了什么，以及如何设置它来开始处理您的代码，您可以开始计算，而不用担心 R 和 Python 之间的冲突。

毕竟，他们现在都站在你这边！

在你的数学和数据科学之旅中，看看我们关于[熊猫](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)和 [matplotlib](https://www.askpython.com/python-modules/matplotlib/python-matplotlib) 的其他模块的其他作品。

## 参考

*   [官方 rpy2 文档](https://rpy2.github.io/doc/latest/html/index.html)
*   [Reticulate 的 GitHub](https://github.com/rstudio/reticulate)
*   [Quora:可以用 Python 运行 R 吗？](https://www.quora.com/Can-I-run-R-in-Python)