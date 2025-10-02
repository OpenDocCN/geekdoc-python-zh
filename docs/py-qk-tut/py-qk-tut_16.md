## Python 进阶 07 函数对象

[`www.cnblogs.com/vamei/archive/2012/07/10/2582772.html`](http://www.cnblogs.com/vamei/archive/2012/07/10/2582772.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

秉承着一切皆对象的理念，我们再次回头来看函数(function)这一结构。函数实际上也是一个对象。既然是一个对象，它也具有属性（可以使用 dir()查询）。作为对象，它还可以赋值给其它变量名，或者作为参数传递给其它函数使用。

1\. lambda
在展开之前，我们先提一下 lambda。lambda 是一种简便的，在同一行中定义函数的方法，其功能可以完全由 def 定义实现。lambda 例子如下：

```py
func = lambda x,y: x + y
print func(3,4)

```

lambda 以及之后的内容实际上生成一个函数对象(也就是函数)。该函数参数为 x,y，返回值为 x+y。该函数对象赋值给函数名 func。func 的调用与正常函数无异。

以上定义完全可以写成以下形式：

```py
def func(x, y): return x + y

```

2\. 函数可以作为参数传递

函数可以作为一个对象进行参数传递。函数名(比如 func)即指向该对象，不需要括号。比如说:

```py
def test(f, a, b): print 'test'
    print f(a, b)

test(func, 3, 5)

```

我们可以看到，test 函数的第一个参数 f 就是一个函数对象。我们将 func 传递给 f，那么 test 中的 f()所做的实际上就是 func()所实现的功能。

这样，我们就大大提高了程序的灵活性。假设我们有另一个函数取代 func，就可以使用相同的 test 函数了。如下:

```py
test((lambda x,y: x**2 + y), 6, 9)

```

思考这句程序的含义。

3\. map 函数

map()是 Python 的内置函数，它的第一个参数是一个函数对象。

```py
re = map((lambda x: x+3),[1,3,5,6])

```

这里，map()有两个参数，一个是 lambda 所定义的函数对象，一个是包含有多个元素的表。map()的功能是将函数对象依次作用于表的每一个元素，每次作用的结果储存于返回的表 re 中。map 通过读入的函数(这里是 lambda 函数)来操作数据（这里“数据”是表中的每一个元素，“操作”是对每个数据加 3）。

(注意，在 Python 3.X 中，map()将每次作用结果 yield 出来，形成一个循环对象。可以利用 list()函数，将该循环对象转换成表)

如果作为参数的函数对象有多个参数，可如下例：

```py
re = map((lambda x,y: x+y),[1,2,3],[6,7,9])

```

map()将每次从两个表中分别取出一个元素，带入 lambda 所定义的函数。

（本小节所使用的 lambda 也完全可以是 def 定义的更复杂的函数）

4\. filter 函数

filter 函数与 map 函数类似，也是将作为参数的函数对象作用于表的各个元素。如果函数对象返回的是 True，则该次的元素被储存于返回的表中。filter 通过读入的函数来筛选数据。(同样，在 Python 3.X 中，filter 返回的不是表，而是循环对象。)

filter 函数的使用如下例:

```py
def func(a): if a > 100: return True else: return False print filter(func,[10,56,101,500])

```

5\. reduce 函数

reduce 函数的第一个参数也是函数，但有一个要求，就是这个函数自身能接收两个参数。reduce 可以累进地将函数作用于各个参数。如下例：

```py
print reduce((lambda x,y: x+y),[1,2,5,7,9])

```

reduce 的第一个参数是 lambda 函数，它接收两个参数 x,y, 返回 x+y。

reduce 将表中的前两个元素(1 和 2)传递给 lambda 函数，得到 3。该返回值(3)将作为 lambda 函数的第一个参数，而表中的下一个元素(5)作为 lambda 函数的第二个参数，进行下一次的对 lambda 函数的调用，得到 8。依次调用 lambda 函数，每次 lambda 函数的第一个参数是上一次运算结果，而第二个参数为表中的下一个元素，直到表中没有剩余元素。

上面例子，相当于(((1+2)+5)+7)+9

(根据 mmufhy 的提醒： reduce()函数在 3.0 里面不能直接用的，它被定义在了 functools 包里面，需要引入包，见评论区)

总结:

函数是一个对象

用 lambda 定义函数

map()

filter()

reduce()