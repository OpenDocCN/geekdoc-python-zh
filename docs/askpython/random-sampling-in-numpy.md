# 在 NumPy 中执行随机采样的 4 种方法

> 原文：<https://www.askpython.com/python/random-sampling-in-numpy>

读者朋友们，你们好！在本文中，我们将关注在 Python [NumPy](https://www.askpython.com/python-modules/numpy/python-numpy-arrays) 中执行随机采样的 **4 种简单方法。**

所以，让我们开始吧！🙂

概括地说，随机抽样实际上是从已定义的数据类型中选择随机值，并将其呈现以备将来使用。

在本主题的课程中，我们将了解以下功能

1.  **NumPy random_sample()方法**
2.  **NumPy ranf()方法**
3.  **NumPy random_integers()方法**
4.  **NumPy randint()方法**

* * *

## 1。用于随机抽样的 NumPy random_sample()方法

使用 **random_sample()方法**，我们可以对数据值进行采样，并轻松选择随机数据。它仅选择[0.0–1.0]之间的随机样本。我们可以基于随机值构建单个样本以及整个数组。

看看下面的语法！

```py
random.random_sample()

```

**举例:**

在下面的例子中，首先，我们执行了随机采样并生成了一个随机值。此外，我们通过将 size 作为参数传递给 random_sample()函数，创建了一个包含随机值的二维数组。

请注意，随机值的范围仅为 0.0 到 1.0。另外，random_sample()函数生成浮点类型的随机值。

```py
import numpy as np

ran_val = np.random.random_sample()
print ("Random value : ", ran_val)

ran_arr = np.random.random_sample(size =(2, 4))
print ("Array filled with random float values: ", ran_arr) 

```

**输出:**

```py
Random value :  0.3733413809567606
Array filled with random float values:  [[0.45421908 0.34993556 0.79641287 0.56985183]
                                        [0.88683577 0.91995939 0.16168328 0.35923753]]

```

* * *

## 2。random _ integers()函数

使用 **random_integers()函数**，我们可以生成随机值，甚至是整型随机值的多维数组。它会生成整型的随机值。此外，它给了我们选择整数值范围的自由，从中可以选择随机数。

**语法:**

```py
random_integers(low, high, size)

```

*   **low** :待选随机值的最低刻度/限值。随机值的值不会低于所提到的低值。
*   **高**:待选随机值的最高刻度/限值。随机值的值不会超过所提到的高值。
*   **size** :要形成的数组的行数和列数。

**举例:**

在本例中，我们创建了一个一维随机值数组，其值仅在范围 5-10 之间。此外，我们使用相同的概念建立了一个多维数组。

```py
import numpy as np

ran_val = np.random.random_integers(low = 5, high =10 , size = 3)
print ("Random value : ", ran_val)

ran_arr = np.random.random_integers(low = 5, high =10 , size = (2,4))
print ("Array filled with random float values: ", ran_arr) 

```

**输出:**

```py
Random value :  [10  5  9]
Array filled with random float values:  [[ 8  8  9  6]
                                        [ 6 10  8 10]]

```

* * *

## 3.randint()函数

**[randint()函数](https://www.askpython.com/python-modules/python-randint-method)** 的工作方式与 random_integers()函数类似。它创建一个数组，其中包含指定整数范围内的随机值。

**举例:**

```py
import numpy as np

ran_val = np.random.randint(low = 5, high =10 , size = 3)
print ("Random value : ", ran_val)

```

**输出:**

```py
Random value :  [5 8 9]

```

* * *

## 4.ranf()函数

同样， **ranf()函数**在功能上类似于 random_sample()方法。它只生成 0.0 到 1.0 之间的 float 类型的随机数。

**举例:**

```py
import numpy as np

ran_val = np.random.ranf()
print ("Random value : ", ran_val)

```

**输出:**

```py
Random value :  0.8328458165202546

```

* * *

## 结论

如果你遇到任何问题，欢迎在下面评论。更多与 Python 编程相关的帖子，敬请关注我们！在那之前，学习愉快！🙂