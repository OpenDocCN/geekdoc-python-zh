# Python 中的优化——完整指南

> 原文：<https://www.askpython.com/python/examples/optimization-in-python>

在本文中，我们将学习优化问题以及如何在 Python 中解决它。优化的目的是在大量的备选方案中选择一个问题的最佳解决方案。

***也读:[如何用 Python 写 Android 应用？](https://www.askpython.com/python/examples/write-android-apps-in-python)***

## 优化问题

让我们看一个简单的优化案例。假设一家面包店每天生产 1000 包面包，每包包含 10 片面包。为了量化生产，每一批面包都是用精确数量的配料如小麦、酵母等来制作的。

在某个财政季度，公司决定在不降低面包质量或尺寸的情况下削减生产成本。管理层决定将其每个面包的对角线长度减少 1 英寸，这并不明显，但在大规模生产中具有广泛的意义。

所以现在，生产小尺寸面包所需的小麦和酵母的精确数量的要求使它成为一个优化问题。一个良好的优化结果可以削减投入成本，同时保持理想的面包大小。

像所有优化问题一样，这个有问题的任务需要一些对所有编程语言都类似的基本要素:

## 解决方案——你要提高的量。

在这个关键时刻，最重要的解决方案是尽可能地削减成本。你必须陈述一种方法，该方法在将解决方案保持在期望的限制下的同时，针对优化问题估计可行的结果。

计算可能解的方法被称为目标函数。在面包尺寸问题中，目标函数将告诉我们，当准备一批新的尺寸减小的面包时，需要多少小麦和酵母。

目标函数旨在为任何问题提供最大值(“最大”在这里意味着值根据问题的需要要么最高要么最低)，面包维度问题是最小化的，因此最终结果将为解决方案提供最大值，意味着最低值。

约束是对目标函数结果的限制，它依赖于问题的需要，这意味着，在需要最高/最低值的问题中，约束充当最终限制，解决方案不能跨越。

例如，制作一批面包所需的原材料的最低数量将作为一个约束，这意味着每批面包需要小麦和酵母的最低限制。最小化解决方案无法估计低于该阈值的结果。

可行的解决方案可以满足问题的所有要求，但不一定是最优的。确定目标和约束是解决优化问题的第一步。

### 使用 python 解决优化问题

让我们解决 Python 中的优化问题。主要有三种优化:

*   **线性优化**

这是从一组参数中寻找最佳可能解决方案的过程。

*   **整数优化**

当问题中涉及的参数多于一个并且涉及整数或布尔参数时，那么它就变成了可通过整数优化来解决的问题。

*   **约束优化**

如果问题涉及一个非常大的参数集，并且需要从这个大的约束集中找到解决方案，那么它就变成了一个约束优化问题。

下面是一个最大化问题的例子，它将通过使用整数优化来解决。

最大化问题是一种整数优化问题，其中为某些参数提供约束，并且通过将这些约束转换成线性方程然后求解它来计算可行解。我们将找出下面等式的可行解。

等式是:3a+6b+2c <= 50

4a- 6b + 8c <= 45

3a+b–5c < = 37

这里我们需要最大化 3*a + 2*b + 2*c

### 解决最大化问题的主要阶段:

**建立和解决问题的基本程序在每种语言中都是相同的:**

*   导入您需要的库。
*   做一个关于求解器的声明。
*   变量和参数声明。
*   标记将用于实现目标的方法。
*   调用求解器并输出结果。

#### 解决这个问题的基本步骤是:

#### 进口

```py
from ortools.linear_solver import pywraplp

```

#### 求解器的声明

```py
solver = pywraplp.Solver.CreateSolver('SCIP')

```

这是一种使用工具计算问题的方法。

SCIP:它是用于解决混合非线性问题的工具箱或工具的参数。

Pywraplp:因为 ortools 是基于 c++的，所以它需要一个包装器才能在 python 上工作。Pywraplp 就是那个包装器。

#### 定义变量和约束

```py
# a, b, and c are non-negative integer variables.

a = solver.IntVar(0.0, solver.infinity(), 'a')

b = solver.IntVar(0.0, solver.infinity(), 'b')

c = solver.IntVar(0.0, solver.infinity(), 'c')

```

**约束将根据等式定义。例如，第一个等式 3a+6b+2c < = 50 将被定义为:**

```py
cons_in1 = solver.Constraint(-solver.infinity(), 50)

cons_in1.SetCoefficient(vara, 3)

cons_in1.SetCoefficient(varb, 6)

cons_in1.SetCoefficient(varc, 2)

```

## 目标函数:

我们需要最大化的等式是 3*a + 2*b + 2*c。下面的代码显示了为该等式创建目标函数的步骤。

```py
obj_prog = solver.Objective()

obj_prog.SetCoefficient(vara, 3)

obj_prog.SetCoefficient(varb, 2)

obj_prog.SetCoefficient(varc, 2)

obj_prog.SetMaximization()

```

## 调用求解器并打印最终结果

```py
solver.Solve()

# Print segment of program

print('Highest objective function value = %d' % solver.Objective().Value())

print()

for variable in [vara, varb, varc]:

    print('%s = %d' % (variable.name(), variable.solution_value()))

```

## 最终代码:

```py
from ortools.linear_solver import pywraplp

def Maximizationproblem():

    solver = pywraplp.Solver.CreateSolver('SCIP')

    vara = solver.IntVar(0.0, solver.infinity(), 'vara')

    varb = solver.IntVar(0.0, solver.infinity(), 'varb')

    varc = solver.IntVar(0.0, solver.infinity(), 'varc')

    # 3*a + 6*b + 2*c <= 50

    cons_in1 = solver.Constraint(-solver.infinity(), 50)

    cons_in1.SetCoefficient(vara, 3)

    cons_in1.SetCoefficient(varb, 6)

    cons_in1.SetCoefficient(varc, 2)

    # 4*a - 6*b + 8*c <= 45

    cons_in2 = solver.Constraint(-solver.infinity(), 45)

    cons_in2.SetCoefficient(vara, 4)

    cons_in2.SetCoefficient(varb, -6)

    cons_in2.SetCoefficient(varc, 8)

    # 3*a + b - 5*c <= 37

    cons_in3 = solver.Constraint(-solver.infinity(), 37)

    cons_in3.SetCoefficient(vara, 3)

    cons_in3.SetCoefficient(varb, 1)

    cons_in3.SetCoefficient(varc, -5)

    # [END constraints]

    # [objective segment of program]

    obj_prog = solver.Objective()

    obj_prog.SetCoefficient(vara, 3)

    obj_prog.SetCoefficient(varb, 2)

    obj_prog.SetCoefficient(varc, 2)

    obj_prog.SetMaximization()

    # Calling solver

    solver.Solve()

    # Print segment of program

    print('Highest objective function value = %d' % solver.Objective().Value())

    print()

    for variable in [vara, varb, varc]:

        print('%s = %d' % (variable.name(), variable.solution_value()))

Maximizationproblem()

```

### 输出

```py
Highest objective function value = 42

vara = 12
varb = 2
varc = 1

Process finished with exit code 0

```

## 结论

在本文中，我们了解了不同类型的优化以及如何在 Python 中实现这些优化。我们还学习了 ortools 和 python 包装器。此外，我们看到了一个完整的工作代码，它最大化了一组三个线性方程中的一个方程。本文将有助于理解 python 中的优化，并为学习者打下基础。

## 参考

[https://developers . Google . com/optimization/introduction/python](https://developers.google.com/optimization/introduction/python)

[https://developers.google.com/optimization/examples](https://developers.google.com/optimization/examples)