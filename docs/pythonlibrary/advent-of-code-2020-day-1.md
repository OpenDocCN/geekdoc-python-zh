# 代码 2020 的出现:第一天

> 原文：<https://www.blog.pythonlibrary.org/2020/12/03/advent-of-code-2020-day-1/>

我决定今年尝试一下代码挑战的[来临。《代码降临》是一系列每天一次的挑战。你可以用任何你想要的编程语言来解决它们。一旦你有了解决方案，你就在网站上输入它来赚取积分。在今年的情况下，你正在赚取星星。每天可以赚两星。](https://adventofcode.com/)

注意:代码的降临不会运行你的代码。它不知道你用的是什么编程语言。它只关心挑战发布后你回答问题的速度。如果你恰好在挑战发布后 5 分钟内回答了问题，你会比那些在挑战出现 5 小时后回答的人得到更好的分数。

我不知道我是否会完成所有的挑战，甚至不知道我能不能完成。如果这些挑战花费了我太多的时间，那么我将不得不放弃它们，因为我还有很多其他的项目要做。但是对于那些我完成的，我会在这里写下来。如果你自己正在解决挑战，你不想看到我的答案，你应该停止阅读！

#### ！！！前方剧透！！！

### 第一部分

每天分为两个问题。问题详述[这里](https://adventofcode.com/2020/day/1)。问题的第一部分是给你一个每行有一个数字的文件。你会发现 t **两个数相加等于 2020** 。然后把这两个数相乘就得到答案了。

当我看到这个问题时，我的第一个想法是使用 Python 的 [itertools](https://docs.python.org/3/library/itertools.html) 库，它有一个方便的[组合](https://docs.python.org/3/library/itertools.html#itertools.combinations)函数。

利用这一点，我想出了以下代码:

```py
from itertools import combinations

result = [pair for pair in combinations(numbers, 2)
          if sum(pair) == 2020]
```

这给了我想要的一对数字，尽管输出看起来像这样:[(number_1，number_2)]

我从中提取了两个数字，将它们相乘，得到了降临节问题第一部分的答案。

### 第二部分

问题的下一部分是，现在他们希望你接受相同的输入，但是找到加起来是 2020 的唯一的**三个数字。然后将这些数字相乘得到答案。**

你可以修改**组合**来寻找两个以上我在上面硬编码的数字。你只是传了 3 个而不是 2 个。

为了让代码更智能，我还导入了 Python 的 **math** 模块，并使用其 **prod()** 函数对结果进行乘法运算。

以下是完整的代码:

```py
import math
from itertools import combinations

def get_answer(numbers: list[int], combos: int = 2) -> int:
    result = [pair for pair in combinations(numbers, combos)
              if sum(pair) == 2020]

    multiplied = math.prod(result[0])

    print(f"The answer is {multiplied}")
    return multiplied

if __name__ == '__main__':
    with open('input.txt') as f:
        numbers = [int(item.strip()) for item in f.readlines()]
    get_answer(numbers, combos=3)
```

我所有的代码也在 Github 上，一些答案包括单元测试。我现在有一些非常基本的测试来应对这个挑战。