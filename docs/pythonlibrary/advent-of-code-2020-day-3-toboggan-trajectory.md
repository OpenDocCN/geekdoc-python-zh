# 代码 2020 的来临:第三天-滑降轨迹

> 原文：<https://www.blog.pythonlibrary.org/2020/12/07/advent-of-code-2020-day-3-toboggan-trajectory/>

当我读到代码出现的第三天[的描述时，我以为它在要求我创建某种寻路算法。这个挑战背后的想法是计算出你滑下格子时会撞到多少棵树。](https://adventofcode.com/2020/day/3)

你应该去检查一下描述，这样你就会明白这个问题了。

#### ！！！前方剧透！！！

如果你还没有解决这个难题，你应该停止阅读！

#### 第一部分

挑战的第一部分是计算你最终会撞上多少棵树。你的斜率看起来像这样:

```py
..##.......
#...#...#..
.#....#..#.
..#.#...#.#
.#...##..#.
..#.##.....
.#.#.#....#
.#........#
#.##...#...
#...##....#
.#..#...#.#
```

您的输入文件告诉您如何沿着斜坡向下行进。所以第一行可能说你向右走 3，向下走 2，而第二行可能说你只向右走 1，向下走 1。树木用“#”标记。

我最终强行得到了答案:

```py
def detect_trees(data: list, right: int = 3, down: int = 1) -> int:
    pos = 0
    trees = 0
    first_row = True
    for line in data[::down]:
        line = line * 100

        if pos != 0:
            char = line[:pos][-1]

        if first_row:
            pos += right+1
            first_row = False
            continue
        else:
            pos += right

        if char == "#":
            trees += 1

    print(f"Number of trees: {trees}")
    return trees
```

虽然我知道这不是很好的代码，甚至没有效率，但我的目标是快速解决这个难题，并基本上放在一个或两个私人排行榜上。

总之，我在这里做的是循环遍历数据文件中的行，并相应地改变我的位置。我用一个小程序告诉我是否在第一行，这帮助我从我想去的地方开始。

每当我最后一个字符是一个“#”的时候，我就增加我的树计数器。这在总体上运行得相当好，即使这是一个相当蹩脚的解决方案。

#### 第二部分

第 2 部分与第 1 部分非常相似，只不过这一次，您需要获得每个斜率的答案，然后将这些答案相乘。为此，我创建了一个新的函数，并使用上面提供的斜率调用了我的函数。

```py
def multiply_slopes(data: list) -> int:
    first = detect_trees(data, right=1, down=1)
    second = detect_trees(data, right=3, down=1)
    third = detect_trees(data, right=5, down=1)
    fourth = detect_trees(data, right=7, down=1)
    fifth = detect_trees(data, right=1, down=2)
    multiplied = first * second * third * fourth * fifth
    print(f'Slopes multiplied: {multiplied}')
    return multiplied
```

然后我把结果相乘，打印出来，还回去。回想起来，我本可以在这里使用 **math.prod()** 来执行乘法步骤，并使代码稍微少一些。

#### 其他解决方案

Matt Harrison (作者和专业 Python 培训师)正在做一个视频，回顾代码出现的每一天。他在文章中包含了很多关于 Python 的技巧。你应该看看这些视频[这里](https://mattharrison.podia.com/advent-of-code-2020-walkthrough)随着价格的上涨，大多数视频被添加。

在写出我的解决方案后，我还发现我可以使用模数运算符来简化整个过程:

[https://www.youtube.com/embed/PBI6rGv9Utw?feature=oembed](https://www.youtube.com/embed/PBI6rGv9Utw?feature=oembed)

那家伙不到 3 分钟就解决了挑战！我喜欢这个解决方案的简洁，尽管这不是我能想到的。