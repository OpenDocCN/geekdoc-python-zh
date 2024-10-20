# Python 中的数独求解器

> 原文：<https://www.askpython.com/python/examples/sudoku-solver-in-python>

今天就让我们用 Python 来构建一个数独求解器吧！数独谜题是一个非常受欢迎的谜题，出现在日报上，吸引了很多人的注意。关于数独谜题和它们的推广有许多困难的、未解决的问题，这使得这个谜题很有趣，特别是对许多数学爱好者来说。

* * *

## 什么是数独谜题？

在数独游戏中，我们需要用 1 到 9 之间的整数填充每个空盒子，这样从 1 到 9 的每个数字在每行、每列和每个用粗边框突出显示的 3 乘 3 的小盒子中都出现一次。

这个难题的难度可能会有所不同。数独谜题的难度级别越多，对计算科学家来说，研究问题就变得越有挑战性。难的谜题大多规定符号较少。

为娱乐而出版的数独游戏有独特的解决方案。如果一个数独谜题有唯一的解，它就被认为是良构的。另一个具有挑战性的研究问题是确定一个数独谜题需要填充多少个盒子才是良构的。有 17 个符号的格式良好的数独存在。未知是否存在一个只有 16 条线索的格式良好的谜题。线索越少，多解的几率越高。

* * *

## 用 Python 解决数独难题的步骤

*   在这个解决数独难题的方法中，首先，我们将 2D 矩阵的大小赋给一个变量 M (M*M)。
*   然后我们分配效用函数(puzzle)来打印网格。
*   稍后，它会将 num 分配给行和列。
*   如果我们在同一行或同一列或特定的 3*3 矩阵中找到相同的 num，将返回“false”。
*   然后，我们将检查是否已经到达第 8 行和第 9 列，并返回 true 以停止进一步的回溯。
*   接下来，我们将检查列值是否变为 9，然后我们移动到下一行和下一列。
*   现在我们进一步查看网格的当前位置是否有大于 0 的值，然后我们迭代下一列。
*   在检查它是否是一个安全的地方之后，我们移动到下一列，然后在网格的当前(行，列)位置分配 num。稍后，我们用下一列检查下一个可能性。
*   由于我们的假设是错误的，我们丢弃了指定的 num，然后我们用不同的 num 值进行下一个假设

## 用 Python 实现数独求解器

我们将使用回溯方法在 Python 中创建数独求解器。回溯意味着一旦我们确定我们当前的解决方案不能延续到一个完整的解决方案，就切换回上一步。我们使用这种回溯原理来实现数独算法。它也被称为解决数独难题的暴力算法。

```py
M = 9
def puzzle(a):
	for i in range(M):
		for j in range(M):
			print(a[i][j],end = " ")
		print()
def solve(grid, row, col, num):
	for x in range(9):
		if grid[row][x] == num:
			return False

	for x in range(9):
		if grid[x][col] == num:
			return False

	startRow = row - row % 3
	startCol = col - col % 3
	for i in range(3):
		for j in range(3):
			if grid[i + startRow][j + startCol] == num:
				return False
	return True

def Suduko(grid, row, col):

	if (row == M - 1 and col == M):
		return True
	if col == M:
		row += 1
		col = 0
	if grid[row][col] > 0:
		return Suduko(grid, row, col + 1)
	for num in range(1, M + 1, 1): 

		if solve(grid, row, col, num):

			grid[row][col] = num
			if Suduko(grid, row, col + 1):
				return True
		grid[row][col] = 0
	return False

'''0 means the cells where no value is assigned'''
grid = [[2, 5, 0, 0, 3, 0, 9, 0, 1],
        [0, 1, 0, 0, 0, 4, 0, 0, 0],
	[4, 0, 7, 0, 0, 0, 2, 0, 8],
	[0, 0, 5, 2, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 9, 8, 1, 0, 0],
	[0, 4, 0, 0, 0, 3, 0, 0, 0],
	[0, 0, 0, 3, 6, 0, 0, 7, 2],
	[0, 7, 0, 0, 0, 0, 0, 0, 3],
	[9, 0, 3, 0, 0, 0, 6, 0, 4]]

if (Suduko(grid, 0, 0)):
	puzzle(grid)
else:
	print("Solution does not exist:(")

```

**输出:**

```py
====================== RESTART: C:/Users/SIDDHI/sudoku.py ===========
2 5 8 7 3 6 9 4 1 
6 1 9 8 2 4 3 5 7 
4 3 7 9 1 5 2 6 8 
3 9 5 2 7 1 4 8 6 
7 6 2 4 9 8 1 3 5 
8 4 1 6 5 3 7 2 9 
1 8 4 3 6 9 5 7 2 
5 7 6 1 4 2 8 9 3 
9 2 3 5 8 7 6 1 4 

```

## 结论

这就是用 Python 构建数独求解器的全部内容！我希望您在通读这篇文章并了解我们如何实现代码时感到愉快。

嘘…用 Python 还有一种更简单的方法来构建数独求解器！

你可以导入数独游戏。来自 https://pypi.org/project/py-sudoku/[的 PyPI 模块。这是一个简单的 Python 程序，可以生成并解决 m x n 数独难题。](https://pypi.org/project/py-sudoku/)

很酷，不是吗？现在是你玩数独游戏的时候了！

## 下一步是什么？

*   [Python 中的简单游戏](https://www.askpython.com/python/examples/easy-games-in-python)
*   [Python 中的猜数字游戏](https://www.askpython.com/python/examples/number-guessing-game-command-line)
*   [给 Python 游戏添加背景音乐](https://www.askpython.com/python-modules/pygame-adding-background-music)
*   [Python 中的石头剪刀布游戏](https://www.askpython.com/python/examples/rock-paper-scissors-in-python-with-ascii-hand)
*   [Python 中的 Hangman 游戏](https://www.askpython.com/python/examples/hangman-game-in-python)

## 资源

*   [数独百科](https://en.wikipedia.org/wiki/Sudoku)
*   [数独求解算法](https://en.wikipedia.org/wiki/Sudoku_solving_algorithms)