# Python 中的刽子手游戏——一步一步的演练

> 原文：<https://www.askpython.com/python/examples/hangman-game-in-python>

在本教程中，我们将学习用 Python 语言创建自己的 hangman 游戏的步骤。

## 关于刽子手

Hangman 是一种猜谜游戏，玩家的目标是找出隐藏的单词。每一次不正确的猜测都会导致留给玩家的机会减少。

剩下的机会以一个绞刑者的形式表现出来。每个英雄的工作就是拯救生命。

* * *

## Python 中 Hangman 游戏的演示游戏

<https://www.askpython.com/wp-content/uploads/2020/06/hangman_game.mp4>

* * *

## 设计绞刑

在我们进入创建游戏逻辑的部分之前，我们首先需要弄清楚游戏对任何玩家来说是什么样子的。这个游戏有两个特别的设计元素:

*   刽子手–我们需要在刽子手的背景下为玩家提供视觉帮助。
*   **单词显示**–在游戏开始时，单词必须显示为空格，而不是字母。

### 刽子手设计

我们知道，在每一个不正确的动作之后，被绞死的人身体的一个新的部分被展示出来。为了实现这一点，我们将身体部位存储在一个列表中。

```py
# Stores the hangman's body values
hangman_values = ['O','/','|','\\','|','/','\\']

# Stores the hangman's body values to be shown to the player
show_hangman_values = [' ', ' ', ' ', ' ', ' ', ' ', ' ']

```

处理这些 hangman 值的函数如下所示:

```py
# Functuion to print the hangman
def print_hangman(values):
	print()
	print("\t +--------+")
	print("\t |       | |")
	print("\t {}       | |".format(values[0]))
	print("\t{}{}{}      | |".format(values[1], values[2], values[3]))
	print("\t {}       | |".format(values[4]))
	print("\t{} {}      | |".format(values[5],values[6]))
	print("\t         | |")
	print("  _______________|_|___")
	print("  ``````py```````py```````py`")
	print()

```

下面的视频显示了游戏中所有可能的刽子手状态。每一个不正确的错误增加一个身体部分，直到身体完整，玩家输。

<https://www.askpython.com/wp-content/uploads/2020/06/hangman_print_all.mp4>

视频中显示的最终状态代表了玩家猜出完整单词后，刽子手逃离绞刑架。

```py
# Function to print the hangman after winning
def print_hangman_win():
	print()
	print("\t +--------+")
	print("\t         | |")

	print("\t         | |")
	print("\t O       | |")
	print("\t/|\\      | |")
	print("\t |       | |")
	print("  ______/_\\______|_|___")
	print("  ``````py```````py```````py`")
	print()

```

上面的函数，`'print_hangman_win()'`负责在玩家获胜时打印逃跑的刽子手。

### 文字显示

在游戏开始时，只有空白必须是可见的。在每个玩家输入之后，我们必须处理需要显示的内容。

```py
# Stores the letters to be displayed
word_display = []

```

最初，列表`'word_display'`包含隐藏单词中每个字母的下划线。以下函数用于显示该列表。

```py
# Function to print the word to be guessed
def print_word(values):
	print()
	print("\t", end="")
	for x in values:
		print(x, end="")
	print()

```

* * *

## 单词数据集

在创造游戏的这一部分，我们可以让我们的想象力自由驰骋。可以有多种方式访问列表单词，如从**导入。csv** 文件，从数据库中提取等。

为了使本教程简单，我们将硬编码一些类别和单词。

```py
# Types of categories
topics = {1: "DC characters", 2:"Marvel characters", 3:"Anime characters"}

# Words in each category
dataset = {"DC characters":["SUPERMAN", "JOKER", "HARLEY QUINN", "GREEN LANTERN", "FLASH", "WONDER WOMAN", "AQUAMAN", "MARTIAN MANHUNTER", "BATMAN"],\
			 "Marvel characters":["CAPTAIN AMERICA", "IRON MAN", "THANOS", "HAWKEYE", "BLACK PANTHER", "BLACK WIDOW"],
			 "Anime characters":["MONKEY D. LUFFY", "RORONOA ZORO", "LIGHT YAGAMI", "MIDORIYA IZUKU"]
			 }

```

让我们理解这里使用的数据结构:

*   `**'topics'**`–这个 Python 字典为每个类别提供了一个数值。这将进一步用于实现一个基于类别的菜单。
*   `**'dataset'**`–这个 Python 字典包含了每个类别的单词列表。当玩家选择一个类别后，我们应该从这里自己选择单词。

* * *

## 游戏循环

每一个依赖于玩家一系列移动的游戏都需要一个游戏循环。这个循环负责管理玩家输入，显示游戏设计，以及游戏逻辑的其他重要部分。

```py
# The GAME LOOP
while True:

```

在这个游戏循环中，我们将处理以下事情:

* * *

## 游戏菜单

游戏菜单负责向玩家提供游戏控制的概念。玩家根据他/她的兴趣决定类别。

```py
# Printing the game menu
print()
print("-----------------------------------------")
print("\t\tGAME MENU")
print("-----------------------------------------")
for key in topics:
	print("Press", key, "to select", topics[key])
print("Press", len(topics)+1, "to quit")	
print()

```

每当创建游戏菜单时，总是提供退出游戏的选项是明智的。

* * *

## 处理玩家的类别选择

一个游戏开发人员，不管他的技术水平如何，都必须时刻关注玩家的输入。游戏不能因为一些错误的玩家输入而崩溃。

```py
# Handling the player category choice
try:
	choice = int(input("Enter your choice = "))
except ValueError:
	clear()
	print("Wrong choice!!! Try again")
	continue

# Sanity checks for input
if choice > len(topics)+1:
	clear()
	print("No such topic!!! Try again.")
	continue	

# The EXIT choice	
elif choice == len(topics)+1:
	print()
	print("Thank you for playing!")
	break

```

在做了一些健全的检查之后，我们已经准备好为游戏选择一个词了。

> **注:**`'clear()'`功能负责清空终端。它利用了 Python 内置的`'os'`库。

* * *

## 选择游戏用词

我们使用内置的 Python 库`'random'`从特定的类别列表中随机选择一个单词。

```py
# The topic chosen
chosen_topic = topics[choice]

# The word randomly selected
ran = random.choice(dataset[chosen_topic])

# The overall game function
hangman_game(ran)

```

在选择这个词之后，是游戏逻辑部分。

* * *

## 刽子手的游戏逻辑

功能`'hangman()'`包含整个游戏功能。包括存储不正确的猜测，减少剩下的机会数，打印刽子手的具体状态。

```py
# Function for each hangman game
def hangman_game(word):

	clear()

	# Stores the letters to be displayed
	word_display = []

	# Stores the correct letters in the word
	correct_letters = []

	# Stores the incorrect guesses made by the player
	incorrect = []

	# Number of chances (incorrect guesses)
	chances = 0

	# Stores the hangman's body values
	hangman_values = ['O','/','|','\\','|','/','\\']

	# Stores the hangman's body values to be shown to the player
	show_hangman_values = [' ', ' ', ' ', ' ', ' ', ' ', ' ']

```

上面的代码片段包含了 hangman 游戏顺利运行所需的所有基本数据结构和变量。

* * *

## 初始化必要的组件

创建游戏最重要的一个方面是游戏组件的初始状态。

```py
# Loop for creating the display word
for char in word:
	if char.isalpha():
		word_display.append('_')
		correct_letters.append(char.upper())
	else:
		word_display.append(char)

```

我们需要初始化单词显示的结构，因为它会随着游戏中每隔一个单词而变化。为了方便起见，我们在同一个循环中初始化容器来存储正确的字母。

> **注:**我们版本的刽子手游戏只支持字母的猜测。如果读者想要添加猜测其他元素(如数字或特殊字符)的功能，必须在这里进行更改。

* * *

## 内部游戏循环

这个内部游戏循环负责控制刽子手游戏的单个游戏流程。它包括显示正确的显示，处理字符输入，更新必要的数据结构，以及游戏的其他关键方面。

```py
# Inner Game Loop			
while True:

	# Printing necessary values
	print_hangman(show_hangman_values)
	print_word(word_display)			
	print()
	print("Incorrect characters : ", incorrect)
	print()

```

* * *

## 玩家的移动输入

我们游戏的这一部分处理玩家与我们游戏的交互。在游戏逻辑中实现之前，输入必须检查几个场景:

*   **有效长度**–因为我们接受单个字符，我们需要检查以防玩家恶意输入多个字符。
*   一个字母表？如前所述，我们的刽子手游戏版本只支持猜字母。
*   已经试过了，作为一个体贴的程序员，如果玩家输入了一个不正确的已经试过的字母，我们必须通知他。

```py
# Accepting player input
inp = input("Enter a character = ")
if len(inp) != 1:
	clear()
	print("Wrong choice!! Try Again")
	continue

# Checking whether it is a alphabet
if not inp[0].isalpha():
	clear()
	print("Wrong choice!! Try Again")
	continue

# Checking if it already tried before	
if inp.upper() in incorrect:
	clear()
	print("Already tried!!")
	continue 	

```

* * *

## 管理玩家的移动

很明显，在管理玩家的移动时，我们只会遇到两种情况。

*   **不正确的字母**–对于不正确的移动，我们更新不正确字母的列表和刽子手显示(添加身体部位)。

```py
# Incorrect character input	
if inp.upper() not in correct_letters:

	# Adding in the incorrect list
	incorrect.append(inp.upper())

	# Updating the hangman display
	show_hangman_values[chances] = hangman_values[chances]
	chances = chances + 1

	# Checking if the player lost
	if chances == len(hangman_values):
		print()
		clear()
		print("\tGAME OVER!!!")
		print_hangman(hangman_values)
		print("The word is :", word.upper())
		break

```

*   **正确的字母表**–如果有能力的玩家输入了正确的字母表，我们会更新单词显示。

```py
# Correct character input
else:

	# Updating the word display
	for i in range(len(word)):
		if word[i].upper() == inp.upper():
			word_display[i] = inp.upper()

	# Checking if the player won		
	if check_win(word_display):
		clear()
		print("\tCongratulations! ")
		print_hangman_win()
		print("The word is :", word.upper())
		break

```

游戏开发者最感兴趣的是每次输入正确的字母时都检查是否获胜。这不是一个硬性的规则，读者可以实现他们自己版本的最终游戏检查。

* * *

## 完整的代码

下面是上面讨论的 hangman 游戏的完整运行代码:

```py
import random
import os

# Funtion to clear te terminal
def clear():
	os.system("clear")

# Functuion to print the hangman
def print_hangman(values):
	print()
	print("\t +--------+")
	print("\t |       | |")
	print("\t {}       | |".format(values[0]))
	print("\t{}{}{}      | |".format(values[1], values[2], values[3]))
	print("\t {}       | |".format(values[4]))
	print("\t{} {}      | |".format(values[5],values[6]))
	print("\t         | |")
	print("  _______________|_|___")
	print("  ``````py```````py```````py`")
	print()

# Function to print the hangman after winning
def print_hangman_win():
	print()
	print("\t +--------+")
	print("\t         | |")

	print("\t         | |")
	print("\t O       | |")
	print("\t/|\\      | |")
	print("\t |       | |")
	print("  ______/_\\______|_|___")
	print("  ``````py```````py```````py`")
	print()

# Function to print the word to be guessed
def print_word(values):
	print()
	print("\t", end="")
	for x in values:
		print(x, end="")
	print()	

# Function to check for win
def check_win(values):
	for char in values:
		if char == '_':
			return False
	return True		

# Function for each hangman game
def hangman_game(word):

	clear()

	# Stores the letters to be displayed
	word_display = []

	# Stores the correct letters in the word
	correct_letters = []

	# Stores the incorrect guesses made by the player
	incorrect = []

	# Number of chances (incorrect guesses)
	chances = 0

	# Stores the hangman's body values
	hangman_values = ['O','/','|','\\','|','/','\\']

	# Stores the hangman's body values to be shown to the player
	show_hangman_values = [' ', ' ', ' ', ' ', ' ', ' ', ' ']

	# Loop for creating the display word
	for char in word:
		if char.isalpha():
			word_display.append('_')
			correct_letters.append(char.upper())
		else:
			word_display.append(char)

	# Game Loop			
	while True:

		# Printing necessary values
		print_hangman(show_hangman_values)
		print_word(word_display)			
		print()
		print("Incorrect characters : ", incorrect)
		print()

		# Accepting player input
		inp = input("Enter a character = ")
		if len(inp) != 1:
			clear()
			print("Wrong choice!! Try Again")
			continue

		# Checking whether it is a alphabet
		if not inp[0].isalpha():
			clear()
			print("Wrong choice!! Try Again")
			continue

		# Checking if it already tried before	
		if inp.upper() in incorrect:
			clear()
			print("Already tried!!")
			continue 	

		# Incorrect character input	
		if inp.upper() not in correct_letters:

			# Adding in the incorrect list
			incorrect.append(inp.upper())

			# Updating the hangman display
			show_hangman_values[chances] = hangman_values[chances]
			chances = chances + 1

			# Checking if the player lost
			if chances == len(hangman_values):
				print()
				clear()
				print("\tGAME OVER!!!")
				print_hangman(hangman_values)
				print("The word is :", word.upper())
				break

		# Correct character input
		else:

			# Updating the word display
			for i in range(len(word)):
				if word[i].upper() == inp.upper():
					word_display[i] = inp.upper()

			# Checking if the player won		
			if check_win(word_display):
				clear()
				print("\tCongratulations! ")
				print_hangman_win()
				print("The word is :", word.upper())
				break
		clear()	

if __name__ == "__main__":

	clear()

	# Types of categories
	topics = {1: "DC characters", 2:"Marvel characters", 3:"Anime characters"}

	# Words in each category
	dataset = {"DC characters":["SUPERMAN", "JOKER", "HARLEY QUINN", "GREEN LANTERN", "FLASH", "WONDER WOMAN", "AQUAMAN", "MARTIAN MANHUNTER", "BATMAN"],\
				 "Marvel characters":["CAPTAIN AMERICA", "IRON MAN", "THANOS", "HAWKEYE", "BLACK PANTHER", "BLACK WIDOW"],
				 "Anime characters":["MONKEY D. LUFFY", "RORONOA ZORO", "LIGHT YAGAMI", "MIDORIYA IZUKU"]
				 }

	# The GAME LOOP
	while True:

		# Printing the game menu
		print()
		print("-----------------------------------------")
		print("\t\tGAME MENU")
		print("-----------------------------------------")
		for key in topics:
			print("Press", key, "to select", topics[key])
		print("Press", len(topics)+1, "to quit")	
		print()

		# Handling the player category choice
		try:
			choice = int(input("Enter your choice = "))
		except ValueError:
			clear()
			print("Wrong choice!!! Try again")
			continue

		# Sanity checks for input
		if choice > len(topics)+1:
			clear()
			print("No such topic!!! Try again.")
			continue	

		# The EXIT choice	
		elif choice == len(topics)+1:
			print()
			print("Thank you for playing!")
			break

		# The topic chosen
		chosen_topic = topics[choice]

		# The word randomly selected
		ran = random.choice(dataset[chosen_topic])

		# The overall game function
		hangman_game(ran)

```

* * *

## 结论

起初，创建刽子手游戏似乎是一项艰巨的任务，但我们希望本教程可以消除读者的误解。如有任何疑问或批评，欢迎在下面评论。

如果你想学习更多关于用 Python 开发基于终端的游戏，你可以看看其他游戏，比如[扫雷](https://www.askpython.com/python/examples/create-minesweeper-using-python)或者[井字游戏](https://www.askpython.com/python/examples/tic-tac-toe-using-python)。