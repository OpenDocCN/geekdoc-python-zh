# Python 刽子手游戏

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/game-hangman>

在 Python 中，我们可以非常快速地创建简单的游戏。在本文中，我们将讨论 Python Hangman 游戏的实现。

## Hangman 游戏是怎么玩的？

刽子手游戏是一个多人游戏。在这个游戏中，一个玩家选择一个单词。其他玩家有一定次数的猜测来猜测单词中的字符。如果玩家能够在一定的尝试中猜出整个单词中的字符，他们就赢了。否则，他们就输了。

## 如何用 Python 创建 Hangman 游戏？

为了用 Python 创建一个刽子手游戏，我们将使用以下步骤。

*   首先，我们将询问用户的姓名。我们将使用 input()方法获取用户输入。执行后，input()方法接受用户的输入并返回一个字符串。
*   接下来，我们将选择一个单词，并要求用户开始猜测单词中的字符。
*   我们还将定义用户可以尝试的最大次数。
*   现在，我们将使用一个 [while 循环](https://www.pythonforbeginners.com/loops/python-while-loop)来反复要求用户猜测字符，直到尝试用尽为止。
*   在 while 循环中，如果用户猜出了正确的字符。我们将在回复中包括它。否则，我们将通知用户他们犯了一个错误。
*   如果用户能够在最大尝试次数内猜出单词的所有字符，他们就赢得了游戏。
*   如果用户在猜出整个单词之前用尽了所有的尝试，他们就输了。

## 刽子手游戏的 Python 代码

以下是经典游戏《刽子手》的一个 Python 脚本。一行破折号代表要猜的单词。如果玩家猜出单词中的一个字母，脚本会将它写在所有正确的位置上。玩家有 10 次机会猜这个单词。你可以很容易地通过改变变量来定制游戏。

确保你理解每一行是做什么的。出于这个原因，所有的 python 语句都已经用 [python 注释](https://www.pythonforbeginners.com/comments/how-to-use-comments-in-python)解释过了。

```py
#importing the time module
import time

#welcoming the user
name = input("What is your name? ")

print ("Hello, " + name, "Time to play hangman!")

#wait for 1 second
time.sleep(1)

print ("Start guessing...")
time.sleep(0.5)

#here we set the secret. You can select any word to play with. 
word = ("secret")

#creates an variable with an empty value
guesses = ''

#determine the number of turns
turns = 10

# Create a while loop

#check if the turns are more than zero
while turns > 0:         

    # make a counter that starts with zero
    failed = 0             

    # for every character in secret_word    
    for char in word:      

    # see if the character is in the players guess
        if char in guesses:    

        # print then out the character
            print (char,end=""),    

        else:

        # if not found, print a dash
            print ("_",end=""),     

        # and increase the failed counter with one
            failed += 1    

    # if failed is equal to zero

    # print You Won
    if failed == 0:        
        print ("You won")
    # exit the script
        break            
    # ask the user go guess a character
    guess = input("guess a character:") 

    # set the players guess to guesses
    guesses += guess                    

    # if the guess is not found in the secret word
    if guess not in word:  

     # turns counter decreases with 1 (now 9)
        turns -= 1        

    # print wrong
        print ("Wrong")  

    # how many turns are left
        print ("You have", + turns, 'more guesses' )

    # if the turns are equal to zero
        if turns == 0:           

        # print "You Lose"
            print ("You Lose"  ) 
```

输出:

```py
What is your name? Aditya
Hello, Aditya Time to play hangman!
Start guessing...
______guess a character:s
s_____guess a character:e
se__e_guess a character:c
sec_e_guess a character:r
secre_guess a character:e
secre_guess a character:t
secretYou won
```

好好享受吧！！

## 结论

在本文中，我们讨论了用 Python 实现 hangman 游戏。要了解更多关于 python 编程的知识，你可以阅读这篇关于[字符串操作](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)的文章。你可能也会喜欢这篇关于 [python 的文章，如果你是简写的话](https://avidpython.com/python-basics/python_if_else_shorthand/)。