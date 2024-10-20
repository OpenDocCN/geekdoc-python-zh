# 用 Python 创建 Wordcounter:如何用 Python 计算文件中的字数和行数？

> 原文：<https://www.askpython.com/python/examples/wordcounter-count-lines-and-words>

读者你好！在本教程中，我们将讨论如何使用 Python 编程计算文件中的行数和字数。

***也读:[如何用 Python 读一个文件？](https://www.askpython.com/python/built-in-methods/python-open-method)***

* * *

## 如何计算字数和行数–Python word counter

假设您有一个大文件，需要计算文件中的字数。除此之外，您还想知道其中有多少行文本。你可以创建一个 wordcounter 程序，用 Python 计算字数和行数。

### 1.创建一个示例文本文件

在创建文本文件的过程中，我们将首先创建一个变量，并为其分配一个字符串。然后，我们将使用 open()函数在只写模式(“w”)下创建一个文件，并将字符串变量的内容写入新创建的文本文件。最后，关闭文本文件。让我们编写一个 Python 程序来创建一个文本文件。

```py
# Create a Python string
string = """Welcome to AskPython!
AskPython is a part of JournalDev IT Services Private Limited."""

# Create a sample text file using open() function
file = open("sample_file.txt", "w", encoding='utf-8')

# Write the above string to the newly created text file
# If it is created successfully
if file != None:
    file.write(string)
    print("Sample text file created and written successfully!!")
else:
    print("OSError: File cannot be created!!")

# Close the above text file using close()
file.close()

```

**输出:**

```py
Sample text file created and written successfully!!

```

### 2.显示示例文本文件的内容

由于我们已经成功创建了一个文本文件，现在我们将使用只读模式下的`read()`函数(**r**’)将示例文本文件的内容读入一个变量。然后，我们将打印 Python 变量的内容，以查看文件中的文本。最后，作为一个好的实践，我们将关闭打开的文本以避免代码中的任何内存泄漏。让我们看看读取给定文本文件的 Python 代码。

```py
# Open the given sample text file using open() function
# In read only mode
file = open("C:path//sample_file.txt", "r", encoding='utf-8')

# Read the sample text file using the read() function
# If it is opened successfully
if file != None:
    file_data = file.read()
    # Print the content of the sample text file
    print("This is the content of the sample text file:\n")
    print(file_data)    
else:
    print("OSError: File cannot be opend!!")

# Close the above opened text file using close() function
file.close()

```

**输出:**

```py
This is the content of the sample text file:

Welcome to AskPython!
AskPython is a part of JournalDev IT Services Private Limited.

```

### 3.计算文件中行数和字数的算法

要计算文件中的行数和字数，我们必须遵循下面给出的步骤:

1.  创建两个变量，比如`line_count` & `word_count`，并用零初始化它们。
2.  创建另一个变量，比如说`file_path`，并用给定文本文件的完整路径初始化它。
3.  使用`open()`功能以只读模式( **r** )打开给定的文本文件。
4.  逐行读取打开的文本文件，并在每次迭代中使`line_count`递增 1。
5.  使用`len()`和`split()`功能计算每行被读取的字数。
6.  将每行的字数加到`word_count`中。
7.  使用`close()`功能关闭打开的文本文件。
8.  打印`line_count`和`word_count`变量的最终值。

### 4.Python 代码计算文件中的行数和字数

让我们通过 Python 代码实现上面的算法来统计行数和字数。

```py
# Create two counter variables
# And initialize them with zero
line_count = 0
word_count = 0

# Open the given sample text file using open() function
file = open("C:path//sample_file.txt", "r", encoding='utf-8')

# Perform all the operations using the sample text file
# If it is opened successfully
if file != None:
    # Iterate over the opened file
    # To the number of lines and words in it
    for line in file:
        # Increment the line counter variable
        line_count = line_count + 1
        # Find the number of words in each line
        words = len(line.split())
        # Add the number of words in each line
        # To the word counter variable
        word_count = word_count + words 
else:
    print("OSError: File cannot be opend!!")

# Close the above opened text file using close() function
file.close()

# Print the final results using the final values 
# Of the line_count and word_count variables
print(f"\nTotal number of lines in the given file: {line_count}")
print(f"\nTotal number of words in the given file: {word_count}")

```

**输出:**

```py
Total number of lines in the given file: 2

Total number of words in the given file: 13

```

## 结论

在本教程中，我们学习了以下内容:

*   如何使用 Python 创建文本文件？
*   如何用 Python 读取一个文本文件的内容？
*   计算给定文本文件中的行数和字数的算法。
*   如何用 Python 统计一个文本文件的行数和字数？

希望你清楚并准备好独立完成这些任务。谢谢你，请继续关注我们的更多 Python 教程。