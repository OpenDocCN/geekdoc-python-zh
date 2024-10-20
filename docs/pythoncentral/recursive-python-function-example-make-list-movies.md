# 递归 Python 函数例子:把你的电影列表！

> 原文：<https://www.pythoncentral.io/recursive-python-function-example-make-list-movies/>

This recipe is a practical example of Python recursive functions, using the `os.listdir` function. However it is not the most effective method to traverse a directory in Python. `os.walk` is generally considered the most *Pythonic* method. For a quick look at how to use `os.walk`, checkout the article [this article for a os.walk example](https://www.pythoncentral.io/how-to-traverse-a-directory-tree-in-python-guide-to-os-walk/ "Python os.walk Example"). If you are after a more in-depth look at `os.walk`, be sure to checkout the article [Python's os.walk: An In-depth Guide](https://www.pythoncentral.io/recursive-file-and-directory-manipulation-in-python-part-1/ "Python's os.walk: An In-depth Guide")

所以。有什么比在你的硬盘上制作一个视频文件列表更好的呢？

让我们列出一个文件夹中的所有视频文件，以及其中的所有其他文件夹！

## Python 中的递归函数是什么？

递归是计算机科学中的一个概念。本质上，它把一个问题分成子问题。Python 中的递归一般与特定的函数、方法或对象有关，它调用自身来分解这些问题。例如，阶乘函数如下所示:

```py

def factorial(n):

    if n == 0:

        return 1

    else:

        return n * factorial(n - 1)

```

注意，`factorial`函数调用自己，将阶乘问题分解成子问题。

## 递归 Python 函数:我们编码吧！

让我们在一个函数中编写遍历代码，如下所示:

```py

import os
def Print _ movie _ files(movie_directory，movie_extensions=['avi '，' dat '，' mp4 '，' mkv '，' vob']): 
' ' '递归打印 movie _ directory 中扩展名为 movie_extensions 的文件''
#获取电影目录参数的绝对路径
电影目录= os.path.abspath(电影目录)
#获取电影目录中的文件列表
电影目录文件= os.listdir(电影目录)
#遍历电影目录文件中文件名的所有文件【T0:
file path = OS . path . join(电影目录，文件名)
#如果 os.path.isfile(filepath)，检查它是否是一个正常的文件或目录
:
#检查文件是否有典型视频文件的扩展名
为 movie_extensions 中的 movie _ extension:
#不是电影文件，如果不是 file path . ends with(movie _ extension):
继续
#我们有一个视频文件！递增计数器
print _ movie _ files . counter+= 1
# Print 它的名字
Print(“{ 0 }”)。format(file path))
elif OS . path . isdir(file path):
#我们得到一个目录，进入其中做进一步处理
print _ movie _ files(file path)

```

代码和注释都是不言自明的。递归 Python 函数`print_movie_files`有两个参数:要搜索的目录路径。然后，它使用`os.listdir`方法获得这个目录中所有文件和文件夹的列表。我们使用一个`for`循环来处理`list,`，使用`os.path.isfile`方法检查文件路径是否是一个正常的文件或目录。如果是扩展名为`movie_extensions`的普通文件，会打印文件路径。如果`filepath`是一个目录，我们递归调用函数本身来进一步处理它。

## 调用递归 Python 函数

现在，我们在`__main__`范围内调用这个函数:

*   [Python 3.x](#custom-tab-0-python-3-x)
*   [Python 2.x](#custom-tab-0-python-2-x)

*   [Python 3.x](#)

[python]
if __name__ == '__main__':

#提供了目录参数，检查并使用是否是目录
if len(sys . argv)= = 2:
if OS . path . isdir(sys . argv[1]):
movie _ Directory = sys . argv[1]
else:
print('错误:“{0}”不是目录。'。format(sys . argv[1])
exit(1)
else:
#将我们的电影目录设置为当前工作目录
movie_directory = os.getcwd()

打印(' \n -在“{0}”中查找电影- \n '。格式(电影目录))

#将已处理文件的数量设置为零
print_movie_files.counter = 0

#开始处理
print_movie_files(电影 _ 目录)

#我们结束了。现在退出。
打印(' \n - {0}个电影文件在目录{1} -'中找到。格式\
(print _ movie _ files . counter，movie_directory))
打印(' \ n 按回车键退出！')

#等到用户按 enter/return，或者 <ctrl-c>尝试:
【input()
除键盘中断:
【exit(0)
[/python]</ctrl-c> 

*   [Python 2.x](#)

[python]
if __name__ == '__main__':

#提供了目录参数，检查并使用是否是目录
if len(sys . argv)= = 2:
if OS . path . isdir(sys . argv[1]):
movie _ Directory = sys . argv[1]
else:
print('错误:“{0}”不是目录。'。format(sys . argv[1])
exit(1)
else:
#将我们的电影目录设置为当前工作目录
movie_directory = os.getcwd()

打印(' \n -在“{0}”中查找电影- \n '。格式(电影目录))

#将已处理文件的数量设置为零
print_movie_files.counter = 0

#开始处理
print_movie_files(电影 _ 目录)

#我们结束了。现在退出。
打印(' \n - {0}个电影文件在目录{1} -'中找到。格式\
(print _ movie _ files . counter，movie_directory))
打印(' \ n 按回车键退出！')

#等到用户按 enter/return，或者 <ctrl-c>try:
raw_input()
除键盘中断:
exit(0)
[/python]</ctrl-c> 

## 运行脚本

1.  下载并解压源代码 zip 文件(见下文)，并将`list-movies.py`复制到您希望搜索的目录中。
2.  **-或-** 将商品代码复制到一个新文件中，并将其作为`list-movies.py`保存在您希望搜索的目录中。
3.  电影所在的目录。例如`cd ~/Movies`或`cd C:\\Users\\Videos`。
4.  使用`/path/to/python list-movies.py`运行`list-movies.py`脚本
    *   Linux/OSX/Unix: `python list-movies.py`
    *   视窗:`C:\\Python34\\python.exe list-movies.py`

提示:在 Linux/OSX/Unix 上你可以将文件标记为可执行，在文件顶部添加一个 Python [*shebang*](https://www.wikiwand.com/en/Shebang_(Unix)) 行，直接运行。例如。

```py

cd ~/Desktop/list-movies.py

chmod +x ./list-movies.py

# Add "#/usr/bin/env python" to the top of the file

./list-movies.py          # Run script, search files in current directory

./list-movies.py ~/Movies # Run script, search for files in ~/Movies

```

脚本中的代码将递归遍历(查找)其中的所有其他文件夹，并检查视频文件。如果您使用的是 Windows 并且安装了 Python IDLE，那么您只需双击该文件并检查输出。

同样，`os.getcwd`方法帮助我们获得当前工作目录(cwd)，即脚本所在的目录。它调用我们刚刚编写的函数，还有一个计数器，它计算找到了多少个视频文件。最后，我们`print`我们拥有的所有信息，并等待用户使用`input()`或`raw_input()`函数终止程序的提示(当从 Python 2 升级到 Python 3 时，Python 将`raw_input()`函数的名称改为`input()`)。