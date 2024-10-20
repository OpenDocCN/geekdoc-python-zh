# 如何在 Python 中遍历目录树 os.walk 指南

> 原文：<https://www.pythoncentral.io/how-to-traverse-a-directory-tree-in-python-guide-to-os-walk/>

当您使用像 Python 这样的脚本语言时，您会发现自己反复做的一件事就是遍历目录树和处理文件。虽然有许多方法可以做到这一点，但 Python 提供了一个内置函数，使这一过程变得轻而易举。

## **什么是 os.walk()函数？**

walk 函数类似于 os.path 函数，但可以在任何操作系统上运行。Python 用户可以利用该函数在目录树中生成文件名。该函数在两个方向上导航树，自上而下和自下而上。

任何操作系统的任何树中的每个目录都有一个作为子目录的基目录。os.walk()函数以三元组的形式生成结果:路径、目录和任何子目录中的文件。

生成的元组有:

1.  **目录路径:** 该字符串将文件或文件夹导向目录路径。
2.  **目录名:** 包含所有不包含“.”的子目录还有“..”。
3.  文件名: 这是系统或用户创建的文件或文件夹列表。它是包含目录文件之外的文件的目录路径。

需要注意的是，列表中的名字不包含路径的任何部分。如果用户想要获取从路径中的目录或文件顶部开始的完整路径，他们必须使用 os.walk.join()，它有 dirpath 和目录名的参数。

如前所述，os.walk()函数可以以自顶向下和自底向上的方式遍历树。top-down 和 bottom-up 是两个可选参数，如果用户想要生成一个目录序列，函数中必须使用其中的一个。

在某些情况下，如果用户没有提及任何与序列相关的参数，则默认使用自顶向下遍历选项。如果 top-down 参数为真，该函数首先为主目录生成三元组，然后是子目录。

另一方面，如果 top-down 参数为 false，函数将为子目录之后的目录生成三元组。简而言之，序列是以自下而上的方式生成的。

此外，当 top-down 参数为真时，用户可以更新目录名列表，os.walk()函数将仅适用于子目录。当 top-down 为 false 时，不可能更新目录名，因为在自下而上模式中，目录名显示在路径之前。

使用 listdir()函数可以消除默认错误。

## **Python OS . walk()函数的工作原理**

在 Python 中，文件系统以特定的方式被遍历。文件系统就像一棵树，只有一个根，这个根将自己分成多个分支，而这些分支又会扩展成子分支，以此类推。

os.walk()函数通过从底部或顶部遍历目录树来生成目录树中文件的名称。

### **OS . walk()**的语法

OS . walk 函数的语法是:

```py
os.walk(top[, topdown=True[ onerror=None[ followlinks=False]]])
```

其中:

*   **Top:** 表示子目录遍历的起点或“头”。如前所述，它会生成三元组。
*   **Topdown:** 当该属性为真时，从上到下扫描目录，为假时，从下到上扫描目录。
*   **Onerror:** 这是一个帮助监控错误的特殊属性。它要么显示一个错误以继续运行该函数，要么引发一个异常以关闭该函数。
*   **跟随链接:** 当设置为 true 时，如果任何链接指向它自己的基目录，该属性将导致无法停止的递归。需要注意的是，os.walk()函数从不记录它以前遍历过的目录。

## **如何使用 os.walk()**

由于 os.walk()与操作系统的文件结构一起工作，用户必须首先将 os 模块导入到 Python 环境中。该模块是标准 Python 安装的一部分，将解决文件列表脚本其余部分中的任何依赖性。

接下来，用户必须定义文件列表功能。用户可以给它起任何名字，但是使用一个能清楚表达其目的的名字是最佳实践。该函数必须给出两个参数: *filetype* 和*filepath*。

*文件路径* 参数将指示函数必须从哪里开始寻找文件。它将使用操作系统格式的文件路径字符串。

**-**

**注意:** 对字符进行适当的转义或编码是必须的。

**-**

当文件列表功能运行时，该参数假定基本目录包含用户需要检查的所有文件和子文件夹。

另一方面， *文件类型* 参数将向函数指示用户正在寻找什么类型的文件。该参数接受字符串格式的文件扩展名，例如，“. txt .”

接下来，用户需要存储脚本在文件列表函数中找到的所有相关文件路径。因此，用户必须创建一个空列表。

使用该函数时，将查找 *文件路径* 中的每个文件，并验证扩展名是否与所需的 *文件类型* 匹配。然后，它会将相关结果添加到空列表中。

因此，要开始迭代过程，我们必须使用 for 循环来检查每个文件。然后 os.walk()函数会在 *filepath* 中找到所有的文件和路径，并生成一个三元组。让我们假设我们将这些组件命名为 *【根】**dirs*和*files*。

由于 *文件* 组件会列出路径内的所有文件名，该函数必须遍历每个文件名。为此，我们必须编写一个 For 循环。

现在，在这个文件级循环下，文件列表函数必须检查每个文件的所有方面。如果您正在编写的应用程序有其他需求，这就是您必须修改脚本的地方。但是，为了便于解释，我们将重点检查所有文件的所需文件扩展名。

在 Python 中，比较字符串是区分大小写的。但是，文件扩展名的写法不同。因此，我们必须使用 lower()方法将 *文件* 和 *文件类型* 转换成小写字符串。这样，我们可以避免由于大小写不匹配而丢失任何文件。

接下来，我们必须使用 endswith()方法将存储文件扩展名的小写 *文件* 属性的末尾与小写 *文件类型* 属性进行比较。该方法将根据是否匹配返回 True 或 False。

布尔结果必须包含在 if 语句中，因此只有当存在匹配的文件类型时，脚本中的下一行才会被触发。

如果文件扩展名符合要求，那么 *文件* 属性及其位置的信息必须添加到 *路径* 组件中，这就是我们的相关文件路径列表。

使用 os.path.join()函数将根文件路径和文件名结合起来，形成一个操作系统可以使用的完整地址。可以使用 append()方法组合数据。

最后，Python 将遍历循环，遍历所有文件夹和文件，毫不费力地构建一个 *路径* 列表。为了使这个列表在文件列表函数之外可用，我们必须在脚本的末尾写 return(paths)。

总的来说，代码应该是这样的:

```py
import os
def list_files(filepath, filetype):
   paths = []
   for root, dirs, files in os.walk(filepath):
      for file in files:
         if file.lower().endswith(filetype.lower()):
            paths.append(os.path.join(root, file))
   return(paths)
```

在这个脚本之后，您必须编写另一个函数，将结果位置保存到您选择的文件中。代码可能是这样的:

```py
my_files_list = list_files(' C:\\Users\\Public\\Downloads', '.csv')
```

现在你的脚本已经准备好找到你需要的文件，你可以专注于分析文本，合并数据，或者任何你需要做的事情。

## 基本 Python 目录遍历

下面是一个非常简单的例子，它遍历一个目录树，打印出每个目录的名称和包含的文件:

```py
[python]
# Import the os module, for the os.walk function
import os

# Set the directory you want to start from
rootDir = '.'
for dirName, subdirList, fileList in os.walk(rootDir):
print('Found directory: %s' % dirName)
for fname in fileList:
print('\t%s' % fname)
[/python]
```

`os.walk`关注细节，在每一次循环中，它给我们三样东西:

*   `dirName`:找到的下一个目录。
*   `subdirList`:当前目录下的子目录列表。
*   `fileList`:当前目录下的文件列表。

假设我们有一个如下所示的目录树:

```py
+--- test.py
|
+--- [subdir1]
|     |
|     +--- file1a.txt
|     +--- file1b.png
|
+--- [subdir2]
|
+--- file2a.jpeg
+--- file2b.html
```

上面的代码将产生以下输出:

```py
[shell]
Found directory: .
file2a.jpeg
file2b.html
test.py
Found directory: ./subdir1
file1a.txt
file1b.png
Found directory: ./subdir2
[/shell]
```

### **改变遍历目录树的方式**

默认情况下，Python 将按照自顶向下的顺序遍历目录树(一个目录将被传递给您进行处理)，*然后* Python 将进入任何子目录。我们可以在上面的输出中看到这种行为；父目录(。)首先被打印，然后是它的 2 个子目录。

有时我们希望自底向上遍历目录树(首先处理目录树最底层的文件)，然后我们沿着目录向上遍历。我们可以通过 topdown 参数告诉`os.walk`这样做:

```py
[python]
import os

rootDir = '.'
for dirName, subdirList, fileList in os.walk(rootDir, topdown=False):
print('Found directory: %s' % dirName)
for fname in fileList:
print('\t%s' % fname)
[/python]
```

这给了我们这样的输出:

```py
[shell]
Found directory: ./subdir1
file1a.txt
file1b.png
Found directory: ./subdir2
Found directory: .
file2a.jpeg
file2b.html
test.py
[/shell]
```

现在，我们首先获取子目录中的文件，然后沿着目录树向上。

### **选择性递归进入子目录**

到目前为止，示例只是遍历了整个目录树，但是`os.walk`允许我们有选择地跳过树的某些部分。

对于`os.walk`给我们的每个目录，它也提供了一个子目录列表(在`subdirList`中)。如果我们修改这个列表，我们可以控制`os.walk`将进入哪个子目录。让我们调整一下上面的例子，跳过第一个子目录。

```py
[python]
import os

rootDir = '.'
for dirName, subdirList, fileList in os.walk(rootDir):
print('Found directory: %s' % dirName)
for fname in fileList:
print('\t%s' % fname)
# Remove the first entry in the list of sub-directories
# if there are any sub-directories present
if len(subdirList) > 0:
del subdirList[0]
[/python]
```

这为我们提供了以下输出:

```py
[shell]Found directory: .
file2a.jpeg
file2b.html
test.py
Found directory: ./subdir2
[/shell]
```

我们可以看到第一个子目录( *subdir1* )确实被跳过了。

这只在自顶向下遍历目录时有效，因为对于自底向上遍历，子目录在它们的父目录之前被处理，所以试图修改`subdirList`是没有意义的，因为到那时，子目录已经被处理了！

就地修改`subdirList`*也很重要，这样调用我们的代码就会看到这些变化。如果我们这样做:*

```py
[python]
subdirList = subdirList[1:]
[/python]
```

...我们将创建一个*新的*子目录列表，一个调用代码不知道的列表。

## **列出目录中文件的四种其他方式**

除了 os.walk()方法之外，还有四种方法可以列出目录中的文件:

### **#1 用 listdir()和 isfile()函数列出一个目录下的所有文件**

串联使用 listdir()和 isfile()函数可以很容易地以目录方式获得文件列表。这两个函数是操作系统模块的一部分。下面是你如何使用它们:

#### **第一步:导入操作系统模块**

操作系统模块是一个标准的 Python 模块，使用户能够使用依赖于操作系统的功能。它包含许多使用户能够与操作系统交互的方法，包括文件系统。

#### **第二步:使用 os.listdir()函数**

使用 os.listdir()函数并向其传递 *路径* 属性将返回由 *路径* 属性提供的文件和目录的名称列表。

#### **第三步:迭代结果**

写一个 for 循环，迭代函数返回的文件。

#### **第四步:使用 isfile()函数**

循环的每次迭代都必须有 os.path.isfile('path ')函数来验证当前路径是文件还是目录。

如果该函数发现它是一个文件，它返回 True，并且该文件被添加到列表中。否则该函数返回 False。

下面是一个 listdir()函数的例子，它只列出了一个目录中的文件:

```py
import os

# setting the folder path
dir_path = r'E:\\example\\'

# making a list to store files
res = []

# Iterating the directory
for path in os.listdir(dir_path):
    # check whether the current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)
print(res)
```

需要注意的是，listdir()函数只列出当前目录中的文件。

如果您熟悉生成器表达式，您可以缩短脚本并使其更简单，就像这样:

```py
import os

def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file
# Now, you can plainly call it whatever you want
for file in get_files(r'E:\\example\\'):
    print(file)
```

函数也可以用来列出文件和目录。这里有一个例子:

| 

```py
import os

# folder path
dir_path = r'E:\\account\\'

# list file and directories; Directly call the listdir() function to get the content of the directory.
res = os.listdir(dir_path)

print(res)
```

 |

### **#2 使用 os.scandir()函数获取一个目录下的文件**

众所周知，scandir()函数比 os.walk()函数更快，而且它迭代控制器的效率也更高。它是一个类似于 listdir()的目录迭代函数；唯一的区别是它产生了包含文件类型数据和名称的 *DirEntry* 对象，而不是返回一个普通文件名的列表。

利用 scandir()函数可以将 os.walk()函数的速度提高 2 到 20 倍，具体取决于操作系统和文件系统的配置。它通过避免对 os.stat()函数的不必要调用来提高速度。

需要注意的是 scandir()函数返回 os 的迭代器。DirEntry 对象，包含文件名。

早在 2015 年 9 月，scandir()函数就包含在 Python 3.5 的标准 Python 库中。

下面是一个使用函数检索目录文件的例子:

```py
import os

# retrieve all the files from inside the specified folder
dir_path = r'E:\\example\\'
for path in os.scandir(dir_path):
    if path.is_file():
        print(path.name)
```

### **#3 使用 Glob 模块**

glob 模块也是标准 Python 库的一部分。用户可以利用该模块来查找其名称遵循特定模式的文件和文件夹。

例如，如果你想获得一个目录下的所有文件，你可以使用 dire_path/*。*模式，其中" *。* "表示具有任何扩展名的文件。

下面是一个使用该模块获取目录中文件列表的例子:

```py
import glob

# search all the files inside a directory
# The *.* indicates that the file name may have any extension
dir_path = r'E:\example\*.*'
res = glob.glob(dir_path)
print(res)
```

您也可以使用该模块，通过将递归属性设置为 true 来列出子目录中的文件:

```py
import glob

# search all the files inside a directory
# The *.* indicates that the file name may have any extension

dir_path = r'E:\demos\files_demos\example\**\*.*'
for file in glob.glob(dir_path, recursive=True):
    print(file)
```

### **#4 使用 Pathlib 模块**

在 Python 3.4 中，pathlib 模块被引入到 Python 标准库中。它为大多数操作系统功能提供了包装器。它包括一些类和方法，使用户能够处理文件系统路径，并为各种操作系统检索与文件相关的数据。

下面是如何使用该模块来检索目录中的文件列表:

1.  导入 pathlib 模块。
2.  写一个 pathlib。Path('path ')行来构造目录路径。
3.  使用 iterdir()函数迭代一个目录中的所有条目。
4.  最后，使用 path.isfile()函数检查当前条目是否是一个文件。

这里有一个利用 pathlib 模块实现这一目的的示例脚本:

```py
import pathlib

# Declaring the folder path
dir_path = r'E:\\example\\'

# making a list to store file names
res = []

# constructing the path object
d = pathlib.Path(dir_path)

# iterating the directory
for entry in d.iterdir():
    # check if it a file
    if entry.is_file():
        res.append(entry)
print(res)
```

## 什么时候使用 os.listdir()函数代替 os.walk()函数比较合适？

这是程序员在了解 os.walk()之后最常问的问题之一。答案很简单:

函数将返回一个文件树中所有文件的列表，而函数将返回一个目录中所有文件和文件夹的列表。

通过一个例子可以更清楚地理解 os.listdir()函数的工作原理:

假设我们有一个目录“Example”，有三个文件夹 A、B 和 C，以及两个文本文件 1.txt 和 2 . txt。A 文件夹有另一个文件夹 Y，其中包含两个文件。B 和 C 文件夹各有一个文本文件。

将“Example”文件夹的目录路径传递给 os.listdir()方法:

```py
import os
example_directory_path = './Example'
print(os.listdir(example_directory_path))
```

上面的脚本将给你一个输出:

```py
['1.txt', '2.txt', 'A', 'B', 'C']
```

换句话说，os.listdir()函数只会生成目录的第一个“层”。这使得它与 os.walk()方法非常不同，后者搜索整个目录树。

简单地说，如果你需要一个根目录下所有文件和目录名的列表，你可以使用 os.listdir()函数。但是，如果您想查看整个目录树，使用 os.walk()方法是正确的方法。

## **结论**

使用 Python 中的 os walk 函数是以自顶向下和自底向上的方式遍历目录中所有路径的简便方法之一。在这篇文章中，我们还介绍了在目录中列出文件的其他四种方法。

现在您已经了解了不同的方法，您可以编写一个脚本来轻松地遍历目录，并将精力集中在分析文本或合并数据上。

要获得关于 Python 的`os.walk`方法的更全面的教程，请查看 Python 中递归文件和目录操作的方法[。或者看看用另一种方式遍历目录(使用递归)，看看 Python 中的递归目录遍历:列出你的电影！](https://www.pythoncentral.io/recursive-file-and-directory-manipulation-in-python-part-1/ "Recursive File and Directory Manipulation in Python")。*