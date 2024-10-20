# Python 中的 Shutil 模块

> 原文：<https://www.askpython.com/python-modules/shutil-module>

嘿！在本教程中，我们将学习 Python 的 shutil 模块中的函数。那么，我们开始吧。

Python 的 shutil 模块为我们提供了许多对文件的高级操作。我们可以复制和删除文件和目录。让我们从模块开始，详细了解每个文件的实际实现。

## 如何用 shutil 模块复制文件？

shutil 模块中有多种方法可以将一个文件的内容复制到另一个文件中。

### 1.shutil.copyfileobj(src，dst)

假设，我们想将文件 *data.txt* 的内容复制到 *data1.txt* 中，我们可以使用下面这段代码:

```py
import shutil

f=open('data.txt','r')
f1=open('data1.txt','w')

# Syntax: shutil.copyfileobj(src,dst)
shutil.copyfileobj(f,f1)

f.close()
f1.close()

```

### **2\. shutil.copy(src,dst)**

将一个文件的数据复制到另一个文件的另一种方法可以是不创建文件对象。这里，我们传递文件的相对路径。

```py
import shutil
#shutil.copy(src.dst)
shutil.copy('data.txt','data1.txt')

```

### 3\. shutil.copy2(src,dst)

*copy* (src，dst)*copy 2(src，dst)* 功能几乎相同，但 *copy2(src，dst)* 也复制源文件的元数据。

元数据包括关于文件何时被创建、访问或修改的信息。

```py
import shutil
#shutil.copy2(src,dst)
shutil.copy2('data.txt','data1.txt')

```

### 4\. shutil.copyfile(src,dst)

这里，源和目标可以是相对路径或绝对路径。假设，我们想将一个文件复制到一个文件夹中，我们可以使用下面的代码片段:

```py
import shutil
import os

path='D:\DSCracker\DS Cracker\Python'
print("Before copying file:") 
print(os.listdir(path)) 

shutil.copyfile('data.txt','Python/data3.txt')

print("After copying file:") 
print(os.listdir(path))

```

**输出:**

```py
Before copying file:
['hey.py']
After copying file:
['data3.txt', 'hey.py']

```

### 5\. shutil.move(src,dst)

假设，我们想从一个位置删除一个文件，并将其移动到另一个位置。这里，让我们将 *shutil.py* 从源文件移动到另一个位置:

```py
import shutil
import os

path='D:\DSCracker\DS Cracker'
print("Source folder:") 
print(os.listdir(path))

path1='D:\DSCracker\DS Cracker\Python'
shutil.move('shutil.py','Python')

print("After moving file shutil.py to destination folder, destination contains:") 
print(os.listdir(path1))

```

**输出:**

```py
Source folder:
['cs', 'data.txt', 'Python', 'ReverseArray', 'ReverseArray.cpp', 'shutil.py']
After moving file shutill.py to destination folder, destination contains:
['data1.txt', 'data3.txt', 'hey.py', 'nsawk.py', 'shutil.py']

```

### 6\. shutil.copytree(src,dst)

如果我们想将一个包含所有文件的完整文件夹复制到一个新位置，我们可以使用 *copytree(src，dst* )函数。

它递归地将以 *src* 为根的整个目录树复制到名为 *dst* 的目录中，并返回目标目录。

让我们将文件夹 *Python* 复制到文件夹 *Newfolder* 中。

**注意:**我们必须在目标文件夹中创建一个新文件夹，因为该功能不允许将内容复制到现有文件夹中。

所以在这里，我们在文件夹 *Newfolder* 中创建了文件夹 *python1* 。

```py
import os
import shutil

path='D:\DSCracker\DS Cracker\Python'
print("Source folder:") 
print(os.listdir(path))

shutil.copytree('Python','NewPython/python1')

path1='D:\DSCracker\DS Cracker\NewPython\python1'
print("Destination folder:")
print(os.listdir(path1))

```

**输出:**

```py
Source folder:
['data1.txt', 'data3.txt', 'hey.py', 'nsawk.py', 'shutill.py']
Destination folder:
['data1.txt', 'data3.txt', 'hey.py', 'nsawk.py', 'shutill.py']

```

## 如何用 shutil 模块移除/删除文件？

既然我们已经学习了如何移动和复制文件，那么让我们学习从 Python 脚本中的特定位置移除或删除文件。

通过使用 shutil.rmtree() ，我们可以删除任何文件夹、文件或目录。让我们删除文件夹 *Python* 。

```py
import os
import shutil

path='D:\DSCracker\DS Cracker'
print("Before deleting:") 
print(os.listdir(path))

shutil.rmtree('Python')
print("After deleting:") 
print(os.listdir(path))

```

输出:

```py
Before deleting:
['cs', 'data.txt', 'NewPython', 'program.py', 'Python', 'ReverseArray', 'ReverseArray.cpp']

After deleting:
['cs', 'data.txt', 'NewPython', 'program.py', 'ReverseArray', 'ReverseArray.cpp']

```

## 如何将一个文件的权限位复制到另一个文件？

复制文件是一部分。如果您只想将一个文件的相同权限复制到所有其他文件，该怎么办？让我们在这里使用 shutil 模块来学习如何做这件事。

### 1.shutil.copymode(夏令时，夏令时)

此方法将权限位从 src 复制到 dst。让我们将 *Python* 目录的权限位复制到 *Python1* 目录。

```py
import shutil
import os
src= 'D:\\DSCracker\\DS Cracker\\Python'
dest='D:\\DSCracker\\DS Cracker\\Python1'

print("Before using shutil.copymode(), Permission bits of destination:")
print(oct(os.stat(dest).st_mode)[-3:])

shutil.copymode(src, dest) 
print("After using shutil.copymode(), Permission bit of destination:")
print(oct(os.stat(dest).st_mode)[-3:])

```

输出:

```py
Before using shutil.copymode(), Permission bits of source:
677
After using shutil.copymode(), Permission bit of destination:
777

```

### 2\. shutil.copystat(src、dst)

shutil.copystat(src.dst)将权限位与元数据一起复制。

```py
import shutil
import os
import time 

src= 'D:\\DSCracker\\DS Cracker\\Python'
dest='D:\\DSCracker\\DS Cracker\\Python1'

print("Before using shutil.copystat():")
print("Permission bits:",oct(os.stat(src).st_mode)[-3:])
print("Last modification time:", time.ctime(os.stat(src).st_mtime)) 

print("Modification time:",time.ctime(os.stat(src).st_mtime))

shutil.copystat(src, dest) 

print("After using shutil.copystat():")
print("Permission bits:",oct(os.stat(dest).st_mode)[-3:])
print("Last modification time:", time.ctime(os.stat(dest).st_mtime)) 
print("Modification time:",time.ctime(os.stat(dest).st_mtime))

```

输出:

```py
Before using shutil.copystat():
Permission bits: 777
Last modification time: Mon Dec  7 02:20:37 2020
Modification time: Mon Dec  7 02:20:37 2020

After using shutil.copystat():
Permission bits: 777
Last modification time: Mon Dec  7 03:43:47 2020
Modification time: Mon Dec  7 03:43:47 2020

```

## shutil 模块中的其他功能

现在让我们来看看 shutil 模块的其他函数。

### 1.shutil.disk_usage(路径)

*shutil.disk_usage(path)* 函数以元组的形式返回给定路径名的磁盘使用统计信息，其属性为 *total* 即内存总量， *used* 即已用空间， *free* 即可用空间(以字节为单位)。

```py
import shutil
import os

path = 'D:\\DSCracker\\DS Cracker\\NewPython\\python1'

statistics=shutil.disk_usage(path)

print(statistics)

```

输出:

```py
usage(total=1000203087872, used=9557639168, free=990645448704)

```

### 2.shutil.which()

*shutil.which()* 函数返回可执行应用程序的路径，如果调用给定命令 cmd，该应用程序将运行。

```py
import shutil
import os

cmd='Python'

locate = shutil.which(cmd) 

print(locate)

```

输出:

```py
C:\Users\AskPython\AppData\Local\Microsoft\WindowsApps\Python.EXE

```

## 结论

在本教程中，我们介绍了如何使用 python 中的 **shutil 模块**来复制、删除和处理文件和文件夹的其他操作。希望你们都喜欢。敬请期待！

## 参考

[shutil-高层文件操作正式文档](https://docs.python.org/3/library/shutil.html)