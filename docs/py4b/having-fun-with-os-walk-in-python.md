# 与操作系统同乐。穿蟒蛇皮走路

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/having-fun-with-os-walk-in-python>

## 概观

```py
 **OS.walk()** generate the file names in a directory tree by walking the tree either
top-down or bottom-up.

For each directory in the tree rooted at directory top (including top itself),
it yields a 3-tuple (dirpath, dirnames, filenames).

**Paths**
root :  Prints out directories only from what you specified
dirs : Prints out sub-directories from root. 
files:  Prints out all files from root and directories 
```

##### walkFileSystem.py

```py
 Open an text editor , copy & paste the code below.

Save the file as walkFileSystem.py and exit the editor.

Run the script:
$ python walkFileSystem.py 
```

```py
import os

os.system("clear")
print "-" * 80
print "OS Walk Program"
print "-" * 80
print "
"

print "Root prints out directories only from what you specified"
print "-" * 70

print "Dirs prints out sub-directories from root"
print "-" * 70

print "Files prints out all files from root and directories"
print "-" * 70

print "This program will do an os.walk on the folder that you specify"
print "-" * 70

path = raw_input("Specify a folder that you want to perform an 'os.walk' on: >> ")

for root, dirs, files in os.walk(path):

    print root
    print "---------------"

    print dirs
    print "---------------"

    print files
    print "---------------"

```

```py
 More reading can be found [here](https://docs.python.org/2/library/os.html "library_os") 
```