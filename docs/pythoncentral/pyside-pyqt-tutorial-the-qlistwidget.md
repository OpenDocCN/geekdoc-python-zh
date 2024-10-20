# PySide/PyQt 教程:QListWidget

> 原文：<https://www.pythoncentral.io/pyside-pyqt-tutorial-the-qlistwidget/>

Qt 有几个允许单列列表选择器控件的小部件——为了简洁和方便，我们称它们为列表框。最灵活的方法是使用 QListView，它在高度灵活的列表模型上提供了一个 UI 视图，这个模型必须由程序员定义；更简单的方法是使用 QListWidget，它有一个预定义的基于项目的模型，允许它处理列表框的常见用例。我们将从更简单的 QListWidget 开始。

## **qlistwidget**

`QListWidget`的构造函数类似于许多 QWidget 派生的对象，并且只接受一个可选的`parent`参数:

```py

self.list = QListWidget(self)

```

### **填充 QListWidget**

用条目填充 QListWidget 很容易。如果您的项目是纯文本，您可以单独添加它们:

```py

for i in range(10):

self.list.addItem('Item %s' % (i + 1))

```

或散装:

```py

items = ['Item %s' % (i + 1)

for i in range(10)]

self.list.addItems(items)

```

您还可以使用`QListWidgetItem`类添加稍微复杂一些的列表项。可以单独创建一个`QListWidgetItem`,然后使用列表的`addItem`方法将其添加到列表中:

```py

item = QListWidgetItem()

list.addItem(item)

```

### **更复杂的 QListWidget I** **tems**

或者它可以以*列表*为父列表创建，在这种情况下，它会自动添加到列表中:

```py

item = QListWidgetItem(list)

```

一个`item`可以通过它的`setText`方法设置*文本*:

```py

item.setText('I am an item')

```

和一个使用其`setIcon`方法设置为`QIcon`实例的图标:

```py

item.setIcon(some_QIcon)

```

您也可以在`QListWidgetItem`的构造函数中指定*文本*或*图标*和*文本*:

```py

item = QListWidgetItem('A Text-Only Item')

item = QListWidgetItem(some_QIcon, 'An item with text and an icon')

```

上述每个构造函数签名也可以选择接受一个父级。

### **使用 QListWidget**

`QListWidget`提供了几个方便的信号，可以用来响应用户输入。最重要的是`currentItemChanged`信号，当用户*改变*所选项目时*发出*；它的槽*接收*两个参数，`current`和`previous`，它们是当前和先前选择的`QListWidgetItems`。当用户*点击*、*双击*、*激活*或*按下*一项时，以及当所选项目组*改变*时，也有信号。

要获取当前选中的项目，您可以使用由`currentItemChanged`信号传递的参数，也可以使用 QListWidget 的`currentItem`方法。

### **关于 QIcons 的说明**

定制`QListWidgetItem`的少数方法之一是添加一个*图标*，所以了解一下`QIcons`是很重要的。有许多方法可以建造一个`QIcon`；您可以通过以下方式创建它们:

*   提供文件名:`icon = QIcon('/some/path/to/icon.png')`。
*   使用主题图标:`icon = QIcon.fromTheme('document-open')`。
*   从一个`QPixMap` : `icon = QIcon(some_pixmap)`。

和许多其他人。对不同方法的一些评论:首先，注意基于文件的创建支持广泛的但不是无限的文件类型集；您可以通过运行`QImageReader().supportedImageFormats()`找到您的版本和平台支持哪些。在我的系统上，它返回:

```py

[PySide.QtCore.QByteArray('bmp'),

PySide.QtCore.QByteArray('gif'),

PySide.QtCore.QByteArray('ico'),

PySide.QtCore.QByteArray('jpeg'),

PySide.QtCore.QByteArray('jpg'),

PySide.QtCore.QByteArray('mng'),

PySide.QtCore.QByteArray('pbm'),

PySide.QtCore.QByteArray('pgm'),

PySide.QtCore.QByteArray('png'),

PySide.QtCore.QByteArray('ppm'),

PySide.QtCore.QByteArray('svg'),

PySide.QtCore.QByteArray('svgz'),

PySide.QtCore.QByteArray('tga'),

PySide.QtCore.QByteArray('tif'),

PySide.QtCore.QByteArray('tiff'),

PySide.QtCore.QByteArray('xbm'),

PySide.QtCore.QByteArray('xpm')]

```

如我所说，选择范围很广。在成熟的平台之外，基于主题的图标创建是有问题的；在 Windows 和 OS X 上你应该没问题，如果你在 Linux 上使用 Gnome 或 KDE 也一样，但是如果你使用不太常见的桌面环境，比如 OpenBox 或 XFCE，Qt 可能找不到你的图标；有一些方法可以解决这个问题，但是没有好的方法，所以你可能只能使用文本。

### **qlist widget 示例**

让我们创建一个简单的列表小部件，显示目录中所有图像的文件名和缩略图图标。因为这些项目很简单，可以创建为一个`QListWidgetItem`，我们将让它从`QListWidget`继承。

首先，我们需要知道您的安装支持什么样的图像格式，这样我们的列表控件就可以知道什么是有效的图像。我们可以用上面提到的方法，`QImageReader().supportedImageFormats()`。在返回之前，我们将把它们都转换成字符串:

```py

def supported_image_extensions():

''' Get the image file extensions that can be read. '''

formats = QImageReader().supportedImageFormats()

# Convert the QByteArrays to strings

return [str(fmt) for fmt in formats]

```

现在我们有了它，我们可以构建我们的图像列表小部件；我们称之为`ImageFileWidget`。它将从`QListWidget`继承，除了一个可选的`parent`参数之外，像所有的`QWidgets`一样，它将需要一个必需的`dirpath`:

```py

class ImageFileList(QListWidget):

''' A specialized QListWidget that displays the list

of all image files in a given directory. '''

def __init__(self, dirpath, parent=None):

QListWidget.__init__(self, parent)

```

我们希望它有一种方法来确定给定目录中的图像。我们将给它一个`_images`方法，该方法将返回指定目录中所有有效图像的文件名。它将使用`glob`模块的`glob`函数，该函数对文件和目录路径进行 shell 风格的模式匹配:

```py

def _images(self):

''' Return a list of file-names of all

supported images in self._dirpath. '''
#从空列表开始
 images = []
#查找每个有效的
 #扩展名的匹配文件，并将它们添加到图像列表中。
对于 supported_image_extensions()中的扩展:
 pattern = os.path.join(self。_dirpath，
 '*。% s ' % extension)
images . extend(glob(pattern))
返回图像

```

既然我们已经有了一种方法来确定目录中有哪些图像文件，那么将它们添加到我们的`QListWidget`中就很简单了。对于每个文件名，我们创建一个以列表为父的`QListWidgetItem`,将其文本设置为文件名，将其图标设置为从文件创建的`QIcon`:

```py

def _populate(self):

''' Fill the list with images from the

current directory in self._dirpath. '''
#如果我们要重新填充，请清除列表
 self.clear()
#为每个图像文件创建一个列表项，
 #为自己的图像设置适当的文本和图标
。_ images():
item = QListWidgetItem(self)
item . settext(image)
item . seticon(QIcon(image))

```

最后，我们将添加一个方法来设置目录路径，每次调用它时都会重新填充列表:

```py

def setDirpath(self, dirpath):

''' Set the current image directory and refresh the list. '''

self._dirpath = dirpath

self._populate()

```

我们将在构造函数中添加一行代码来调用`setDirpath`方法:

```py

self.setDirpath(dirpath)

```

这就是我们`ImageFileList`类的最终代码:

```py

class ImageFileList(QListWidget):

''' A specialized QListWidget that displays the

list of all image files in a given directory. '''
def __init__(self，dirpath，parent=None): 
 QListWidget。__init__(self，parent)
self . setdirpath(dirpath)
def setDirpath(self，dirpath): 
' ' '设置当前图像目录并刷新列表'‘
自我。_dirpath = dirpath 
 self。_ 填充()
def _images(self): 
' ' '返回 self 中所有
支持的图像的文件名列表。_dirpath。''
#从空列表开始
 images = []
#为每个有效的
 #扩展名找到匹配的文件，并将它们添加到 supported _ image _ extensions():
pattern = OS . path . join(self。_dirpath，
 '*。% s ' % extension)
images . extend(glob(pattern))
返回图像
def _populate(self): 
' ' '用 self 中的
当前目录中的图像填充列表。_dirpath。''
#如果我们要重新填充，请清除列表
 self.clear()
#为每个图像文件创建一个列表项，
 #为自己的图像设置适当的文本和图标
。_ images():
item = QListWidgetItem(self)
item . settext(image)
item . seticon(QIcon(image))

```

因此，让我们将`ImageFileList`放在一个简单的窗口中，这样我们就可以看到它的运行。我们将创建一个`QWidget`作为我们的窗口，在其中添加一个`QVBoxLayout`，并添加`ImageFileList`，以及一个*入口小部件*，它将显示当前选中的项目。我们将使用`ImageFileList`的`currentItemChanged`信号来保持它们同步。

我们将创建一个`QApplication`对象，传递给它一个空列表，这样我们就可以使用`sys.argv[1]`来传递图像目录:

```py

app = QApplication([])

```

然后，我们将创建窗口，设置最小尺寸并添加布局:

```py

win = QWidget()

win.setWindowTitle('Image List')

win.setMinimumSize(600, 400)

layout = QVBoxLayout()

win.setLayout(layout)

```

然后，我们将实例化一个`ImageFileList`，传递接收到的图像目录路径和我们的窗口作为它的父节点:

```py

first = ImageFileList(sys.argv[1], win)

```

并添加我们的入口小部件:

```py

entry = QLineEdit(win)

```

并将这两个小部件添加到我们的布局中:

```py

layout.addWidget(first)

layout.addWidget(entry)

```

然后，我们需要创建一个*槽*函数，在当前项改变时调用；它必须接受参数`curr`和`prev`、当前和先前选择的项目，并且应该将条目的文本设置为当前项目的文本:

```py

def on_item_changed(curr, prev):

entry.setText(curr.text())

```

然后，我们把它接到信号上:

```py

lst.currentItemChanged.connect(on_item_changed)

```

剩下的就是显示窗口和运行应用程序:

```py

win.show()

app.exec_()

```

我们的最后一部分，包装在标准的`if __name__ == '__main__'`块中，是:

```py

if __name__ == '__main__':

# The app doesn't receive sys.argv, because we're using

# sys.argv[1] to receive the image directory

app = QApplication([])
#创建一个窗口，设置它的大小，给它一个布局
win = q widget()
win . setwindowtitle('图像列表')
 win.setMinimumSize(600，400)
layout = QVBoxLayout()
win . set layout(布局)
#使用从命令行
lst = ImageFileList(sys . argv[1]，win)传入的 image 
 #目录创建我们的 image filelist 对象之一
layout.addWidget(lst)
entry = QLineEdit(win)
layout.addWidget(条目)
def on_item_changed(curr，prev):
entry . settext(curr . text())
lst . currentitemchanged . connect(on _ item _ changed)
win.show() 
 app.exec_() 

```

运行我们的整个示例要求您有一个装满图像的目录；我在我的 Linux 发行版的`/usr/share/icons`目录中使用了一个作为例子:

```py

python imagelist.py /usr/share/icons/nuoveXT2/48x48/devices

```

但是你必须找到你自己的。几乎任何图像都可以。

很明显,`QListWidget`是一个非常简单的小部件，没有提供很多选项；对于很多用例来说，这是不够的。对于这些情况，您可能会使用一个`QListView`，我们将在下一期讨论这个问题。