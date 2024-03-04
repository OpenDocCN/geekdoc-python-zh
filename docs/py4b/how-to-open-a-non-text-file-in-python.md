# 如何在 Python 中打开非文本文件

> 原文：<https://www.pythonforbeginners.com/files/how-to-open-a-non-text-file-in-python>

Python 标准库提供了几个读写文本文件的工具，但是非文本文件呢？使用 Python，也可以打开非文本文件。我们可以用多种方式来做这件事。

在大多数情况下，使用标准库可以打开 Python 中的非文本文件。但是在一些模块的帮助下，我们可以更进一步。

在本教程中，我们将介绍几种用 Python 打开文本文件的方法。使用 Python 代码示例，我们将探索读取和写入二进制文件，这个过程包括将 ASCII 数据转换为字节。

我们还将向您展示如何使用用于创建和编辑图像文件的**枕头**模块。接下来，我们将看看如何使用**波**模块来读取音频数据。作为奖励，我们提供了一个用 Python 从头开始生成音频文件的例子。

我们还将研究如何使用 **open** ()函数来读取。csv 文件，并将其与打开非二进制文件进行比较。如果您对学习如何在 Python 中读取非文本文件感兴趣，那么您来对地方了。

## 文本文件和非文本文件有什么区别？

文本文件通常由单词和数字组成。这些文件有文本行，通常是用人们能读懂的语言写的(相对于机器)。

文本文件使用 ASCII ( **A** 美国 **S** 标准**C**ode for**I**information**I**interchange)字符来表示字母和数字。文本文件通常以扩展名*结尾。虽然情况并不总是如此。*

另一方面，非文本文件是包含 ASCII 文本以外的数据的文件。有很多这样的文件。通常，用 Python 打开一个非文本文件所需要的就是随该语言发布的标准库。但是在一两个模块的帮助下，我们可以将我们的 Python 技能提升到另一个水平。

## 用 Python 读写二进制(非文本)文件

二进制文件也称为非文本文件，是包含多组二进制数字(位)的文件。二进制文件将比特分成八个序列，称为一个**字节**。通常，这些字节代表的不是文本数据。

有了 Python，我们可以使用标准函数读写二进制文件。例如，我们可以使用 **open** ()函数创建一个新的二进制文件。为此，我们需要向函数传递一些特殊字符。这告诉函数我们想要在*写模式* (w)和*二进制模式* (b)下打开文件。

打开一个新文件后，我们可以使用 **bytearray** ()函数将一列数字转换成字节。bytearray()函数用于将对象转换为字节数组。可以使用**写**()功能将二进制数据保存到磁盘。

#### 示例:将二进制文件写入磁盘

```py
f = open("binary_file", 'w+b')
byte_arr = [1,2,3,4]
# convert data to a byte array
binary_format = bytearray(byte_arr)
f.write(binary_format)
f.close() 
```

同样，我们可以使用 **open** ()函数在 Python 中打开一个非文本文件。当读取二进制文件类型时，需要将字符“r”和“b”传递给 open()函数。这告诉 open()函数我们打算以二进制模式读取文件。

#### 示例:读取二进制文件

```py
file = open("binary_file",'rb')
number = list(file.read())
print("Binary data = ", number)
file.close() 
```

**输出**

```py
Binary data =  [1, 2, 3, 4]
```

## 如何将二进制数据转换为文本

使用标准的 Python 方法，可以将文本转换成二进制数据，反之亦然。使用 **decode** ()方法，我们可以将二进制数据转换成 ASCII 文本。这样做可以让我们获取二进制信息，并将其转换为人类可以阅读的内容。

#### 示例:在 Python 中转换二进制数据

```py
# binary to text
binary_data = b'Hello World.'
text = binary_data.decode('utf-8') # Change back into ASCII
print(text)

# convert text to bytes
binary_message = text.encode('utf-8')
print(type(binary_message))

binary_data = bytes([65, 66, 67])  # ASCII values for the letters A, B, and C
text = binary_data.decode('utf-8')
print(text) 
```

## 在 Python 中打开图像文件

可以使用 open()函数打开图像文件。通过将结果赋给一个新变量，我们可以打开一个图像文件。在下面的例子中，我们将尝试打印以读取模式打开的图像文件的内容。

```py
file = open("boat.jpg",'r')
print(file) 
```

**输出**

```py
<_io.TextIOWrapper name='boat.jpg' mode='r' encoding='cp1252'>
```

使用 Python 模块，我们可以做的不仅仅是打开图像文件。有了枕头模块，我们可以随心所欲地处理图像。

在使用枕头模块之前，您需要安装它。安装 Python 模块最简单的方法是从命令提示符或终端运行 pip。

使用以下命令安装 pip。

```py
pip install pillow
```

在电脑上安装 Pillow 后，您可以打开并编辑照片和其他图像文件。使用 Pillow，我们可以读取图像文件并将其尺寸打印到控制台。

### 如何使用枕头模块

在我们可以使用 pillow 模块之前，我们必须让 Python 知道我们想在我们的程序中使用它。我们通过导入模块来实现这一点。在下面的例子中，我们使用来自的**和来自**的**来包含来自 PIL (Pillow)的**图像**模块。**

#### 示例:用 PIL 打开图像文件

```py
from PIL import Image

img = Image.open("boat.jpg")

#open the original image
size = img.size

print(size)
# opens the image in the default picture viewer
img.show() 
```

Pillow 包括编辑照片的模块。使用 Pillow 附带的 **ImageOps** ，我们可以反转照片，并使用您计算机上的默认照片查看器显示它。

#### 示例:使用 PIL 反转图像

```py
from PIL import Image, ImageOps

img = Image.open("boat.jpg")

#open the original image
size = img.size

print(size)
# invert the image with ImageOps
img = ImageOps.invert(img)
# opens the image in the default picture viewer
img.show() 
```

## 在 Python 中打开音频文件

Python 提供了读取和创建音频文件的工具。使用**波**模块，我们可以打开。wav 音频文件并检查它们的数据。在下面的例子中，我们将打开一个名为“portal.wav”的. wav 文件，并将其采样率打印到控制台。

#### 示例:用波形模块打开一个. wav 文件

```py
import wave

audio = wave.open("portal.wav",'r')

print("Sample Rate: ", audio.getframerate()) 
```

**输出**

```py
Sample Rate:  44100
```

更进一步，我们可以使用 Wave 模块从头开始生成我们自己的音频文件。通过给音频帧分配一个随机值，我们可以生成一个静态噪声的音频文件。

#### 示例:使用 Wave 模块生成音频文件

```py
import wave, struct, math, random
# this example will generate an audio file of static noise

sample_rate = 44100.0 # hertz
duration = 1.0 # seconds
frequency = 440.0 # hertz

sound = wave.open("sound.wav",'w')
sound.setnchannels(1) # mono
sound.setsampwidth(2)
sound.setframerate(sample_rate)

for i in range(99999):
    # 32767 is the maximum value for a short integer.
   random_val = random.randint(-32767, 32767)
   data = struct.pack('h', random_val)
   sound.writeframesraw(data)

sound.close() 
```

## 如何在 Python 中打开 CSV 文件

逗号分隔值文件(通常称为 CSV 文件)是存储和交换数据的便捷方式。这些文件通常包含由逗号分隔的数字和/或字母。

即使 CSV 文件不以。txt 扩展名，它们被视为文本文件，因为它们包含 ASCII 字符。作为 Python 开发人员，学习如何打开 CSV 文件是一项有用的技能。使用下面的例子来比较在 Python 中打开非文本文件和文本文件。

使用下面的示例 CSV 文件，我们将探索如何使用 Python 读取 CSV 数据。

*animal_kingdom.csv*

```py
"amphibians","reptiles","birds","mammals"
"salamander","snake","owl","coyote"
"newt","turtle","bald eagle","raccoon"
"tree frog","alligator","penguin","lion"
"toad","komodo dragon","chicken","bear" 
```

#### 示例:使用 Open()读取 CSV 文件

```py
with open("animal_kingdom.csv",'r') as csv_file:
    file = csv_file.read()

    print(file) 
```

运行上面的代码会将 *animal_kingdom.csv* 的内容打印到控制台。然而，有一种更好的方法来读取 Python 中的 CSV 文件。

### 如何使用 CSV 模块

CSV 模块预装在 Python 中，所以不需要安装它。使用 CSV 模块可以让我们更好地控制 CSV 文件的内容。例如，我们可以使用 **reader** ()函数从文件中提取字段数据。

#### 示例:使用 csv 模块读取 Python 中的 csv 文件

```py
import csv

with open("animal_kingdom.csv",'r') as csv_file:
    csvreader = csv.reader(csv_file)

    # read the field names
    fields = csvreader.__next__()
    print('Field Names\n----------')
    for field in fields:
        print(field)

    print('\nRows\n---------------------')
    for row in csvreader:
        for col in row:
            print(col,end="--")
        print("\n") 
```

## 结论

我们研究了 Python 中处理非文本文件的各种方法。通过将 ASCII 文本转换成字节数组，我们可以创建自己的二进制文件。在 read()函数的帮助下，我们可以读取二进制数据并将其打印到控制台。

Python 模块 Pillow 是打开和编辑图像文件的绝佳选择。如果您有兴趣了解更多关于图像处理的知识，枕头模块是一个很好的起点。

Wave 模块是用 Python 发布的。用它来读写波形数据。通过一点数学知识，Wave 模块可以用来生成各种声音效果。

## 相关职位

如果您觉得本教程很有帮助，请点击下面的链接，了解更多关于 Python 编程的精彩世界。

*   学习 [Python 读取文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)操作与我们的操作指南
*   [Python 列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)适合初学者