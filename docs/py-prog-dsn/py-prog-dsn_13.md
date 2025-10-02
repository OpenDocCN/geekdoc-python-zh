# 附录

# 1 Python 异常处理参考

## 1 Python 异常处理参考

本节简单罗列 Python 语言中与异常处理有关的常用语句形式及用法。 发生错误时通常由系统自动抛出异常，但也可由程序自己抛出并捕获。

捕获并处理异常：try-except

发生错误时，如果应用程序没有预定义的处理代码，则由 Python 的缺省异常处理机制 来处理，处理动作是中止应用程序并显示错误信息。如果程序自己处理异常，可编写 try-except 语句来定义异常处理代码。详见前面各节。

手动抛出异常：raise 异常可以由系统自动抛出，也可以由我们自己的程序手动抛出。Python 提供 raise 语句

用于手动抛出异常。下面的语句抛出一个 ValueError 异常，该异常被 Python 的缺省异常处 理程序捕获：

```py
>>> raise ValueError
Traceback (most recent call last): File "&lt;stdin&gt;", line 1, in &lt;module&gt;
ValueError 
```

除了错误类型，raise 语句还可以带有错误的描述信息：

```py
>>> raise ValueError, "Wrong value!"
Traceback (most recent call last): File "&lt;stdin&gt;", line 1, in &lt;module&gt;
ValueError: Wrong value! 
```

当然也可以由程序自己处理自己抛出的异常，例如

```py
>>> try:
        raise ValueError 
    except ValueError:
        print "Exception caught!"
Exception caught! 
```

用户自定义异常

前面程序例子中抛出的都是 Python 的内建异常，我们也可以定义自己的异常类型。为 此目的，需要了解 Python 的异常类 Exception 以及类、子类、继承等面向对象程序设计概念， 这些概念将在第 x 章中介绍。这里我们用下面的简单例子演示大致用法，以使读者先获得一 个初步印象：

```py
>>> class MyException(Exception): 
        pass 
```

这是一个类定义，它在 Python 内建的 Exception 类的基础上定义了我们自己的异常类 MyException。虽然语句 pass 表明我们并没有在 Exception 类的基础上添加任何东西，但 MyException 确实是一个新的异常类，完全可以像 Python 内建的各种异常一样进行抛出、捕 获。例如：

```py
>>> try:
        raise MyException 
    except MyException:
        print "MyException caught!"
MyException caught! 
```

确保执行的代码：try-finally

一般来说，发生异常之后，控制都转到异常处理代码，而正常算法部分的代码不再执行。

Python 的异常处理还允许我们用 try-finally 语句来指定这样的代码：不管是否发生异常，这 些代码都必须执行。这种机制可以用来完成出错后的扫尾工作。例如：

```py
>>> try:
        x = input("Enter a number: ") 
        print x
    finally:
        print "This is final!"
Enter a number: 123
123
This is final! 
```

本例中，我们为 x 输入了一个正常数值 123，故 try 语句块没有发生异常，显示 123 后 又执行了最后的 print 语句。为什么不写成如下形式呢？

```py
x = input("Enter a number: ") 
print x
print "This is final!" 
```

区别在于，当发生错误时，这种写法就有可能未执行最后的 print 语句，而 try-finally 的写法则在发生异常的情况下也会确保执行最后的 print 语句。例如我们再次执行上面的语 句：

```py
>>> try:
        x = input("Enter a number: ") 
        print x
    finally:
        print "This is final!"
Enter a number: abc
This is final!
Traceback (most recent call last): File "&lt;stdin&gt;", line 2, in &lt;module&gt; File "&lt;string&gt;", line 1, in &lt;module&gt;
NameError: name 'abc' is not defined 
```

可见，由于输入数据错误，导致 try 语句块发生异常而无法继续，但 finally 下面的 语句却得到了执行。仅当 finally 部分确保执行之后，控制才转到（缺省）异常处理程序 来处理捕获到的异常。

一般形式：try-except-finally

这种形式的异常处理语句综合了 try-except 和 try-finally 的功能。首先执行 try 部分，如 果一切正常，再执行 finally 部分。try 部分如果出错，则还是要执行 finally 部分，然后再由 except 部分来处理异常。

# 2 Tkinter 画布方法

## 2 Tkinter 画布方法

本节罗列 Canvas 对象的方法，供需要的读者编程时参考。具体用法请查阅参考资料。

创建图形项的方法

*   create_arc(<限定框>, <选项>)：创建弧形，返回标识号
*   create_bitmap(<位置>, <选项>)：创建位图，返回标识号
*   create_image(<位置>, <选项>)：创建图像，返回标识号
*   create_line(<坐标序列>, <选项>)：创建线条，返回标识号
*   create_oval(<限定框>, <选项>)：创建椭圆形，返回标识号
*   create_polygon(<坐标序列>, <选项>)：创建多边形，返回标识号
*   create_rectangle(<限定框>, <选项>)：创建矩形，返回标识号
*   create_text(<位置>, <选项>)：创建文本，返回标识号
*   create_window(<位置>, <选项>)：创建窗口型构件，返回标识号

操作画布上图形项的方法

*   delete(<图形项>)：删除图形项
*   itemcget(<图形项>, <选项>)：获取某图形项的选项值
*   itemconfig(<图形项>, <选项>)：设置图形项的选项值
*   itemconfigure(<图形项>, <选项>)：同上
*   coords(<图形项>)：返回图形项的坐标
*   coords(<图形项>, x0, y0, x1, y1, ..., xn, yn)：改变图形项的坐标
*   bbox(<图形项>)：返回图形项的界限框（坐标）
*   bbox()：返回所有图形项的界限框
*   canvasx(<窗口坐标 x>)：将窗口坐标 x 转换成画布坐标 x
*   canvasy(<窗口坐标 y>)：将窗口坐标 y 转换成画布坐标 y
*   type(<图形项>)：返回图形项的类型
*   lift(<图形项>)：将图形项移至画布最上层
*   tkraise(<图形项>)：同上
*   lower(<图形项>)：将图形项移至画布最底层
*   move(<图形项>, dx, dy)：将图形项向右移动 dx 单位，向下移动 dy 单位
*   scale(<图形项>, <x 比例>, <y 比例>, <x 位移>, <y 位移>)：根据比例缩放图形项

查找画布上图形项的方法

下列方法用于查找某些项目组。对每个方法，都有对应的 addtag 方法。不是处理 find 方法返回的每个项目，而是为一组项目增加一个临时标签、一次性处理所有具有该标签的项 目、然后删除该标签，常常可以得到更好的性能。

*   find_above(<图形项>)：返回位于给定图形项之上的图形项
*   find_all() ：返回画布上所有图形项的标识号构成的元组，等于 find_withtag(ALL)
*   find_below(<图形项>)：返回位于给定图形项之下的图形项
*   find_closest(x, y)：返回与给定位置最近的图形项，位置以画布坐标给出
*   find_enclosed(x1, y1, x2, y2)：返回被给定矩形包围的所有图形项
*   find_overlapping(x1, y1, x2, y2)：返回与给定矩形重叠的所有图形项
*   find_withtag(<图形项>)：返回与给定标识匹配的所有图形项

操作标签的方法

*   addtag_above(<新标签>, <图形项>)：为位于给定图形项之上的图形项添加新标签
*   addtag_all(<新标签>)：为画布上所有图形项添加新标签，即 addtag_withtag(<新 标签>, ALL)
*   addtag_below(<新标签>, <图形项>)：为位于给定图形项之下的图形项添加新标签
*   addtag_closest(<新标签>, x, y)：为与给定坐标最近的图形项添加新标签
*   addtag_enclosed(<新标签>, x1, y1, x2, y2)：为被给定矩形包围的所有图形项添 加新标签
*   addtag_overlapping(<新标签>, x1, y1, x2, y2) ：为与给定矩形重叠的所有图 形项添加新标签
*   addtag_withtag(<新标签>, <标签>)：为具有给定标签的所有图形项添加新标签
*   dtag(<图形项>, <标签>)：为给定图形项删除给定标签
*   gettags(<图形项>：返回与给定图形项关联的所有标签

# 3 Tkinter 编程参考

## 3 Tkinter 编程参考

# 3.1 构件属性值的设置

### 3.1 构件属性值的设置

Tkinter 构件对象有很多属性，这些属性的值可以在创建实例时用关键字参数指定（未 指定值的属性都有缺省值）：

```py
<构件类>(<父构件>,<属性>=<值>,...) 
```

也可以在创建对象之后的任何时候通过调用对象的 configure（或简写为 config）方法来更改 属性值：

```py
<构件实例>.config(<属性>=<值>,...) 
```

构件类还实现了一个字典接口，可使用下列语法来设置和查询属性：

```py
<构件实例>["<属性>"] = <值> value = <构件实例>["<属性>"] 
```

由于每个赋值语句都导致对 Tk 的一次调用，因此若想改变多个属性的值，较好的做法 是用 config 一次性赋值。

有些构件类的属性名称恰好是 Python 语言的保留字（如 class、from 等），当用关键词 参数形式为其赋值时，需要在选项名称后面加一个下划线（如 class*、from*等）。

# 3.2 构件的标准属性

### 3.2 构件的标准属性

Tkinter 为所有构件提供了一套标准属性，用来设置构件的外观（大小、颜色、字体等） 和行为。

设置构件的长度、宽度等属性时可选用不同的单位。缺省单位是像素，其他单位包括 c（厘米）、i（英寸）、m（毫米）和 p（磅，约 1/72 英寸）。

颜色

多数构件具有 background（可简写为 bg）和 foreground（可简写为 fg）属性，分别用于 指定构件的背景色和前景（文本）色。颜色可用颜色名称或红绿蓝（RGB）分量来定义。

所有平台都支持的常见颜色名称有"white"、"black"、"red"、"green"、"blue"、"cyan"、"yellow"、"magenta"等，其他颜色如 LightBlue、Moccasin、PeachPuff 等等也许依赖于具体的安装平台。颜色名称不区分大小写。大多数复合词组成的颜色名称也可以在使用单词间加 空格的形式，如"light blue"。

通过 RGB 分量值来指定颜色需使用特定格式的字符串："#RGB"、"#RRGGBB"、 "#RRRGGGBBB"和"#RRRRGGGGBBBB"，它们分别用 1～4 个十六进制位来表示红绿蓝分 量值，即分别将某颜色分量细化为 16、256、4096 和 65536 级。如果读者不熟悉十六进制， 可以用下面这个方法将十进制数值转换成颜色格式字符串，其中宽度可选用 01～04：

```py
my_color = "#%02x%02x%02x" % (128,192,200) 
```

字体

多数构件具有 font 属性，用于指定文本的字体。一般情况下使用构件的缺省字体即可， 如果实在需要自己设置字体，最简单的方法是使用字体描述符。

字体描述符是一个三元组，包含字体族名称、尺寸（单位为磅）和字形修饰，其中尺寸 和字形修饰是可选的。当省略尺寸和字形修饰时，如果字体族名称不含空格，则可简单地用 字体族名称字符串作为字体描述符，否则必须用元组形式（名称后跟一个逗号）。例如下列 字体描述符都是合法的：

```py
("Times",10,"bold") ("Helvetica",10,"bold italic") ("Symbol",8) ("MS Serif",) "Courier" 
```

Windows 平台上常见的字体族有 Arial、Courier New（或 Courier）、Comic Sans MS、Fixedsys、Helvetica（同 Arial）、MS Sans Serif、MS Serif、Symbol、System、Times New Roman（或 Times）和 Verdana 等。字形修饰可以从 normal、bold、roman、italic、underline 和 overstrike 中选用一个或多个。

除了字体描述符，还可以创建字体对象，这需要导入 tkFont 模块，并用 Font 类来创建 字体对象。在此不详述。

边框

Tkinter 的所有构件都有边框，某些构件的边框在缺省情形下不可见。边框宽度用 borderwidth（可简写为 border 或 bd）设置，多数构件的缺省边框宽度是 1 或 2 个像素。可 以用属性 relief 为边框设置 3D 效果，可用的 3D 效果有'flat'或 FLAT、'groove'或 GROOVE、 'raised'或 RAISED、'ridge'或 RIDGE、'solid'或 SOLID、'sunken'或 SUNKEN（见图 8.28）。

![](img/程序设计思想与方法 347519.png)

图 8.28 按钮边框 3D 效果

文本

标签、按钮、勾选钮等构件都有 text 属性，用于指定有关的文本。文本通常是单行的， 但利用新行字符\n 可以实现多行文本。多行文本的对齐方式可以用 justify 选项设置，缺省 值是 CENTER，可用值还有 LEFT 或 RIGHT。

图像

很多构件都有 image 属性，用于显示图像。例如命令按钮上可以显示图像而不是文本，标签也可以是图像而非文本，Text 构件可以将文本和图像混合编辑。

image 属性需要一个图像对象作为属性值。图像对象可以用 PhotoImage 类来创建，图像 的来源可以是.gif 等格式的图像文件。

例如：

```py
>>> root = Tk()
>>> img ＝ PhotoImage(file="d:\mypic.gif")
>>> Button(root,image=img).pack() 
```

# 3.3 各种构件的属性

### 3.3 各种构件的属性

除了标准属性，每种构件类还有独特的属性。这里仅以 Button 类为例列出按钮构件的 常用属性，其他构件类仅列出类名，具体有哪些属性请查阅 Tkinter 参考资料。

Button

构造器：`Button(parent, option = value, ... )`

常用选项：

*   anchor：指定按钮文本在按钮中的位置（用方位值表示）。
*   bd 或 borderwidth：按钮边框的宽度，缺省值为 2 个像素。
*   bg 或 background：背景色。 command：点击按钮时调用的函数或方法。
*   default：按钮的初始状态，缺省值为 NORMAL，可改为 DISABLED（不可用状态）。
*   disabledforeground：不可用状态下的前景色。
*   fg 或 foreground：前景色（即文本颜色）。
*   font：按钮文本字体。 height：按钮高度（对普通按钮即文本行数）。
*   justify：多行文本的对齐方式（LEFT，CENTER，RIGHT）。
*   overrelief：当鼠标置于按钮之上时的 3D 风格，缺省为 RAISED。 padx：文本左右留的空白宽度。
*   pady：文本上下留的空白宽度。 relief：按钮边框的 3D 风格，缺省值为 RAISED。
*   state：设置按钮状态（NORMAL，ACTIVE，DISABLED）。
*   takefocus：按钮通常可成为键盘焦点（按空格键即为点击），将此选项设置为 0 则不能成为 键盘焦点。
*   text：按钮上显示的文本，可以包含换行字符以显示多行文本。
*   textvariable：与按钮文本关联的变量（实为 StringVar 对象），用于控制按钮文本内容。
*   underline：缺省值为-1，意思是按钮文本的字符都没有下划线；若设为非负整数，则对应, 位置的字符带下划线。
*   width：按钮宽度（普通按钮以字符为单位）。

Checkbutton

Entry

Frame

Label

LabelFrame

Listbox

Menu

Menubutton

Message

OptionMenu

PanedWindow

Radiobutton

Scale

Scrollbar

Spinbox

Text

Toplevel

# 3.4 对话框

### 3.4 对话框

GUI 的一个重要组成部分是弹出式对话框，即在程序执行过程中弹出一个窗口，用于与 用户的特定交互。Tkinter 提供了若干种标准对话框，用于显示消息、选择文件、输入数据 和选择颜色。

tkMessageBox 模块

本模块定义了若干种简单的标准对话框和消息框，它们可通过调用以下函数来创建：askokcancel、askquestion、askretrycancel、askyesno、showerror、showinfo 和 showwarning。 这些函数的调用语法是：

```py
function(title, message, options) 
```

其中 title 设置窗口标题，message 设置消息内容（可用\n 显示多行消息），options 用于设置 各种选项。

这些函数的返回值依赖于用户所选择的按钮。函数 askokcancel、askretrycancel 和 askyesno 返回布尔值：True 表示选择了 OK 或 Yes，False 表示 No 或 Cancel。函数 askquestion 返回字符串 u"yes"或 u"no" ，分别表示选择了 Yes 和 No 按钮。

参数 options 可设置以下选项：

```py
default = constant 
```

指定缺省按钮。其中 constant 的值可以取 CANCEL、IGNORE、NO、OK、RETRY 或 YES。如果未指定，则第一个按钮（"OK"、"Yes"或"Retry"）将成为缺省按钮。

```py
icon = constant 
```

指定用什么图标。其中 constant 的值可以取 ERROR、INFO、QUESTION 或 WARNING。

```py
parent = window 
```

指定消息框的父窗口。如果未指定，则父窗口为根窗口。关闭消息框时，焦点返回到父 窗口。

tkFileDialog 模块

本模块定义了两种弹出式对话框，分别用于打开文件和保存文件的场合。通过调用函数 askopenfilename 和 asksaveasfilename 来创建所需对话框，调用语法是：

```py
function(options) 
```

如果用户选择了一个文件，则函数的返回值是所选文件的完整路径；如果用户选择了“取 消”按钮，则返回一个空串。参数 options 可用的选项包括：

```py
defaultextension = string 
```

缺省文件扩展名。string 是以"."开头的字符串。

```py
filetypes = [(filetype,pattern),...] 
```

用若干个二元组来限定出现在对话框中的文件类型。每个二元组中的 filetype 指定文件 类型（即扩展名），pattern 指定文件名模式。这些信息将出现在对话框中的“文件类型”下 拉框中。

```py
initialdir = dir 
```

指定初始显示的目录路径。缺省值为当前工作目录。

```py
initialfile = file 
```

指定在“文件名”域初始显示的文件名。

```py
parent = window 
```

指定对话框的父窗口。缺省值为根窗口。

```py
title = string 
```

指定对话框窗口的标题。

tkSimpleDialog 模块

本模块用于从用户输入数据。通过调用函数 askinteger、askfloat 和 askstring 弹出输入对 话框。这些函数的调用语法是：

```py
function(title, prompt, options) 
```

其中 title 指定对话框窗口的标题，prompt 指定对话框中的提示信息，options 是一些选项。 返回值是用户输入的数据。参数 options 可设置的一些选项包括：

```py
initialvalue = value 
```

指定对话框输入域中的初始值。

```py
minvalue = value 
```

指定合法输入的最小值。

```py
maxvalue = value 
```

指定合法输入的最大值。

tkColorChooser 模块

本模块提供选择颜色的对话框。通过调用函数 askcolor 即可弹出颜色对话框：

```py
result = askcolor(color,options) 
```

其中参数 color 指定显示的初始颜色，缺省值为淡灰色。参数 options 可设置的选项包括：

```py
title = text 
```

指定对话框窗口的标题，缺省为“颜色”。

```py
parent = window 
```

指定对话框的父窗口。缺省为根窗口。 如果用户点击“确定”按钮，返回值为元组(triple, color)，其中 triple 是包含红绿蓝分量的 三元组(R, G, B)，各分量值的范围是[0,255]，color 是所选颜色（Tkinter 颜色对象）。如果用 户点击“取消”按钮，则返回值为(None, None)。

# 3.5 事件

### 3.5 事件

事件描述符是一个字符串，由修饰符、类型符和细节符三个部分构成：

```py
<修饰符>-<类型符>-<细节符> 
```

类型符

事件类型有很多，下面列出较常用的类型符：

Activate

构件从无效状态变成激活状态。

Button

用户点击鼠标按键。具体按键用细节符描述。

ButtonRelease

用户释放鼠标按键。在多数情况下用这个事件可能比 Button 更好，因为如果用户无意 点击了鼠标，可以将鼠标移开构件再释放，这样就不会触发该构件的点击事件。

Configure

用户改变了构件（主要是窗口）大小。

Deactivate

构件从激活状态变成无效状态。

Destroy

构件被撤销。

Enter

用户将鼠标指针移入构件的可见部分。

FocusIn

构件获得输入焦点。通过 Tab 键或 focus_set()方法可使构件获得焦点。

FocusOut

输入焦点从构件移出。

KeyPress

用户按下键盘上的某个键。可简写为 Key。具体按键用细节符描述。

KeyRelease

用户松开按键。

Leave

用户将鼠标指针移开构件。

Motion

用户移动鼠标指针。

修饰符

下面是常用的修饰符：

Alt

用户按下并保持 alt 键。

Control

用户按下并保持 control 键。

Double

在短时间内连续发生两次事件。例如<Double-Button-1>表示快速双击鼠标左键。

Shift

用户按下并保持 shift 键。

Triple

在短时间内连续发生三次事件。

细节符

鼠标事件的细节符用于描述具体绑定的是哪一个鼠标键，1、2、3 分别表示左、中、右 键。

键盘事件的细节符用于描述具体绑定的是哪一个键。对键的命名有多种方式，它们分别对应于 Event 对象中的如下几个属性：

char

如果按下 ASCII 字符键，此属性即是该字符；如果按下特殊键，此属性为空串。

keycode

键码，即所按键的编码。注意，键码未反映修饰符的情况，故无法区分该键上的不同字 符，即它不是键上字符的编码，故 a 和 A 具有相同的键码。

keysym

键符。如果按下普通 ASCII 字符键，键符即是该字符；如果按下特殊键，此属性设置 为该键的名称（是个字符串）。

keysym_num

键符码，是等价于 keysym 的一个数值编码。对普通单字符键来说，就是 ASCII 码。与 键码不同的是，键符码反映了修饰符的情况，因此 a 和 A 具有不同的键符码。

除了可打印字符，常见的特殊按键的键符包括：Alt_L，Alt_R，BackSpace，Cancel， Caps_Lock，Control_L，Control_R，Delete，Down，End，Escape，F1～F12，Home，Insert， Left，KP_0～KP_9，Next，Num_Lock，Pause，Print，Prior，Return，Right，Scroll_Lock， Shift_L，Shift_R，Tab，Up 等等。

常用事件

根据以上介绍的事件描述符的组成，可以构造如下常用事件：

*   <Button-1>：左键点击
*   <Button-2>：中键点击
*   <Button-3>：右键点击
*   <Double-Button-1>：左键双击
*   <Triple-Button-1>：左键三击
*   <B1-Motion>：左键按下并移动，每移一点都触发事件
*   <B2-Motion>：中键按下并移动，每移一点都触发事件
*   <B3-Motion>：右键按下并移动，每移一点都触发事件
*   <ButtonRelease-1>：左键按下并释放
*   <ButtonRelease-2>：中键按下并释放
*   <ButtonRelease-3>：右键按下并释放
*   <Enter>：进入按钮区域
*   <Leave>：离开按钮区域
*   <FocusIn>：键盘焦点移到构件或构件的子构件上
*   <FocusOut>：键盘焦点从本构件移出 a：用户按下小写字母“a” 可打印字符（字母、数字和标点符号）都类似字母 a 这样使用。只有两个例外：空格键 对应的事件<space>，小于号对应的事件是<less>。
*   <Shift-Up>：同时按下 Shift 键和↑键。
*   与<Shift-Up>类似的还有利用 Shift、Alt 和 Ctrl 构成的各种组合键，例如<Control-a>，
*   <Control-Alt-a>等等。
*   <Key>：按下任意键。
*   具体按下的键值由传递给回调函数的事件对象的 char 属性提供。如果是特殊键，char 属性值为空串。注意，如果输入上档键（如@#$%^&*之类），当按下 Shift 键时就触发了<Key> 事件，再按下上档键又会触发<Key>。
*   <Configure>：构件改变大小或位置。构件的新尺寸由事件对象的 width 和 height 属性传递。

事件对象

每个事件都导致系统创建一个 Event 对象，该对象将被传递给事件处理程序，从而事件 处理函数能够从该对象的属性获得有关事件的各种信息。事件对象的属性包括：

x，y

鼠标点击位置坐标（相对于构件左上角），单位是像素。

x_root，y_root

鼠标点击位置坐标（相对于屏幕左上角），单位是像素。

num char

鼠标键编号，1、2、3 分别表示左、中、右键。

如果按下 ASCII 字符键，此属性即是该字符；如果按下特殊键，此属性为空串。

keycode

所按键的编码。注意，此编码无法区分该键上的不同字符，即它不是键上字符的编码。

keysym

如果按下普通 ASCII 字符键，此属性即是该字符；如果按下特殊键，此属性设置为该 键的名称（是个字符串）。

keysym_num：这是 keysym 的数值表示。对普通单字符键来说，就是 ASCII 码。

width，height

构件改变大小后的新尺寸（宽度和高度），单位是像素。仅适用于<Configure>事件。

widget

生成这个事件的构件实例。