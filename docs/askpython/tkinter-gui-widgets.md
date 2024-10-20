# Tkinter GUI 部件–完整参考

> 原文：<https://www.askpython.com/python/tkinter-gui-widgets>

欢迎来到这个关于 tkinter GUI 部件的教程。在本文中，我将简要地向您介绍所有 Tkinter 小部件，并提供一些简单的代码片段来创建这些小部件。在本文快结束时，您将能够使用所有代码并使用 GUI 部件构建一个迷你服务。

## 创建主 Tkinter 窗口

这一步对于任何 Tkinter GUI 小部件都是必要的，不管它的特性如何。

```py
from tkinter import *

root = Tk()

#optional
root.configure(background='yellow')
root.geometry('600x800)

#place any of your code lines here

#end of code
root.mainloop()

```

上面显示的代码片段是一个基本结构，在面板上放置任何小部件之前，必须使用它来定义 tkinter GUI 框。

在代码片段中，`tkinter`库已经被导入，并且使用`root`对象实例化了`Tk()`构造函数。在代码的末尾，我使用`root.mainloop()`调用了整个程序。

`root.configure()`用于向您的主机添加额外的属性。在这个例子中，我用它来添加属性`background`和`root.geometry()`来确保主框架是指定的期望尺寸。然而，这些是可选使用的。

## 放置 Tkinter GUI 小部件

现在我们已经初始化了 Tkinter 的主机，我们将看看不同的小部件。

我将介绍最常用的小部件，包括标签、按钮、复选按钮、条目、滑块(在 Tkinter 中称为标尺)、列表框和单选按钮。

将下面给出的代码片段添加到主窗口的代码中。

### 1.Tkinter 标签小部件

对于这里的[标签小部件](https://www.askpython.com/python-modules/tkinter/tkinter-frame-and-label)，我们将使用`Label`构造函数本身来定义它。标签将放在根主窗口中，文本将显示“嘿，欢迎使用我的 GUI”。

然后我们把标签放在窗口里面，我们和 pady 进行了讨论，在 y 轴上给了我们更多的空间。

```py
label=Label(root,text="Hey, welcome to this my GUI")
label.pack(pady=10)

```

### 2.Tkinter 按钮小工具

[按钮](https://www.askpython.com/python-modules/tkinter/tkinter-buttons)将被放置在同一个主窗口上，并使用 button()构造函数创建。按钮的文本将显示“按下按钮”。请注意，文本颜色是绿色的。为此，我们将绿色指定为前景。

当按钮被按下时，我们想要激活一个功能，我们将把该功能分配给命令参数。函数的名字是`button_trigger()`。当按钮被按下时，它将激活此功能，并打印一条消息说“按钮被按下”。

我们已经将按钮打包到主根窗口中。因此，当我们按下这个按钮时，它将激活这个功能，并在控制台中打印消息。

```py
def button_trigerr():
    print("Button Pressed")

button = Button(root,text="press button", foreground="green", command=button_trigger)
button.pack(pady=10)

```

### 3.Tkinter 检查按钮小工具

在下一个例子中，我们有[检查按钮。](https://www.askpython.com/python-modules/tkinter/tkinter-checkbox-and-checkbutton)

当我们选中这个框或按钮时，它会将背景变成白色，就像打开灯一样。如果我们取消选中它，它会像关灯一样把背景变成黑色。让我们试一试。

```py
def check_button_action():
    print(check_button_var.get())

    if check_button_var.get()==1:
        root.configure(background='white')
    else:
        root.configure(background='black')

check_button_var = IntVar()
check_button = tk.Checkbutton(root, text="on/off", variable=check_button_var, command= button_action)
check_button.pack(pady=10)

```

所以首先，使用`Checkbutton()`创建检查按钮。它将在根主窗口上运行。**文字为“开/关”。**

我们已经将一个变量与这个复选按钮相关联，它是一个整数。将由复选按钮激活的功能，名为 button_action。

复选按钮有两个默认状态，即 0 和 1，这些默认状态将被分配给此处的变量。这个变量将跟踪复选按钮的状态，并获取复选按钮的状态。

我们只是继续引用`variable.get()`。如果复选按钮的状态为 1，这相当于复选框被选中，我们将使窗口的背景为白色。

如果它是 0，我们将把根窗口的背景设为黑色，这给了我们开灯或关灯的效果。然后，我们将它打包到 pady 为 10 的“主”框架中。

### 4.Tkinter 入口小部件

[条目小部件](https://www.askpython.com/python-modules/tkinter/tkinter-entry-widget)允许我们输入文本，并将文本从文本框或条目传输到控制台，并在控制台上显示消息。

为了创建入口小部件，我们已经创建了一个框架。为了创建框架，我们使用 frame()。

框架将放在主根窗口中，边框宽度为 5，具有凹陷效果。我们引用框架包，这将把框架打包到主窗口中。

```py
entry_frame = Frame(root, borderwidth=5, relief = SUNKEN)
entry_frame.pack()

text_box=Entry(entry_frame)
text_box.pack()

def get_entry_text():
    print(text_box.get())
    label.configure(text=text_box.get())

button = Button(entry_frame, text='get entry', command=get_entry_text)
button.pack(pady=10)

```

然后，我们创建了我们的条目文本框，条目将进入框架。我们把条目装进了相框。所以框架将进入主窗口，条目将进入框架。

然后，我们继续创建一个按钮，将文本从条目传输到控制台。现在请注意，我们的条目消息被打印到控制台，并且还更新了我们在大型机中的标签。要获取文本，我们只需使用 get()方法。

### 5.Tkinter 缩放微件

接下来，让我们来看看这里的滑块或[缩放小部件](https://www.askpython.com/python-modules/tkinter/tkinter-scale-widget)。对于这个小部件，假设您有一张餐馆账单，金额为 100 美元，您想看看不同的小费金额会如何影响总账单。

我们可以用滑块来显示小费，用输入框来显示账单，然后标签会显示账单总额。标签会给我们显示总账单。

```py
slider_frame = Frame(root, borderwidth=5, relief=SUNKEN)
slider_frame.pack(pady=10)

def calculate_total_bill(value):
    print(value)
    if bill_amount_entry.get()!=' ':
        tip_percent=float(value)
        bill=float(bill_amount_entry.get())
        tip_amount=tip_percent*bill
        text=f'(bill+tip_amount)'
        bill_with_tip.configure(text=text)

slider = Scale(slider_frame, from_=0.00, to=1.0,orient=HORIZONTAL, length=400, tickinterval=0.1, resolution=0.01, command=calculate_total_bill)
slider.pack()

```

好的，为了创建比例，我们使用 scale()，然后在这里我们为条目文本框输入所有的参数或自变量。我们在上面创建了它。

我们已经将滑块、条目文本框和标签打包到主窗口的框架中。我们像上一个例子一样创建了框架。

对于滑块所做的更改，`calculate_total_bill()`将被激活。该功能基本上是从输入框中提取账单金额的文本。然后，它将从滑动标尺中获取小费百分比，将小费计入账单，然后给我们显示在标签上的总账单。

### 6\. Tkinter ListBox Widget

接下来我们来看一下[列表框小部件。](https://www.askpython.com/python-modules/tkinter/tkinter-listbox-option-menu)这里我们有一个包含五个项目的列表框。在本例中，我们将只选择其中一项。然后，我们将按下按钮，我们希望将文本从项目转移到标签。

```py
listbox_frame=Frame(root,borderwidth=5, relief=SUNKEN)
listbox_frame.pack(pady=10)

listbox=Listbox(listbox_frame)
listbox.pack()

listbox.insert(END,"one")

for item in ["two","three","four","five"]:
    listbox.insert(END, item)

def list_item_selected():
    selection=listbox.curselection()
    if selection:
        print(listbox.get(selection[0]))
        list_box_label.configure(text=listbox.get(selection[0]))

list_box_button = Button(listbox_frame,text='list item button', command=list_item_selected)
list_box_button.pack()

```

为了创建列表框，我们使用了`Listbox()`。我们将把列表框放在一个框架内。创建列表框后，我们可以继续将项目插入列表框。

如果你想插入几个条目，你可以用 for 循环来完成。这里我们创建了一个按钮，当我们按下按钮时。它将激活创建的`list_item_selected()`。

为了访问选择，我们引用`listbox.curselection()`。为了确保我们确实选择了一些东西，我们使用`if selection:`，如果我们选择了一个项目，我们引用列表框项目，然后我们得到实际的项目。

我们在方括号中使用零的原因是，项目通常是一个数字元组，这将只给我们数字。然后，我们想继续用我们选择的项目更新我们的标签。

### 7.Tkinter 单选按钮小部件

对于最后一个例子，我们来看一下[单选按钮](https://www.askpython.com/python-modules/tkinter/tkinter-messagebox-and-radiobutton)。现在，根据选择的单选按钮，我们将在这里显示一个图像。所以这里我们有山，划船和露营。

我已经创建了三个单选按钮。所有的单选按钮都将被放置在主根窗口中。对于这篇课文，我们已经指定了“山、划船和野营”。

所有单选按钮都将有一个与一个变量相关联的值，我们已经创建了该变量。

```py
Label(root, text="choose icon")

def radio_button_func():
    print(rb_icon_var.get())
    if(rb_icon_var.get())==1:
        radio_button_icon.configure(text='\u26F0')
    elif rb_icon_var_get()==2:
        radio_button_icon.configure(text='\u26F5')
    elif rb_icon_var_get()==3:
        radio_button_icon.configure(text='\u26FA')

rb_icon_var=IntVar()

Radiobutton(root,text="mountains",variable=rb_icon_var, value=1, command=radio_button_func).pack()
Radiobutton(root,text="boating",variable=rb_icon_var, value=2, command=radio_button_func).pack()
Radiobutton(root,text="camping",variable=rb_icon_var, value=3, command=radio_button_func).pack()

radio_button_icon = Label(root, text=' ', font=("Helvetica",150))
radio_button_icon.pack()

```

在这种情况下，因为您一次只能单击一个单选按钮，所以与任一单选按钮相关联的值(我们在这里指定为 1、2 或 3)将被指定给变量。

当单击一个单选按钮时，它将激活或调用“radio_button_func()”。

因此，如果为 mountains 单击第一个单选按钮，值 1 将被赋给该变量，然后我们将获取该值并测试它是否等于 1。

如果它等于 1，我们将对山脉使用 Unicode 文本表示。

## 结论

快速总结一下，我们已经了解了一些常用的小部件，它们的用法如下:

*   **标签**–显示文本或信息
*   **按钮**–用于工具栏、应用程序窗口、弹出窗口和对话框
*   **检查按钮**–用于执行开关选择。
*   **输入部件**–用于输入或显示单行文本
*   **Scale widget**–当您希望用户输入一个有界的数值时，用来代替 Entry widget。
*   **列表框**–用于显示备选项列表。
*   **单选按钮**–用于向用户提供多种可能的选择，但用户只能选择其中之一。

一定要试试这些不同的部件，并在下面的评论区让我们知道你最喜欢的部件！！