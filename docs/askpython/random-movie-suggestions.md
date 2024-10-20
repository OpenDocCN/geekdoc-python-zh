# Python Tkinter:随机电影建议

> 原文：<https://www.askpython.com/python-modules/tkinter/random-movie-suggestions>

嘿伙计们！今天在本教程中，我们将使用 [Python tkinter](https://www.askpython.com/python-modules/tkinter/tkinter-buttons) 构建一个简单的 GUI 随机电影建议应用程序。

## 1.数据准备

为了获得包含大量电影名称的大型数据集，我们使用了`kaggle`。我们在项目中使用的数据集可以从这里的

### 1.1 导入模块

对于数据准备，我们需要两个模块，即`numpy`和`pandas`。除了这些模块，我们还将导入`random`和`tkinter`模块，我们将在后面的章节中用到它们。

导入模块的代码如下所示。

```py
import numpy as np
import pandas as pd
import random
import tkinter as tk

```

### 1.2 数据加载

为了加载`csv`格式的数据文件，我们将使用[熊猫模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)的 [`read_csv`函数](https://www.askpython.com/python-modules/python-csv-module)，并将所有信息存储到一个变量中。

为了查看数据，我们将使用显示数据集前 5 行的`head`函数。

数据集包含 50602 个电影名称，这显然是一个巨大的数字。加载初始数据的代码如下所示。

```py
data = pd.read_csv("indian movies.csv")
data.head()

```

### 1.3 数据准备

既然已经加载了全部数据，我们需要观察哪些列是需要的，哪些是不需要的。我们为这个项目需要的唯一的列是**电影名称**和上映的**年份**。

首先，我们使用 numpy 模块将整个数据转换成数组，以便更容易地遍历数据。然后，我们创建一个空列表来存储必要的信息。

下一步涉及按行遍历数据，只将电影名称和年份以元组的形式一起存储到一个公共列表中。

相同的代码如下所示。

```py
data = np.array(data)
l_movies=[]
for i in data:
    l_movies.append((i[1],i[2]))
print(l_movies[:5])

```

让我们使用列表切片打印前 5 部电影的名称和年份。其输出如下所示:

```py
[('Dr. Shaitan', '1960'), ('Nadir Khan', '1968'), ('Apna Sapna Money Money', '2006'), ('Aag Aur Sholay', '1987'), ('Parivar', '1956')]

```

## 2.创建 Tkinter 应用程序窗口

整个窗口包含标签、输出文本框和按钮，全部放在一个屏幕上。我们还将使用不同的颜色和字体定制整个窗口。

整个设计过程的代码如下所示:

```py
import tkinter as tk
window = tk.Tk()
window.geometry("600x200")
window.config(bg="#ABEBC6")
window.resizable(width=False,height=False)
window.title('Movie Name Suggestor')

l1 = tk.Label(window,text="Click on the button to suggest you a movie",font=("Arial",20),fg="Black",bg="White")
b1 = tk.Button(window,text="Suggest Movie",font=("Arial",15),bg="darkgreen",fg="white")
t1 = tk.Text(window,width=50,height=1,font=("Arial",15),state='disabled')

l1.place(x=30,y=10)
b1.place(x=200,y=60)
t1.place(x=15,y=120)
window.mainloop()

```

如果你在设计过程中有疑问，可以参考这里提到的[教程。最终的输出屏幕如下所示。](https://www.askpython.com/python/tkinter-gui-widgets)

![Final Screen Movie Suggest App](img/d7877e55fc1beb5f4b78a61a3ec97fe8.png)

Final Screen Movie Suggest App

## 3.向按钮添加功能

为了给“建议电影”按钮添加功能，我们将创建一个新的函数，从我们在步骤 1 中准备的列表中选择一个随机的电影数据。

所选的电影名称和上映年份将被添加到输出文本框中。该函数的代码如下所示:

```py
def suggest():
    t1.config(state='normal')
    t1.delete('1.0', tk.END)
    r = random.choice(l_movies)
    name = r[0]
    year = r[1]
    msg = r[0] +"(" + r[1] + ")"
    t1.insert(tk.END,msg)
    t1.config(state='disabled')

```

创建函数后，我们需要做的就是将`command`属性添加到按钮声明中。这就对了。您的 GUI 应用程序现在已经完成了！

## 实现随机电影建议应用程序的完整代码

该应用程序的完整代码如下所示:

```py
import numpy as np
import pandas as pd
import random
import tkinter as tk

data = pd.read_csv("indian movies.csv")
data = np.array(data)
l_movies=[]
for i in data:
    l_movies.append((i[1],i[2]))

def suggest():
    t1.config(state='normal')
    t1.delete('1.0', tk.END)
    r = random.choice(l_movies)
    name = r[0]
    year = r[1]
    msg = r[0] +"(" + r[1] + ")"
    t1.insert(tk.END,msg)
    t1.config(state='disabled')
window = tk.Tk()
window.geometry("600x200")
window.config(bg="#ABEBC6")
window.resizable(width=False,height=False)
window.title('Movie Name Suggestor')

l1 = tk.Label(window,text="Click on the button to suggest you a movie",font=("Arial",20),fg="Black",bg="White")
b1 = tk.Button(window,text="Suggest Movie",font=("Arial",15),bg="darkgreen",fg="white",command=suggest)
t1 = tk.Text(window,width=50,height=1,font=("Arial",15),state='disabled')

l1.place(x=30,y=10)
b1.place(x=200,y=60)
t1.place(x=15,y=120)
window.mainloop()

```

## 样本输出

下图显示了当用户请求应用程序获取一部电影来观看时所生成的输出。

![Output 1 Screen Movie Suggest App](img/4a18b56b37e9d22cb62f73a36f4322d4.png)

Output 1 Screen Movie Suggest App

![Output 2 Screen Movie Suggest App](img/8cab0eedda61c7ecd07c0e4093c9306a.png)

Output 2 Screen Movie Suggest App

## 结论

就这样，伙计们！我们构建了一个令人惊叹的完美的 [Tkinter GUI 应用程序](https://www.askpython.com/python/examples/gui-calculator-using-tkinter)。希望你明白一切。

自己也试试吧！编码快乐！