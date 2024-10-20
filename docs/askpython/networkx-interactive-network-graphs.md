# 用 Python 创建交互式网络图

> 原文：<https://www.askpython.com/python-modules/networkx-interactive-network-graphs>

我相信在使用一个叫做 Networkx 的特殊库之前，您已经用 python 构建了网络图。你有没有想过是否有一种方法可以与图形互动？你猜怎么着？！有一个名为 Pyvis 的库，它有助于提高 Python 编程语言中网络图的交互性。

***也读作: [NetworkX 包——Python 图库](https://www.askpython.com/python-modules/networkx-package)***

Pyvis 库支持可视化，并为网络图增加了交互性。该库构建在强大而成熟的 VisJS JavaScript 库之上。这允许快速响应的交互，并以低级 JavaScript 和 HTML 的形式提取网络图。

安装 Pyvis 库简单明了，可以使用下面的命令在系统的命令提示符下使用 pip 命令完成。

* * *

## 代码实现

现在让我们继续使用 Python 编程语言中的 Pyvis 库来实现交互式网络图的代码。我们将从使用下面的代码片段导入所有必要的库/模块开始。

```py
from pyvis import network as net
from IPython.core.display import display, HTML
import random

```

我们将从创建一个只有节点而没有边的网络图开始。空图的创建可以使用 Network 函数来完成，该函数指定其中网络图的属性，包括背景颜色、标题、高度和宽度。

接下来，我们将利用`add_node`函数向网络图添加节点。我们将添加 10 个节点(从 1 到 10)，然后将网络图转换为 HTML 格式，以增加交互性并保存 HTML 文件。

```py
g_only_nodes =  net.Network(height='600px',width='90%',
                  bgcolor='white',font_color="red",
                  heading="Networkx Graph with only Nodes")

for i in range(1,11):  
  g_only_nodes.add_node(i)

g_only_nodes.show('Only_Nodes.html')
display(HTML('Only_Nodes.html'))

```

看看只有节点的网络图是什么样子的。

创建网络图的下一步是在节点之间添加边。我们将在随机节点之间添加随机边。下面同样来看看这个功能。

```py
def generate_edge():
  s = random.randint(1,10)
  d = random.randint(1,10)
  return (s,d)

```

在函数中，我们将使用`random.randint`函数生成随机的源和目的节点对。我们将得到 1 到 10 之间的随机节点。以确保我们有足够的优势；我们将生成 20 条随机边。为了确保同一条边不会反复出现，我们将记录(源、目的地)节点对。看看下面的代码。

```py
g =  net.Network(height='600px',width='90%',
                  bgcolor='white',font_color="red",
                  heading="A Simple Networkx Graph")

for i in range(1,11):  
  g.add_node(i)

i=0
chosen_set = []
while(i!=20):
  eg = generate_edge()
  if(eg[0]!=eg[1] and not (eg in chosen_set)):
      chosen_set.append(eg)
      g.add_edge(eg[0],eg[1])
      i+=1

g.show('Simple_Network_Graph.html')
display(HTML('Simple_Network_Graph.html'))

```

添加边之后，我们将得到一个类似下图的网络图。看看网络图变得多么神奇，多么具有互动性！

* * *

## 结论

Pyvis 是一个强大的 python 模块，用于使用 Python 编程语言可视化和交互式操作网络图。我希望您能够使用该库构建网络图，并喜欢与这些图进行交互。

感谢您的阅读！

编码快乐！😃

***也可阅读:[Python 中的网络分析——完全指南](https://www.askpython.com/python/examples/network-analysis-in-python)***

* * *