# 定制 Pyvis 交互式网络图

> 原文：<https://www.askpython.com/python/examples/customizing-pyvis-interactive-network-graphs>

在本教程中，我们将学习如何通过向网络图添加可用属性来定制 Python 中的交互式网络图，并使其看起来更好。

***也读作:[用 Python 创建互动网络图](https://www.askpython.com/python-modules/networkx-interactive-network-graphs)***

有许多节点属性可以使可视化变得非常有趣，下面列出了这些属性:

1.  大小-节点的半径
2.  value–节点的半径，但根据传递的值进行缩放
3.  标题–标题意味着当用户悬停在节点上时，显示在节点上的文本
4.  X 和 Y 值–提及节点的 X 和 Y 坐标。
5.  标签–标签是显示在节点旁边的文本。
6.  color–该属性中提到了节点的颜色。

我们开始吧！

* * *

## 代码实现

对于本教程，我们将从一个由 10 个节点组成的简单图形开始，这些节点具有随机边，可以使用下面的 python 代码来构建。

```py
def generate_edge():
  s = random.randint(1,10)
  d = random.randint(1,10)
  return (s,d)

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

生成的网络图看起来有点像下图。

现在，我们将在接下来的章节中一次处理一个节点属性。

### 向图表添加标签

我们可以在 add_node 函数中添加标签作为标签属性。在这种情况下，使用下面的代码将标签设置为节点号。label 参数是最终可视化中节点旁边可见的字符串。

```py
def generate_edge():
  s = random.randint(1,10)
  d = random.randint(1,10)
  return (s,d)

g_labels =  net.Network(height='600px',width='90%',
                  bgcolor='white',font_color="red",
                  heading="A Simple Networkx Graph with Labels")

for i in range(1,11):  
  g_labels.add_node(i,label=str(i))

i=0
chosen_set = []
while(i!=20):
  eg = generate_edge()
  if(eg[0]!=eg[1] and not (eg in chosen_set)):
      chosen_set.append(eg)
      g_labels.add_edge(eg[0],eg[1])
      i+=1

g_labels.show('Simple_Network_Graph_labels.html')
display(HTML('Simple_Network_Graph_labels.html'))

```

生成的网络图如下所示。

### 增加节点的大小

在本节中，我们将以 value 属性的形式添加节点的大小，以便将节点缩放到特定的值。为了得到随机比例因子，我们将使用下面的函数。

```py
def generate_size_node():
  v = random.randint(5,20)
  return v

```

接下来，我们将把 value 属性添加到 add_node 函数中，并将比例因子作为 value 属性的值，就像我们在下面的代码中所做的那样。

```py
def generate_size_node():
  v = random.randint(5,20)
  return v

g_sizes = net.Network(height='600px',width='90%',
                bgcolor='white',font_color="red",
                heading="Network Graph with Different Sizes")

for i in range(1,11):  
  val = generate_size_node()
  g_sizes.add_node(i,label=str(i),value=val)

i=0
chosen_set = []
while(i!=20):
  eg = generate_edge()
  if(eg[0]!=eg[1] and not (eg in chosen_set)):
      chosen_set.append(eg)
      g_sizes.add_edge(eg[0],eg[1])
      i+=1

g_sizes.show('Simple_Network_Graph_sizes.html')
display(HTML('Simple_Network_Graph_sizes.html'))

```

这是令人惊叹的视觉效果。

### 向节点添加颜色

本节将重点介绍如何给节点添加各种颜色。我们将使用下面的函数以六进制编码的形式生成随机颜色。我们将以颜色属性的形式添加颜色，它也可以采用普通的 HTML 颜色，如红色或蓝色。我们也可以指定完整的 RGBA 或 hexacode 规格作为下面的颜色。

看看下面的代码和输出。

```py
def generate_color():
  random_number = random.randint(0,16777215)
  hex_number = str(hex(random_number))
  hex_number ='#'+ hex_number[2:]
  return hex_number

g_colors =net.Network(height='600px',width='90%',
              bgcolor='white',font_color="red",
              heading="Network Graph with Different Colors")

colors=[]
for i in range(1,11):  
  c = generate_color()
  colors.append(c)
  while(c in colors):
      c = generate_color()
  colors.append(c)

  val = generate_size_node()

  g_colors.add_node(i,label=str(i),color=c,value=val)

i=0
chosen_set = []
while(i!=20):
  eg = generate_edge()
  if(eg[0]!=eg[1] and not (eg in chosen_set)):
      chosen_set.append(eg)
      g_colors.add_edge(eg[0],eg[1])
      i+=1

g_colors.show('Simple_Network_Graph_colors.html')
display(HTML('Simple_Network_Graph_colors.html'))

```

### 指定节点的形状

节点的形状定义了节点在最终可视化中的样子。有许多可用的形状，包括正方形、星形、多边形等。有两种类型的节点。一种类型里面有标签，另一种类型下面有标签。

看看下面的代码，它会将形状分配给节点。看看最终的可视化。

```py
def get_random_shape():
  shapes = ['box','polygon','triangle','circle','star','cylinder']
  r = random.randint(0,len(shapes)-1)
  return shapes[r]

g_shapes =net.Network(height='600px',width='90%',
              bgcolor='white',font_color="red",
              heading="Network Graph with Different Shapes")

colors=[]
for i in range(1,11):  
  c = generate_color()
  colors.append(c)
  while(c in colors):
      c = generate_color()
  colors.append(c)

  val = generate_size_node()
  s = get_random_shape()

  g_shapes.add_node(i,label=str(i),color=c,value=val,shape=s)

i=0
chosen_set = []
while(i!=20):
  eg = generate_edge()
  if(eg[0]!=eg[1] and not (eg in chosen_set)):
      chosen_set.append(eg)
      g_shapes.add_edge(eg[0],eg[1])
      i+=1

g_shapes.show('Simple_Network_Graph_shapes.html')
display(HTML('Simple_Network_Graph_shapes.html'))

```

### 添加了节点边框宽度

节点的边框宽度定义了节点边框的宽度。看看下面的代码，它将为节点分配边框宽度。看看最终的可视化。

```py
g_borders =net.Network(height='600px',width='90%',
              bgcolor='white',font_color="red",
              heading="Network Graph with Different BorderWidths")

colors=[]
for i in range(1,11):  
  c = generate_color()
  colors.append(c)
  while(c in colors):
      c = generate_color()
  colors.append(c)

  val = generate_size_node()
  s = get_random_shape()
  b = random.randint(3,5)

  g_borders.add_node(i,label=str(i),color=c,
                    value=val,shape=s,borderWidth=b)

i=0
chosen_set = []
while(i!=20):
  eg = generate_edge()
  if(eg[0]!=eg[1] and not (eg in chosen_set)):
      chosen_set.append(eg)
      g_borders.add_edge(eg[0],eg[1])
      i+=1

g_borders.show('Simple_Network_Graph_Borders.html')
display(HTML('Simple_Network_Graph_Borders.html'))

```

## 完整的代码

让我们看一下下面的代码，它将涵盖上面提到的所有自定义属性以及一些额外的属性。除此之外，我们还将显示网络图的物理按钮。我们还将为节点添加标题，并将网络图定向为显示边的箭头。

```py
def generate_edge():
  s = random.randint(1,10)
  d = random.randint(1,10)
  return (s,d)

def generate_size_node():
  v = random.randint(5,20)
  return v

def generate_color():
  random_number = random.randint(0,16777215)
  hex_number = str(hex(random_number))
  hex_number ='#'+ hex_number[2:]
  return hex_number

g_complete =net.Network(height='600px',width='50%',
              bgcolor='white',font_color="red",notebook=True,
              heading="A Complete Networkx Graph",directed=True)

colors=[]
for i in range(1,11):  
  c = generate_color()
  colors.append(c)
  while(c in colors):
      c = generate_color()
  colors.append(c)

  val = generate_size_node()
  b = random.randint(3,5)

  g_complete.add_node(i,label=str(i),color=c,value=val,
                      title="Hello! I am Node "+str(i),borderWidth=b)

i=0
chosen_set = []
while(i!=20):
  eg = generate_edge()
  if(eg[0]!=eg[1] and not (eg in chosen_set)):
      chosen_set.append(eg)
      g_complete.add_edge(eg[0],eg[1])
      i+=1

g_complete.show_buttons(['physics'])

g_complete.show('A_Complete_Networkx_Graph.html')
display(HTML('A_Complete_Networkx_Graph.html'))

```

* * *

## 结论

在本教程中，我们学习了节点的自定义属性，只需添加一些内容就可以使交互图形更加漂亮。我希望你喜欢网络图以及它们的互动性！

感谢您的阅读！

编码快乐！😃

* * *