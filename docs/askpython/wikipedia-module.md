# Python:维基百科模块

> 原文：<https://www.askpython.com/python-modules/wikipedia-module>

你好，学习伙伴！在今天的教程中，我们将学习一个名为维基百科的新模块，它可以用来获取任何需要的信息。

所以让我们开始吧。

## Python 中的维基百科模块简介

Python 的`Wikipedia module`可以用来从我们都很熟悉的维基百科网站上获取一堆信息。

我们将从把`wikipedia`模块导入我们的程序开始。如果导入命令出错。确保使用`pip`命令安装模块。

## 从维基百科模块获取数据

现在让我们学习如何用 Python 实际实现维基百科模块。

### 1.获取随机页面名称

选择好的标题进行搜索有时是一项艰巨的任务。人们可以使用`random`方法获得随机标题。

如果我们需要一个以上的随机标题，该方法可以将页数作为参数。该函数返回标题列表。

相同的代码如下所示。

```py
import wikipedia 
print(wikipedia.random(pages=5))

```

该函数的输出如下所示。

```py
['Bharathi Kannamma', 'Sancergues', 'Live in Gdańsk', 'Allery Sandy', 'Ronald (disambiguation)']

```

### 2.获取摘要

方法可以用来获得任何标题的摘要。使用下面的代码也可以做到这一点。

`summary`方法将一个字符串作为参数，指定要搜索的标题。它为提到的标题返回一些句子。

我们还可以添加我们需要的句子数量作为参数来限制存储的数据。相同的代码如下所示。

```py
s1 = wikipedia.summary('Frank Johnson (musician)',sentences=50) 
print(s1) 

```

代码的输出如下所示。

```py
Frank Johnson (c. 1789 – 1871) was an American popular fiddle player and brass band leader based in North Carolina, near Wilmington, United States, for most of the nineteenth century. Although largely forgotten by history books and often confused with composer Francis "Frank" Johnson, he helped define the sound of African-American fiddle and brass-band music in the mid-19th century.

== Personal life ==
Johnson was born into slavery circa 1789, in North Carolina, and became a free man sometime before 1830\. He showed a talent for music early on and established himself as a popular fiddle player for dances. Using money he earned from performances, he bought the freedom of himself, his wife and his children.
A contemporary account of Johnson while performing at a "pic nic" describes him: "To say that he is handsome would not be strictly true, and still, when he is living so full of music that his features follow the changes of his tune, it is fair to say he looks very 'becoming'."He was buried in Pine Forest Cemetery, Wilmington, after a well-attended funeral: "the largest, we think, that has ever occurred in this city, it being estimated that there were at least two thousand persons in the procession, including the colored fire companies in uniform, with standards draped in mourning, the colored Masonic fraternity in regalia, etc., the whole preceded by a brass band."

== Career ==
Johnson assembled his freed sons and various nephews into an eponymous brass band by 1830\. The band consisted of about 15 members. Johnson himself played many instruments, but was known for his mastery of the fiddle, clarinet, and cornet. The Frank Johnson Band was popular with white planters and often played for state fairs, picnics, cotillions, college commencement balls (e.g., at Chapel Hill, North Carolina), and political rallies (but only for Democrats).

```

### 3.获取整个维基百科页面

为了从 Wikipedia 获得整个页面，我们使用了`page`函数，该函数将页面的标题作为参数。

该函数为提到的标题返回一个页面对象。我们可以进一步从创建的页面对象中提取数据。相同的代码如下所示。但是打印创建的页面对象不会产生任何信息。

为了从页面对象中获取数据，我们需要从页面中引用我们需要的确切信息。

查看下面的代码。

```py
page_obj = wikipedia.page('Yarwil')
print(page_obj)
print("TITLE OF THE PAGE:\n",page_obj.original_title)
print("\n\n")
print("CATEGORIES OF THE PAGE CHOOSEN:\n",page_obj.categories)
print("\n\n")
print("CONTENTS OF THE PAGE INCLUDE:\n",page_obj.content)

```

上面提到的代码的输出如下所示。

```py
<WikipediaPage 'Yarwil'>
TITLE OF THE PAGE:
 Yarwil

CATEGORIES OF THE PAGE CHOOSEN:
 ['All stub articles', 'Articles with short description', 'Companies based in Bærum', 'Norwegian company stubs', 'Short description matches Wikidata', 'Technology companies of Norway', 'Use dmy dates from January 2014']

CONTENTS OF THE PAGE INCLUDE:
 Yarwil AS is a joint venture between Yara International and Wilhelmsen Maritime Services. The Norwegian registered company provides systems for reduction of NOx emissions from ship engines. The technology is based on the Selective Catalytic Reduction (SCR) method using Urea as a reactant. This method can reduce NOx emissions from ships by as much as 95%.
The company was established as a reaction to the increased focus by the global community on emissions to air from the maritime industry.  New IMO regulations, MEPC 58, are in place, which demand a reduction in NOx emissions from ships globally of 20% by 2011 and 80% by 2016.
There are several different technologies available for the reduction of NOx, however the Selective Catalytic Reduction method is the only known technology that can reach the 2016 target of 80%.
Yarwil was registered on 22 August 2007 and has its headquarters at Lysaker just outside Oslo in Norway.On 21 October 2013 a press release was issued by Yara International stating they had acquired full ownership of Yarwil and that the company would become part of their NOxCare initiative as of 1 January 2014.

== References ==

== External links ==
Acticle about Yarwil in Emissions Worldview
Article about Yarwil by Lloyd's List
Article on NOx reduction by Bellona
NOxCare.com

```

### 4.获取不同语言的数据

为了获得不同语言的信息，我们将使用`set_lang`函数并将语言作为参数。

该函数将数据转换成上述语言。相同的代码如下所示。在下面的代码中，我们将获得法语语言的信息。

```py
wikipedia.set_lang("fr")
print(wikipedia.summary('Mickey',sentences="5"))

```

输出结果如下所示。

```py
Mickey Mouse [mikɛ maus] (en anglais : [ˈmɪki maʊs] ) est un personnage de fiction américain appartenant à l'univers Disney, apparaissant principalement dans des dessins animés, dans des bandes dessinées et des jeux vidéo. Véritable ambassadeur de la Walt Disney Company, il est présent dans la plupart des secteurs d'activité de la société, que ce soit l'animation, la télévision, les parcs d'attractions ou les produits de consommation. Mickey est utilisé comme un vecteur de communication et ses qualités doivent respecter la morale prônée par « Disney », que ce soit par Walt ou par l'entreprise elle-même. Mickey Mouse est connu et reconnu dans le monde entier, sa célèbre silhouette formée de trois cercles étant devenue indissociable de la marque Disney.
Mickey a été créé en 1928, après que Walt Disney eut dû laisser son premier personnage créé avec Ub Iwerks, Oswald le lapin chanceux, à son producteur.

```

## 结论

所以，今天在本教程中，我们了解了一个名为维基百科的新图书馆，它可以收集关于某个主题的信息。

希望你学到了新的东西！感谢您的阅读！