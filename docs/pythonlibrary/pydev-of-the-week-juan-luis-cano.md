# 本周 PyDev:胡安·路易斯·卡诺

> 原文：<https://www.blog.pythonlibrary.org/2018/02/19/pydev-of-the-week-juan-luis-cano/>

本周我们欢迎胡安·路易斯·卡诺( [@astrojuanlu](https://twitter.com/astrojuanlu) )成为我们的本周 PyDev！他是 Python Spain 非营利组织的主席，也是 [poliastro](https://github.com/poliastro/poliastro) 项目的发起人。如果你能看懂西班牙语，那么你可能想看看他的[网站](http://www.pybonacci.org/)。否则你绝对可以看看他的 Github 简介，看看他在做什么或者对什么感兴趣。让我们花些时间去了解更多关于胡安的事情吧！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

I'm an Aerospace engineer with a passion for space, programming and open source. I really enjoy solving Physics or Math problems using computers, and the more my job requires me reading scientific papers and derive mathematical equations, the happier I am. I am an open culture advocate and also chair of the Python Spain non-profit, which organizes the PyCon in our country (I have been involved one way or another since the first edition, in 2013). When I am not in front of the computer, I love to listen to all kinds of music (electronic, '70s rock, opera, blues), go everywhere with my bicycle and travel.![](img/695727dd31398a5a5235d7da53b9092c.png) **你为什么开始使用 Python？** **One snowy winter morning in my first year of University, our Physics professor wanted us to derive whether it's better to climb a frozen ramp using the reverse gear of the car or not. However, he gave us the wrong solution and I struggled the whole weekend to understand it, without knowing it was wrong. I wanted to interactively displace the center of gravity of the car and the slope of the ramp without doing the calculations every time and visualizing it in some way, so I bought a 6 month student license of Mathematica. It was totally awesome, but I could not afford buying a complete license and it worked so so on Linux, so I typed "free alternative to Mathematica" and [Sagemath](http://www.sagemath.org/) appeared. This was 2011, and the rest is history. **你还知道哪些编程语言，你最喜欢哪一种？** **My first programming experience was with a game called [Tzar](https://www.wikiwand.com/en/Tzar:_The_Burden_of_the_Crown). It's kind of like Age of Empires, but allowed you to create your own campaigns, define events, write dialogs... I copied and pasted $myVar everywhere without knowing what I was doing, but the result was super exciting. After that, I learned ActionScript 3 and Flash to create my own games, HTML and CSS to create my own website, PHP to give it some logic... I have to admit that I somewhat miss the object oriented nature of AS3\. Also, they taught us FORTRAN 90 (all caps) in University, but since I started with Python I never had to go back to FORTRAN again. As any engineer out there, I also had to do a thing or two in MATLAB, but I oppose its closed nature and pervasiveness. Overall, my favorite is Python, of course 🙂

你现在在做什么项目？

I just finished a freelance project with Boeing and the European Comission to try to predict aircraft trajectories based on historical data using machine learning. In my spare time, I maintain an open source library called poliastro (more on that later) and I dedicate a lot of time to promotion, documentation, prepare talks about it for conferences... I'm also trying to gather data from the Spanish railway company, RENFE, which is a bit challenging because one has to do some ugly web scraping tricks on their website. I love challenges 🙂

哪些 Python 库是你最喜欢的(核心或第三方)？

My favorite libraries are Astropy and SymPy, hands down. The latter constitutes what for me is one of the most successful Python libraries when we talk about new contributors, Google Summer of Code projects... And its LaTeX output mode in the Jupyter notebook is just marvelous. The former is a rare gem in which very talented astronomers have managed to put together a big, complicated library that acts as a foundation to many different projects. I use its unit handling capabilities and kinematic reference frame conversion functions a lot in poliastro.

**poli astro****项目是如何产生的？**

In 2013, when I was an Erasmus (visiting) student in Politecnico di Milano, an Italian colleague and I had to optimize an orbital trajectory from the Earth to Venus and then analyze the orbital perturbations of a Venus-orbiting spacecraft. My colleague started his part in MATLAB, and one afternoon he sent me an email with some non-working scripts and gave up. So I rewrote some parts in Python, kept some others in MATLAB (using Octave, since I didn't have a license), and even included a FORTRAN part as well. This mess worked beautifully in my computer but was probably impossible to install on Windows, soÂ two years laterÂ I rewrote all the algorithms in Python using numba, threw the MATLAB and FORTRAN away and became a happy man again. Now the project is more mature: [we had funding from the European Space Agency last year](http://blog.poliastro.space/2017/09/15/2017-09-15-poliastro-070-released-ready-pycones/), [we presented it at the first Open Source Cubesat Workshop](https://oscw.space/program/) that took place at the European Space Operations Centre, and [this year we have been accepted as mentoring organization in Google Summer of Code as part of the Open Astronomy umbrella](https://summerofcode.withgoogle.com/organizations/5078690623389696/), so I hope I get tons of applications from students!

 **你从运行这个开源项目中学到了什么？**

还有其他类似 poliastro 的项目，我想说其中一些有更好的算法和更多的功能。然而，我要说的是，他们都没有在 API 设计、文档、示例和推广上花费这么多时间(当然，相对于项目的规模而言)。此外，Python 不是世界上最快的语言，但它完成了工作，它很容易学习，而且它是趋势。我在 poliastro 学到的最重要的经验是:

*   有时候，语言胜于表现。
*   有时候，文档胜于特性。
*   有时候，营销胜于质量。

我最喜欢的一句 Python 开发人员的名言是 matplotlib 的创始人 John Hunter(愿他安息):[“一个开源项目要取得成功，最重要的商品是用户”](http://nipy.org/nipy/faq/johns_bsd_pitch.html)。我花了很多时间让 poliastro 易于安装和使用，现在这是许多用户第一次接触这种语言，他们对新功能提出了有见地的建议，推动了项目的进程，给了我继续下去的动力。

你还有什么想说的吗？

Python and the Open Source/Hardware/Knowledge mentality are now making its way into science, but there's still a long road ahead in Engineering. I would like to encourage other engineers out there who are tired of paying expensive MATLAB licenses against their will, or tired of endlessly looking for papers that everybody cites and nobody can find, to embrace the Open movement, engage in open source, and make its contribution. Per Python ad astra!**Thanks for doing the interview!******