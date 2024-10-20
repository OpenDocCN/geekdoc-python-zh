# PyDev of the Week: Pradyun Gedam

> 原文：<https://www.blog.pythonlibrary.org/2021/11/29/pydev-of-the-week-pradyun-gedam/>

This week we welcome Pradyun Gedam ([@pradyunsg](https://twitter.com/pradyunsg)) as our PyDev of the Week! Pradyun works on pip, Python Packaging Index (PyPI), the Python Packaging Authority (PyPA), Sphinx, TOML and more! Pradyun blogs and does talks. You can find out more about those on his [website](https://pradyunsg.me/). If you're more interested in seeing Pradyun's work, then you ought to head on over to [GitHub](https://github.com/pradyunsg) or [GitLab](https://gitlab.com/pradyunsg). 

Let's spend a few moments getting to know Pradyun better!

## Can you tell us a little about yourself (hobbies, education, etc.)?

I grew up in Mumbai, but recently moved to the UK, where I’m working for Bloomberg. I’m part of the company’s Python Infrastructure team, mainly focusing on the software infrastructure and developer experience for Python within the organization.

I generally like sports. I’ve played a lot of cricket, basketball, and football throughout most of my life. Unfortunately, I haven’t been able to play any of those team sports recreationally for over a year now.

However, I have played a *lot* of video games lately. I have a strong bias toward systemic simulation games. Rimworld has been slurping up a lot of my free time over the past few weeks – perhaps, to the point of being unhealthy, now that I think about it. In the recent past, Steam tells me I’ve been playing Parkitect, Dyson Sphere Program, Oxygen Not Included, Hitman 2, and Rocket League.

I feel like I’m a weird breed of video gamer though – my last two gaming systems have been a Linux Desktop and a MacBook Air. My only Windows-based gaming happens through my GeForce NOW subscription.

Education-wise, I have a bachelor’s degree in “Computer Science Engineering.”

## Why did you start using Python?

To make my own video game! Python was the first programming language I learnt, and I learnt it to make games.

I was in 8th grade, if I remember correctly, when my parents gave me the book “Core Python Programming” with the suggestion: How about you learn how to make games instead of just playing them?

That was a very sticky idea. So, after putting it off for a while, I did indeed pick up the book and learnt from it. By the end of it, I had learnt a lot about computers in general, and then went on to build a 2-player 2D tank shooter as well!

I found the process of writing my own game more fun than actually playing it, so I started looking around for more things to make with this new skill I had learnt.

## What other programming languages do you know, and which is your favorite?

I definitely spend a lot of my time working with Python, largely biased by the fact that it’s the one I’m most familiar with. That is followed by web technologies (JavaScript, CSS, HTML). I also work a decent amount with C++ code at work. I’ve enjoyed working with Rust at every opportunity/excuse I’ve had to do so; especially since the entire development experience and tooling around the language are great. I’ve also attended beginner sessions for Clojure, Ruby, and Haskell in the past, but I won’t say I know them.

To be honest, I don’t think I have any favorites here – to me, programming languages are tools in the toolbox of a software engineer. The tools I’m most familiar with are Python, web technologies (JS, CSS, HTML, and their various relatives), C++, and Rust.

## What projects are you working on now?

Lately, it’s been a mix of working on Python Packaging tooling, Sphinx documentation stuff, and fiddling around with speech recognition on my Raspberry Pi.

I ended up updating [the page on my website about what I’m working on](https://pradyunsg.me/stuff/) while writing my response here, so I guess that’s the rest of my answer. ![:sweat_smile:](img/9cee26febb5237097224cf413f442c7e.png)

## Which Python libraries are your favorite (core or third-party)?

The most fun I’ve had with Python has been with the [`ast`](https://docs.python.org/3/library/ast.html) module – mostly because it led to me learn *way more than I needed to* about language grammars, parsers, parser generators, and language design.

[PursuedPyBear](https://ppb.dev/) (ppb) is an educational game engine and the project that I’m most excited to see growing. Video games is how I got started with Python, and ppb provides a much better foundation for learning Python (and game development) than what I had!

[`rich`](https://rich.readthedocs.io/en/stable/) is also high up on the list of packages that I like right now!

## How did you get started with the Python Packaging Authority (PyPA)?

Honestly, I feel like I stumbled my way into it.

That was purely curiosity driven for me – “oh, GitHub is a thing” ? “oh, I can read source code of things I use” ? “oh, the people who wrote this have discussions in this issues tab”. So… I ended up reading on some of the long-standing issues on the project. And then, making comments in those issues and summarizing the discussions so far. Something I didn’t realise back then: When you write a summary of a *really* long GitHub discussion, you also retrigger the whole discussion based on that summary.

The first major discussion I got involved in like this was about how `pip install --upgrade <something>` behaves, which quickly got entangled with how `pip install <directory>` should reinstall the project from the directory. It just spiraled from there.

This was around May or June 2016\. At this point, I had been writing Python code for fun for about two or three years. It was just after I’ve given my final exams for high school (12th grade). Given that I wasn’t exactly preparing for “Joint Entrance Examination” (JEE) full time – unlike most of my peers – I had a bunch of free time on my hands.

So, I ended up putting that time into what ended up being a few months of technical discussion about what-do-we-do-here. The whole thing resulted in a `+163 ?13` PR at the end – and I learnt very quickly that a lot of working on software is a people thing.

I actually enjoyed the process and learnt a lot of new things as a result. There was a certain dopamine rush associated with that the whole thing, so I ended up spending more time on the project. A few months later, I got into a college and realised that I can apply for Google Summer of Code (GSoC)! I did that in 2017, and a couple of months after that was over, I got offered the commit bit on pip and virtualenv.

Since then, I’ve been involved in some form with *nearly* every PyPA project on GitHub – either as a direct code contributor, engaging in discussions on their issue tracker, or as someone providing inputs in the standardisation process.

And, since I looked these up while writing this: My first commit in pip was updating a bunch of `http` URLs to `https`.

```py
e04941bba Update http urls to use https instead (#3808) 
```

## What have you been doing with Sphinx lately?

Actually, quite a lot, now that this question has made me think about it.

Documentation is a big part of software’s usability story, and a large part of how users interact with a given project. I’ve always felt that Sphinx-based documentation looked *very* dated, thereby contributing to lower quality content and a poor user experience. These feelings were validated and boosted by the user experience research that was conducted for pip throughout 2020.

Long story short, I ended up writing a Sphinx theme ([Furo](https://pradyunsg.me/furo/)) that’s become quite popular (if I may say so myself). In the process of writing that theme, I discovered a bunch more things I could do. So… since then, I have:

*   worked with the original author of [sphinx-themes.org](https://sphinx-themes.org/) and completely revamped the website (twice!), while also making it significantly easier to maintain.
*   picked up the maintenance of [sphinx-autobuild](https://pypi.org/project/sphinx-autobuild/), which provides a really nice live-reloading experience for writing Sphinx documentation.
*   collaborated with various folks from [ReadTheDocs](https://readthedocs.org/) to improve things around Sphinx on their platform.
*   gotten involved in conversations about Sphinx, both in the issue tracker of Sphinx, as well as in the technical writing spaces (like [Write The Docs](https://www.writethedocs.org/)).
*   become a member of the [Executable Books Project](https://executablebooks.org/), where I’m trying to help out with all the cool things that everyone else is building.
*   started writing a second theme and a bunch of theme-related tooling for Sphinx (though, I’ve not posted the code publicly yet).

The first public beta of Furo was [2020.9.2.beta1](https://pradyunsg.me/furo/changelog/#beta1), which was over a year ago!

## Of the packages that you help maintain, what’s your favorite and why?

Hmm… this is a tricky question to answer. It doesn’t help that I genuinely have so many things to choose from.

Furo comes to mind first. It is probably the only project I work on where I make the final call on everything unilaterally. It helps that it looks pretty and that it’s visible when someone uses it.

However, the project I spend the most time on is definitely pip. That is not the most fun project to work on though, even though it can be the most rewarding in terms of the sheer number of users I can help through my involvement with it.

## What are the biggest challenges that the PyPA is facing?

Oh wow, that’s a heavy question. There’s a lot to unpack here.

I think one of the unique things about Python’s packaging tooling, compared to most other popular languages, is that it’s *almost* entirely driven by volunteers. This contrasts drastically with the investments and resources that other ecosystems have put toward their tooling. There are many consequences of this, but this question is about the challenges, so I’ll stick to those aspects.

There are definitely a lot of **known functionality improvements that are difficult to get done with only volunteers**. It is often tricky to coordinate across projects since availability is a hit-or-miss story. Improvements take time and generally progress slowly, both due to the volunteer nature of the labour, as well as the scope and scale of the work. Broadly, the PyPA operates on consensus and that is often not straightforward to establish. Even in those cases where we all agree and know *exactly* what needs to be done, we simply don’t have the ability to “throw” developer time at it to solve it.

There’s also the problem of **poor documentation and communication channels**. As with the functionality improvements, there’s limited availability of folks with the right skills and knowledge to invest the necessary energy into these areas. This isn’t helped by the fact that these areas are fairly complicated and nuanced in their own right, often rivaling the code changes in the amount of energy and understanding necessary to get things right.

Paying down our **accrued technical debt** doesn’t really attract as much energy as it should IMO, especially for something so foundational to the Python language ecosystem. A lot of improvements are blocked on this, since many foundational projects have a whole lot of technical debt which hasn’t been paid down. This sort of long-term low-effort/low-reward stuff is difficult to do when everyone’s a volunteer. Some things are easier when you’re getting paid to do it. Paying down technical debt is certainly one of these. Conversely, paying down technical debt is also tricky to get funding for on its own. 🙂

The **“shackles” of backwards compatibility** is another challenge (I consider this distinct from technical debt). There are many behaviours, CLI options, API designs, names for parameters/functions, etc. that people rely on in these tools. We’ve also learnt that some of these are confusing, frustrating, poorly named, and overall horrible for the user experience. Or, they are sub-optimal choices/names/approaches, especially given that we have the context we do today. Or, there’s a better approach for solving these issues, since we’ve made improvements in other parts of the ecosystem. But we can’t do much about them anytime soon, since we can’t break the parts of the Python ecosystem that depend on these behaviours.

All of this isn’t news to the folks who are actively involved in this space, or to many of your readers. Some of these are constraints that any long-standing software system has. That said, there has been substantial effort put toward improving this ecosystem as a whole over the last few years!

The PyPA’s **standards-based approach** is starting to show its benefits, through the ecosystem of alternatives to setuptools and overall workflow tooling that has been built over the last few years – these don’t have the same technical debt or backwards compatibility concerns and can innovate at a much faster pace. There are also multiple efforts underway to create **re-usable libraries** that implement functionality that, earlier, was only available within the internals of long-standing tools (e.g., pip, setuptools, distutils, etc.).

The Python Software Foundation’s Packaging Working Group (Packaging-WG) has been working toward securing **funding for the Python ecosystem**. They actually maintain a list of [fundable Python Packaging improvements](https://github.com/psf/fundable-packaging-improvements/) which are well-scoped, have clear consensus among the community, and are clearly going to improve an end-users’ experience – all of which are things that organisations that could fund this kind of work would want.

**The Packaging-WG’s efforts have been very fruitful**, leading to some *very* visible and impactful work being funded, like the new [pypi.org](http://pypi.org/) site and the pip resolver rewrite (I worked on this one!). These were the sorts of things that would’ve taken many more years had they been undertaken only by volunteers and we hadn’t gotten this sort of funding. This has also led to establishing better documentation and communication around changes, including setting up “infrastructure” for doing a better job on these fronts going forward.

## Thanks so much for doing the interview, Pradyun!