# PyDev of the Week: Alex Hall

> 原文：<https://www.blog.pythonlibrary.org/2021/12/13/pydev-of-the-week-alex-hall/>

This week we welcome Alex Hall as our PyDev of the Week! Alex is the creator of [https://futurecoder.io/](https://futurecoder.io/), a platform for learning Python. You can see what other projects Alex works on over at [GitHub](https://github.com/alexmojaki).

Let's spend some time getting to know Alex better!

**Can you tell us a little about yourself (hobbies, education, etc):**

I do a lot of coding in my spare time, mostly my own open source Python projects. That's my only regular hobby, when people ask what I do for fun I get a little stressed and think of [https://xkcd.com/1423/](https://xkcd.com/1423/). I'm interested in maths - I majored in Computer Science and Maths in university - but these days I do little more than watch maths YouTube videos and fiddle with random ideas.

**Why did you start using Python?**

I initially taught myself programming as a teenager in a very ad-hoc manner using GameMaker. At first I just used it to generate loads of pretty geometrical patterns by drawing lines and circles according to some mathematical rules. Probably the most significant thing I made with it was some games for practicing basic arithmetic. One of my teachers recommended that I enter a programming competition and suggested Python. It was my first 'real' programming language. I sped through the official tutorial and absolutely loved everything about it. I didn't get far in the competition because I still didn't really know what I was doing and hadn't had any formal training, but I had fun in the process. In particular when I looked up how to generate permutations and discovered recursion my mind was absolutely blown by the idea of a function calling itself, it was so exciting!

**What other programming languages do you know and which is your favorite?**

Python is by far my favourite language, no other language comes close. I think a big part of that is because I know it so well, again much better than any other language. I love metaprogramming, and that requires knowing intimately the details of how the language works, but it also seems like Python is much better suited to this than other languages. For example, decorators are perfectly natural in Python. In some statically typed languages like Java, I think no close equivalent is possible.

Apart from Python, most of the code I write these days (both in my job and my personal projects) is JavaScript and TypeScript. Some other languages that I've built significant projects in but are fading from memory are Kotlin, Java, Scala, and Ruby. Beyond that I've dabbled with more languages than I can count, there was a time when learning new languages seemed super fun and cool.

**What projects are you working on now?**

I'm mostly working on [futurecoder](https://futurecoder.io/), a 100% free and interactive Python course for beginners. Apart from the website, I've described the features and advantages in [this reddit post](https://www.reddit.com/r/learnprogramming/comments/qjs3wh/i_built_futurecoder_a_100_free_and_interactive/) which is also a good place to see what others think of it. futurecoder also uses several Python libraries that I've written and which are used in other places, particularly birdseye, snoop, executing, stack_data. I give an overview of these projects on my [GitHub profile](https://github.com/alexmojaki). They all involve the kind of metaprogramming that I love: analysing code and doing deep introspection to extract interesting information for debugging and understanding. So even when I'm not working directly on futurecoder, I'm often working on something related.

**Which Python libraries are your favorite (core or 3rd party)?**

A library that means a lot to me is [asttokens](https://github.com/gristlabs/asttokens). It makes it easy to retrieve the source code (and its location) associated with a node of the AST (Abstract Syntax Tree) which isn't something many people need, but I use it all the time. Recent versions of Python have a function for that but not for everything I need, so it's an essential dependency of my libraries. I made a bunch of contributions to the project, both because I was grateful for its existence and to make sure my own projects that depended on it kept working well. Some time later I shared futurecoder with the creator of asttokens so that they could see what it was being used for. They responded with a job offer for their company [Grist](https://getgrist.com/), it looked interesting, and now I work there! Occasionally I even get to do some fun Python metaprogramming involving asttokens!

**How did you futurecoder come about?**

I've long daydreamed about making educational software, although originally it was for maths - remember that was one of my first projects. As my interests turned in general from maths to programming, so did my interest in education. At one point I got really into answering questions on StackOverflow, it was basically a hobby.

Somehow I came across [Thonny](https://thonny.org/), a Python IDE for beginners, and found the debugger really interesting, particularly its ability to step through one expression at a time. I looked at its source code to figure out how it worked, and that was when I first learned about the AST. I adapted the ideas to build birdseye.

At that time I already had something like futurecoder in mind, with the birdseye debugger being a primary feature, but I decided to build birdseye as a library that could be used on its own, not only as part of some learning resource. That turned out to be a much bigger project than I'd expected. Years later, after having built the other libraries I mentioned, I decided to actually start on futurecoder. It's a good thing I waited because I was now much more experienced in both Python and JavaScript, and could build a proper interface in React instead of some awkward spaghetti jQuery code like what birdseye still uses.

My motivation was and still is based on the observation that other similar resources (e.g. Codecademy) seem really lacking in features and could be improved in so many ways, and that's what futurecoder is doing. Plus, the best resources generally require payment, and I want good education to be freely available to all.

**What are some challenges you faced working on futurecoder and how did you overcome them?**

The biggest technical challenge was letting people run arbitrary Python code. Originally this meant running the code in a server, which was a concern for security, devops, architecture, scalability, and funding. I sort of hoped that I would figure it out (maybe with help) before futurecoder got too popular, and that people would donate enough to keep servers running. It wasn't a good plan.

Fortunately I found a much better solution: run Python in the browser using [Pyodide](https://pyodide.org/). There's several older Python interpreters that run in the browser like Skulpt and Brython, but they wouldn't support the heavy metaprogramming like birdseye and snoop which rely on a lot of CPython-specific details. Pyodide is basically CPython running in the browser, so all that introspection still works perfectly, even down to disassembling bytecode instructions.

Pyodide is still pretty new and comes with challenges on its own, [like reading from stdin (e.g. the input() function)](https://github.com/pyodide/pyodide/issues/1219), but with a lot of work I managed to get it working. I did a [massive overhaul](https://github.com/alexmojaki/futurecoder/pull/157) of the futurecoder codebase, completely removing the Django backend servers and converting the whole thing to a 'static' website that costs very little in hosting and makes life much easier. It now uses a firebase database to store user data, which was also very helpful, there's some very cool technology there.

A challenge I'm struggling with is making the site look good. I hate CSS with a fiery passion and generally spend as little time on it as I can, which is why the course uses bootstrap a lot and looks generic and sad. I'm hugely grateful to a friend and coworker for making the awesome homepage, my own attempts looked hideous. But it still has issues, especially on mobile, and I dread the thought of trying to deal with them myself.

The other problem is marketing and promotion. I get lots of compliments from users who love learning on the site, but I don't know how to spread the word more widely. In particular I'm struggling to find people to help me build futurecoder further. I've been working on futurecoder for about 2 years now, mostly on my own, and it's getting exhausting. So I'm trying to figure out how other people advertise free stuff and attract open source contributors, and it's hard.

**Thanks for doing the interview, Alex!**