# PyDev of the Week: Yury Selivanov

> 原文：<https://www.blog.pythonlibrary.org/2021/10/25/pydev-of-the-week-yury-selivanov/>

This week we welcome Yury Selivanov ([@1st1](https://twitter.com/1st1)) as our PyDev of the Week! Yury is the founder of CEO of [EdgeDB](https://www.edgedb.com/). Yury is also a core developer of the Python programming language. One of his most well known recent contributions is adding async to Python in [PEP 492](https://www.python.org/dev/peps/pep-0492/). You can see what else he has been working on in Python on [GitHub](https://github.com/1st1).

Let's spend some time getting to know Yury better!

**Can you tell us a little about yourself (hobbies, education, etc):**

Hi there! My name is Yury and I’m a software engineer, also wearing a hat of co-founder and CEO of EdgeDB.

I graduated from Moscow State Technical University named after Bauman with a Bachelor degree in Computer Science in 2006\. Shortly after that, I moved to Toronto, Canada, where I lived until relocating to San Francisco in 2020.

My hobbies include photography, UI design, and most recently I’ve been trying to teach myself guitar. Last but not least, somehow programming still feels like a hobby to me after all these years.

**Why did you start using Python?**

In summer 2008 I co-founded [MagicStack](http://magic.io). My co-founder [Elvis](https://twitter.com/elprans) and I did not have a clear business plan back then, but we did have a clear vision of how we want to write software. We were obsessed with metaprogramming, data modeling, and domain-specific languages and wanted to push the limits of what was possible in web frameworks back in the day. We spent weeks, if not months, researching the state of the art and ultimately decided to start from a clean state and build the foundation for our future projects ourselves.

We looked around for a programming language that was aligned with our vision and philosophy and Python was the best match. To make our life more challenging we started with Python 3.0 alpha 6.

**What other programming languages do you know and which is your favorite?**

I’m very comfortable with TypeScript/JavaScript and C. I used to know C#, C++, Java, and dabbled with many other languages, like Erlang and Go. Recently I’ve been playing with Rust a little bit.

Out of all programming languages I know, Python is my most favorite simply because I know its ins and outs.

**What projects are you working on now?**

EdgeDB.

EdgeDB is the project that underpins my entire career. It drived pretty much all of my contributions to Python. Some examples:

*   When I saw a demo of Google Wave I was completely blown away. Turned out they used non-blocking IO to make it, so naturally, we had to start using it too! Years later I’ve acquired some knowledge and started to help Guido with asyncio.
*   At some point we used asyncio heavily but Python lacked an asynchronous “with” block and “yield from” felt weird to me. So I proposed to add async/await to Python.
*   I knew that IO performance would be a bottleneck eventually so I looked into ways of improving the performance; that’s how uvloop was born.
*   We needed an async driver to talk to PostgreSQL, which led us to create asyncpg.
*   Later we needed an equivalent of thread local storage for async/await, that’s how we ended up proposing and implementing Python’s contextvars.

The story continues to this day. I have a unique luxury of being able to build an open source project I love day by day and contribute to Python anytime we need to push it a little further. And I’m very grateful for this opportunity.

**Which Python libraries are your favorite (core or 3rd party)?**

Core: asyncio. I use it every day and I’m one of the maintainers (even though not a very active one lately).

3rd party: click. mypy. Even though mypy isn’t exactly a library it is the tool that has impacted my workflow a lot, so it deserves to be mentioned.

**How did you become a core developer of the Python programming language?**

Around 2012 Brett Cannon asked on the python-dev mailing list if someone wants to help with PEP 362\. And it just happened that I looked at that PEP a year before that and implemented its API for myself, making some changes along the way. I offered help and started working on it. In a few months, Larry Hastings, Brett, and I made progress, and got the PEP accepted. A year later, I noticed that the API needed some updates for the upcoming Python release and I submitted a few PRs. That’s when I was promoted to a core developer.

If you have some spare time and love Python: consider contributing! Becoming a core developer isn’t that hard but it would help the project, benefit the community, and open many doors for you.

**Is there anything else you’d like to say?**

If you follow my work or use uvloop or asyncpg, please give EdgeDB a try and share some feedback at [https://github.com/edgedb/edgedb](https://github.com/edgedb/edgedb).

EdgeDB is our attempt to rethink what a relational database is. How can we streamline them? How can we make developers fall in love with databases?

To answer those questions we had to start with the very fundamentals. EdgeDB has a high-level data model with object types that map naturally to Python and JavaScript. It has a pretty cool querying language -- EdgeQL -- that aims to surpass SQL in power, composability, and expressiveness. It’s high performance, supports GraphQL, and has amazing client libraries. We have been working hard to ensure that EdgeDB is a joy to use. Follow EdgeDB on [Twitter](https://twitter.com/edgedatabase) to stay tuned!

**Thanks for doing the interview, Yury!**