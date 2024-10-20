# PyDev of the Week: Fanilo Andrianasolo

> 原文：<https://www.blog.pythonlibrary.org/2021/12/06/pydev-of-the-week-fanilo-andrianasolo/>

This week we welcome Fanilo Andrianasolo ([@andfanilo](https://twitter.com/andfanilo)) as our PyDev of the Week! Fanilo is a core developer for several [Streamlit](https://streamlit.io/) extensions and a Community Ambassador for Streamlit itself. He also has several [tutorials](https://linktr.ee/andfanilo) on Streamlit that you can check out.

![Fanilo Andrianasolo](img/09fe724731cbdded6722a1f2609698e2.png)

Let's spend some time getting to know Fanilo!

### Can you tell us a little about yourself (hobbies, education, etc)

Hello everyone! My name is Fanilo, I've been doing data analysis for around 8 years now, and am currently working as a Data Science & Business Intelligence Advocate / Tech Lead at Worldline.

I graduated from Ecole Centrale Lyon, one of the French "grandes ecoles" where we are taught a broad set of scientific domains, from acoustic engineering to fluid mechanics. I liked most of the tutorials around algorithms and numerical computing, so I decided to take a semester at the University of Queensland in Australia to study software engineering and machine learning. I loved this course abroad so much I decided to make a career out of data analytics (I like to think koalas and surfing also contributed to this enthusiasm).

Aside from work, I play and coach in badminton on a regional competitive level, try to play jazz piano while sipping a cup of tea and am learning video & music production.

### Why did you start using Python?

Years ago, our Data Mining stack was gravitating around SAS and R. One of my main activities was converting R code to Spark in Scala for production on a Hadoop cluster. At times it was challenging as you are juggling between two very different paradigms.

I knew there was a Python binding to Spark and I wanted to find an easier bridge between Data Scientists and Software Engineers in the Hadoop ecosystem; so I started rebuilding some of our data mining projects in Python with a senior data scientist colleague.

We both grew very fond of the language! The syntax felt simple and readable, yet we could build powerful and complex data processing pipelines. What struck me the most was the ecosystem: did you know you could "pip install wget" on Windows to have a pseudo-wget command? That day I jokingly messaged my colleague "Python has a library for everything!", and still regularly browse Pypi for niche and useful packages.

### What other programming languages do you know and which is your favorite?

I've done my share of Scala during my Apache Spark years, and know my way around Java as it is the predominant language in the company I currently work in. In the JVM space, I'd like to try Kotlin one day, it looks like their community and Data Science ecosystem are growing and the syntax looks nice.

I'm also a fan of building web applications to showcase my works. I don't pretend to be a Frontend engineer, but I can write small Typescript/Vue/React apps. I find the Javascript world has matured a lot those past years and the Typescript compiler ranting about my code has definitely helped.

Favorite language? I've been using Python pretty much everywhere now, from "check the quality of merchant data in our master customer database" to "downloading attachments from your email" processes. I have to thank the book "Automate the Boring Stuff with Python" for opening my eyes to using Python for every daily task. Who knows if Go, Rust, or Julia challenges it someday, and I'd like to add in C++ to build fancy audio processing tools.

### What projects are you working on now?

I'm mostly involved with prototyping data-driven features for projects, reviewing and deploying Python code on an online learning project, as well as promoting Business Intelligence and Machine Learning to internal product/engineering teams and external customers.

Outside of work, I started editing and publishing tutorials as slide carousels, as well as short Data Science skits with the hopes to build educational yet entertaining longer videos about Data Analysis later on. I also contribute a lot to the Streamlit community, but we will talk about this in a few questions.

### Which Python libraries are your favorite (core or 3rd party)?

I am a big fan of Streamlit ([https://streamlit.io/](https://streamlit.io/)) as it enables me to quickly showcase and share visual data analysis projects. For example, one of my Machine Learning demos involved using a Tensorflow model to recognize live drawings in the same vein as the "Quick, Draw" game ([https://quickdraw.withgoogle.com/](https://quickdraw.withgoogle.com/)). I struggled for 5 days with ECharts, Fabric.js, and Tensorflow.js, having to convert Python models to their JS counterpart and agonizing on CSS. Today with Streamlit I think this would take me less than 5 hours to build. Now I pretty much build a CLI and a Streamlit app as interfaces for every data quality and processing app I create at work.

I like using Plotly Express and Altair for interactive plots, and FastAPI/Pydantic are pretty high on my list too. The collections and itertools core libraries have a lot of hidden gems I rediscover now and then.

### What are your contributions to Streamlit?

I had never really contributed to any open source project or online community before. A year and a half ago when I first toyed with Streamlit, the forum and core team were still small. I would regularly see the founders Adrien, Amanda, and Thiago, along with some colleagues advise to other users on the 2-month old forum. The tone was very open and friendly, so I decided to help users on the forum too. I became very active there (I almost got the "365 days in a row" award!), so much that I got contacted by the team, became a forum moderator, was later invited as a guest on their chatroom, and participated in multiple beta testings. I am now part of the "Streamlit Creators" program ([https://streamlit.io/creators](https://streamlit.io/creators)) which is like being a Community Ambassador for Streamlit, and it comes with nice goodies!

Today I am still very involved in the community in different ways:

*   I maintain several Streamlit extensions: streamlit-echarts ([https://github.com/andfanilo/streamlit-echarts](https://github.com/andfanilo/streamlit-echarts)) to display ECharts plots in an app, streamit-drawable-canvas ([https://github.com/andfanilo/streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)) to integrate a live drawing component, and streamlit-lottie ([https://github.com/andfanilo/streamlit-lottie](https://github.com/andfanilo/streamlit-lottie)) to display Lottie animations
*   I am still hanging around the Streamlit forum ([https://discuss.streamlit.io/](https://discuss.streamlit.io/)) and Discord ([https://discord.gg/bTz5EDYh9Z](https://discord.gg/bTz5EDYh9Z)) helping users with their issues
*   I write tutorials like the Streamlit Components Hands-On ([https://streamlit-components-tutorial.netlify.app/](https://streamlit-components-tutorial.netlify.app/)), and started doing a weekly Streamlit tip series on Twitter ([https://twitter.com/andfanilo](https://twitter.com/andfanilo))

### If you needed to create a full-fledged website with Python, which framework would you use and why?

There are a lot of options nowadays to build a web application in Python, from the top of my head I can think of Streamlit, Dash, Panel, Gradio, Voilà, Django, FastAPI delivering static pages...they all serve different use cases and come with different constraints regarding the mapping between widget and state.

Whenever I need to show off and interact with some data processing or analysis, I will use Streamlit. I love the simplicity of its design and the low barrier of entry, and I believe you can still do very complex tools with it. But I also understand developers who are put off by the "rerun the whole script on every user interaction, put into cache or session state any heavy computation" lifecycle and prefer Dash or Panel for callbacks to define the mapping between user interaction and backend computation, especially for bigger, multipage web apps. To choose between those libraries, I don't usually give recommendations (and there are plenty of articles on the web on this), rather I ask users to test each library, get a feeling of their API, and ask the community if some more advanced tasks you would need to dig into are possible in this framework.

I did not have the opportunity to use Django yet, as my usual ML demos are single-page static apps without authentication, so worst-case scenario I can get by with React and FastAPI. I'm pretty sure Django is here to stay as one of the preferred frameworks for building "full-fledged websites with administration tools" though, whereas Streamlit/Dash/Panel/Gradio/Voilà would tend towards "providing Python users a way to create a web UI for their works".

### Is there anything else you’d like to say?

Have fun in what you do, don't be scared to contribute in online communities and build a lot of small and silly projects to improve at first, as consistency beats intensity!

### Thanks for doing the interview, Fanilo!