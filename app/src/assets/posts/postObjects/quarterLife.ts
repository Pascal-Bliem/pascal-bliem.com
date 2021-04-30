import Post from "../postModel";

export default new Post(
  // title
  "How to Quarter-Life-Crisis your Way into Data Science",
  // subtitle
  "My personal story of how I left academia, traveled around Asia, and got into Data Science on the way",
  // publishDate
  new Date("2020-03-27"),
  // titleImageUrl
  "https://www.datapine.com/blog/wp-content/uploads/2016/05/funny-pie-chart-example.png",
  // titleImageDescription
  "When you've seen all sides of the pyramid, it's time to reach for the sky!",
  // tags
  ["Learning", "Non-Tech"],
  // content
  `So this is going to be an article about how I got into data science and how that's the result of a series of slightly chaotic events. When I was starting out back then, I came across a lot of blog post from people who where documenting their stories, how they made the decision to change their career path, how they studied, and how - despite their struggles - their journey was usually very rewarding. Those posts always motivated me to keep going when I was questioning if I had set the right goals. Hence, I though I may just contribute such a piece of motivation for those who find themselves in the same situation as I did almost a year ago. A friend of mine recently wrote a great blog post titled ["Quarter Life Crisis" ](http://regitamhrdk.wixsite.com/website/post/quarter-life-crisis) (in Bahasa Indonesia), in which she pointed out that this phenomenon of anxiety about not knowing where to go in life, is actually a great opportunity to grow and advance beyond what we've done so far. I do absolutely agree with this and I am going to tell you how this worked out for me. 

**A word of warning:** This post is way too long, and not really the type of post from which you learn much about tech-stuff. It's my personal story and it is written as such. If you're (understandably) not interested in my personal background, feel free to skip to the section where I write about how I [learned data science](#learn).

### I graduated, and now what? 

At the time of writing this, I'm 27 years old. About two years ago, I was at a point at which everything seemed to be amazing. I had just graduated with honors from my master program in materials engineering from one of Germany's top engineering universities, after finishing a very though but successful master thesis, which even got [published](https://www.nature.com/articles/s41598-018-34042-1) in a peer-reviewed journal. I was looking back at 5 years of successful scholarly work and materials science research, many [amazing research topics](https://github.com/Pascal-Bliem/my-papers-and-theses), and research stays at universities in Sweden and Taiwan, as well as industrial internships. For the past years I had always imagined myself doing a PhD in materials science and becoming a senior scientist. Now, there I was at the point in life I had imagined for so long, having several PhD offers at hand but...somehow...something didn't feel right.

![Watching the sunrise from within the laboratory tells you that you spent too much time in there.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/sunriselab.jpg)

I looked back at the last couple of years and realized I had done most of the *PhD-kind-of-things* already; spending days and nights in the lab working on my projects, visiting fancy research facilities, (co)authoring papers, giving talks about my research etc. Did I really want to spend the next 4 years with what I already had been doing for the last 4 years? I did not. Yet, I had gotten so used to the idea of becoming a scientist and holding a PhD degree (maybe a little ego was involved there) that it wasn't easy at all to reconsider my options. I didn't know what to do - the PhD or...or what else? - but I knew that I wasn't happy the way that I though I would be. In addition, I was so tired of living in the same environment for years and just went through a break-up of a three years relationship. I didn't really have a backup plan at that point, but I just felt that I had to get out somehow. So when my friend told me, "If you don't know what to do, just go travel the world!", that's exactly what I did.

### Let's not decide what to do with my life just now

Obviously, some people in my social surroundings thought I went downright insane, but there was also a lot of positive feedback for my idea from my friends and family. I sold pretty much everything I couldn't fit in one backpack, gave up my apartment, and booked a one-way-ticket to Bali. I told myself that I could just postpone the decision of what to do with my life for a couple of months and that I probably will be wiser after having seen more of the world. I backpacked through 10 countries in southeast Asia and the Indian subcontinent, snorkeling around tropical islands, driving motorbikes through tea fields and along coastlines, taking part in silent meditation retreats, hiking over the Himalayas, and floating down the Mekong river for days. I saw a lot of amazing stuff, but at some point I started feeling exhausted again of seeing new things everyday without really doing anything productive. I felt like I wasn't generating much value for anyone else, so I started looking for some useful work to do.

![On the way to hopefully finding an idea what to do next...](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/world_map.png)

Since I had always enjoyed teaching stuff, I thought I might give it a try and started volunteering as an English teacher in a school in a remote Vietnamese mountain town. That was a great experience, but I realized quickly that trying to explain the English language to little kids who don't speak a word of English yet, *using English*, didn't make too much sense. Next I tried volunteering in a [fantastic English school](http://ilec.edu.vn/) next to the technological university in Hanoi, talking English to young adult engineering students. This worked a lot better, but still, I realized that I really missed doing something science and tech related. So I changed countries and started volunteering as a Science and Maths teacher in a [school](https://jogjacommunityschool.org/) in Jogjakarta, Indonesia. The pedagogical part of that adventure was quite challenging and rewarding, the "sciency" part, however, was not. The concepts taught at a high school level were just too shallow and too far away from application to keep me interested. At that point I had to admit to myself that I was and always will be that tech guy who only feels fulfilled when wrapping his brains around complex problems and crafting solutions for them.

### Maybe it's about time to come up with a plan now

At that point, about 9 months into my trip, I faced the postponed decision again. What am I going to do with my live? I was still quite sure about not wanting back into academic research. So, start working as a materials engineer in industry? I had done internships in that field before and I didn't really feel like I wanted to be working in industrial production again. Actually, I realized that most of the engineering job descriptions I found interesting, required decent coding skills. Maybe that was even the very reason why I found them so interesting. I was really interested into programming since I took a Java class in my bachelors and I had been using quite a bit of Python for automation and data processing when I was working on molecular dynamics or quantum mechanics simulations on a super-computer during my materials research. But next to my regular lab work back then, I never had the time to really professionalize my coding skills far enough to make myself employable in a coding job. One of those backlog items that never had made to *work-in-progress*.

Was it the right time to tackle this now? Could I even do this by myself, without a formal education in the field? Many blog post authors and digital nomads I met on my travels told me that they had done something like that, so yeah, I should be able to succeed at it as well. But again, my ego (and some relatives of mine) had some doubts: "You spend about 5 years becoming good at scientific research, had mentionable successes, and now you want to leave all that behind and start off as a beginner again?" Anyway, there I was, sitting in my hot and tiny Kos (an indonesian boarding house) room next to an active volcano, in the wonderful and very cheap city of Jogjakarta, still some money on my bank account, basically no current job commitment, and already speaking decent Indonesian. If there was one right point in space and time to do it, it was this one. After having to go on a visa run to Malaysia shortly, I had made up my mind. I changed to another volunteering organization where I could work part-time and came back to Jogja with the ambition to spend the next couple of months self-teaching myself to ramp up my coding skills, become an actual programmer, and get a job within half a year.


### How I figured out what I want and how to get there <a id="learn"></a>
Where the heck should I start?! Computer science seemed (and still does seem) so gigantic and overwhelming, so many sub-disciplines to chose from, and half of the vocabulary sounded like alien language to me at that time. I probably spent a whole week or so searching Google and YouTube for phrases like *how to teach yourself programming*, *what discipline of CS is right for me*, *what programming language to start with*, *please end my suffering* and so on. I wanted to be able to properly build cool software solutions, so the software engineering/developing aspect sounded appealing to me. I also wanted to keep doing experimental, investigative things, like I had done when I was still working in research. Luckily, I came across **Data Science** which seemed to perfectly unite those parts. The often cited quote "A Data scientist is someone who is better at statistics than a software engineer and better at software engineering than a statistician" appealed to me as a job description and, furthermore, I actually already had quite some suitable experience: I had been analyzing my data with Python (the most popular language in Data science) before, lots of experience in looking for patterns in data, and a good understanding of the linear algebra and differential calculus behind most machine learning models. So now I knew *where* to go, I just had to figure out *how* to get there.

### Massive Open Online Courses (MOOCs)
I definitely wasn't going to go back to university for a whole program, but I also didn't get much out of all the tiny YouTube and Medium blog post tutorials that started treating a problem somewhere in the middle and left off before the end. In needed something more holistic which I could still work on flexibly, following my own schedule (and budget). Some presumably great options like [Udacity](https://www.udacity.com/) nano degrees or [Data Camp's](https://www.datacamp.com/) different tracks were unfortunately out of the price range I was looking for (which meant for free). There was unfortunately not much about data science on [FreeCodeCamp](https://www.freecodecamp.org/). [Udemy](https://www.udemy.com/) also seemed to have some great courses on a discount price sometimes. Nowadays I'm using Udemy a lot with a business account, but back then (without a salary, living on less then 10 dollars per day) I had to get it cheaper. Luckily, I came across MOOCs (massive open online courses) on platforms like [Coursera](https://www.coursera.org/) and [EdX](https://www.edx.org/), which demand payment for getting a course certificate, but let you audit the courses for free. 

![A small selection of all the amazing sources of data science and software development knowledge.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/courses.png)

After browsing through their catalogues I decided to go with Coursera because they offered some specializations which seemed to suit my demands quite well. And maybe also because they offered 7 days free trials which I shamelessly abused to cram some courses and get the certificates for free. Anyway, I went with the [Applied Data Science with Python](https://www.coursera.org/specializations/data-science-python) specialization offered by the University of Michigan, which consisted of five courses covering data manipulation and processing, visualization, applied machine learning, text mining, and network analysis. This gave me a pretty good overview about what different branches data science consists of, and especially the machine learning part got me hooked. Theses courses did, however, not elaborate on any general programming or software engineering concepts. To make up for this shortcoming, I also took the [Object Oriented Java Programming: Data Structures and Beyond](https://www.coursera.org/specializations/java-object-oriented) specialization offered by truly fantastic instructors of the University of California San Diego. The individual courses covered the most common algorithms and data structures for sorting, searching, and graph operations, and I finished the specialization with a [project on social network analysis](https://github.com/Pascal-Bliem/social-network-analysis---communities-and-user-importance).


### Podcasts - Your best friends while on the road

I was very impatient with learning as much as I could in as little time as possible because I wanted to feel less like an absolute beginner as soon as possible and, also, I was running out of money and needed to get a job at some point. I wondered how I could keep learning about data science and coding when I couldn't do my online courses, for example while I was sitting on my motorbike or strolling over the market to get some veggies and tofu. I started looking for podcasts and found quite a handful that were really good. I loved listening to [Partially Derivative](http://partiallyderivative.com/) or [Data Framed](https://www.datacamp.com/community/podcast) (which are unfortunately discontinued) while I was driving through Javanese country roads and rice fields and I'm still listening to [Data Skeptic](https://dataskeptic.com/), [Linear Digressions](http://lineardigressions.com), the [Data Engineering Podcast](https://www.dataengineeringpodcast.com/) and [Developer Tea](https://spec.fm/podcasts/developer-tea) almost every day. Despite not paying perfect attention all the time and initially not understanding most of the concepts, it was great to be fully immersed in data science topics all the time and it helped to expand my mental map of all the topics encompassed under that big umbrella term of *Data Science*. After every new episode, I had a dozen new concepts to look up and it always gave me a little confidence boost when I realized that I already understood a little more when listening to the next episode.

![Some of the podcasts that I used for learning data science when I could have my eyes on a screen.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/podcasts.png)

### Building a portfolio with real projects
About two and a half months into the endeavour, I got to the point where I had finished up most of the course work that I had planned for and had gotten a good theoretical grasp of most things I wanted to learn, but I hadn't actually build a lot of original stuff. The next step was to apply what I had learned in some end-to-end (yet manageably small) projects. That was particularly important to me because I figured that - having hold no prior job with the title of Data Scientist and not having a perfectly matching university degree - I definitely needed some portfolio projects to present to potential employers. Okay, but what should I do? When you google for "how to come up with project ideas", you'll find a lot of posts telling you that you should build applications that are going to solve your personal problems or cater your needs in some way. But most *data products* I could think of would probably take a handful of developers a couple of months to build. What *smaller* problems did I want to solve? And which of those were original? I had seen the Titanic, Iris flower, and housing price data sets so often that I wondered if there aren't any other data sets out there. I searched Kaggle and the [UCI Machine Learning Repository](https://github.com/Pascal-Bliem/exploring-the-UCI-ML-repository) and did some machine learning tasks on what I found, but with those prepared data sets, it didn't feel like solving real world problems either. Should I be building a visual WhatsApp [chat log analysis](https://github.com/Pascal-Bliem/whatsapp-chatlog-analysis) to disprove my friend's false claims about what she had said when and in which way? That wasn't really enough. Trying to understand certain concepts better that I had struggled with before, such as [error rate control](https://github.com/Pascal-Bliem/error-control-statistical-tests) in statistical significance testing? Very interesting but not really end-to-end either. 

![It's easy to get a little confused about hypothesis testing and how false positives, false negatives, and statistical power influence each other.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/falsepositivenegative.png)

Then it struck me that I was (presumably) going back to Europe fairly soon to look for work. I had been away from home for more than a year and I hardly ever hung out with other Europeans during that time. Even before I departed, I spend most of my time in my very academic social bubble, and I felt a little detached from the European people. Time to reintegrate, I thought! After some searching, I came across the amazing [European Social Survey](https://www.europeansocialsurvey.org/), which describes itself as "an academically driven cross-national survey that has been conducted across Europe since its establishment in 2001". Extracting information from this data would be a great way to mentally prepare myself for getting back to Europe and get an original [portfolio project](https://github.com/Pascal-Bliem/european-social-survey) which I could present to potential future employers. The fairly large data sets held the answers of tens of thousands respondents from to face-to-face interviews with hundreds of questions in over 20 participating countries. It had all sorts of info on social topics, climate change, politics, immigration, individual well-being, and much more. And also, being scrambled together from dozens of universities, it was messy: nonsensical variable encodings, inhomogeneous scales, country specific questions and so on. I spend probably most of the time digging through the documentation and cleaning/preprocessing the data. But then, I could do exploratory analysis of variable correlations with interactive visualizations, investigate differences between countries with statistical hypothesis testing and determine their effect sizes, and - most amazingly - predict individual peoples' personally perceived happiness with machine learning, using their other answers to survey questions as features. It turned out to be that proper end-to-end project I had been looking for and I learned a lot along the way, both regarding data science methodology, as well as the social insights of the survey.

### The job search
Now, about three and a half months into the entire process, I felt like I really had the must-haves for being employable in my repertoire, but I caught myself having the same thoughts over and over: I gotta study one more of these statistical concepts, then I'll apply for jobs, got to check out one more of these cool libraries, understand this family of algorithms, read this one paper or post, then I'll start applying for jobs. I was just in the process of trying to get a convolutional neural network to tell images of cats and dogs apart, when I finally got to the conclusion that it may be a little more important to care about my future than building a silly pet detector. It was still one and a half months to go until my flight back to Germany and I wouldn't be able to go to interviews right away if I were to be invited, so I first aimed at looking for jobs in Singapore, the closest industrialized country. Looking for jobs is always a daunting process but I wasn't quite prepared for how bad it would start off for me. Most companies didn't even reply. On top of that, tons of (probably automated) rejection mails. I though that maybe the Singaporean job boards were useless so I tried to track hiring people from Asian tech giants like Grab, Gojek, Shopee etc. on LinkedIn - and no one replied. I felt like all the work of the last couple of months had been in vain. Apparently no one considered me employable, no one seemed to be interested in my projects or even wanted to talk to me. Was the entire decision one big mistake? I felt defeated. Until now, I don't get what was the secret sauce of the Singaporean job market; maybe it's personal referrals, which are important in a strongly Chinese influenced culture; maybe its the fact that one is always competing with lots of local university graduates for entry-level positions. Having wasted my time with sending applications to Singapore, I finally turned my focus towards Europe again, and Germany in particular. 

Again, I wasn't quite prepared for the feedback I got, but now it was the other way around. I had applied to many vacancies and put up my job search request on several European job boards, and suddenly, a ton of recruiting agencies and hiring managers wanted to talk to me. Now I had to stay up late at night (due to the different time zones) to attend  several rounds of phone or video call interviews with about 30 companies and agencies. I even had to start turning people down because I couldn't fit them into my calendar anymore. Apparently the German IT job market at that time was in very strong demand of skilled workers so that people wanted to hire me for all sorts of developer jobs, not only for data science and AI. That also meant that the interviews were of very varying quality. Many of these third-party recruiting agencies had clearly no idea about the technologies or even about the clients they were hiring for. Neither did they bother to thoroughly read my CV or check my LinkedIn before pulling me into hour-long phone calls, which would have answered most of their questions in advance. I often had to narrate my job-search-story over and over again until I felt like a stuck vinyl record. Some other interviews were great though, despite sometimes non-ideal conditions. The company that ended up hiring me, e.g., interviewed me while I was sitting in a Hanoi co-working space with a horrible internet connection and lots of motorcycle noise from the busy streets outside. The month or so that I spent on so many calls was tough; not only because of the job search itself, but also because I didn't have much time to keep working on my personal project or got to spent more time with my friends in Indonesia and Vietnam whom I wouldn't be seeing again for a long while. However, the efforts did pay off, as I was finally ready to return to Germany and face the final stage of my job search.

### Europe - The Final Countdown (pun intended)
After a couple of awfully long flights and stopovers, for the first time in almost one and a half years, I stepped on winterly German soil again. Everything felt a little unreal to me; the temperature drop of about 30°C, how *orderly* everything was compared to Indonesia, temporarily moving in with my family, and drinking amazing German beer again. I wasn't left with much time to acclimatize though, as I was thrown into a marathon of on-sight job interviews all over the country. It was two weeks till Christmas, everyone wanted to get their hiring processes finished before the holidays, and that meant that I spent those two weeks entirely on the car or train, driving from one city to another. After having two interviews in one day in the south of Germany, I took a night train to the mid-west, slept a few hours in a train station hostel, and squeezed in the interview with the company that ended up hiring me, only to rush on to the next city and next interview. Most of the job interviews went pretty well, despite the rush and my (perceived) lack of preparation. Nonetheless, many of them didn't really go into that generalist data science direction I was looking for the most. One of the large insurance companies I interviewed with, however, was just in the process of building up a new data science team that was supposed to be employed in a very wide range of use cases. Needless to say, I was very happy when they called and told me that they wanted me for the job and that they'd send me the contract...after the works council had signed the hiring decision...which wasn't going to happen until two days before Christmas. The next days, in which I tried stalling the other offers I had already gotten, were quite tense until I finally got the  call saying that I got the job, literally on the last day before the holidays. 

I was able to relax a bit...for about half a day, until I realized that now, I had to start looking for a place to live in a city with one of Germany's toughest housing markets. That city also happened to my place of birth - kind of went full circle here. I was, again, incredibly lucky and found a very tiny and expensive room in a shared apartment very close to my future office. I moved there with everything I owned at that point; my laptop, one suitcase, one backpack, and a box full of hats form seven different countries. And there I was, in the new Data Scientist job for which I had been working so hard for the last six months. I got a massive amount of input within the first couple of weeks, talked to dozens of people and tried to get an overview over all the different IT systems I'd have to be working with. I can confidently claim that an enterprise with almost 120 years of history and about 20000 employees is nothing short of complexity. Despite the inertia that such a large company can have at times, there is a huge wealth of potential in its data and plenty of possibilities to contribute business value with data science methods. I now work in an awesome team of data scientists, data engineers, and data strategists on use cases about churn and retention, customer value models, recommender systems, marketing support, process automation with text mining, and many other fascinating topic. This work gives me a great chance to take the research mindset I cultivated in academia, combine it with powerful software engineering, and transform it into real value for the business. Looking back now at everything I have done since my master graduation, I can say that all the decisions and efforts - not going for the PhD, traveling the world, and breaking into data science - were totally worth it. It was an amazing time and experience and it makes me even more look forward to all the great thing to come.

![The view from besides my new office onto my place, or as people here say: Home is where the Dom is.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/rhineanddome.jpg)

In case you made it this far, thanks a lot for reading! If you have a similar story or you are planning to change your career path into data science or development, feel free to contact me and exchange some ideas on this exciting topic.

Cheers,
Pascal`
);
