import Post from "../postModel";

export default new Post(
  // title
  "Thanks for the feedback",
  // subtitle
  "A summary and discussion of the book of same title",
  // publishDate
  new Date("2021-06-28"),
  // titleImageUrl
  "https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/feedback/feedback.jpg",
  // titleImageDescription
  "How to best use feedback to grow?",
  // tags
  ["Non-Tech", "Learning"],
  // content
  `This is gonna be on feedback: how to get the most out of the feedback you receive and use it for your personal development, as discussed by Douglas Stone and Sheila Heen in their [book](https://www.amazon.com/Thanks-Feedback-Science-Receiving-Well/dp/0670014664) _Thanks for the Feedback - The Science and Art of receiving Feedback well (even when it is off base, unfair, poorly delivered, and, frankly, you're not in the mood)_. Yep, pretty long title, but it conveys the message quite clearly. Stone and Heen are both lecturers at the Harvard Law School and have studied how to get most value out of feedback in their research in the Harvard Negotiation Project. When I read non-fiction books, I usually like to take notes. This time, I thought I should condense my notes into a blog post, because I really enjoyed this book and found it super useful for extending my own understanding of feedback, both in terms of professional and personal relationships. So, this post is basically a summary and light discussion of the aforementioned book.

Besides learning new technologies, programming languages, or frameworks, I think it is important for developers to sharpen their soft skills as well. Especially the once concerning interpersonal aspects. After all, most of us don't work completely isolated from other humans; we interact with our colleagues and customers and there's a lot they can teach us about ourselves and about what we could potentially improve in this interaction or the way we do our work in general. Probably most businesses have understood by now that it's important to give feedback to their employees if they want them to improve and managers are often trained in giving some sort of feedback to their teams. The receiving end is often overlooked, even though, as Stone and Heen argue in their book, it is the more important side for making feedback effective. Just because you give someone feedback, it doesn't mean that they'll act on it. Actually, receiving critical feedback can be pretty difficult, because it shows us that what we've been doing so far was not perceived as ideal. We often feel triggered by feedback we receive, which makes it hard to rationally look at it, but very easy to discard it. In the following, we'll have a look at why exactly we have a hard time with receiving feedback and what we can do to change that and use is for our personal growth instead.

### Why can it be hard to receive feedback well?

Feedback comes in many forms, some of it is certainly welcome. A clap on the shoulder, or being told "hey you did great, keep going" sure feels nice. But critical feedback? Getting an evaluation that does not match our expectations? Being told that the way we usually do our work, the way we've perfectionized over a long time, is actually suboptimal and requires some adjustments? Doesn't feel so nice. But what exactly is it that makes us feel bad about it? Stone and Heen have identified three main groups of triggers that can make it hard for us to receive feedback well. _Truth Trigger_ revolve around the actual content of the feedback. We may feel that it i simply wrong, unhelpful, or unaligned with our expectations. In consequence, we feel wronged and discard the feedback. _Relationship Triggers_ are activated by the particular person that is giving us the feedback. Our focus shifts from the feedback itself to how we thing that this person, in our relationship, shouldn't be giving us such feedback. Lastly, there are _Identity Triggers_. We feel attacked when the feedback contradicts the image of how we perceive ourselves. Let's discuss these three categories in more detail below.

### Truth Triggers

To understand why we often feel that feedback is wrong, we should try to first understand what feedback actually is. According to the authors, there are three general types of feedback that have different purposes:

- **Appreciation**: to motivate and encourage
- **Coaching**: to helps increase knowledge, skill, capability etc. or raise feelings in relationships
- **Evaluation**: to tell us where we stand, inform decision making, and align expectations

While all of these are useful and necessary, we don't always want or expect all of them. If the giver and the receiver of feedback talk cross-purpose because they have different ideas on which type the conversation should be about, the feedback surely won't be received well. So the first thing we should be doing before even getting or giving feedback is to be mindful about what need/want or what we're being offered, and align our expectations. It might also be a good idea to actually cover the different purposes in separate conversations. Evaluation, especially if it's not the evaluation we were hoping for, can drown the rest. Imagine you get a bad performance review and then your manager tries to coach you on what you should improve. Probably you won't be listening to a single word of that coaching because your mind is occupied with thinking about what an idiot your boss is for giving you a bad review. Let some time pass so that you'll (hopefully) understand that it's a good idea to receive coaching so that your next performance review will be better.

Feedback can be pretty vague; certainly not everyone is giving it to us in a fully comprehensible manner. Hence, it can be hard to see their point and see what may be true about their feedback. It is, however, very easy to spot what we think is wrong about it right away. But if it is obviously wrong, why would they give it in the first place? Maybe there is a tiny bit of truth in it. To find that tiny bit, discuss and understand where the feedback is coming from and where it is going to. May they have different data than us? Or a different interpretation of it? It's easy to see the same thing but put very different labels on it. Do they want to give us advice or inform us about expectations or consequences? We need to ask ourselves what is right and legit about their feedback, and see what concerns we have in common. If we can work together with the feedback giver, we can get a more complete picture and maximize what we'll (both) get out of the conversation.

Another reason that makes wrong-spotting in others' feedback for us so easy is that we simply don't see all the things about ourselves that others may be seeing. We all have some blind spots that are obvious to others or we may be unconsciously sending some signals that we are not aware of but that are easily picked up by others. Most of the time we are not aware of our facial expression or tone of voice. Maybe we think we're saying really nice things, but our face and voice are telling everybody else that we mean the opposite. We may be unaware of even big patterns of behavior. Let's think about what factors could cause or amplify our blind spots. When it comes to feedback, often we tend to discount our own emotions, but others may count them double. We also tend to attribute positive outcomes to our own efforts while we try to excuse negative outcomes with the situation or circumstances. Others may do the exact opposite and attribute negative outcomes to our character. They saw that we messed it up; must be our fault, why care about the circumstances. Another problem may be a gap between the intention of our actions and the actual impact they have on others. We judge ourselves by our intention, while others, of course, only care about our impact on them. It doesn't matter how good our will was if we try to fix a situation for others but end up making it worse. The problem is that in all these situations, we're usually not aware of the fact that others may perceive our actions so differently from how we do. To identify our own blind spots, we need the help of others. So let's invite our feedback givers to help us see ourselves and answer the question of how we are standing in our own way.

### Relationships and Systems

As mentioned before, we're often more triggered by who is giving the feedback, rather than by its actual content. How we receive feedback depends a lot on how we think of the giver. Do we trust them or think that they're credible and have good judgement and skill? Imagine the "business" guy tells you that your code performs bad and needs to run faster. What does he know? He doesn't even know how to program and has never worked on a software development project. Easy to discard it right away. But hey, maybe you could optimize it and make it run a bit faster, regardless of who is telling you to do so. We also tend to care a lot about how we are treated by the feedback giver. Do we feel that we are accepted, appreciated, and that our autonomy is respected? If not, we're likely to not take the feedback well. Who wants to listen to a jerk's advice anyways. We'd rather give them some of our feedback, telling them how obnoxious we think they are. Before we know it, we're suddenly having a conversation with two totally different topics and are talking past each other. Stone and Heen call this phenomenon switch-tracking. Maybe topics in a feedback conversation need to be discussed both ways, but better not simultaneously. If we realize that we're switch-tracking, we need to give both topics their own tracks. It can be challenging to take feedback from people that we don't trust, e.g. strangers, and especially from people we find difficult. But people who see us at our worst may be especially good at pointing out the areas where we have the most room to grow; hence, we should not avoid these conversations. We should be vigilant, though, for relationship issues hiding under coaching. Not everyone who's trying to "fix" us has our wellbeing in mind.

If we're triggered by feedback due to the person delivering it, it is a good idea to look out for relationship systems that may be the cause of our discomfort. We can take three steps back here. First, look at the Us + Them intersection. Are the differences between us creating friction? How are we in the relationship? E.g., does your business partner want you to be really resourceful and not spend one cent too much, whereas you think that you need to invest more courageously, and are you both criticizing each other for your respective lack of initiative and wastefulness? Instead of focusing on what the other person is doing wrong, step back and notice what each one is doing in reaction to the other. Next, we step back further and see if it's our respective roles that cause the friction. Do our roles clash? Are the incentives that come with our roles opposing each other or are our responsibilities overlapping or poorly defined? If any of that is the case, the problem probably doesn't lie with any individual, but rather with how their roles are organized. If we step back even further, we can have a similar look at processes, policies, or the physical environment reinforcing the problem. Are centralized processes too inflexible for local needs? Are different times zones making it difficult to cooperate with your overseas colleagues? Are you being pulled into a conflict with another team because new cooperate policy wants them to kind of replace your team? Again, probably not any individuals fault, but a result of the systems we're all in. Looking at systems helps us to reduce judgement, enhance accountability, and uncover root causes of problems.

### Identity and Growth Mindset

The last category of triggers is probably the most difficult to accept, because they can attack our very own identity and the way we see ourselves. If people criticize what we do, we may think they are criticizing us as a person, the way we are, they way we live our lives. Different people react very differently to feedback that could be interpreted as attacking their identity. We have different baselines, and the emotional impact (up or down) of feedback as well as how fast we recovery from it, can vary a lot. Our emotions can distort our stories about the feedback itself significantly. According to Stone and Heen, about half of the influence on how we perceive feedback is genetically wired into our predisposition. While some people with a high baseline are hardly bothered by critical feedback, others with a lower baseline may feel that the negative feedback they got now is what they've always been getting and always will be getting for the rest of their future. To mitigate the emotional impact of feedback, we first need to dismantle the distortions we perceive. There are a couple of things we can do about that: We should be prepared and mindful when we're getting feedback and separate the strands. What is the actual statement, and what are our feelings about it and the story we craft from them? Those are different things. We can try to contain the story that forms in our mind by making clear what the feedback actually is about and what it isn't about. If we did bad at one particular task, and we're being told so, it doesn't mean we're horrible humans or that the feedback giver thinks so. But no matter what others think about us, we need to accept that we cannot control the way they see us. And we don't need to fully buy their story about us either - other's views on us are input, not imprint.

The probably most important strategy for receiving critical feedback well and using it for our personal development is to cultivate a growth identity. How do we tell our own story? Do we think that we're either all-or-nothing good or bad at something? Then we should probably shift to appreciating how complex reality is. If we think that all of our abilities are given, predetermined by our genetics, our upbringing, or sociocultural background, we leave no room for development. When we fail at something that we'd like to be succeeding at, instead of thinking "oh well, I guess I suck at this and there's nothing I can do about it", rather shift to "I might suck at this now, but I can certainly get better at it if I work on it hard". We must shift to seeing challenges as opportunities and feedback as useful information for learning and growth. When cultivating a growth mindset, try to really accept the coaching you get as such, and try to even get some coaching out of plain evaluation. When being evaluated, try to separate the judgement from the actual assessment and consequences. Also, don't judge yourself only based on the first evaluation you get, but rather on how well you responded to it and how much you managed to grow from that feedback - give yourself a second score for how you handled the first score! I personally do fully agree with Stone and Heen on this point here. The right mindset is the single most important thing for personal and professional development. I can observe it in myself; I can welcome even painful feedback as long is I'm convinced that it'll give me a chance to improve myself, and I've observed the same with my students.

### How good do I have to get?

Okay, so should we all grow infinitely? If there's always more to fix, to improve, if the criticism never ends, we may eventually just burn out. Therefore, setting boundaries on feedback is also critical to healthy relationships and life-long learning. Stone and Heen define three types of boundaries that can be useful to not be overwhelmed by feedback: _Thanks and No_, we're happy to hear you advice, but we may not take it. So please don't complain that we never listen to you; we do, but we consider your advice as only one possible option. Next there is _Not now, not about that_, we need some time and space now and not talk about that sensitive topic. We just got fired and need to have a couple of drinks before we want to listen to why we messed it up again. Lastly, _No Feedback at all_, I know what you'd like to say and I don't want to hear it. That's my decision and you need to respect it. If our relationship continues depends on if you can keep your judgement to yourself or not. When we feel like we have to turn down feedback, we should try to be appreciative and firm, and rather use the word "and" instead of "but". We don't necessarily want to disagree with what the feedback giver said _and_ we also don't want to discuss it further for now. Let's be specific about our exact request, the time frame, the consequences, and their assessment. Also, if we decide that we won't be changing based on their feedback, we should at least try to reduce the negative impact that we have on them. We need to understand that impact, and then try to problem-solve together, maybe even coach them in turn on how to deal with the unchanged us.

### Conclusion

Based on the work of Douglas Stone and Sheila Heen, we've discussed that feedback is a great opportunity for growth, but can often be hard to receive well. We've seen that three main groups of triggers concerning the truth of the feedback, the relationship to the person giving it, and the impact on or identity, can make it difficult for us to accept the feedback we get. Feedback can be roughly categorized into appreciation, coaching, and evaluation, and comes in vague labels; hence, intentions are often confused and expectations not met. We often have blind spots that we're not aware of and need the help of others to see them. Sometimes the relationship to the feedback giver is problematic because our roles are opposed or processes and policies are causing our conflict. We often have feedback for each other and it is important to not switch-track in one conversation, but give each topic its own track. Receiving feedback can be particularly tricky when we feel that our identity is under attack. To not feel emotionally assaulted, we need to dismantle the distortions that we build around the actual feedback itself, and cultivate a growth mindset that allows us to see feedback as an opportunity to develop ourselves. Lastly, we've also seen that it's important to be able to draw boundaries when we're not able or ready to receive certain feedback. I really enjoyed this book and I'll probably have a look at the authors' other book, [_Difficult Conversations: How to discuss what matters most_](https://www.amazon.com/Difficult-Conversations-Discuss-What-Matters/dp/0670921343) soon. So, thanks to Douglas and Sheila for the book and thank you for reading this post!
`
);
