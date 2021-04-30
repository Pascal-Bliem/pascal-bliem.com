import Post from "../postModel";

export default new Post(
  // title
  "Machine Learning, NLP, and Toxic Language Classification",
  // subtitle
  "An introductory talk I gave at the CODEx2020 Developer & IT-Expert Conference",
  // publishDate
  new Date("2020-11-18"),
  // titleImageUrl
  "https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/codex2020/codex_thumbnail.png",
  // titleImageDescription
  "Come learn about machine/deep learning and natural language classification!",
  // tags
  ["Data Science", "Learning"],
  // content
  `### Check out the talk
At the CODEx2020 Developer & IT-Expert Conference I gave a talk about Machine Learning, Natural Language Processing (NLP), and Toxic Language Classification with [ToxBlock](http://www.pascal-bliem.com/tox-block), a Python deep learning application I built last spring.

I give an introduction to machine/deep learning, go through the basics of natural language processing, talk about [ToxBlock](http://www.pascal-bliem.com/tox-block), my toxic language classification package, and finally give a demo. If you're not a machine learning or NLP practitioner already, this recording may be a decent explanation of the fundamentals to you.

You can find a ToxBlock project page with links to Github repositories, PyPI, and blog posts [here](http://www.pascal-bliem.com/tox-block). And here's the recording of the talk:

<div style="text-align: center;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/y8pvfu4F2uo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
Feel free to contact me if you have any question, remarks, or ideas on the topic.  

The icons in the presentation were kindly provided free of cost by Eucalyp, Freepik, and Nhor Phai at [FlatIcon](https://www.flaticon.com/).
`
);
