import Post from "../postModel";

export default new Post(
  // title
  "ToxBlock: using AI to keep discussions clean",
  // subtitle
  "A deep learning application that recognizes verbal toxicity in text",
  // publishDate
  new Date("2020-06-15"),
  // titleImageUrl
  "https://wp.stanforddaily.com/wp-content/uploads/2018/03/AL.030818.Clickbait.crop_-1.jpg",
  // titleImageDescription
  "Cut the bulls%@$t !",
  // tags
  ["Data Science & AI/ML"],
  // content
  `**TL;DR:** I built an open-source Python package called \`tox-block\` which can classify English text into six different categories of verbal toxicity. You can download it from the Python package index ([PyPI](https://pypi.org/project/tox-block/)) with \`pip install tox-block\` and find some usage examples in its [repo on Github](https://github.com/Pascal-Bliem/tox-block). If you're interested in the motivation and story behind it, just keep on reading.

*Disclaimer: Since this is a post on classification of toxic language, it inevitably contains examples of toxic language that may be considered profane, vulgar, or offensive. If you do not wish to be exposed to toxic language, DO NOT proceed to read any further.*

### Why care about toxic language?

There is no place like the internet when it comes to observing what the Germans call *verrohung der Gesellschaft*, which roughly translates to *brutalization of society*. Being caught in confrontation-lacking echo bubbles and hidden behind the anonymity of a screen and a fake user name, people can say very nasty things when they suddenly find themselves confronted with options or views that differ form their own. While some social networking companies thrive on the user engagement that comes with dirty verbal fights, most moderate users suffer. And not just on a level of personal sentiment. Extreme verbal toxicity scares away moderate users and effectively shuts down any proper discussion or exchange of opinion; a process that is absolutely crucial for the development of educated and fact-based opinions and viewpoints in an open and free society.

So what can be done to foster reasonable discussions on social media or in the comment sections of news sites, blogs, and video streaming platforms? Of course, toxic contents can usually be reported to the platform hosts (in many countries even to the police) by other users, but that usually requires a manual review by the hosts' staff; a task for which many small publishers simply don't have the necessary capacities. Larger companies can often outsource this tasks to low-wage countries, but that doesn't necessarily speed up the review process and, frankly, it's not a nice job to have. I have personally talked to so-called *content moderators* in the Philippines and Thailand, and they have confirmed that seeing the darkest parts of the web all day long is impacting their mental health. No perfect solution, by far.

But hey, we live in the age of deep learning. Shouldn't AI be able to take care of this by know? In fact, sentiment classification is a classic task in the field of natural language processing (NLP) and I suppose that classifying toxicity falls somewhere along those lines. About two years ago in spring 2018, Kaggle concluded a [competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview) on toxic comment classification hosted by Conversation AI / Jigsaw. They provided labeled data of about 250000 comments from Wikipedia articles of which about 10% contained some type of the toxicity categories toxic, severe toxic, obscene, insult, threat, and identity hate. Many of the competitors managed to build very well performing models, so I thought this may be a great chance for me to learn a thing or two about NLP and maybe end up building something interesting or even useful.

The next two section will go deeper into what can be done with the given data and how the neural network model behind ToxBlock works. If you're less interested into getting lost in the technical details and more into finding out how you can actually use ToxBlock, feel free to fast-forward to the [corresponding section](#usage).

### Taking a look at the data

The [data set](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) I used already came in a very usable form; columnar data with the comment text the the respective labels. Each comment could have multiple labels if belonging to multiple categories of toxicity. The classes are distributed quite unevenly, which can make it more challenging for a machine learning algorithm to recognize instances of the minority classes. Only about 10% of the comments in the data set belong to any of the toxic classes. Also among those classes the distribution is not balanced, as can be seen in the figure below.

<img src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/tox-block/class_distribution.png" alt="Distribution of classes in the comment data set." style="width: 60%;">

If we look at the 10 most commonly appearing words in toxic comments (I've censored them here to not have my site flagged as NSFW by search engines), some are a bit of a surprise:

- fu***k
- like
- sh***t
- wikipedia
- ni***er
- fu***king
- suck
- ass
- hate
- u

I really hope my model won't end up recognizing the word *Wikipedia* as an insult.
Let's randomly pick a non-toxic and a toxic comment to get an idea for how they look like.  

A non-toxic comment:  
" Main Page 

I think whoever is really writing this article should try to get it featured on the main page before the election, because after the election who cares? "  

A toxic comment:  
" Cu###ts

Way to discuss, eh?  Protect the article and then fu###k it up.  Despicable.  Well if you're the kind of people who think there is some special meaning to being blown up in a Montego, I haven't got the fu###king time to argue.  190.46.108.141"  

We can see in the toxic example that it contains the user's IP address. Besides of IPs some of the comments also contain user names and hyperlinks - all features that should be removed because they are not useful for general text classification and can even cause data leakage from the training into the test set, in case the same users show up in both sets. 

Besides dataset-specific cleaning, there are some common an general preprocessing strategies for text data in NLP. As in any machine learning problem, one wants to maximize the signal to noise ratio. For text that means that there is a lot that we can sort out before before we train a model or make inferences. In most cases, punctuation and numbers don't play a huge role in classifying the meaning in text, hence, they're often removed. But there are also regular words that used so frequently that they rarely contribute much information. Such words are called stop words and are often removed in NLP preprocessing. In English those could be words such as personal pronouns ("I, you, he"), "otherwise", "regarding", "again", "into", "further", you get the idea. You can find a collection of typical stop words for over 40 languages [here](https://www.ranks.nl/stopwords). It sometime makes sense to summarize different word with a similar or same word stem into the same word if they convey the same meaning. This is called *text normalization* and two popular methods are called *stemming* and *lemmatization*. Stemming or lemmatization reduce inflection in words to their root forms, with the difference that lemmatization produces stems that actually exist in the language. All these NLP task are readily implemented in NLP Python packages like \`spacy\` or \`nltk\`, but I actually ended up using none of them in the final model. The performance gain resulting from these methods was so small that I decided to go for less computations and, hence, faster inference.

### How the model works

Before deep learning became so popular, features were usually extracted from text by *tokenizing* the text of each document into words or [n-grams](https://en.wikipedia.org/wiki/N-gram) and creating some kind of token frequency matrix (e.g. by count vectorization or [TI-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vectorization). This would then be used as the input for some classifying algorithm. In the aforementioned Kaggle competition, a combination of naive Bayes and linear support vector machine classifiers, as described by [Wang and Manning](https://github.com/sidaw/nbsvm/blob/master/wang12simple.pdf), performed very well. The other go-to approach is to rely on deep learning, or recurrent neural networks to be more specific. These type of networks are suited for processing sequence data, as their units possess an internal state that can be passed from one unit to the next. As text can be considered a sequence of words, and the meaning of one part of a sentence probably depends on another part of the sentence (or sequence), recurrent neural networks are great for NLP tasks. The two most popular types from this family are called gated recurrent units (GRU) or long short-term memory (LSTM) networks. I went for the latter. If I'd try to describe the whole model in one gibberish sentence it would be something like: tokenize and sequentialize the input text, feed it into a pre-trained embedding layer, then into a bidirectional LSTM layer, use global 1D max-pooling, then into a fully-connected layer, add some dropout, into a final fully-connected layer with Sigmoid activations, and train the whole thing with its 1,045,756 parameters.

But what the heck does that mean? Let's look at it step by step. First we have to tokenize the sequences of text and turn them into a numerical form so that an algorithm can process them. Imagine we have a collection of documents, e.g. short sentences:
\`\`\`python
["These are not the droids you are looking for.",
 "Actually, I think these are exactly the droids we are looking for."]
\`\`\`
Now we take each document apart word by word, and turn each word into a number. Of course, the same word needs to be assigned the same number across all documents:
\`\`\`python
[
    [2, 1, 7, 3, 4, 8, 1, 5, 6], 
    [9, 10, 11, 2, 1, 12, 3, 4, 13, 1, 5, 6]
]
\`\`\`
Finally, we need cut off or pad these numerical sequences with zeroes so that they all have the same length:
\`\`\`python
[
    [ 0,  0,  0,  0,  0,  0,  2,  1,  7,  3,  4,  8,  1,  5,  6],
    [ 0,  0,  0,  9, 10, 11,  2,  1, 12,  3,  4, 13,  1,  5,  6]
]
\`\`\`
Now these sequences can be fed into an embedding layer. The idea behind word embeddings is to encode the meaning of words as vectors in high dimensional space. Word vectors of words with a similar meaning (e.g. bee, wasp, bumblebee, hornet) should be close together in this space, vectors of words that have nothing to do with each other should be far apart, and vectors of words with opposite meanings like "front" or "back" should ideally pointing into the opposite direction. If you would subtract the "male" vector from the word "king" and add the "female" vector, you should end up somewhere close to the word "queen" (as sketched in the figure below). To train such embeddings well, one usually needs **a lot** of text; hence, it often makes sense to initialize the parameters of the embedding layer with pre-trained embeddings and then fine-tune them on the given task. There are many pre-trained embeddings available such as [ELMo](https://allennlp.org/elmo), [BERT](https://pypi.org/project/bert-embedding/), [Word2vec](https://code.google.com/archive/p/word2vec/), or [fastText](https://fasttext.cc/); I decided to go with [GloVe](https://nlp.stanford.edu/projects/glove/) which has been pre-trained on the Wikipedia and Gigaword 5 corpora. 

![Word embeddings encode the meaning of word into n-dimensional vectors.](https://miro.medium.com/max/3010/1*sXNXYfAqfLUeiDXPCo130w.png)

After having transformed the sequence of words into sequences of word vectors, we can pass them on to the core piece of the model: the recurrent layer. [Simple recurrent units](https://en.wikipedia.org/wiki/Recurrent_neural_network) do often have a problem of [vanishing gradients](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) during training. To avoid this problem, gated recurrent neural networks, such as the LSTM, have been introduced. Besides getting a sequence as input and also outputting a sequence, a layer of LSTM units (or cells) also propagates a cell state vector from one cell to the next (see figure below). Each cell can read from it, write to it, or reset itself, and it does so by so-called gates. An LSTM unit has three of these gates called *forget*, *input*, and *output* which decide how much information of the cell state is kept, what values will be updated, and what parts of the cell state will be the output. I don't want to discuss the ins and outs of LSTMs in length here, but let me point you to this great [article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). The LSTM layer in this model is bidirectional, which means it parses the input sequences both ways, front to back and back to front. This can be very useful when dealing with natural language, as the different parts of a sentence may depend on each other in both directions (You don't see what I mean? Try learning German...we chop up verbs into several pieces and scatter them all over the sentence).

![Recurrent units in an LSTM avoid the vanishing gradient problem by using gates to learn what values of the cell state should be forgotten, updated, and outputted.](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/10131302/13.png)

The recurrent layer will output another sequence, but at this point we are going to down-sample it a bit. We apply global-max-pooling over the sequence, which means of all the steps (or n-dimensional word vectors) in the sequence, we only take the maximum value fo each dimension. That sounds a bit confusing, so let's visualize it. Imagine we have a tensor of shape (3,3,1) corresponding to (batch size, sequence steps, vector dimensions):
\`\`\`python
[[[1.], [2.], [3.]],
 [[4.], [5.], [6.]],
 [[7.], [8.], [9.]]]
\`\`\`
One-dimensional global max-pooling would reduce it to a tensor of shape (3,1) corresponding to (batch size, vector dimensions):
\`\`\`python
[[3.],
 [6.],
 [9.]]
\`\`\`
Why would we do this in the first place? We'll feed it into a fully-connected layer next, for which the tensors need to be flattened in some way. By global max-pooling, we can greatly reduce the dimensionality and try to "extract signal from noise". The argument goes something like: the sequence part of interest is likely to produce the largest value, hence, it shall be most interesting to only take the maximum value.

Next stop is a fully-connected (or dense) layer. All of the values we got from the previous layer will be multiplied with specific weights (which are learned during training) and serve as inputs for all units in the fully-connected layer (as sketched in the figure below). The outputs of this layer will be passed through a [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation and some [dropout](https://en.wikipedia.org/wiki/Dilution_(neural_networks)). In the context of neural networks, dropout means that some of the connections between the layers are randomly set to zero, which forces the network to robustly spread the information flow over more units and, hence, serves as a regularization technique to reduce over-fitting. Finally, we enter a last fully-connected layer with only six units, corresponding to the six categories of verbal toxicity that we are trying to predict. Each of their outputs is passed to a [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation function that forces the output between zero and one - a value that corresponds to the predicted probability of the respective category.

<img src="https://www.researchgate.net/profile/Srikanth_Tammina/publication/337105858/figure/fig3/AS:822947157643267@1573217300519/Types-of-pooling-d-Fully-Connected-Layer-At-the-end-of-a-convolutional-neural-network.jpg" alt="In fully-connected layers, each output of a unit is multiplied with specific weights (learned during training) to serve as inputs for all units in the next layers. Thicker connections in this image imply large weights, meaning that much information is propagated over these connections." style="width: 50%;">

### How you can use ToxBlock
<a id="usage"></a>

You can find the source code and usage examples in the [ToxBlock repo](https://github.com/Pascal-Bliem/tox-block) on Github, from where you can also clone it. To make it easily accessible, I also put \`tox-block\` on the Python package index ([PyPI](https://pypi.org/project/tox-block/)) from where you can download and install it via
\`\`\`python
pip install tox-block
\`\`\`
The methods for prediction are contained in the module \`tox_block.prediction\`. Predictions for single strings of text can me made via \`tox_block.prediction.make_single_prediction\`:

\`\`\`python
from tox_block.prediction import make_single_prediction

make_single_prediction("I will beat you up, you f***king idiot!")
\`\`\`
It will return a dictionary with the original text and the predicted probabilities for each category of toxicity:
\`\`\`python
{'text': 'I will beat you up, you f***king idiot!',
 'toxic': 0.9998680353164673,
 'severe_toxic': 0.7870364189147949,
 'obscene': 0.9885633587837219,
 'threat': 0.8483908176422119,
 'insult': 0.9883397221565247,
 'identity_hate': 0.1710592657327652}
\`\`\`
To make bulk predictions for several texts, they can be passed as a list of strings \`into tox_block.prediction.make_predictions\`:
\`\`\`python
from tox_block.prediction import make_predictions

make_predictions(["Good morning my friend, I hope you're having a fantastic day!",
                  "I will beat you up, you f***king idiot!",
                  "I do strongly disagree with the fascist views of \\
                   this joke that calls itself a political party."])
\`\`\`
It will return a dictionary of dictionaries of which each contains the original text and the predicted probabilities for each category of toxicity:
\`\`\`python
{
0: {'text': "Good morning my friend, I hope you're having a fantastic day!",
  'toxic': 0.05347811430692673,
  'severe_toxic': 0.0006274021579883993,
  'obscene': 0.004466842859983444,
  'threat': 0.009578478522598743,
  'insult': 0.00757843442261219,
  'identity_hate': 0.002106667961925268},
 1: {'text': 'I will beat you up, you f***king idiot!',
  'toxic': 0.9998679757118225,
  'severe_toxic': 0.7870362997055054,
  'obscene': 0.9885633587837219,
  'threat': 0.8483908176422119,
  'insult': 0.9883397221565247,
  'identity_hate': 0.171059250831604},
 2: {'text': 'I do strongly disagree with the fascist views of this joke that calls itself a political party.',
  'toxic': 0.026190076023340225,
  'severe_toxic': 7.185135473264381e-05,
  'obscene': 0.0009493605466559529,
  'threat': 0.00012321282702032477,
  'insult': 0.0029190618079155684,
  'identity_hate': 0.0022098885383456945}
}
\`\`\`
That's basically the core functionality. For more details, please refer to the [repo](https://github.com/Pascal-Bliem/tox-block). How about you try to integrate this Python package into you own applications, or use it to keep your own website, forum, or blog clean from nasty language that ruins everyone elses experience? 

Having a python package for toxic language recognition is cool, but maybe it's not the most elegant solution to integrate it into your code base, especially if your application is written in a language other than Python. We're on the web anyway, so we might as well handle all of this with a simple HTTP request. That's why I'm currently working on the **ToxBlock API**, a REST API that will make ToxBlock's predictive capabilities available for any other application that can wrap nasty texts in JSON and POST it to an API endpoint. Stay tuned for more and thanks for reading!
`
);
