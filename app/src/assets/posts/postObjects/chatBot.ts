import Post from "../postModel";

export default new Post(
  // title
  "Starfleet assimilated by the ChatBorg",
  // subtitle
  "A sequence-to-sequence deep learning approach to generating Star Trek dialogues",
  // publishDate
  new Date("2020-08-18"),
  // titleImageUrl
  "https://i.ytimg.com/vi/Hno3K8H7U5g/maxresdefault.jpg",
  // titleImageDescription
  "We are the ChatBorg. Training your neural networks is futile. Your vanishing gradients will be assimilated.",
  // tags
  ["Data Science", "Learning"],
  // content
  `**TL;DR:** In this project I tried to train a sequence-to-sequence deep learning model on scripts of all the Star Trek TV shows with the aim to let it generate artificial Star Trek dialogues. I used an encoder-decoder long short-term memory (LSTM) network architecture, which is commonly found in task such as machine translation, question answering, and simple chatbots. The idea is to pass a dialogue line into the encoder and let the decoder generate an appropriate response. Long story short, it didn't work very well and only generates very short sentences without much meaning. I think this may be due to the data set being heavily dominated by few-words-long dialogue lines. Furthermore, free dialogue generation, as opposed to tasks with a clear request-response targets (such as translation or question answering) requires much more of a language understanding. For this, my model (due to my computational resource limitations) is way too simple, and large pretrained, transformer-based models may be a more suitable choice. So if you were looking for cool results, you may stop reading here. However, I found this project to be very educational and a great introduction to seq2seq models, so I decided to turn it into a blog post anyway. If you're interested in it for the sake of learning, please keep reading.

### Introduction

Being a scientist, I guess it's not much of a surprise that I'm a big nerd and I love all sorts of Scify stuff. Star Trek is definitely one of my favorite franchises (though I love Star Wars as well) and I was very happy when I came across a [data set on Kaggle](https://www.kaggle.com/gjbroughton/start-trek-scripts) that contained raw text scripts of all Star Trek series episodes, scraped from [this](http://www.chakoteya.net/StarTrek/index.html) website. I recently took an online course titled [Deep Learning: Advanced NLP and RNNs](https://www.udemy.com/course/deep-learning-advanced-nlp/) in which the lecturer gave an introduction to sequence-to-sequence (seq2seq) models and used an LSTM-based encoder-decoder model to do neural machine translation from English to Spanish. It worked reasonably well for sentence to sentence translation and I wondered if it may be able to generate dialogues based on the same kind of request-response pattern. Wouldn't it be cool if we could generate new Star Trek scripts? Well, as I already mentioned in the TL;DR, it didn't work very well, but I found it very educative and wanted to share it. Below, I will discuss the following sections:

1. [Seq2seq Theory](#theory)
2. [Preprocessing the Data](#data)
3. [Building the Model for Training](#training)
4. [Modifying the Model to Generate Text](#decoding)
5. [Generating Dialogues](#generate)
6. [Conclusion](#conclusion)

<a id="theory"></a>

### Seq2seq Theory

When working with text, we usually consider a sentece (or a document) as a sequence of words, much like a time series of words. Usually, some kind of recurrent neural network (RNN) architecture is used to process sequences in deep learning (though transformer-based models have become the state of the art in natural language processing lately). These kind of networks carry a hidden or state, sort of a context, from processing one part of the sequence to the next, which allows them to account for dependencies throughout the sequence. This is conceptually shown in the figure below.

![A RNN "unrolled" - passing a state or context from one part of the sequence to the next.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/startrek-chatbot/RNN-unrolled.png)

The RNNs we are using are called long short-term memory (LSTM) networks, which are a little more complicated than in the figure above, but essentially are based on the same principles: they carry states (called hidden and cell states for the LSTM) from one part of the sequence to the next to understand long-range dependencies. With these kind of networks, we can build models that have different input-output-relationships, some of which are depicted in the figure below. We may want to input maybe just one topic and get a matching paragraph generated (one-to-many) or read a paragraph and classify its topic (many-to-one). We may also want to train a model to find the most important signal among all sequence steps (many-to-many with same input and output lengths, potentially with global max pooling) or we want to input a sentence and get a different one as a response, such as in machine translation or question answering (many-to-many with variable input and output lengths).

![Different relationships of input and output in sequence processing models.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/startrek-chatbot/reltionships.jpg)

Our example of generating dialogue belongs to the latter set of problems. We want to put in a dialogue line and get a different one as an output, similar to a question answering scenario. For these type of problems, an encoder-decoder architecture is commonly applied. The idea behind it is the following: We have an encoder LSTM that takes in the input sequence and encodes it. Instead of caring about the regular output of this encoder we just take it's final hidden and cell state, which is basically the context of the input sequence, and use them as initial states for the decoder. The decoder is another LSTM layer which takes these states and its own predicted word from the previous sequence step as input. During training, we try to make it a bit easier for the decoder to learn the correct output sequences by applying a method called teacher forcing. In teacher forcing, we feed the target output sequence offset by one as input into the encoder. During prediction, however, the every next prediction step in the sequence gets the actual prediction (and hidden and cell states) from the previous step as an input. The concept of an encoder-decoder model with teacher forcing is illustrated in the figure below for the example application of machine translation.

![An example of an encoder-decoder sequence-to-sequence model for machine translation.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/startrek-chatbot/seq2seq.svg)

To make the encoding of words into a numerical format a bit more effective than just using an integer, we'll use en embedding layer in front of both the encoder and decoder. The idea behind word embeddings is to encode the meaning of words as vectors in high dimensional space. Word vectors of words with a similar meaning (e.g. bee, wasp, bumblebee, hornet) should be close together in this space, vectors of words that have nothing to do with each other should be far apart, and vectors of words with opposite meanings like "front" or "back" should ideally pointing into the opposite direction. If you would subtract the "male" vector from the word "king" and add the "female" vector, you should end up somewhere close to the word "queen" (as sketched in the figure below). To train such embeddings well, one usually needs a lot of text; hence, it often makes sense to initialize the parameters of the embedding layer with pre-trained embeddings and then fine-tune them on the given task. I use the [GloVe](https://nlp.stanford.edu/projects/glove/) embedding here, which has been pre-trained on the Wikipedia and Gigaword 5 corpora.

![Word embeddings encode the meaning of word into multi-dimensional vectors.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/startrek-chatbot/wordembedding.svg)

Finally we put a dense layer with a softmax activation behind the decoder to find the most probable word from the vocabulary to chose. The model I'll use here is (partially to restrictions in computational resources or rather my unwillingness to spend a lot of money on them) is pretty simple. It is only one LSTM layer in both the encoder and decoder and dimensionalities in both the embedding layer and the LSTMs latent space were held fairly low. This may make it difficult for the model to actually learn some language understanding from the dialogue line. Anyway, we'll see - let's get started!

\`\`\`python
# import libraries

# computation & data
import numpy as np
import scipy
import sparse
import pandas as pd
import unicodedata
# plotting
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
#plt.rcParams['figure.figsize'] = [5.0, 7.0]
import seaborn as sns
sns.set_style("darkgrid")
# deep learning
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# standard python
import os, sys
import re
from typing import List
\`\`\`

<a id="data"></a>

### Preprocessing the Data

Here we'll load in the data, have a look at it and do some preprocessing before we feed it into the model.

\`\`\`python
# read in the data
data = pd.read_json("all_scripts_raw.json")
\`\`\`

\`\`\`python
# a look into the raw text of an episode of DS9
data["DS9"].loc["episode 101"][:1000]
\`\`\`

    "\\n\\n\\n\\n\\nThe Deep Space Nine Transcripts - Trials and\\nTribble-ations\\n\\n\\n\\nTrials\\nand Tribble-ations\\nStardate:\\nUnknown\\nOriginal Airdate: 4 Nov, 1996\\n\\n\\n\\n\\n\\n\\n  [Ops]\\n\\n(A pair of dour pin-striped bureaucrats arrive on\\nthe turbolift.) \\nKIRA: Welcome to Deep Space Nine. I'm Major Kira. \\nDULMUR: I'm Dulmur. \\n(An anagram of Muldur, and yes - ) \\nLUCSLY: Lucsly. Department of Temporal Investigations. \\nKIRA: We've been expecting you. \\nDAX: I guess you boys from Temporal Investigations are always on time. \\nDULMUR: Where's Captain Sisko?\\n\\n [Captain's office]\\n\\nSISKO: Are you sure you don't want anything? \\nDULMUR: Just the truth, Captain. \\nSISKO: You'll get it. Where do you want to start? \\nDULMUR: The beginning. \\nLUCSLY: If there is such a thing. \\nDULMUR: Captain, why did you take the Defiant back in time? \\nSISKO: It was an accident. \\nLUCSLY: So you're not contending it was a predestination paradox? \\nDULMUR: A time loop. That you were meant to go back into the past? \\nSISKO: Erm, no. \\nDULMUR: Good. \\nLUCSLY:"

We can see that every speakers’ line starts with a new line and their name in all caps followed by a colon, e.g. \\nSISKO:  
So, we can use this pattern to split the individual lines:

\`\`\`python
 # use a regex to split the raw script into lines of individual speakers
 # the ' character in the regex is for O'BRIEN, the [OC] means over comm, I think
 def split_lines(raw_script: str) -> List[str]:
     return re.split(r"(?:\\n[A-Z']+: )|(?:\\n[A-Z']+ \\[OC\\]: )", raw_script)

 lines = split_lines(data["DS9"].loc["episode 101"])
\`\`\`

\`\`\`python
# let's have a look
lines[1:25]
\`\`\`

    ["Welcome to Deep Space Nine. I'm Major Kira. ",
     "I'm Dulmur. \\n(An anagram of Muldur, and yes - ) ",
     'Lucsly. Department of Temporal Investigations. ',
     "We've been expecting you. ",
     'I guess you boys from Temporal Investigations are always on time. ',
     "Where's Captain Sisko?\\n\\n [Captain's office]\\n",
     "Are you sure you don't want anything? ",
     'Just the truth, Captain. ',
     "You'll get it. Where do you want to start? ",
     'The beginning. ',
     'If there is such a thing. ',
     'Captain, why did you take the Defiant back in time? ',
     'It was an accident. ',
     "So you're not contending it was a predestination paradox? ",
     'A time loop. That you were meant to go back into the past? ',
     'Erm, no. ',
     'Good. ',
     'We hate those. So, what happened? ',
     'This may take some time. ',
     'Is that a joke? ',
     'No. ',
     'Good. ',
     "We hate those too. All right, Captain. Whenever you're ready. ",
     'Two weeks ago the Cardassian Government contacted me and wanted\\nto return an Orb to the Bajorans. ']

There is still some cleaning to do here. We want to unicode-normalize everything to remove weird accents on alien names and such. There are still a couple of newlines \\n within the text and locations \\[Bridge\\] and scene descriptions (The turbo lift is full of bananas) are included in parenthesis. We'll also put whitespace between the last word of a sentence and the punctuation, remove trailing white spaces, and lowercase everything.

\`\`\`python
def preprocess_lines(lines: List[str]) -> List[str]:
    clean_lines = []
    for line in lines:
        # nomralize
        line = (unicodedata.normalize(u'NFKD', line).encode('ascii', 'ignore').decode('utf8'))
        # remove stuff in parenthesis
        line = re.sub(r"\\(.*\\)", "", line)
        line = re.sub(r"\\[.*\\]", "", line)
        # replace \\n  and weird chars with space
        line = re.sub(r"\\n", " ", line)
        line = re.sub(r"[^a-zA-Z?.!,¿]+", " ", line)
        # put space before punctuation
        line = re.sub(r"([?.!,¿])", r" \\1 ", line)
        line = re.sub(r'[" "]+', " ", line)
        # strip and lowercase
        line = line.strip().lower()
        clean_lines.append(line)
    return clean_lines
\`\`\`

\`\`\`python
# let's have a look at the cleaned lines
clean_lines = preprocess_lines(lines)
clean_lines[1:25]
\`\`\`

    ['welcome to deep space nine . i m major kira .',
     'i m dulmur .',
     'lucsly . department of temporal investigations .',
     'we ve been expecting you .',
     'i guess you boys from temporal investigations are always on time .',
     'where s captain sisko ?',
     'are you sure you don t want anything ?',
     'just the truth , captain .',
     'you ll get it . where do you want to start ?',
     'the beginning .',
     'if there is such a thing .',
     'captain , why did you take the defiant back in time ?',
     'it was an accident .',
     'so you re not contending it was a predestination paradox ?',
     'a time loop . that you were meant to go back into the past ?',
     'erm , no .',
     'good .',
     'we hate those . so , what happened ?',
     'this may take some time .',
     'is that a joke ?',
     'no .',
     'good .',
     'we hate those too . all right , captain . whenever you re ready .',
     'two weeks ago the cardassian government contacted me and wanted to return an orb to the bajorans .']

Cool, that seems to have worked, so let's apply it to all the data:

\`\`\`python
# we'll store all episodes' processed scripts in here
all_episodes = []

for col in data.columns:
    for raw_script in data[col][data[col].notna()].values:
       all_episodes.append(preprocess_lines(split_lines(raw_script)))
\`\`\`

Let's explore our corpus a bit. I'd like to know how many lines we have in total and how long they usually are.

\`\`\`python
num_lines = 0
line_lengths = []

for clean_script in all_episodes:
    num_lines += len(clean_script)
    for line in clean_script:
        line_lengths.append(len(line.split()))
print(f"Number of lines: {num_lines}")
\`\`\`

    Number of lines: 250708

\`\`\`python
# let's see how the line lengths are distributed (we wont )
line_lengths = np.array(line_lengths)
sns.distplot(line_lengths[line_lengths<1000], kde=False)
print(f"Number of words at the 0.99 quantile: {np.quantile(line_lengths, 0.99)}")
\`\`\`

    Number of words at the 0.99 quantile: 80.0

![The distribution of dialogue line lengths in words.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/startrek-chatbot/seqlenhist.png)

We can see that we have about 250000 lines in total and that in 99% of the cases, the lines are not longer than 80 words. Therefore, I'll choose 80 as the maximum sequence length here and will later pad all shorter sequences with zeroes to this length. We can also see that half of the lines have only 11 words or less. Most of the dialogue lines seem to be very short, which means they'll need a lot of zero-padding later, which doesn't convey any information. This may make it difficult for the model to predict longer, more interesting sentences later.

Here is probably also a good point to define the other parameters that will be used for the model. Besides the sequence length, I'll also define the maximum vocabulary size, dimensions for the word embedding and the latent LSTM encoding as well as training parameters such as batch size and number of epochs.

\`\`\`python
BATCH_SIZE = 32           # batch size for training
EPOCHS = 10               # number of training epochs
LATENT_DIM = 128          # latent dimensionality of the LSTM encoding space.
MAX_SEQ_LEN = 80          # the maximum length for the text sequences
MAX_NUM_WORDS = 20000     # maximum vocabulary size for the word tokenizer
EMBEDDING_DIM = 100       # dimensionality for the word embedding
\`\`\`

Since we want to build a dialog system, we need to train it in a way that it'll output a sentence based on the previous sentence, and so on and so forth. That means we'll create pairs of input-target sequences form all lines and add start of sentence <START\\> and end of sentence <END\\> tokens to them. We also want to train the decoder using teacher forcing, so we'll need the target sequences offset by one with a <START\\> token as well.

\`\`\`python
# we'll store the input, target, and target_inputs for
# teacher forcing in these lists
input_lines = [] # input lines
target_lines = [] # target lines
target_lines_inputs = [] # target lines offset by 1

# for each episode
for clean_script in all_episodes:
    # get all input and target lines (skip first one
    # because it is episode title line, and last one
    # because no response line will follow)
    for i, input_line in enumerate(clean_script[1:-1]):
        # if shorter equal than MAX_SEQ_LEN
        if (len(input_line.split()) <= MAX_SEQ_LEN and
            len(clean_script[i+1].split()) <= MAX_SEQ_LEN):
            # add start/end token and append to respective list
            input_lines.append("<START> " + input_line)
            target_lines.append(clean_script[i+1] + " <END>")
            target_lines_inputs.append("<START> " + clean_script[i+1])
\`\`\`

Now we have all the lines we need, but they're still strings, and a neural net cannot process them like this. We'll tokenize (split up by words and punctuation in this case) the text and turn it into numeric sequences.

\`\`\`python
# fit tokenizer on vocabulary
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer.fit_on_texts(input_lines + target_lines)

# the word index will allow is to translate from numbers
# back to actual words
word_index = tokenizer.word_index

# tokenize all three seq types
input_sequences = tokenizer.texts_to_sequences(input_lines)
target_sequences = tokenizer.texts_to_sequences(target_lines)
target_sequences_inputs = tokenizer.texts_to_sequences(target_lines_inputs)

# to feed the sequences into the encoder and decoder,
# we'll have to pad them with zeroes (left and right, respectively)
encoder_inputs = pad_sequences(input_sequences, maxlen=MAX_SEQ_LEN, padding="pre")
decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=MAX_SEQ_LEN, padding='post')
decoder_targets = pad_sequences(target_sequences, maxlen=MAX_SEQ_LEN, padding='post')
\`\`\`

Eventually, we want to have a model that can predict which of the MAX_NUM_WORDS is the most likely to come next. It's basically a many-class classification problem and we should probably use some form of categorical cross entropy as a loss function. Keras has a loss function called \`SparseCategoricalCrossentropy\` which allows to just encode the targets as integers, but as of now, it unfortunately doesn't work with sequence outputs. Therefore, I tried to use the regular \`CategoricalCrossentropy\`, which means that we'll have to one-hot encode the targets. I ended up using a custom version for the crossentropy to account for the zero-padding, but the one-hot encoding still applies. If we'd encode this in a dense array it wouldn't fit into memory, so we'll have to put it into a sparse tensor.

\`\`\`python
# assign the one-hot values
coords = [[],[],[]]
data = []
for i, d in enumerate(decoder_targets):
    dim1 = i    # which sequence
    for j, word in enumerate(d):
        dim2 = j  # which position in the sequence
        if word != 0:
            dim3 = word   # which word in the vocabulary
            coords[0].append(dim1)
            coords[1].append(dim2)
            coords[2].append(dim3)
            data.append(1.0)

# pass values to a sparse tensor of the right shape
# len(decoder_targets) x MAX_SEQ_LEN x MAX_NUM_WORDS
decoder_targets_one_hot = sparse.COO(coords, data,
                                     shape=(len(decoder_targets),
                                                MAX_SEQ_LEN,
                                                MAX_NUM_WORDS))
\`\`\`

<a id="training"></a>

### Building the Model for Training

We will try to improve our models understanding of the English language by using pretrained word embeddings, GloVe in this case. We'll load in the word vectors and set them as weights for the word embedding layers in front of the encoder and decoder.

\`\`\`python
# load the GloVe word vectors from file
glove_vectors = {}
with open(f"./glove.6B.{EMBEDDING_DIM}d.txt", encoding="utf8") as f:
  # the format of the file is: word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    glove_vectors[word] = vec

# create an embedding matrix from the vectors
embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))
for word, i in word_index.items():
  if i < MAX_NUM_WORDS:
    embedding_vector = glove_vectors.get(word)
    if embedding_vector is not None:
      # words not contained in embedding will be initialized as zero
      embedding_matrix[i] = embedding_vector
\`\`\`

We'll use this matrix as starting weights for the embedding layer.

\`\`\`python
# the embedding layer
embedding_layer = Embedding(
  MAX_NUM_WORDS,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQ_LEN,
  trainable=True   # we'll fine tune it
)
\`\`\`

Now it's finally time to build the actual model. We'll start with the encoder. It'll just be an LSTM layer, but instead of making any use of the outputs, we'll just keep the hidden and cell state of the LSTM, which we will later feed into the decoder as its initial states.

\`\`\`python
# the encoder
encoder_inputs_placeholder = Input(shape=(MAX_SEQ_LEN,))
encoder_inputs_x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(
  LATENT_DIM,
  return_state=True,
  # dropout=0.5
)
encoder_outputs, h, c = encoder(encoder_inputs_x)

# keep the hidden and cell states as
# initial states for the decoder
encoder_states = [h, c]
\`\`\`

The decoder will also be an LSTM layer which will predict a sequence (the text we want to generate) and uses the hidden and cell states from the encoder as initial states for itself. The decoder layer will return its states as well, which do not play any role during training, but they’ll be needed when we reuse this layer to make predictions later. We'll also add a final dense layer with a softmax activation to choose predictions from the vocabulary.

\`\`\`python
# the decoder
decoder_inputs_placeholder = Input(shape=(MAX_SEQ_LEN,))
decoder_inputs_x = embedding_layer(decoder_inputs_placeholder)

# Contrary to the encoder, we want to generate text
# as output, hence, return_sequences=True.
decoder_lstm = LSTM(
  LATENT_DIM,
  return_sequences=True,
  return_state=True,
  # dropout=0.5
)

# use the encoder states as the inital states
decoder_outputs, _, _ = decoder_lstm(
  decoder_inputs_x,
  initial_state=encoder_states
)

# dense layer with softmax to make predictions
decoder_dense = Dense(MAX_NUM_WORDS, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
\`\`\`

Now we can plug all components together to build the model, compile it, and train it. We will also implement custom definitions of the loss function as well as the accuracy, which account for the zero paddings on the sequences. I've tried the regular categorical crossentropy before, but it perormed much worse than the custom version here.

\`\`\`python
# combine components to a model
model = Model(
    inputs=[encoder_inputs_placeholder, decoder_inputs_placeholder],
    outputs=decoder_outputs
)

# a custom loss that accounts for the zero-padding
def custom_loss(y_true, y_pred):
  mask = K.cast(y_true > 0, dtype='float32')
  out = mask * y_true * K.log(y_pred)
  return -K.sum(out) / K.sum(mask)

# a custom accuracy that accounts for the zero-padding
def acc(y_true, y_pred):
  targ = K.argmax(y_true, axis=-1)
  pred = K.argmax(y_pred, axis=-1)
  correct = K.cast(K.equal(targ, pred), dtype='float32')

  # 0 is padding, don't include those
  mask = K.cast(K.greater(targ, 0), dtype='float32')
  n_correct = K.sum(mask * correct)
  n_total = K.sum(mask)
  return n_correct / n_total

# compile the model
model.compile(optimizer=Adam(learning_rate=0.005), loss=custom_loss, metrics=[acc])
model.summary()
\`\`\`

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_2 (InputLayer)            [(None, 80)]         0
    __________________________________________________________________________________________________
    input_1 (InputLayer)            [(None, 80)]         0
    __________________________________________________________________________________________________
    embedding (Embedding)           (None, 80, 100)      2000000     input_1[0][0]
                                                                     input_2[0][0]
    __________________________________________________________________________________________________
    lstm (LSTM)                     [(None, 128), (None, 117248      embedding[0][0]
    __________________________________________________________________________________________________
    lstm_1 (LSTM)                   [(None, 80, 128), (N 117248      embedding[1][0]
                                                                     lstm[0][1]
                                                                     lstm[0][2]
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 80, 20000)    2580000     lstm_1[0][0]
    ==================================================================================================
    Total params: 4,814,496
    Trainable params: 4,814,496
    Non-trainable params: 0
    __________________________________________________________________________________________________

Since the targets are still in that huge sparse tensor, which apparently cannot directly be used for training, we'll supply the training batches via a data generator that densifies the targets in each batch. During training, Keras will call this generator function to get the training data and targets, batch by batch.

\`\`\`python
# generator function to supply the training batches
def generate_data():
  counter=0
  while True:
    # select the batches
    encoder_batch = encoder_inputs[counter:counter+BATCH_SIZE,:]
    decoder_batch = decoder_inputs[counter:counter+BATCH_SIZE,:]
    target_batch = (decoder_targets_one_hot[counter:counter+BATCH_SIZE,:,:]
                    .todense()
                    .astype(np.float32))
    counter += BATCH_SIZE
    yield [encoder_batch, decoder_batch], target_batch

    # restart counter to yeild data in the next epoch as well
    if counter >= 1000:#len(decoder_targets):
        counter = 0
\`\`\`

\`\`\`python
model_checkpoint_callback = ModelCheckpoint(
    filepath="./star_trek_chatbot_checkpoint.h5",
    save_weights_only=False,
    monitor="loss",
    mode="min",
    save_best_only=True)

reduceLR = ReduceLROnPlateau(
    monitor="loss", factor=0.5, patience=2, verbose=0, mode="auto",
    min_delta=0.0001, cooldown=0, min_lr=0
)

# train the model
history = model.fit(
  generate_data(),
  steps_per_epoch=len(decoder_targets) // BATCH_SIZE,
  epochs=EPOCHS,
  callbacks=[model_checkpoint_callback, reduceLR],
  verbose=0
)
\`\`\`

\`\`\`python
# Save model
model.save("./star_trek_chatbot.h5")
\`\`\`

<a id="decoding"></a>

### Modifying the Model to Generate Text

Now that the model is trained, we want to make predictions. We still have to set up the generative functionality though. We need to take the components we already used and put them together to a new model in which the decoder will always predict one word at a time, taking the hidden and cell states as well as the previously predicted word as an input for the next prediction. This will be repeated until the end-of-sentence <END\\> token appears.

The encoder will look just like before, but it will stand alone now and just be used for getting the initial states for the decoder.

\`\`\`python
# the encoder consists of the same components as before
encoder_model = Model(
    inputs=encoder_inputs_placeholder,
    outputs=encoder_states
)
\`\`\`

Here we have to set up the decoder components which will later be used in the prediction loop. We are using the previously trained decoder_lstm layer here and feed it the states and the outputs of the previous prediction in the sequence.

\`\`\`python
# the hidden and cell states
decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# the regular input of length 1 (the previously predicted word)
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = embedding_layer(decoder_inputs_single)

# getting the outputs (a predicted word) and the states,
# which will both be used as input for predicting the next word
decoder_outputs, h, c = decoder_lstm(
  decoder_inputs_single_x,
  initial_state=decoder_states_inputs
)

decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)

# the decoder sampling model
# inputs: y(t-1), h(t-1), c(t-1)
# outputs: y(t), h(t), c(t)
decoder_model = Model(
    inputs=[decoder_inputs_single] + decoder_states_inputs,
    outputs=[decoder_outputs] + decoder_states
)
\`\`\`

The model will, of course, output numbers, not strings. Therefore, we will have to reverse the word index of the tokenizer to be able to translate from indicies back to actual words.

\`\`\`python
# map indicies back to actual words
index_to_word = {value:key for key, value in word_index.items()}
\`\`\`

The last step is to set up the text sequence prediction loop. We'll pass the input text through the encoder to get the initial states for the decoder and always start by feeding a start-of-sentence token <START\\> as input to the decoder. The loop will then generate predictions, one word at a time, until the end-of-sequence token <END\\> is predicted.

\`\`\`python
def decode_sequence(input_seq):
    # encode the input to get the states
    states_value = encoder_model.predict(input_seq)

    # start with empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # and fill with <start>
    target_seq[0, 0] = word_index["<start>"]

    # if end is predicted, we break the loop
    end = word_index["<end>"]

    # the predicted output text
    output_sentence = []

    for _ in range(MAX_SEQ_LEN):
        output_tokens, h, c = decoder_model.predict(
          [target_seq] + states_value
        )

        # get next word index
        idx = np.argmax(output_tokens[0, 0, :])

        # if index corresponds to end, break
        if end == idx:
            break

        word = ""
        if idx > 0:
          word = index_to_word[idx]
          output_sentence.append(word)

        # update the decoder input with the word just predicted
        target_seq[0, 0] = idx

        # update states
        states_value = [h, c]

    return ' '.join(output_sentence)
\`\`\`

<a id="generate"></a>

### Generating Dialogues

Now that everything is set up, let's try to produce some artificial neural Star Trek dialogues! I'll pick some examples that could be from Deep Space 9.

\`\`\`python
texts = ["<start> hello captain, how do you like the station ?",
         "<start> have you seen jake ? ",
         "<start> the cardassians are heading towards the wormhole !",
         "<start> after fifty years of occupation ,  bajor is finally indipendent ."]

for text in texts:
    input_seq = pad_sequences(tokenizer.texts_to_sequences(text),maxlen=MAX_SEQ_LEN)
    print(text[8:])
    print(decode_sequence(input_seq) + "\\n")
\`\`\`

    hello captain, how do you like the station ?
    the ship .

    have you seen jake ?
    jennifer ?

    the cardassians are heading towards the wormhole !
    cancelled ?

    after fifty years of occupation ,  bajor is finally indipendent .
    dabo !

While the results are generally rather disappointing (I've tried a bunch of other examples as well), I do love the last onw here; the Cardassians are gone, let's play Dabo! :D

<a id="conclusion"></a>

### Conclusion

In conclusion, I've tried to use a very simple seq2seq model based on an LSTM encoder-decoder architecture. It didn't work really well in terms of producing intersting new Star Trek dialougues that even Gene Roddenberry couldn't have dreamt of. Instead it spits out very short replies that do not neccessarily have much to do with the input. I think this may be due to the data set being heavily dominated by few-words-long dialogue lines. Furthermore, free dialougue generation, as opposed to tasks with a clear request-response targets (such as translation or question ansering) requires much more of a language understanding. For this, my model is way too simple. One the one hand, because I intentionally kept the number of components and parameters relatively small to keep it computationally tractable, on the other hand because the architecture itself may be a little to simple for free language generation. 

With only the final cell state being passed from the encoder to the decoder, it means that the entire context from the encoding of the input has to be compressed into a single vector. Much of what was relevant along the sequence may have gotten lost at the final state, hence, not being available as useful input for the decoder. Using a seq2seq architecture that applies a [attention mechanism](https://arxiv.org/pdf/1902.02181.pdf) may perform better. Especially, very large [transformer](https://arxiv.org/abs/1706.03762)-based language models (which also use attention), such as Google's [BERT](https://arxiv.org/abs/1810.04805) or OpenAI's [GPT3](https://arxiv.org/abs/2005.14165) are excelling at language understanding and text generation. Maybe I'll try to pick this project up again, starting from such a giant, pretrained transformer model. Nonetheless, I found this project to be very educative and a great introduction to seq2seq RNN-based models. I've you made it until here, thanks a lot for reading, and may you boldly go where no human has gone before :)
`
);
