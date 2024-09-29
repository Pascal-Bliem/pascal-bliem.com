import Post from "../postModel";

export default new Post(
  // title
  "Classifying Dog Breeds with Deep Learning",
  // subtitle
  "Specialized CNN architectures, transfer learning, and lots of data",
  // publishDate
  new Date("2021-02-24"),
  // titleImageUrl
  "https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/dog-classifier/dogbreeds.png",
  // titleImageDescription
  "Which breed is this?",
  // tags
  ["Data Science & AI/ML"],
  // content
  `**TL;DR:** I scraped a large data set containing about 121k images of 121 different dog breeds to train a CNN deep learning classification model with PyTorch. Starting from a model pretrained on ImageNet, the model performs at about 89% accuracy on validation data, with relatively simple additional training.

I haven't build any cool deep learning app in quite a while, and I felt it's about time. I already gathered some experience in the field of natural language processing with my [ToxBlock](http://www.pascal-bliem.com/blog/tox%20block%20using%20ai%20to%20keep%20discussions%20clean) app, so now I'd like to do something in the field of computer vision. Quite some time ago I though about building an app for [Chinese character OCR](http://www.pascal-bliem.com/blog/a%20data%20set%20of%20handwritten%20chinese%20characters), but had to give up the idea when I found out that training a model that could recognize thousands of different characters was a bit to much for the resources available to me. I still like the idea of building an app that people can have on their mobile phones and use for classifying stuff they come across. If not Chinese characters, what else would be a good target for image classification? I asked a few people what they would like to take pictures of when walking around outside, and "definitely DOGS!" was among the replies I got. And I thought that was a great idea; I see tons of cute dogs when taking a walk in the park, but I only know like 5 dog breeds by name. Having an app on my phone that tells me which breed it is would be amazing! In this post I'll talk about how I obtained a data set and how the actual neural network that does the classification is build and trained. Anything related to mobile development won't be in this post, but I'll probably cover it in a later post.

### Getting the data

Okay, so first thing in every machine learning project is data. I'm probably not the first person thinking about classifying dog breeds, so I'd assume there are some data sets available online. And, in fact, there is a pretty famous one called [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) (from Stanford University). It contains about 150 pictures for each of 120 different dog breeds. This data set even got its own [Kaggle competition](https://www.kaggle.com/c/dog-breed-identification) and is a part of [ImageNet](http://www.image-net.org/), a large image classification data set. A similar data set is the [Tsinghua Dogs](https://cg.cs.tsinghua.edu.cn/ThuDogs/) (by Tsinghua University), which contains images of the same 120 breeds plus 10 new classes. The number of images is in proportion to their frequency of occurrence in China, some classes are still quite underrepresented.

Using the Stanford Dogs data set as a starting point is great for several reasons: As I mentioned earlier, I don't know a lot of dog breeds by name, and here, someone has preselected 120 breeds that I can use as well. Furthermore, the fact that these 120 breeds are also a part of the ImageNet data set makes it ideal for transfer learning. Many models that have been trained on ImageNet are available online and can be used as a starting point for further, more specialized training. However, only using the two aforementioned data sets won't be sufficient. They still don't contain enough training examples to distinguish all breeds reliably and they're also not licensed for commercial use. But hey, people love their dogs and they love to take photos of them and put them on the internet under a public licence. And a really convenient way to get these images is over an image search API, such as the one from [Bing](https://www.microsoft.com/en-us/bing/apis/bing-image-search-api). I will use the 120 dog breed names plus [Shiba Inu](https://en.wikipedia.org/wiki/Shiba_Inu) (I really like this breed but it wasn't in the Stanford data set) as search terms for the image search API calls and try to get about 1000 images per breed.

In case you'd like to do something like that and use the Bing image search API, here's a code snippet that you could use for downloading all the image data. Keep in mind that you'd have to sign up for [Microsoft Azure](https://azure.microsoft.com/) to get an API key to use with the API. As of now, I got some free credit when signing up which I used for calling the API, so I didn't actually spend any of my own money. Here, \`search_terms\` is a Python list of search terms (dog breed names) to look up images for, \`directory_names\` is a list of directories into which I'll save the Images, and \`subscription_key\` is the API key.

\`\`\`python
import requests
from PIL import Image
from IPython.display import clear_output

search_url = "https://api.bing.microsoft.com/v7.0/images/search"
headers = {"Ocp-Apim-Subscription-Key" : subscription_key}

# loop through all dog breeds
for count, search_term, directory_name in zip(
    range(len(search_terms)),
    search_terms,
    directory_names
):
    # print progress
    print(f"Entering {directory_name} {count+1}/{len(search_terms)}")

    # create directory for dog breed if it doesn't exist yet
    directory_path = os.path.join("data", "bing_image_api", "images", directory_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # get 50 images at a time
    for offset in range(0, 1001, 50):

        # log progress
        clear_output(wait=True)
        print(f"Entering Offset {offset}")

        # Set up params for the request to the Bing Image Search Api;
        # NOTE that you can set the license type to public and the
        # image type to photo - this is great because it will mostly
        # prevent you from getting drawings or clip art of dogs!
        params  = {
            'q': search_term,
            'offset': offset,
            'count': 50,
            'license': "public",
            'imageType': "photo"
        }

        # make the request to the API
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()

        # for each search result that came back from the API call
        for i, result in enumerate(search_results['value']):

            # get each image by URL and write to file
            image_url = result['contentUrl']
            image_path = os.path.join(directory_path, f"{offset+i}.jpg")
            # open a file handle here to write the image to
            with open(image_path, "wb") as handle:

                try:
                    # log progress
                    clear_output(wait=True)
                    print(f"Getting {directory_name} {count+1}/{len(search_terms)} image {offset+i}")

                    # send request to get the image
                    response = requests.get(image_url, timeout=30)

                    # skip and remove the image file if something's wrong
                    if not response.ok:
                        os.remove(image_path)
                        continue


                    clear_output(wait=True)
                    print(f"Writing {directory_name} {count+1}/{len(search_terms)} image {offset+i}")

                    # write to file in blocks until response is empty
                    for block in response.iter_content(1024):
                        if not block:
                            break
                        handle.write(block)

                # if anything goes wrong, delete the image file and move on
                except:
                    os.remove(image_path)
                    continue

            # validate if the image is uncorrupted; if not, delete it
            try:
                im = Image.open(image_path)
                im.verify()
            except Exception as e:
                os.remove(image_path)
            # also make sure the image is not an animated GIF
            try:
                im.seek(1)
            except EOFError:
                is_animated = False
            else:
                is_animated = True
            if is_animated:
                os.remove(image_path)
\`\`\`

Now you should have thousands of dog images available to train on. They may, of course come in all sorts of sizes, mostly larger than you would need them for your neural network anyway, so you may want to resize them already at this point. I found this useful for me because I wanted to train the model on a Cloud service that offers GPU infrastructure, namely [Google Colab](https://colab.research.google.com/), and it goes much faster to upload and unpack the images when they're already down-sized. You could resize all the images like this:

\`\`\`python
# the size the image will be resized to
NEW_SIZE = 256

# loop through all image directories
for count, directory_name in enumerate(directory_names):
    source_path = os.path.join("data", "bing_image_api", "images", directory_name)
    target_path = os.path.join("data", "bing_image_api", "small_images", directory_name)

    # create target directory if it doesn't exist yet
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # loop through each file in the directory
    file_names = os.listdir(source_path)
    for i, file_name in enumerate(file_names):

        # log progress
        clear_output(wait=True)
        print(f"Moving {directory_name} {count+1}/{len(directory_names)} image {i+1}/{len(file_names)}")

        # load the image
        im = Image.open(os.path.join(source_path, file_name))

        # resize the image
        im = im.resize([NEW_SIZE, NEW_SIZE])

        # save the resized image to target directory
        im.save(os.path.join(target_path, file_name))
\`\`\`

Now we should have the data in a neat format for zipping it up and loading it to where ever we have some GPUs available for training a model.

### Building the model

When it comes to computer vision tasks, one usually doesn't have to start dreaming up models from scratch. Pretty much all networks that work well with image data are convolutional neural networks (CNNs). Among those, there are many known model architectures that have achieved great results on classification task such as ImageNet. For a while, the way to go to improve accuracy was to make the models deeper and deeper, but that also means they have more parameters. On the one hand, this means it takes longer to train the network, on the other hand, it will also make inference slower, which can be critical on edge devices that have relatively little computing power. Considering that I'd like to run the model inference within a mobile app, it would be desireable to keep the number of parameters as small as possible.

Luckily, some very smart people came up with some very smart ideas to reduce the amount of parameters while keeping a high model performance, such as in the [ShuffleNet](https://arxiv.org/abs/1707.01083), [SqueezeNet](https://arxiv.org/abs/1602.07360), or [MobileNetV2](https://arxiv.org/abs/1801.04381). All these models are also freely available online (in deep learning frameworks such as [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/)), pretrained on ImageNet. I decided to go for MobileNetV2 as a starting point for my model. This architecture applies some cool tricks (which you can read about in the [paper](https://arxiv.org/abs/1801.04381)) such as inverted residuals and linear bottle necks, but what really shrinks down the number of parameters is using depth-wise separable convolutions. Basically, these convolutions treat every channel of the image separately and then combine the results with point-wise convolutions. It's not trivial to explain how exactly that works, so I won't do it here, but [this blog post](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728) does a great job at it.

For this Project I will use the PyTorch deep learning framework. I will now walk you through the process of building the model:

\`\`\`python
# import the libraries we'll need
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output
import time
import os
import math
from copy import copy, deepcopy
\`\`\`

Now let's define some config variables that we'll need throughout the rest of the code.

\`\`\`python
# Use a GPU if it's available - if not, use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path to the directory the image data resides in
data_dir = "/content/images/small_images"

# path to where the model checkpoints are saved
checkpoint_dir = "/content/gdrive/MyDrive/dog_classifier_checkpoint.p"

# fraction of the data that will be used for training (rest is for validation)
train_frac = 0.8

# batch size for training the model
batch_size = 64

# for evaluating accuracy on only high-probability predictions
cutoff_prob = 0.7
\`\`\`

Before working on the actual model, we should prepare the data set for both training and validation. Since we're working on image data, it is easy to use some data augmentation by applying some random transformations to the images. In this case, we'll randomly crop out a 224x224 patch out of the (previously 256x256) images, maybe flip it horizontally and apply some small rotation to it. This way, the model will never see the exact same image twice, which should help to make generalize better. Note that these transformations are applied to the training data only; for the validation data, we just crop out a center patch in the right size. For both parts of the data set, we'll also normalize the data on a per-channel basis. We'll subtract the mean value of each channel from each pixel and divide by the standard deviation (which I've calculated for this data set beforehand). Centering the data around zero usually helps with convergence and hence, speeds up training.

\`\`\`python
# here we got the transforms and normalization I talked about
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.512, 0.489, 0.422], [0.267, 0.263, 0.271])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.512, 0.489, 0.422], [0.267, 0.263, 0.271])
    ]),
}

# load the whole data set from the folders that contain
# the images of each dog breed
whole_dataset = datasets.ImageFolder(data_dir)
class_names = whole_dataset.classes

# calculate train and validation set sizes
train_size = math.floor(len(whole_dataset) * train_frac)
val_size = len(whole_dataset) - train_size

dataset_sizes = {'train': train_size, 'val': val_size}

# do a train-validation-split
train_set, val_set = torch.utils.data.random_split(whole_dataset, (train_size, val_size))
train_set.dataset = copy(whole_dataset)

# apply the data transforms to each data set
train_set.dataset.transform = data_transforms['train']
val_set.dataset.transform = data_transforms['val']

# create data loaders from the data sets
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader}
\`\`\`

To ensure that what we coded above makes sense, let's have a look at a batch of images that'll be used for training now:

\`\`\`python
def imshow(inp):
    # transpose tensor to match matplotlib's image format
    inp = inp.numpy().transpose((1, 2, 0))

    # reverse the data normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    # plot the image
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(inp, interpolation="nearest")
    plt.pause(0.001)


# get a batch of training data
inputs, classes = next(iter(train_loader))

# make an image grid from batch
out = torchvision.utils.make_grid(inputs)

# plot
imshow(out)
\`\`\`

![A training batch of dog images.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/dog-classifier/doggrid.png)

This looks like everything works well with the data preparation. Now that we have the data sets prepared, we can set up the model. This step is actually really simple as we'll just slightly modify the MobileNetV2. Let's first get the pretrained model, which is readily available in PyTorch.

\`\`\`python
model = models.mobilenet_v2(pretrained=True)
model
\`\`\`

\`\`\`
MobileNetV2(
  (features): Sequential(
    (0): ConvBNReLU(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  # [...] Here would be 17 more repeating layers of InvertedResiduals,
  # but I left them out for a better readability of this post
  )

  (classifier): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=1280, out_features=1000, bias=True)
  )
)
\`\`\`

Now if we have a look at all the layers in the model, we can see that the last part is a classifier consisting of a Linear/Dense layer with 1000 output features. This is because there are 1000 classes in the ImageNet data set on which this model has been pretrained. We do, however, only have 121 dog breeds to classify, so we have to adjust the classifier of this model to suit our problem. In addition, we'll add a LogSoftmax layer to transform the output of the model to log probabilities. Using log probabilities rather than regular probabilities between 0 and 1 does generally lead to easier training of the model as log probabilities penalize larger errors more, lead to less arithmetic operations, and have better numerical stability (because they avoid very small probabilities close to 0). We also move the model to the \`device\`, meaning the GPU, if one is available.

\`\`\`python
# replace the classifier-part of the model that has been pretrained on ImageNet
# by a new, untrained classifier which we will then train to classify dog breeds
num_clf_in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
  nn.Dropout(p=0.2, inplace=False),
  nn.Linear(in_features=num_clf_in_features, out_features=len(class_names), bias=True),
  nn.LogSoftmax(dim=1)
)

# move model to device
model = model.to(device)
\`\`\`

### Training the model

Now that we have the model ready, what do we need to train it? We need to define a loss function (in PyTorch usually called \`criterion\`), which is a differentiable function that states how far off our predictions are. The choice of loss function depends on what the model outputs: if the last layer of the model was a Softmax function that outputs probabilities (between 0 and 1), we'd probably be using a cross-entropy loss. But since we are using a LogSoftmax that outputs log probabilities, we'll use a negative log likelihood loss (\`NLLLoss\`) instead. An optimizer will, starting from this loss function, compute the gradients of all parameters and perform the gradient decent to reduce the loss in the next iteration. How far down the gradient the optimizer goes at each step is (initially) determined by the learning rate. If the learning rate is too small, the model will take very long to train - if it's too high, the optimizer may miss a loss minimum. Hence, it makes sense to start with a higher learning rate and reduce it when we realize that the loss is no longer decreasing. In PyTorch, the learning rate can be manipulated with a variety of learning rate schedulers; the one we use here, \`ReduceLROnPlateau\`, will reduce the learning rate if the validation loss doesn't decrease anymore. In terms of optimizers, there, as well, is a large palette of algorithms to choose from (you can look them up [here](https://pytorch.org/docs/stable/optim.html)). What worked well for me is starting with [Adam](https://arxiv.org/abs/1412.6980), an optimizer that adapts the learning rate throughout training. However, when training with Adam started showing diminishing returns, I switched to using [stochastic gradient decent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (\`SGD\`) with momentum and was able to improve the model performance a little further.

\`\`\`python
# the loss function
criterion = nn.NLLLoss()

# the optimizer (I started with Adam and went on with SGD)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# the learning rate scheduler
rrp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.001, verbose=True)
\`\`\`

Everything is set up now to enter the training loop. Within this (fairly large) loop, both model training and validation is going to happen. Note that this is all squeezed into one block of code because this is written for a blog post; if this was a production environment, I'd highly recommend splitting it up into several functions, maybe even separate files. Anyways, let's define the training loop function:

\`\`\`python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, only_classifier=False):
    since = time.time()

    # storing current best model weights and corresponding accuracy
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    # track validation score over the epochs
    val_losses = []
    val_accs = []
    prev_epochs_printout = ""

    for epoch in range(num_epochs):
        # log progress
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # set model to training mode
            else:
                model.eval()   # set model to evaluate mode

            # keep track of metric during epoch
            running_loss = 0.0
            running_corrects = 0

            # iterate over the data
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history only if in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # if only the classifier should be trained, disable
                    # all gradients and only enable them for classifier
                    if only_classifier:
                        for param in model.parameters():
                            param.requires_grad = False
                        for param in model.classifier.parameters():
                            param.requires_grad = True
                    else:
                        for param in model.parameters():
                            param.requires_grad = True

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # update metrics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # print out status
                epoch_loss = running_loss / ((i+1)*batch_size)
                epoch_acc = running_corrects.double() / ((i+1)*batch_size)

                clear_output(wait=True)
                print(f"{prev_epochs_printout}\nEpoch {epoch}/{num_epochs - 1} Progress {(i+1)*batch_size}/{dataset_sizes[phase]} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # calculate metric for the whole current epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

            # if in validation phase, step the learning rate scheduler
            if (phase == "val") and (scheduler != None):
                scheduler.step(epoch_loss)

            # update validation metrics
            if phase == "val":
                val_losses.append(round(epoch_loss, 4))
                val_accs.append(round(epoch_acc, 4))
                prev_epochs_printout += f"Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n"

            # print out status
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # if the model performed better than at any previous stage,
            # set new best and save the model weights
            if (phase == "val") and (epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())
                torch.save(best_model_wts, checkpoint_dir)

        print()

    # print out final status
    time_elapsed = time.time() - since
    print(f"Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # restore best model weights
    model.load_state_dict(best_model_wts)

    return model
\`\`\`

Now we can train the model in two steps. As mentioned above, the current model is already pretrained on the ImageNet data set, meaning all its convolutional layers have probably already learned features that may be relevant to our image data as well. However, we replaced the last section of the model, the classifier. I will attempt to train only the classifier part and leaving the rest of the model as it is. By doing this, I hope to leverage the image-related features that the pretrained model has already learning and not "sending a shock" through the whole model, which may mess up the parameters of the convolutional layers. Once the model performance doesn't improve anymore while only training the classifier, I will make all parameters trainable and resume training the entire model.

\`\`\`python
# train classifier only
model = train_model(
    model, criterion, optimizer,
    scheduler=rrp_lr_scheduler,
    num_epochs=5,
    only_classifier=True
)

# train the whole model
model = train_model(
    model, criterion, optimizer,
    scheduler=rrp_lr_scheduler,
    num_epochs=25,
    only_classifier=False
)

\`\`\`

The (truncated) output looks something like this:

\`\`\`
Epoch 0/9 Loss: 0.8426 Acc: 0.7649
Epoch 1/9 Loss: 0.7726 Acc: 0.7850
Epoch 2/9 Loss: 0.7064 Acc: 0.8017
Epoch 3/9 Loss: 0.7089 Acc: 0.8013
Epoch 4/9 Loss: 0.6813 Acc: 0.8113
Epoch 5/9 Loss: 0.6557 Acc: 0.8207
Epoch 6/9 Loss: 0.6662 Acc: 0.8211
Epoch 7/9 Loss: 0.6543 Acc: 0.8234
Epoch 8/9 Loss: 0.6406 Acc: 0.8275
# [...] Cut out some lines here for readability
Epoch 24/24 Progress 31552/31521 Loss: 0.4532 Acc: 0.8901
val Loss: 0.4532 Acc: 0.8901

Training complete in 97m 55s
Best val Acc: 0.8901
\`\`\`

After many epochs and playing around with optimizers, learning rates, and data augmentation, the dog breed classifier seems to perform decently. We'll go into more detailed evaluation in the next section.

### Evaluating the model

What I want to do here is to run the validation set (on which the model has not been trained) through prediction again and score not only its absolute accuracy, but also consider its performance when only high-probability-predictions are taken into account. The motivation for this is as follows: if someone takes a really crappy picture of a dog, one can't really expect a model to classify it correctly. I didn't have the time to go through all the dog images I scraped manually to see if they look like proper pictures of that dog breed. I noticed that e.g. among the pug images, there were a lot of mixed breeds that presumably contained pug DNA, but didn't really look like a pug. Also pictures taken from weird perspectives or with many different dogs in them cannot be expected to be classified correctly. If this model is served in a mobile app, that wouldn't really be a problem; if the certainty of a prediction is below a certain percentage, say 70%, one could prompt the user to take a better picture of the dog. Therefore, I'll also check how the model performs if only predictions with a certainty over 70% (\`cutoff_prob\`) are considered.

\`\`\`python
# deactivate gradient tracking in evaluation mode
with torch.no_grad():
    model.eval()

    # keep track of metrics
    running_corrects = 0
    running_high_prob_corrects = 0
    total_high_prob = 0

    all_preds = []
    all_labels = []

    # iterate over validation data
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # run inputs through model
        outputs = model(inputs)

        # make predictions, regular and for high certainty
        max_vals, preds = torch.max(outputs, 1)
        high_prob_mask = (max_vals >= cutoff_prob)
        high_prob_preds = preds[high_prob_mask]
        high_prob_labels = labels[high_prob_mask]

        all_preds += preds.tolist()
        all_labels += labels.tolist()

        # update metrics
        running_corrects += torch.sum(preds == labels.data)
        running_high_prob_corrects += torch.sum(high_prob_preds == high_prob_labels.data)
        total_high_prob += len(high_prob_preds)

# print out result
print(f"Acc: {running_corrects/dataset_sizes['val']:.4f}\nHigh Prob Acc: {running_high_prob_corrects/total_high_prob:.4f}")
\`\`\`

\`\`\`
Acc: 0.8901
High Prob Acc: 0.9535
\`\`\`

We can see that if we only consider predictions with a high certainty, the model performs really well at 95% accuracy. We have to keep in mind though that this doesn't mean that the model recognizes all dog breeds with such a high accuracy. There are significant differences between the metrics for individual dog breeds, as can bee seen by looking at the classification report:

\`\`\`python
from sklearn.metrics import classification_report
cp = classification_report(
    all_labels,
    all_preds,
    labels=list(range(len(class_names))),
    target_names=class_names
)
print(cp)
\`\`\`

\`\`\`
                          precision    recall  f1-score   support

                 affenpinscher       0.95      0.97      0.96       204
                  afghan_hound       0.96      0.95      0.95       234
           african_hunting_dog       0.98      0.96      0.97       225
                      airedale       0.95      0.97      0.96       201
american_staffordshire_terrier       0.78      0.67      0.72       211

                    # [...] I cut out most lines here for readability

                       whippet       0.85      0.87      0.86       234
       wire_haired_fox_terrier       0.92      0.94      0.93       218
             yorkshire_terrier       0.90      0.79      0.84       240

                      accuracy                           0.89     31521
                     macro avg       0.89      0.89      0.89     31521
                  weighted avg       0.89      0.89      0.89     31521
\`\`\`

As we can see, the performance differs quite a lot among the different breeds. Some show an F1 score as high as 98%, others as low as 70%.
We can have a look at which dog breeds are often mixed up by looking at the confusion matrix.

\`\`\`python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels, all_preds)
# set all the correctly classified entries to zero
# because we're not interested in them
np.fill_diagonal(cm, 0)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
\`\`\`

I'll save you looking at the entire confusion matrix in detail here, but as an example we can pick out the breeds that have been mixed up particularly often. It's not very surprising that it's breeds that look very similar and are even closely related to each other.
Within the validation data, the breed that was most often misclassified (80 times) as one particular other breed, was the Pembroke Welsh Corgi, which has been mistaken for the Cardigan Welsh Corgi. As you can see in the picture below, these two really do look pretty similar:

![The Cardigan Welsh Corgi and Pembroke Welsh Corgi in comparison.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/dog-classifier/corgicomparison.jpg)

Overall, I got the impression that the model is performing pretty decently at classifying breeds and only fails often on images that do either not capture the dog properly or for dog breeds that are very similar to each other. Even I as a human would have some trouble telling a bunch of Corgis apart.

### Conclusion

In conclusion, we've seen how to build a decently performing dog breed classifier by getting a lot of image data via an image search and using it to fine tune a pretrained convolutional neural network. Via a free Azure account and the Bing image search API, it was fairly easy to get more than a hundred thousand photos of he 121 different dog breeds. Since most of the dog breeds were already a part of the ImageNet data set, it was ideal to start from a model that has been pretrained on ImageNet. The chosen model architecture was MobileNetV2, which is designed for performing well while using relatively few parameters so that it is feasible to run inference on mobile devices. The final model reaches an overall validation accuracy of about 89% (up to 95% when only predictions with more than 70% certainty are considered), but the performance varies quite a bit among the different breeds. I'll probably try to incorporate this model in a mobile app soon, so look out for an upcoming post on that. Thanks a lot for reading!
`
);
