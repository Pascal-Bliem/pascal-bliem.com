import Post from "../postModel";

export default new Post(
  // title
  "Object Detection: YOLO",
  // subtitle
  'Understand and implement the "You only look once" model',
  // publishDate
  new Date("2021-07-11"),
  // titleImageUrl
  "https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/object-detection/yolocover.png",
  // titleImageDescription
  "You only look once, so you better make the most of it!",
  // tags
  ["Data Science", "Learning"],
  // content
  `As I already mentioned in a [previous post](https://pascal-bliem.com/blog/object%20detection%20metrics), after creating the [Doggo Snap](https://pascal-bliem.com/doggo-snap) app, I've become a lot more interested in computer vision tasks that can be solved by deep learning. One obvious example is object detection, where we want to detect certain objects in an image and figure out where exactly they are in the image. Imagine you're a robot with a camera as eyes and you need to figure out what to pick up and where it stands. That's where deep learning with convolutional neural network comes in very handy. In the [previous post](https://pascal-bliem.com/blog/object%20detection%20metrics), I've cover the concept of bounding boxes in object detection in detail, now we'll have a look at how we can build a model that predicts these bounding boxes around objects.

Object detection used to be performed by a sliding window approach, in which a predefined box is slid over the image with a certain stride and every crop defined by the current position of the box is individually classified. This approach is, however, very computationally expensive because we have to "look" many time, for each new crop. [Region-proposing neural networks](https://arxiv.org/abs/1506.01497) were a bit faster but still slow. In the paper ["You Only Look Once: Unified, Real-Time Object Detection"](https://arxiv.org/abs/1506.02640), or short YOLO, the authors came up with a much more efficient way to predict bounding boxes. Since the original paper was published in 2016, there have been several updates to the YOLO architecture (I think there are 5 versions of it as of now), but I want to stick to the original version here to understand the fundamentals. In the following, I want to go through the idea behind YOLO, the implementation of the architecture, its loss function, and how it would be trained. But first, I want to say thanks a lot to [Aladdin Persson](https://www.youtube.com/channel/UCkzW5JSFwvKRjXABI-UTAkQ), a YouTuber who publishes really insightful deep learning videos, from which I learned a lot about deep learning, especially for computer vision and this implementation of YOLO. I'll assumer we'll be using the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset here, which was also used in the original paper.

### So how does that thing work?

I think the authors describe what they did quite clearly in the [original paper](https://arxiv.org/abs/1506.02640), so go ahead and have a look at it if you're somewhat familiar with reading deep learning literature. Or just keep reading here, I'll try to summarize the basic idea behind the YOLO algorithm, and I'll try to use the same nomenclature as in the paper. So, we want to detect objects in images, which means we need to find the objects (predict bounding boxes) and figure out what kind of objects they are (perform a classification). What makes the YOLO algorithms particularly efficient is that, instead of performing several runs through a neural net for different parts of the image, we try to put out everything we need in one pass. This is done by splitting the image into a grid that has SxS cells (they used S=7, so 7x7 in the paper), and making each of these cells responsible for detecting only one object and outputting the corresponding bounding box. That means that if we would need to detect more smaller objects, we would need a finer grid (increase S). You can see this idea visualized in the figure below.

![The concept behind the YOLO model (figure from the paper by Redmon et al.)](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/object-detection/yoloconcept.png)

Of course, an object might be in several of these grid cells, so we need to find the one cell that contains the center point of the object, which will be responsible for outputting the bounding box for that object (the bounding box itself can reach beyond the cells boundaries). Imagine, each grid cell gets its own coordinate system and the predicted coordinates and dimensions are relative to it. So, if we output something like [x, y, width, height], the coordinates (x, y) of the objects center point will be within the cell in relative coordinates (which means between 0 and 1), but the relative width and height may be larger than the cell (which means they could be larger than 1). Even though only one object can be detected in each cell, several bounding boxes could be predicted for that object. In the paper the number of predicted bounding boxes is called B and set to equal 2. The idea behind that is that the different boxes could specialize on different characteristics, e.g. wide vs. tall objects. The output of one of the grid cells would then look something like [class_1, ..., class_c, confidence_1, x_1, y_1, width_1, height_1, confidence_2, x_2, y_2, width_2, height_2], if we have C classes and the class label is one-hot encoded. One of the class labels will be one, all others zero, depending on which class the object belongs to. A certainty as well as the center point coordinates, width, and height are outputted for each of the two predicted bounding boxes. Then the shape of the whole model output would be (S, S, C + 5 * B). Okay, that's the output, but how does the rest of the model look like?

### The architecture

The model is a fairly standard and large convolutional neural network (CNN), inspired by [GoogleLeNet](https://arxiv.org/abs/1409.4842) as the authors say, with 24 convolutional layers followed by 2 fully connected layers. The architecture is shown in the figure below.

![The YOLO model architecture (figure from the paper by Redmon et al.)](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/object-detection/yoloarchitecture.png)

The network's convolutional part consists of several blocks of convolutional and max-pooling layers. The dimensions of the filter kernels, input and output channels, and strides can be read from the figure. The figure doesn't show padding thought. To satisfy the relationship between input and output dimensions of the convolution blocks, some padding is necessary. We can calculate the dimensions according to the formula \`output_size = [( input_size – kernel_width + 2 * padding) / stride] + 1\`. There needs to be a padding of 3 on the first input layer and same-padding for the rest of the convolutional blocks. Note that the final output now coincides with the shape of the predictions we've shown above; width S=7, B=2, and C=20, that's (7, 7, 30).

Let's see how to implement this in code, using [PyTorch](https://www.pytorch.org), my favorite Python deep learning framework. We'll slightly deviate from the original implementation and use batch norm here, which was not used in the original paper.Batch normalization can usually speed up training massively by preventing internal covariate shift and making the optimization function space simpler. We will first define the architecture. Below is a list containing further lists of which each represents a layer in the convolutional stack. The entries in these list stand for [kernel_size, out_channels, stride, padding] of each layer. Max-pooling layer, which will always have a kernel size of 2x2 and a stride of 2, are represented by a "M". Not in this list, but implemented later, are also the two fully connected layers.

\`\`\`python
yolo_architecture = [
    [7, 64, 2, 3],
    "M",
    [3, 192, 1, 1],
    "M",
    [1, 128, 1, 0],
    [3, 256, 1, 1],
    [1, 256, 1, 0],
    [3, 512, 1, 1],
    "M",
    [1, 256, 1, 0],
    [3, 512, 1, 1],
    [1, 256, 1, 0],
    [3, 512, 1, 1],
    [1, 256, 1, 0],
    [3, 512, 1, 1],
    [1, 256, 1, 0],
    [3, 512, 1, 1],
    [1, 512, 1, 0],
    [3, 1024, 1, 1],
    "M",
    [1, 512, 1, 0],
    [3, 1024, 1, 1],
    [1, 512, 1, 0],
    [3, 1024, 1, 1],
    [3, 1024, 1, 1],
    [3, 1024, 2, 1],
    [3, 1024, 1, 1],
    [3, 1024, 1, 1],
]
# fully connected layer are not considered here yet, they'll be implemented individually
\`\`\`

Now let's build a network from this architecture. I'll try to keep telling the story with in-line comments in the code.

\`\`\`python
from typing import List, Union, Tuple
import torch
import torch.nn as nn

# a class that represents one of the CNN blocks in the network
class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super(CNNBlock, self).__init__()

        # the convolution itself
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # followed by batch normalization
        self.batchnorm = nn.BatchNorm2d(out_channels)
        # and the activation function
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.leakyrelu(x)

# this is the actual model itself
class Yolo(nn.Module):
    # all parameters have defaults according to the paper
    # in_channels are the color channels of RGB images
    # grid_size, num_boxes, num_classes are S, B, C
    def __init__(
        self,
        architecture: List[Union[List[int], str]] = yolo_architecture,
        in_channels: int = 3,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20,
        **kwargs
    ) -> None:
        super(Yolo, self).__init__()

        self.architecture = architecture
        self.in_channels = in_channels
        # we'll have separate functions for creating the layers
        self.conv_layers = self._create_conv_layers(self.architecture)
        self.fc_layers = self._create_fully_connected_layers(
            grid_size, num_boxes, num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers
        # start_dim=1 because we don't want to flatten the batch size
        x = torch.flatten(x, start_dim=1)
        return self.fc_layers(x)

    # create the convolutional part of the network
    def _create_conv_layers(
        self,
        architecture: List[Union[List[int], str]]
    ) -> nn.Sequential:
        layers = []

        in_channels = self.in_channels

        # we loop through the architecture list and create each layer
        for layer in architecture:
            # if layer is a list, we know it's a conv layer
            if type(layer) == list:
                # add a CNNBlock
                layers += [
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=layer[1],
                        kernel_size=layer[0],
                        stride=layer[2],
                        padding=layer[3])
                ]
                # set in_channels for the next layer
                # to out_channels of current layer
                in_channels = layer[1]
            # if it's a max-pooling layer
            elif type(layer) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # return all layers as a sequential model part
        return nn.Sequential(*layers)

    # create the convolutional part of the network
    def _create_fully_connected_layers(
        self,
        grid_size: int,
        num_boxes: int,
        num_classes: int
    ) -> nn.Sequential:
        S, B, C = grid_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            # this will be reshaped later to be of shape (S, S, C + B * 5)
            nn.Linear(4096, S * S (C + B * 5))
        )
\`\`\`

That's basically the model architecture. Not actually that complicated, right? The interesting part comes now, where we'll see which loss function is responsible for training the model.

### The loss function

Let's think again about what exactly the model is doing. For each cell in the SxS grid, we want to see if there's an object or not and if yes, classify to which class it belongs. We also want to finds the objects center point and draw a bounding box around it. All these parts of the problem are cast into the the loss function, which can be seen in the figure below. It looks complicated at first, but it makes a lot of sense when we go through it bit by bit.

![The YOLO loss function (figure from the paper by Redmon et al.)](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/object-detection/yololoss.png)

The overall loss function is composed of several contributing parts of square losses. The squaring of the loss terms has the effect, that large deviations from the ground truth will be penalized much more. Some of the loss terms are prefixed by \`lambda_coord\`, which is just a multiplication constant to prioritize these loss terms a bit higher, because we want the model to put particular emphasize on getting the location of the boxes right. We'll set it to 5. The first term is for the center point coordinates; we can see we sum over the amount or grid cells (SxS) and number of predicted boxes per cell (B=2 in our case). The identity function in front of the loss term is either 1, if there was a target bounding box in the i-th cell and the j-th predicted box was responsible for outputting that box (meaning it had the highest IOU out of all predictors in that grid cell), else it is 0. The second term is basically the same, but for the width and height of the bounding box. Note that we take the square roots of height and width to make sure that we prioritize smaller bounding boxes equally much to larger ones. In the third term, \`C_i\` is either 1 or zero, depending on if there is an object in the cell or not, and \`C^_i\` is the predicted probability that there is an object in the cell. The fourth term is basically the same as the third, but for the case that there is no object in the cell. We want to penalize a prediction for an object if there actually is none. There's again a multiplication constant to prioritize this term a bit less; in this case we'll set it 0.5. The last term is for the classification, so if we get the class of the object right. Interestingly, instead of a common cross-entropy loss, the authors use a simple regression loss here as well.

Now let's have a look at how we can implement this custom loss function in code. We'll also use the \`intersection_over_union()\` function from the [previous post](https://pascal-bliem.com/blog/object%20detection%20metrics).

\`\`\`python
class YoloLoss(nn.Module):
    def __init__(
        self,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20
    ) -> None:
        # Note that I'm pretending that these parameters could be varied
        # from their defaults, but actually I'm treating them as if
        # they're hard-coded constants here to make the implementation
        # simpler and make the concepts clearer to understand.
        super(YoloLoss, self).__init__()

        # we use summed square losses
        self.mse = nn.MSELoss(reduction="sum")
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        # the prioritization multipliers as described above
        self.lambda_noobj = 0.5
        self.lamda_coord = 5

    def forward(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # reshape the predictions to (batch_size, S, S, C + B * 5)
        predictions = predictions.reshape(
            -1, self.S, self.S, self.C + self.B * 5
        )

        # calculate the IOU of the two predicted bboxes with the target bbox
        iou_b1 = intersection_over_union(
            predictions[..., 21:25], target[..., 21:25]
        )
        iou_b2 = intersection_over_union(
            predictions[..., 26:30], target[..., 21:25]
        )
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # best_box is the argmax and will either be 0 or 1 if B=2,
        # depending which of the two predicted bboxes has the higher IOU
        iou_maxes, best_box = torch.max(ious, dim=0)
        # in the paper, this is the identity function I_obj_i
        # that tells us if there is an object in cell i
        exists_box = target[..., 20].unsqueeze(3)

        ### Loss for box coordinates ###
        # select predicted coordinates, width, and hight for the best_box
        box_predictions = exists_box * (
            # this is 0 if the 0th bbox was best
            best_box * predictions[..., 26:30]
            # this is 0 if the 1st bbox was best
            + (1 - best_box) * predictions[..., 21:25]
        )
        # same for target box
        box_targets = exists_box * target[..., 21:25]

        # we take the sqrt of width and height
        box_predictions[..., 2:4] = (
            torch.sign(box_predictions[..., 2:4])
            * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # calculate the summed squared loss
        box_loss = self.mse(
            # we flatten this here from (batch_size, S, S, 4) to
            # (batch_size*S*S, 4) because the MSE will sum up the
            # losses of all batch_size examples and all S*S cells
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        ### Loss for object ###
        # get prediction probability/confidence of best_box
        pred_box = (
            best_box * predictions[..., 25:26]
            + (1 - best_box) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            # same as above, we flatten to (batch_size*S*S*1)
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        ### Loss for no object ###

        # if there is no object, both predicted boxes should know that
        # there is no object, hence, we consider loss from both boxes here
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21]),
            torch.flatten((1 - exists_box) * target[..., 20:21])
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26]),
            torch.flatten((1 - exists_box) * target[..., 20:21])
        )

        ### Loss for classification ###
        class_loss = self.mse(
            # flatten to (batch_size*S*S, 20)
            torch.flatten(exists_box * predictions[...,:20], end_dim=-2),
            torch.flatten(exists_box * target[...,:20], end_dim=-2)
        )

        ### Final Loss ###
        # combine all the loss terms
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
\`\`\`

That was pretty much the trickiest part. Now we can almost start training, just have to the the data first.

### Getting the training data

As I mentioned in the introduction, the original YOLO was trained on the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset, which I also like because it has my name in it. This is a dataset with around 43000 images and labels that specify to which of the 20 classes the object belongs and where the bounding boxes should be located. You can have a look at some examples from the 20 classes (airplanes, people, plants, chairs etc.) in the image below. Getting the data from the original source is a bit of a hassle, but luckily, [Aladdin Persson](https://www.youtube.com/channel/UCkzW5JSFwvKRjXABI-UTAkQ) has uploaded a prepared version of the dataset on [Kaggle](https://www.kaggle.com/dataset/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2), from where you can download it.

![Some samples from the PASCAL VOC dataset.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/object-detection/pascalvocimages.png)

The dataset is distributed over two directories, one which holds the bare images, and one which has one text file for each image containing the labels for that image. There also is a CSV file, mapping the image files to their respective label files. For each object in an image, there is one line in the label file that has 5 columns, one for the class label, the center point coordinates (x, y), width, and height. The coordinates and dimensions are relative (between 0 and 1) to the whole image, which is convenient, because we'll be rescaling the images. We will need to convert the coordinates for the whole image to coordinates relative to the cells in the SxS grid, though. PyTorch allows us to define custom datasets, so we can implement all our special needs into its \`__getitem__()\` function. Let's code it.

\`\`\`python
import os
import pandas
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    # the mapping file is a csv that provides a
    # mapping from the image to its label file
    def __init__(
        self,
        mapping_path: str,
        img_path: str,
        label_path: str,
        S: int = 7,
        B: int = 2,
        C: int = 2,
        transform: nn.Module = None
    ) -> None:
        # read the image-label-mapping from csv
        self.mapping = pd.read_csv(mapping_path)
        self.img_path = img_path
        self.label_path = label_path
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.mapping)

    # we only have to implement what to do for a single item
    # in the dataset and can then get it by index
    def __getitem__(self, index: int) -> Tuple[Image, torch.Tensor]:
        # get the label path for item of index
        label_path = os.path.join(
            self.label_path, self.mapping.iloc[index, 1]
        )

        # get all objects and their bboxes for that item
        bboxes = []
        with open(label_path) as file:
            for line in file.readlines():
                class_label, x, y, width, height = [
                    float(l) for l in line.replace("\n", "").split()
                ]
                class_label = int(class_label)
                bboxes.append([class_label, x, y, width, height])

        # get the image path for item of index
        img_path = os.path.join(
            self.img_path, self.mapping.iloc[index, 0]
        )

        image = Image.open(img_path)
        # cast to tensor in case we need to do transformations
        bboxes = torch.tensor(bboxes)

        # if there are any transformations, apply them to both
        # the image and the bounding boxes
        if self.transform:
            image, bboxes = self.transform(image, bboxes)

        # The following corresponds to the output of the model,
        # the SxS grid times number of classes plus B-times prediction
        # certainty, midpoint coordinates (x, y), width and height.
        # From the B boxes, actually only one is used because we only
        # have one ground truth label here, but we need its shape to match
        # the predictions, where we output B=2 bounding box candidates.
        target_matrix = torch.zeros([self.S, self.S, self.C + 5 * self.B])
        for bbox in bboxes:
            class_label, x, y, width, height = bbox.tolist()
            class_label = int(class_label)
            # in the SxS grid, i is row index, j is column index
            # we cast to int the get the cell the center point is in
            i, j = int(self.S * y), int(self.S * x)
            # calculate (x,y) relative to the cell coordinate system
            x_cell, y_cell = self.S * x - j, self.S * y - i
            # get width and height relative to the cell coordinate system
            width_cell, height_cell = width * self.S, height * self.S

            # one cell can be responsible for only one object,
            # so if there is currently no object in cell (i,j)
            if target_matrix[i, j, 20] == 0:
                # then there is one now
                target_matrix[i, j, 20] == 1
                # set the bbox coordinates and class label
                bbox_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                target_matrix[i, j, 21:25] = bbox_coordinates
                target_matrix[i, j, class_label] = 1

        # return image and target which now has the (S, S, C + B * 5)
        # shape that we've discussed in the architecture section
        return image, target_matrix

\`\`\`

Now that we've got the data ready, let's train the model.

### Training

We have set up almost everything we need already. We'll set some hyperparameters, instantiate our model, loss function, an optimizer and data loader and run a fairly standard PyTorch training loop.

\`\`\`python
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
# tqdm is a convenient package for a dynamic progress bar
from tqdm import tqdm

torch.manual_seed(42)

# hyperparameters
learning_rate = 0.0005
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
weight_decay = 0
epochs = 100
num_workers = 2
pin_memory = True
image_path = "path/to/images"
label_path = "path/to/labels"

# write a custom compose for transforms that not only transforms
# the image but also the bounding boxes accordingly
class Compose(object):
    def __init__(self, transforms: List[nn.Module]) -> None:
        self.transforms = transforms

    def __call__(
        self,
        img: Union[torch.Tensor, Image],
        bboxes: torch.Tensor
    ) -> None:
        for transform in transforms:
            img, bboxes = transform(img, bboxes)

# resize the image to match the input size of the model
transform = Compose([transforms.Resize([448, 448]), transforms.ToTensor()])

# the training function
def train(
    train_loader: DataLoader,
    model: Yolo,
    optimizer: optim.Optimizer,
    loss_function: YoloLoss
) -> None:
    model = model.to(device)
    # set up progress bar
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    # iterate over each batch, x and y are the
    # model inputs and targets, respectively
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        # make prediction
        out = model(x)
        # calculate loss
        loss = loss_function(out, y)
        mean_loss.append(loss.item())
        # step gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update progressbar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss is {sum(mean_loss)/len(mean_loss)}")

# instantiate model, optimizer, and loss function
model = Yolo(grid_size=7, num_boxes=2, num_classes=20).to(device)
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)
loss_function = YoloLoss()

# prepare training dataset
train_dataset = Dataset("path/to/mapping", image_path, label_path)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True,
    drop_last=True
)

# train the model
for epoch in range(epochs):
    train(train_loader, model, optimizer, loss_function)

\`\`\`

And that's the training. We can set up a data loader for the test set and perform model evaluation on it in almost the same way, except that we don't need to optimize anything then and should set the model to evaluation mode with \`model.eval()\` and disable gradient tracking by performing the calculation in a \`with torch.no_grad():\` block. Additionally, we could also take into account some learning rate scheduling, which may help with training. As you may have imagined, I'm not actually fully training the model here, because it is pretty huge and I don't want to spend the time and resources on it. The authors of the YOLO paper pretrained their CNN on [ImageNet](https://www.image-net.org/) for a week before they even started switching the training task to detection. Instead of wasting a lot of time and money here, if we had a real task at hand now, we could start from an already pretrained model. The YOLO architectures are pretty well know and you can download the latest version 5 in different sizes and with pretrained weights from the PyTorch [model hub](https://pytorch.org/hub/ultralytics_yolov5/). Anyways, the output would look something like what you can see in the image below :)

![Some examples of object detection on artwork and natural images (figure from the paper by Redmon et al.)](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/object-detection/yolooutput.png)

### Conclusion

In this post, we've explored the famous "You only look once" (YOLO) model for object detection with deep learning. We've discussed the general working principle behind YOLO, how we manage to detect multiple objects in one "run" through the network by splitting the images into cells in a grid and predicting bounding boxes for each of these cells. We have implemented the architecture of the model and understood the individual terms contributing to the special YOLO loss function. We've then set up a custom dataset implementation for the PASCAL VOC dataset as well as a simple training loop to train the model. Since the original YOLO, several improvements have been made to the algorithm, so if you want to use YOLO in a project, check out the latest version. I, again, want to thank [Aladdin Persson](https://www.youtube.com/channel/UCkzW5JSFwvKRjXABI-UTAkQ), whose videos on deep learning have helped me a lot to understand the whole topic of object detection. And, of course, thank you for reading!
`
);
