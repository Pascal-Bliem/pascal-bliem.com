import Post from "../postModel";

export default new Post(
  // title
  "Instance segmentation for fashion",
  // subtitle
  "Finding and classifying apparel in images with Mask R-CNNs",
  // publishDate
  new Date("2021-08-12"),
  // titleImageUrl
  "https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/mask-rcnn/fashion_segmentation.png",
  // titleImageDescription
  "Segment fashion images to let AI figure out what people are wearing!",
  // tags
  ["Data Science", "Learning"],
  // content
  `At the moment, I'm super exited about computer vision with Deep Learning, as you can probably tell by looking at most of my last posts. I after my [dog breed classifier](https://pascal-bliem.com/doggo-snap), I recently had a look at object detection with [YOLO](https://pascal-bliem.com/blog/object%20detection%20yolo), and now I'm looking at the somewhat similar problem of instance segmentation. A friend of mine is co-founder of the sustainable fashion startup [La Vouga](https://lavouga.com/), which connects customers in search of ethical and sustainable fashion with independent artisan makers across Europe. We recently talked about how it would be great to find clothing matching certain search terms on their partners' websites, without having to rely on (sometimes weird and heterogenous) product descriptions. If we could just let an AI tag relevant images with the type of apparel and descriptive attributes, this would greatly help customers to find the hand-crafted slow-fashion they're looking for, by just typing what they're looking for or uploading images with similar clothing at La Vouga's search.

If we want to treat this scenario as a computer vision task, we're dealing with instance segmentation. That means we're not just detection an object (clothing in this case) or semantically segmenting the entire image into different classes, but we want to find the exact pixel locations of the different instances of clothing in the image so that we can have a look at them individually and extract descriptive attributes for each instance. To see if this is promising in a proof of concept, we of course don't want to collect our own annotated training data set yet; hence, I'll describe in the following what kind of data is out there that can be used for this task, and what kind Deep Learning architectures may be able to solve the problem. In the end I will train a Mask R-CNN to perform instance segmentation on fashion images, but only with classifying the instances into apparel categories - not yet with predicting descriptive attributes. I'm planning the modify some of the detection models' source code in [torchvision](https://github.com/pytorch/vision) to incorporate the attribute prediction as well, but this may take some time, so I'll postpone that part to a future post. Okay, let's get started.

### Fashion image segmentation datasets

When I started researching this topic, I was really surprised how many different datasets and papers were out there that dealt with computer vision tasks on fashion images. But actually, it makes a lot of sense. The clothing e-commerce industry is huge, and there's a huge amount of money to earn; naturally, people are investing in AI research to find out what people are wearing, how it looks on them, track rising fashion trends, or matching products from commercials with real people wearing them.

Most machine learning practitioners have probably encounter the Fashion-MNIST dataset (not the one with handwritten digits) at some point, as it is a popular Kaggle [competition](https://www.kaggle.com/zalando-research/fashionmnist) for beginners. The dataset contains small gray scale images of clothing products from 10 different categories, but no humans are attached to these pieces of apparel. Some people have tried identifying clothing on people as early as 2012 at [chictopia.com](http://www.chictopia.com/). In the paper [“Parsing clothing in fashion photographs” by Yamaguchi et al. 2012](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.233.840&rep=rep1&type=pdf), the authors set up the Fashionista dataset containing ca. 650 manually annotated images with clothing segmentation and pose estimation. They extended it with the Paperdoll dataset from the paper ["Retrieving Similar Styles to Parse Clothing" by Yamaguchi et al. 2014](http://vision.is.tohoku.ac.jp/~kyamagu/papers/yamaguchi2014retrieving.pdf), in which they automatically/weakly annotated a million images. Again from chictopia.com, the [Chictopia10K dataset](https://github.com/lemondan/HumanParsing-Dataset) was presented in ["Deep Human Parsing with Active Template Regression" by Liang et al. 2015](https://arxiv.org/abs/1503.02391). You can find an example of the segmentation masks in the figure below. This latter dataset was e.g. used in ["A Generative Model of People in Clothing" by Lassner et al. 2017](https://arxiv.org/abs/1705.04098) to build generative models that can produce realistic images of humans in different poses (though their faces look like horrifying demons from hell 😅).

![Segmentations predicted on the Chictopia10K dataset in Liang et al. 2015](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/mask-rcnn/image_masks.png)

These datasets were all pretty interesting, but I was looking for something a bit more modern, with more classes, attributes and especially more training examples. Ebay's researchers have released a large (ca. 50k images) dataset called ModaNet in ["ModaNet: A Large-scale Street Fashion Dataset with Polygon Annotations" by Zheng et al. 2019](https://arxiv.org/abs/1807.01394), which contains 13 meta-categories of clothing. Another large (ca. 44k images) dataset is [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) described in ["DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images" by Ge et al. 2019](https://arxiv.org/abs/1901.07973v1), which also contains 13 different categories of clothing. Those two datasets are amazing, but I was still looking for something with a more detailed classification and descriptive attributes. Luckily, I found exactly that on Kaggle in the [iMaterialist (Fashion) 2019 for Fine-Grained segmentation competition](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/overview). This dataset (an image from it is displayed as the title image of this post) contains 46 apparel objects (27 main apparel items and 19 apparel parts), and 92 related fine-grained attributes in ca. 50k clothing images. This sounds like just the right dataset to do the proof of concept for our fashion image tagging AI. So now that we got the data, its time to think about the model.

### Finding a model for instance segmentation

As mentioned shortly in the introduction, we're dealing with the problem of instance segmentation, which we can consider a combination of object detection, classification, and semantic segmentation. That means we'll need a couple of things to be achieved by the model architecture. One the one hand, object detection usually involves finding suitable bounding boxes for objects and classifying what is inside of those boxes. We don't really need to output another image on the backside of the model, but rather find the right class id and regress the bounding box coordinates. We don't really need to care about if the object was in a certain pixel of the original image and, hence, can usually use some kind of fully connected layers as a predictor for the neural network. Some examples of this would be the [YOLO](https://arxiv.org/abs/1506.02640) model (on which I've previously written a [post](https://pascal-bliem.com/blog/object%20detection%20yolo)) which splits the original image into a grid and makes predictions for each cell in only one pass through the network, or regional convolutional neural networks (R-CNNs), which perform classifications for a certain number of region proposals and which I'll explain in more detail later. On the other hand, semantic segmentation really wants to know which pixel in the original image belongs to a certain class of object or background; we basically need to find an image mask that represents the presence of each class in the original image. One choice for this task would be [fully convolutional networks](https://arxiv.org/abs/1411.4038) (FCNs), in which the final dense layers are replaced by convolutions that output a feature map, which can be upsampled to the original images size and act as a heat map for the presence of a certain class. Another, more complex choice could be [U-Net](https://arxiv.org/abs/1505.04597), a architecture that combines two CNN parts in an encoder-decoder structure to output segmentation maps.

If we combine these two tasks, we end up with instance segmentation: locate the object, classify it, and find its segment on the original image. The model I'll be using for this is called [Mask R-CNN](https://arxiv.org/abs/1703.06870). It basically starts from a R-CNN for object detection and adds another branch to the model that outputs the segmentation masks. But do understand this thoroughly, we should go way back to the question "What are R-CNNs?". [Regional convolutional neural networks](https://arxiv.org/abs/1311.2524) have a region proposal mechanism which identifies regions of interest (ROI) in the original image (more details will follow) and then sends this region through a convolutional neural network that acts mostly as a classifier. This is performed on every proposed region, so that means for maybe 2000 proposals, it needs to run 2000 times. Obviously, that will take a lot of time. An improved version, called [Fast R-CNN](https://arxiv.org/abs/1504.08083) was proposed, which passed the entire image through a CNN once and then projects the proposed regions of interest on the output feature map of the CNN, and pools/maps the regions into a fully connected predictor. You can see that visualized in the figure below.

![The concept of the Fast R-CNN (from Girshick et al. 2015)](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/mask-rcnn/fast_rcnn.png)

This is already a lot faster than the original R-CNN, but there is still a bottle neck that makes it slow and that is the region proposal mechanism. It is usually some kind of selective search in which some initial regions are proposed and then iteratively merged by a greedy algorithm into larger regions based on heuristics such as similarity in color, texture, or size, until the wanted amount of proposals is reached. This takes relatively long, cannot always be performed on GPU with the rest of the network, and basically prevents real-time application. Luckily, there's another improvement called [Faster R-CNN](https://arxiv.org/abs/1506.01497), which comes with its own region proposal network (RPN), a fully convolutional network that simultaneously predicts object bounds and objectness (how much does this seem like an object?) scores at each position. This RPN shares its convolutional layers with the object detection network. This idea is visualized in the figure below.

<img src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/mask-rcnn/rpn_faster_rcnn.png" style="width: 70%;" alt="The region proposal network (RPN) in Faster R-CNN (from Ren et al. 2016)"/>

During training, the objective alternates between fine-tuning for the region proposal task and then fine-tuning for object detection, while keeping the proposals fixed. This scheme produces a unified network with convolutional features that are shared between both tasks. Region proposals are generated by sliding a smaller network over the convolutional feature maps, and at each position of the window, up to k anchor boxes are proposed, which have different scales and aspect ratios (the authors chose k=9 with 3 scales and 3 aspect ratios). These k anchors are then send to two fully connected layers, a bounding box regressor, and a classifier, which are then fine tuned (remember that the conv layers are shared between the region proposal network and the Fast R-CNN detector). You can see this anchor region proposal scheme visualized below.

![The region proposal mechanism with anchor boxes in Faster R-CNN (from Ren et al. 2016)](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/mask-rcnn/anchors_faster_rcnn.png)

This now means that the entire process of region proposal, classification, and bounding box regression can be done in one neural network and it is fast enough for real-time applications. Now we're almost there. We got the object detection part covered, now comes the segmentation part. Another development on top of Faster R-CNN ist [Mask R-CNN](https://arxiv.org/abs/1703.06870). It looks almost exactly like Faster R-CNN except one obvious and one less obvious difference. In addition to the class and bounding box prediction, it has an additional, fully decoupled head that adds another convolutional layer to predict a segmentation mask. When predicting masks, it is important to preserve the exact pixel locations. The Faster R-CNN's ROI pooling, however, is a coarse spatial quantization for feature extraction, which had to be replaced by a ROI alignment layer that preserves the exact pixel locations. You can see the architecture below. Now, I think, we got all the theoretical knowledge we need and can implement it.

![The Mask R-CNN architecture (from He et al. 2018)](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/mask-rcnn/mask_rcnn.png)

### Setting up a Mask R-CNN

This architecture is already pretty advanced and there are a lot of elements that have to be plugged together: the backbone CNN, the [feature pyramid network](https://arxiv.org/abs/1612.03144), region proposal network, ROI alignment, the predictor heads for classification, bounding box regression, and mask prediction, as well as the respective losses. Coding this from scratch would be a lot of code. But luckily, [torchvision](https://github.com/pytorch/vision), PyTorch's library for computer vision, has a bunch of models already pre-implemented. The have a sub-module \`torchvision.models.detection\`, which hosts a variety of different R-CNN models, including Mask R-CNN. It also comes as a pretrained version with a [ResNet50](https://arxiv.org/abs/1512.03385) backbone that has been pretrained on the [COCO](https://cocodataset.org/#home) dataset.

On PyTorch's website, we can also find a really useful [tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) that shows how we can use their Mask R-CNN and set up some new predictor heads for a custom dataset:

\`\`\`python
# as in the PyTorch tutorial https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer_channels = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer_channels,
                                                       num_classes)

    return model
\`\`\`

This implementation expects a certain input. As targets, we need to pass a class label, a tensor with bounding box coordinates, as well as a tensor representing the masks. Therefore, we first need to set up a custom PyTorch dataset to get our data into the right format. I got the implementation below mostly form [this Kaggle Kernel](https://www.kaggle.com/abhishek/mask-rcnn-using-torchvision-0-17) from Abhishek Thakur and made some adjustments to incorporate the descriptive attributes in the data as well. Some of the modules I import here are utilities that can be found in the torchvision Github repo at [\`vision/references/detection/\`](https://github.com/pytorch/vision/tree/master/references/detection). First of all, we need to convert the mask information, which is present in run-length encoding, into an array/tensor that represents a binary image mask.

\`\`\`python
import numpy as np

def rle_decode(mask_rle, shape):
    """Returns binary numpy array according to the shape,
    1 for the mask, 0 for the background.

    Args:
        mask_rle: in 1d array of run-length encoding as string
                  [start0] [length0] [start1] [length1]...
        shape: Shape of array to return (height,width)

    Returns:
        mask: The image mask as a numpy array of shape (height, width)

    """
    shape = (shape[1], shape[0])
    s = mask_rle.split()
    # gets starts & lengths 1d arrays
    starts, lengths = [
      np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])
    ]
    starts -= 1
    # gets ends 1d array
    ends = starts + lengths
    # creates blank mask image 1d array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    # sets mask pixels
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # reshape as a 2d mask image, the transpose
    # is needed to align to RLE direction
    return img.reshape(shape).T
\`\`\`

We can then implement the custom dataset:

\`\`\`python
import collections
import import pandas as pd
from PIL import Image

# the implementation of the dataset is similar to Abhishek Thakur's kernel https://www.kaggle.com/abhishek/mask-rcnn-using-torchvision-0-17
class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, df_path, height, width, num_attributes, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = pd.read_csv(df_path)
        self.height = height
        self.width = width
        self.image_info = collections.defaultdict(dict)
        # ClassId contains categories as well as attributes,
        # we extract category here
        self.df['CategoryId'] = self.df.ClassId.apply(
          lambda x: str(x).split("_")[0]
        )
        # add the descriptive attributes as well
        self.df['AttributesIds'] = (
            self.df['AttributesIds']
            .apply(lambda x: str(x).split(","))
            .apply(lambda idx: [
                1 if (str(i) in idx) else 0
                for i in range(num_attributes)
            ])
        )

        # for each image, put all encodings and corresponding categories in lists
        temp_df = (self.df.groupby('ImageId')['EncodedPixels', 'CategoryId']
                       .agg(lambda x: list(x)).reset_index())
        # the image dimensions
        size_df = (
            self.df.groupby('ImageId')['Height', 'Width']
            .mean()
            .reset_index()
        )
        temp_df = temp_df.merge(size_df, on='ImageId', how='left')

        # store all the relevant infos for each image in the image_info dict
        for index, row in tqdm(temp_df.iterrows(), total=len(temp_df)):
            image_id = row['ImageId']
            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            self.image_info[index]["image_id"] = image_id
            self.image_info[index]["image_path"] = image_path
            self.image_info[index]["width"] = self.width
            self.image_info[index]["height"] = self.height
            self.image_info[index]["labels"] = row["CategoryId"]
            self.image_info[index]['attributes'] = row['AttributesIds']
            self.image_info[index]["orig_height"] = row["Height"]
            self.image_info[index]["orig_width"] = row["Width"]
            self.image_info[index]["annotations"] = row["EncodedPixels"]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        info = self.image_info[idx]
        # create a mask for all objects in the image of shape
        # (num_obj, width, height)
        mask = np.zeros(
          (len(info['annotations']), self.width, self.height),
          dtype=np.uint8
        )

        labels = []
        attributes = []
        # create the submasks for each object by decoding them from run_length
        # format to array of shape (orig_width, orig_height) and then resize
        # to (width, height)
        for m, (annotation, label, attribute) in enumerate(zip(
            info['annotations'], info['labels'], info['attributes']
        )):
            sub_mask = rle_decode(
                annotation,
                (info['orig_height'], info['orig_width'])
            )
            sub_mask = Image.fromarray(sub_mask)
            sub_mask = sub_mask.resize(
                (self.width, self.height),
                resample=Image.BILINEAR
            )
            mask[m, :, :] = sub_mask
            # here we +1 the category label because the label numbering
            # starts at 0 but we want to consider 0 to be the background
            labels.append(int(label) + 1)
            attributes.append(attribute)

        # create bounding boxes for the objects and filter out objects
        # that are very small (blelow 20*20 pixels)
        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_attributes = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_attributes.append(attributes[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        # if there are no labels left, put in a background dummy
        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_attributes.append(np.zeroes(num_attributes))
            new_masks.append(mask[0, :, :])

        # recombine the new masks into one array
        nmx = np.zeros(
            (len(new_masks), self.width, self.height),
            dtype=np.uint8
        )
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        # convert bounding boxes, masks, labels and idx to torch tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)
        attributes = torch.as_tensor(new_attributes, dtype=torch.int64)
        image_id = torch.tensor([idx])
        # calculate area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # in the example from the PyTorch tutorial, people are segmented and
        # there is a flag for crowds of people - this is irrelevant here,
        # so we'll set it to zero
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target['attributes'] = attributes
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_info)
\`\`\`

For data augmentation purpose, we might want to include some image transforms in the dataset. Since the normal trochvision transform only operate on images, but we also need to transform the masks and bounding boxes accordingly, we will use custom transforms here which can be found under [\`vision/references/detection/transforms.py\`](https://github.com/pytorch/vision/tree/master/references/detection/transforms.py).

\`\`\`python
import transforms as T

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
\`\`\`

Now we can set up a dataset and data loader for training.

\`\`\`python
DATA_DIR = "path/to/data"
# +1 because we consider the background to be class 0
num_classes = 46 + 1
num_attributes = 341
batch_size = 4

dataset = FashionDataset(
    image_dir=os.path.join(DATA_DIR, "train"),
    df_path=os.path.join(DATA_DIR, "train.csv"),
    height=512,
    width=512,
    num_attributes=num_attributes,
    transforms=get_transform(train=True)
)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=lambda x: tuple(zip(*x))
)
\`\`\`

The rest will be a fairly standard PyTorch training loop, which is relatively simple because all losses are calculated within the model and returned as a dictionary during training.

\`\`\`python
# get the model
model = get_instance_segmentation_model(num_classes)
model.to(device)

# set up optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

model.train()
num_epochs = 10

for epoch in range(1, num_epochs+1):

    for i, (images, targets) in enumerate(data_loader):
        # move tensors to GPU
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # in training, the model returns a dict with all individual losses for
        # the classifier, bounding boxes, masks, region proposals and objectness
        loss_dict = model(images, targets)

        # sum the losses to one value
        losses = sum(loss for loss in loss_dict.values())

        # backprop and optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # step the learning rate scheduler
        lr_scheduler.step()

        print(
            f"Epoch {epoch}/{num_epochs} Batch {i}/{len(dataset)//batch_size}"\
            f", Loss: {losses.item()}"
        )
\`\`\`

If we put the model into evaluation mode with \`model.eval()\`, it will not anymore output the loss dictionary, but the predictions for class, bounding boxes, and masks. Perfect segmentations would look something like in the image below.

![Segmentation examples form the iMaterialist Fine-Grained segmentation dataset.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/mask-rcnn/segmentation.jpg)

Up to this point it was fairly easy because PyTorch provided these useful implementations out of the box. But as I mentioned earlier. I'd also like to add the prediction of descriptive attributes to the model which is not implemented yet. I'll probably rewrite some of the torchvision detection model code to include another predictor head or modify the existing Fast R-CNN predictor to give me another output. This may take a while though, as I'm fairly busy these days. Look out for it in a future post!
Thanks a lot for reading, I hope you had as much fun learning about instance segmentation and Mask R-CNNs as I had. Cheers!
`
);
