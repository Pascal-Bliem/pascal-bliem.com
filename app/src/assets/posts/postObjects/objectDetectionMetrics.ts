import Post from "../postModel";

export default new Post(
  // title
  "Object Detection: Metrics",
  // subtitle
  "A couple of things we need to know for evaluating object detection models",
  // publishDate
  new Date("2021-08-16"),
  // titleImageUrl
  "https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/object-detection/boundingboxes.png",
  // titleImageDescription
  "Object detection in action!",
  // tags
  ["Data Science & AI/ML", "Learning"],
  // content
  `After my last big project, [Doggo Snap](https://pascal-bliem.com/doggo-snap), a mobile app which uses deep learning to classify dog breeds from photos, I've gotten really exited about the field of computer vision. There is still a lot to explore beyond simple image classification. After being able to tell if something, such as a certain dog breed, is in an image, it would also be nice to tell where in the image it is. This is the problem of object detection or image segmentation. We want to be able to not only classify one or more objects or a section in an image, but also to draw an accurate bounding box around them or detect if a pixel belongs to a certain class. I want to focus on object detection with bounding boxes for now, as can be seen in the title image above.

Traditionally this used to be performed by a sliding window approach, in which a predefined box is slid over the image with a certain stride and every crop defined by the current position of the box is individually classified. This approach is, however, very computationally expensive because we have to "look" many time, for each new crop. [Region-proposing neural networks](https://arxiv.org/abs/1506.01497) were a bit faster but still slow. Would be nice if we only had to look once, right? This is also what the authors of the paper ["You Only Look Once: Unified, Real-Time Object Detection"](https://arxiv.org/abs/1506.02640), or short YOLO, thought when they came up with a much more efficient algorithm to perform object detection. I want to get into the YOLO neural network architecture in a later post though. For now, it is important to cover some metrics first, which are crucial for understanding how we can evaluate the bounding boxes that our model predicts. Before I continue here I want to say huge thanks to [Aladdin Persson](https://www.youtube.com/channel/UCkzW5JSFwvKRjXABI-UTAkQ), who has a YouTube channel on which he publishes fantastically educative videos on deep learning, from which I learned a lot about computer vision with deep neural networks.

In the following, we'll cover three metrics that kind of build up on each other and, eventually, will allow us to score the performance of the predictions we get, and on the way there, solve the problem that we may potentially get multiple bounding boxes for the same object. Those metrics are

- Intersection over Union (IOU)
- Non-max suppression
- Mean Average Precision (mAP)

Let's have a look at them one by one.

### Intersection over Union

Given that we have labeled data to train our models, we'll need to compare the label/target classifications and bounding boxes with the predictions made by the model. The classification part is easy, either we predicted the right class or not, maybe taking the prediction certainty into account. How about the bounding boxes? We want the predictions to overlap as much as possible with the targets. So, we want the common area of the two boxes, their intersection, to be relatively large compared to the combined area of the target and prediction box, their union. The ration of the two is the intersection-over-union (IOU). You can see this in the example below. The green shaded area, which both the target bounding box and the predicted bounding box have in common, is the intersection. When calculating the union, we have to keep in mind to only count this area once in total, not once per bounding box. You can now see that if the boxes wouldn't overlap at all, the IOU would equal 0, if they would overlap perfectly, the IOU would equal 1. I guess somewhere around 0.5, the prediction would become passable.

![A target and a predicted bounding box for detecting a beer glass. From this we can calculate the IOU](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/object-detection/koelschbox.png)

Calculating the union is trivial, but how do we calculate the intersection? We need to find the box coordinates that define it. To do so we get the top left edge by taking the maximum of each the x-coordinates and the y-coordinates of the target and predicted box. For the lower right point, we do the same but take the minimum instead. Let's see how we'd implement this in code. Throughout this (and the next) post, I'll use [PyTorch](https://pytorch.org/), my favorite tensor library and deep learning framework in Python. Note that the bounding box coordinate format, [x, y, w, h], I will use here is not describing the top left and bottom right edge of the box but rather the center point of the box and its width and height. This is the format also used in the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset, which was used in the original YOLO paper (and carries my name :p). I'll try to keep telling the story with inline-comments:

\`\`\`python
from typing import List
import torch

def intersection_over_union(
    bboxes_pred: torch.Tensor,
    bboxes_target: torch.Tensor
) -> torch.Tensor:
    """Calculates the intersection-over-union (IOU)
    of target and predicted bounding boxes.

    Args:
        bboxes_pred: Tensor of shape (N, 4), containing N
                     predicted bounding boxes, each [x, y, w, h]
        bboxes_target: Tensor of shape (N, 4), containing N
                       target bounding boxes, each [x, y, w, h]

    Returns:
        iou: The intersection-over-union metric
    """

    # convert the center-point-width-height representation of the
    # boxes to a top-left-edge-bottom-right-edge representation
    box1_x1 = bboxes_pred[..., 0:1] - bboxes_pred[..., 2:3] / 2
    box1_x2 = bboxes_pred[..., 1:2] - bboxes_pred[..., 3:4] / 2
    box1_x2 = bboxes_pred[..., 0:1] + bboxes_pred[..., 2:3] / 2
    box1_y2 = bboxes_pred[..., 1:2] + bboxes_pred[..., 3:4] / 2
    box2_x1 = bboxes_target[..., 0:1] - bboxes_target[..., 2:3] / 2
    box2_x2 = bboxes_target[..., 1:2] - bboxes_target[..., 3:4] / 2
    box2_x2 = bboxes_target[..., 0:1] + bboxes_target[..., 2:3] / 2
    box2_y2 = bboxes_target[..., 1:2] + bboxes_target[..., 3:4] / 2

    # get the top-left (x1, y1) and bottom-right (x2, y2)
    # points that define the intersection box
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # calculate the intersection area by multiplying the sides of the rectangle
    # (the clamp(0) is for the case that the boxes don't intersect at all)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # get the individual box areas
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 -box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 -box2_y1))

    # calculate the union (don't forget to not count the intersecting area double)
    union = box1_area + box2_area - intersection

    return intersection / union
\`\`\`

### Non-max Suppression

The next concept we're going to talk about is non-max suppression. This isn't really a metric, but rather a technique to solve the following problem: What is we get many bounding box predictions that look like they're largely overlapping and meant for the same object? How are we going to choose the right one? We can see this example in the figure below. We will usually build our models in a way that they output a certainty score for a classification, a probability that states how likely our model thinks that the object actually belongs to the predicted class. The higher this score, the more certain the model is that what is inside of the box actually belongs to the predicted class. We can describe each prediction as an array with 6 elements: bbox = [class, certainty, x, y, width, height]. Knowing this, non-max suppression is actually pretty easy to understand: For all boxes predicted for a certain class, we check them pair-wise (starting from the one with the highest certainty) and calculate their IOU. If it is over a certain threshold (e.g. 0.5) we consider those boxes redundant and just remove the one with the lower predicted certainty. We then repeat this for all classes. We can also set a certainty/probability threshold to filter out unlikely boxes right from the start.

![Non-max suppression: from bounding boxes that have a minimum overlap (IOU), we only take the one with the highest predicted certainty.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/object-detection/koelschnonmax.png)

Now let's see how we could implement this in code:

\`\`\`python
def non_max_suppression(
    bboxes: List[List[float]],
    iou_threshold: float,
    cert_threshold: float,
) -> List[List[float]]:
    """Perform non-max suppression on predicted bounding boxes.

    Args:
        bboxes: List of shape (N, 6), containing N predicted
                bounding boxes, each [class, certainty, x, y, w, h]
        iou_threshold: IOU Threshold
        iou_threshold: Certainty Score Threshold

    Returns:
        remaining_bboxes: Bounding boxes remaining after
                          non-max suppression is performed
    """

    # bboxes will be of shape [[class, certainty, x, y, width, height], ...]
    # filter out all boxes below certainty threshold
    bboxes = [box for box in bboxes if box[1] > cert_threshold]

    # this will be the output after non-max suppression
    remaining_bboxes = []

    # put bbox with highest certainty to the beginning of the list
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # take bboxes one by one starting from highest certainties
    while bboxes:
        chosen_box = bboxes.pop(0)

        # filter out lower-certainty boxes of the same class
        # that have sufficiently high IOU with the chosen box
        bboxes = [
            box for box in bboxes
            # keep them if they're of different classes
            if box[0] != chosen_box[0]
            # or if they're below the iou_threshold
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
            ) < iou_threshold
        ]

        remaining_bboxes.append(chosen_box)

    return remaining_bboxes
\`\`\`

### Mean Average Precision (mAP)

Now this final metric is what we'll actually use for scoring the performance of our object detection model. The mean Average Precision (mAP) is basically the integration of the precision-recall-curve of a model. Anyone who as done anything with machine learning has probably encountered precision and recall, but let's quickly recap it here. Precision tells us, among all positive predictions we make, how many of those are true positives: precision = true positives / (true positives + false positives). The recall tells us, among all positive examples in the data, how many did the model predict to be positive: recall = true positives / (true positives + false negatives). The tradeoff between these two metrics is obvious. If we want a higher precision, we can try to make our model stricter so that it will make positives predictions with more care, only predicting the ones with high certainty. At the same time, that means that probably more data points will be missed by it and false negatives increase, hence, recall decreases. A generally good model should have a good precision and a good recall at the same time, or in other words, if precision and recall are plotted against each other on a curve, the area under that curve should be as large as possible. This area is also called the average precision (AP). If we calculate this AP for each class that the model can potentially classify and take the mean of those results, we end up with the mean average precision (mAP). You can see this visualized in the image below; the precision-recall-curves for different classes are plotted and the overall model performance can be evaluated by the mean of the integrals of these curves.

![Precision-Recall-Curves of different classes classified by model are used to calculate the mean Average Precision (mAP)](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/object-detection/mAP.png)

But how to we apply this to the problem of object detection. As mentioned before, we'll probably be in a situation in which we have labeled data with target bounding boxes given as ground truth and bounding boxes predicted by our model. We want to know how close to the targets the predictions are. As we know now, we need to calculate the precision-recall-curve per class, and for that we need to first know which predictions we consider a true or false positives. For that we can simply calculate the IOU of the target and predicted box and see if it lies above a defined threshold. Now we calculate the curve by sorting the predictions by confidence score from the highest to the lowest, and then we go through that sequence and calculate precision and recall cumulatively, always up to the current step in the sequence. The resulting values are points on the precision-recall-curve and we can integrate the curve. We then repeat the procedure for all classes and take the mean. We may even run this entire algorithm for several different IOU thresholds.

Now let's see how we would implement that in code. Note that now (different from the previous code cell), each prediction will be a list of seven elements because here we score on all images in a dataset and we want to make sure that the target and prediction boxes actually belong to the same image, therefore, we include an image index:

\`\`\`python
from collections import Counter

def mean_average_precision(
    pred_boxes: List[List[float]],
    target_boxes: List[List[float]],
    iou_threshold: float = 0.5,
    num_classes: int = 20
) -> float:
    """Calculate mean Average Precision (mAP) score.

    Args:
        pred_boxes: List of shape (N, 7), containing N predicted
                bounding boxes, each [img_idx, class, certainty, x, y, w, h]
        target_boxes: List of shape (N, 7), containing N target
                bounding boxes, each [img_idx, class, certainty, x, y, w, h]
        iou_threshold: Threshold for IOU between target and predicted
                       bounding box
        num_classes: The number of classes used by the model

    Returns:
        mAP: mean Average Precision score
    """

    average_precisions = []

    # loop through all classes
    for class_idx in range(num_classes):
        predictions = []
        targets = []

        # get all the predicted and target bboxes
        # belonging to current class
        for prediction in pred_boxes:
            if prediction[1] == class_idx:
                predictions.append(prediction)

        for target in target_boxes:
            if target[1] == class_idx:
                targets.append(target)

        # count how many target boxes appear in each image of the dataset
        count_target_boxes = Counter([t[0] for t in targets])

        # we need to keep track of the target bounding box that have already
        # been covered, so that in case there are several bbox predictions for
        # the same target bbox, only the first one with the highest certainty
        # will be considered true and the others false;
        # If the first img has 2 bboxes and the second img has 4 bboxes,
        # after the loop below, count_target_boxes will look like
        # {0:torch.tensor([0,0]), 1:torch.tensor([0,0,0,0]), ... }
        for key, val in count_target_boxes.items():
            count_target_boxes[key] = torch.zeros(val)

        # sort predictions over the certainties, starting from highest
        predictions = sorted(predictions, key=lambda x: x[2], reverse=True)

        # tensors to keep track of true and false positives
        true_positives = torch.zeros(len(predictions))
        false_positives = torch.zeros(len(predictions))

        all_target_bboxes = len(targets)

        # now we pick one prediction at a time
        for prediction_idx, prediction in enumerate(predictions):

            # and get the target bboxes from the corresponding image
            targets_corresponding_img = [
                bbox for bbox in targets if bbox[0] == prediction[0]
            ]

            # now this is the length of all target bboxes in that image
            num_targets = len(targets_corresponding_img)

            # for each of the target bboxes, calculate its IOU with
            # the current prediction and find the best
            best_iou = 0
            for target_idx, target in enumerate(targets_corresponding_img):
                iou = intersection_over_union(
                    torch.tensor(prediction[3:]),
                    torch.tensor(target[3:]),
                )
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx

            if best_iou > iou_threshold:
                if count_target_boxes[prediction[0]][best_target_idx] == 0:
                    # has not yet been covered, so its a true positive
                    true_positives[prediction_idx] = 1
                    # now we check it as covered so that
                    # it won't get covered again later
                    count_target_boxes[prediction[0]][best_target_idx] = 1
                else:
                    # its a false positive
                    false_positives[prediction_idx] = 1
            else:
                # its a false positive
                false_positives[prediction_idx] = 1

        # calculate the cumulative sums to be able to calculate
        # precision and recall up to each prediction in the sequence
        true_positives_cumsum = torch.cumsum(true_positives, dim=0)
        false_positives_cumsum = torch.cumsum(false_positives, dim=0)
        recalls = torch.divide(true_positives_cumsum, (all_target_bboxes))
        precisions = torch.divide(
            true_positives_cumsum, 
            (true_positives_cumsum + false_positives_cumsum)
        )

        # for integrating the precision-recall curve, we'll also
        # need a point (0,1), so we need to concat these values
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))

        # integrate the curve
        average_precisions.append(torch.trapz(precisions, recalls))

    # the the mean and return
    mAP = sum(average_precisions) / len(average_precisions)
    return mAP
\`\`\`

And that's basically it. As mentioned above, we can now also calculate this metric for different IOU thresholds e.g. from 0.5 to 0.95 in steps of 0.05.

### Conclusion

Let's quickly recap the concepts we covered: We've seen that the fundamental metric that tells us how much predicted bounding boxes overlap with target boxes is the intersection-over-union (IOU). We solved the problem of having potentially many bounding boxes predicted for the same object with non-max suppression. Finally we've seen how the overall model can be scored with mean Average Precision (mAP). Now that we know how to handle bounding boxes that are predicted by an object detection model, we can have a look at an actual model. In an upcoming post I will have a look at the YOLO architecture, so stay tuned. Thanks a lot for reading!
`
);
