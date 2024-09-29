import Post from "../postModel";

export default new Post(
  // title
  "Transfer ML Models easily with ONNX",
  // subtitle
  "Moving a Deep Learning Model from Python to JavaScript",
  // publishDate
  new Date("2021-03-25"),
  // titleImageUrl
  "https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/onnx/onnx-logo.png",
  // titleImageDescription
  "The Open Neural Network Exchange (ONNX)",
  // tags
  ["Data Science & AI/ML", "Web Development"],
  // content
  `**TL;DR:** The Open Neural Network Exchange (ONNX) allows to transfer machine learning models into a common format that can easily be exchanged between different frameworks or runtimes. Here, I will show how to export a deep learning model for classifying dog breeds from PyTorch (Python) into the ONNX format and run inference in NodeJS (a JavaScript runtime).

Insufficient interoperability in machine learning has been a pain point for a long time. There are many different frameworks for machine learning or deep learning that can usually not be used interchangeably. Most of these frameworks, which are used for the development of machine learning models, are written in the Python programming language (or in C/C++ but at least have a Python wrapper). However, the system on which the model should eventually perform inference, might be written in a different programming language. Think about execution in a mobile app or the browser; Python is usually not used there. In the past, this often meant that a model would have to be reimplemented in the target systems.

To avoid the overhead of reimplementation and promote interoperability between different frameworks, tools, runtimes, and compilers, Microsoft and Facebook started developing the [Open Neural Network Exchange (ONNX)](https://onnx.ai/) in 2017. On their [Github](https://github.com/onnx/onnx), they describe it as follows: "ONNX provides an open source format for AI models, both deep learning and traditional ML. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types." Since every machine/deep learning model can be represented as a combination of operators, models from any machine learning framework can be transferred into the ONNX format on be executed by the ONNX runtime, which is available in a couple of programming languages. In the following, I will export a model from PyTorch and run inference on it in NodeJS.

### Exporting a model to ONNX

In a [previous blog post](http://www.pascal-bliem.com/blog/classifying%20dog%20breeds%20with%20deep%20learning), I described how I built a CNN deep learning model to perform image classification on 121 different dog breeds. I developed this model in PyTorch, a Python deep learning framework. PyTorch has some build in capabilities to export its models to ONNX and also offers a [tutorial](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) on how to do it. In the previous post, we created the model like this:

\`\`\`python
import torch
import torch.nn as nn
from torchvision import models

# get predefined model architecture
model = models.mobilenet_v2(pretrained=True)

# replace the classifier-part of the model that has been pretrained on ImageNet
# by a new classifier for classifying dog breeds
num_clf_in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
  nn.Dropout(p=0.2, inplace=False),
  nn.Linear(in_features=num_clf_in_features, out_features=len(class_names), bias=True),
  nn.LogSoftmax(dim=1)
)

model.to("cuda")
\`\`\`

We can then load the trained model parameters by:

\`\`\`python
model.load_state_dict(torch.load(checkpoint_dir))
\`\`\`

In this case it would be relatively straight forward, if we wanted to run the model on the current Python implementation of the ONNX runtime. However, earlier versions of the runtime seem to be lacking certain operators, and it appears that the runtime currently implemented for JavaScript in the library [ONNX.js](https://github.com/microsoft/onnxjs) is lacking some of the operators that our model uses. If we would just export the current model as it is, we would later (when running it in NodeJS) get errors that would look like:

\`\`\`
TypeError: cannot resolve operator 'LogSoftmax' with opsets: ai.onnx v9
\`\`\`

and

\`\`\`
TypeError: cannot resolve operator 'Shape' with opsets: ai.onnx v9
\`\`\`

Fixing the first one is easy: The final LogSoftmax layer in the model doesn't have any weights, so we can just replace it with regular Softmax layer without having to retrain the model. A Softmax operator is implemented in ONNX.js at the moment. The only difference is that the model will now output regular probabilities between 0 and 1, instead of log probabilities. Fixing the "Shape" error was a little more tricky. Apparently, ONNX.js currently does not support dynamic shape calculation, but the \`torchvision.models.mobilenet_v2\` class implements a method \`_forward_impl\` which has a dynamic shape operation:

\`\`\`python
def _forward_impl(self, x):
    # This exists since TorchScript doesn't support inheritance, so the superclass method
    # (this one) needs to have a name other than \`forward\`that can be accessed in a subclass
    x = self.features(x)
    # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
    x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
    x = self.classifier(x)
    return x
\`\`\`

If we replace this operation with a \`flatten\` operation (like in [this](https://github.com/pytorch/vision/blob/c991db82abba12e664eeac14c9b643d0f1f1a7df/torchvision/models/mobilenetv2.py#L103) implementation), the method \`_forward_impl\` will look like this:

\`\`\`python
def _forward_impl(self, x: Tensor) -> Tensor:
    # This exists since TorchScript doesn't support inheritance, so the superclass method
    # (this one) needs to have a name other than \`forward\` that can be accessed in a subclass
    x = self.features(x)
    # Cannot use "squeeze" as batch-size can be 1
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x
\`\`\`

Since the \`flatten\` operator is implemented in ONNX.js, this should work.

Now let's export the model to ONNX. We need to define and output path \`onnx_model_dir\` and some dummy input data, so that the model knows which input shape to expect:

\`\`\`python
import onnx
import onnxruntime as ort

# dummy data to define the input shape
# (batch size 1, 3 channels, height 224, width 224)
dummy_input = torch.randn(1, 3, 224, 224, device='cuda')

torch.onnx.export(
    model,
    dummy_input,
    onnx_model_dir,
    verbose=False,
    # input_names= we can name the input and output
    # output_names= layers here, if we want to
)
\`\`\`

We can check if the export worked by validating the model:

\`\`\`python
# load the ONNX model
onnx_model = onnx.load(onnx_model_dir)

# check that the model is well formed
onnx.checker.check_model(onnx_model)

# print a representation of the graph (very long)
# print(onnx.helper.printable_graph(onnx_model.graph))
\`\`\`

If everything works fine until here, we can now try to do some inference with the ONNX runtime in Python. The \`val_loader\` is the validation data loader (see [previous blog post](http://www.pascal-bliem.com/blog/classifying%20dog%20breeds%20with%20deep%20learning)).

\`\`\`python
# create an inference session based on the exported model
ort_session = ort.InferenceSession(onnx_model_dir)

# get a batch from the data loader
batch, labels = next(iter(val_loader))

# run a prediction on the first image in the batch
# (note that, since I didn't name the input or output
# layers when exporting the model, the first input
# layer was automatically named "input.1")
outputs = ort_session.run(None, {"input.1": batch.detach().numpy()[:1]})

# print the output
outputs
\`\`\`

\`\`\`
[array([[2.27155489e-10, 6.50463699e-06, 1.23810617e-09, 9.81572441e-08,
1.22788846e-08, 2.33644937e-06, 1.34210339e-07, 19075965e-10,

# [...] I cut out most entries for readability

3.75615491e-05, 2.95915470e-09, 7.11135826e-08, 79975457e-09,
6.07401551e-09]], dtype=float32)]
\`\`\`

We get back a list containing an array with predicted probabilities for each of the 121 dog breeds - everything seems to work well.

### Running inference in JavaScript

Now the real magic happens when we change from one programming language to another. Since I am planning to serve my dog classifier in a mobile app, which I'd like to write in [React Native](https://reactnative.dev/), I want to try out running it in JavaScript. Specifically, I will use the [NodeJS](https://nodejs.org/en/) runtime. Besides the JavaScript libraries [ONNX.js](https://www.npmjs.com/package/onnxjs) and [onnxjs-node](https://www.npmjs.com/package/onnxjs-node), we'll also need [NumJs](https://www.npmjs.com/package/numjs) (a JavaScript equivalent of NumPy) for processing image data. Just like I did when training the model in PyTorch, I need to normalize the image data based on the mean and standard deviation I calculated for each channel.

\`\`\`javascript
// import libraries
require("onnxjs");
require("onnxjs-node");
const nj = require("numjs");

// import a mapping from class numbers to class names
const classMap = require("./class_mapping");

const IMAGE_SIZE = 224;

// load an image and resize to IMAGE_SIZE
let img = nj.images.read("./sample_images/1.jpg");
img = nj.images.resize(img, IMAGE_SIZE, IMAGE_SIZE);

// a function that performs per-channel normalization of each pixel
function normalizePerChannel(img, channel, mean, std) {
  // extract all pixels for the given channel
  img = img
    .slice(null, null, [channel, channel + 1])
    .reshape(IMAGE_SIZE * IMAGE_SIZE);

  // make sure under-laying data type is float32
  img.selection.data = new Float32Array(img.selection.data);

  // normalize pixel values
  img = img.divide(255.0).add(-mean).divide(std);

  return img.tolist();
}

// normalize each RGB channel
const channelR = normalizePerChannel(img, 0, 0.512, 0.267);
const channelG = normalizePerChannel(img, 1, 0.489, 0.263);
const channelB = normalizePerChannel(img, 2, 0.422, 0.271);

// combine all channels into one array
const imgData = [...channelR, ...channelG, ...channelB];
\`\`\`

Now that the image data is prepared, we'll perform inference on it:

\`\`\`javascript
// open an ONNX inference session with CPU backend
const session = new onnx.InferenceSession({ backendHint: "cpu" });

session.loadModel("./ml_models/dog_classifier.onnx").then(() => {
  // build input data tensor by providing the image data,
  // data type, and input dimensions
  const inferenceInputs = [
    new onnx.Tensor(imgData, "float32", [1, 3, 224, 224]),
  ];

  // run inference on the input tensor
  session.run(inferenceInputs).then((output) => {
    const outputTensor = output.values().next().value;
    // get index of highest probability prediction
    const pred = outputTensor.data.indexOf(Math.max(...outputTensor.data));
    // log prediction to console
    console.log(\`Prediction: \${classMap[pred]}.\`);
  });
});
\`\`\`

\`\`\`
Prediction: golden_retriever
\`\`\`

Great! The model still works and classifies dog breeds correctly, without having to reimplement any part of it in JavaScript. This greatly simplifies the usage of machine learning models across different programming languages and devices.

### Conclusion

We have learned about the Open Neural Network Exchange (ONNX), which allows to transfer machine learning models into a common format that can easily be exchanged between different frameworks or runtimes. We have seen how to export a model that has been developed in Python in the PyTorch framework into the ONNX format and the use this common format to perform inference with this model in a JavaScript runtime. In the future, I'm planning to use this model and the ONNX.js runtime to build a mobile app for classifying dog breeds on photos. Stay tuned for a blog post on that topic. Thanks for reading!
`
);
