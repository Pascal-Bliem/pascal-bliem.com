import Post from "./postModel";

const posts: Post[] = [
  require("./postObjects/decisionTrees").default,
  require("./postObjects/languageLearning").default,
  require("./postObjects/activityRecognition").default,
  require("./postObjects/maskRCNN").default,
  require("./postObjects/yolo").default,
  require("./postObjects/objectDetectionMetrics").default,
  require("./postObjects/feedback").default,
  require("./postObjects/doggoSnap").default,
  require("./postObjects/onnx").default,
  require("./postObjects/dogClassifier").default,
  require("./postObjects/headUp").default,
  require("./postObjects/mlNlp").default,
  require("./postObjects/diarystaFront").default,
  require("./postObjects/diarystaBack").default,
  require("./postObjects/chatBot").default,
  require("./postObjects/toxBlockAPI").default,
  require("./postObjects/toxBlock").default,
  require("./postObjects/chineseOCR").default,
  require("./postObjects/quarterLife").default,
  require("./postObjects/mlDatasets").default,
  require("./postObjects/bikeRental").default,
  require("./postObjects/errorRate").default,
  require("./postObjects/socialSurvey").default,
  require("./postObjects/socialNetwork").default,
];

export default posts;
