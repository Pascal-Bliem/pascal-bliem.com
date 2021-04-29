import Post from "./postModel";

const posts: Post[] = [
  require("./postObjects/doggoSnap").default,
  require("./postObjects/onnx").default,
  require("./postObjects/dogClassifier").default,
];

export default posts;
