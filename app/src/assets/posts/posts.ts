import marked from "marked";
import markdownPath from "raw-loader!./markdown/test.md";

fetch(markdownPath)
  .then((res) => res.text())
  .then((text) => console.log(marked(text)));

export default [{}];
