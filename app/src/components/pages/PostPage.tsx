import React, { Fragment, useEffect } from "react";
import styles from "./PostPage.module.scss";
import { useParams } from "react-router-dom";
import Navbar from "../layout/Navbar";
import Footer from "../layout/Footer";
import marked from "marked";
import _ from "lodash";
import posts from "../../assets/posts/posts";
import createTOC from "../../utils/postTOC";

export interface PostPageProps {
  postName: string;
}

const PostPage = () => {
  const { postName }: PostPageProps = useParams();

  const post = posts.find(
    (post) => _.lowerCase(post.title) === _.lowerCase(postName)
  );

  useEffect(() => {
    document.title = post
      ? post.title + " - Pascal Bliem"
      : "Blog - Pascal Bliem";
  }, [post]);

  // this useEffect is auto-captioning the images from their alt text
  useEffect(() => {
    const images = document.querySelectorAll(".container img");
    let L = images.length;
    const fig = document.createElement("figure");
    let which: Element;
    let temp: Node;

    while (L) {
      temp = fig.cloneNode(false);
      which = images[--L];
      const caption = which.getAttribute("alt");
      //@ts-ignore
      which.parentNode.insertBefore(temp, which);
      const content = document.createElement("figcaption");
      content.innerHTML = caption ? caption : "";
      temp.appendChild(which);
      temp.appendChild(content);
    }
  }, []);

  // this useEffect calls a util function which adds the table of contents (TOC)
  useEffect(() => {
    if (post) {
      const { maxWidthHandler, scrollHideHandler, scrollCurrentHandler } =
        createTOC();

      return () => {
        window.removeEventListener("resize", maxWidthHandler);
        window.removeEventListener("scroll", scrollHideHandler);
        window.removeEventListener("scroll", scrollCurrentHandler);
      };
    }
  }, [post]);

  // this useEffect adds the scripts for MathJax
  useEffect(() => {
    const mathJaxCdn = document.createElement("script");
    mathJaxCdn.type = "text/javascript";
    mathJaxCdn.src =
      "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML";
    mathJaxCdn.async = false;
    document.body.appendChild(mathJaxCdn);

    const mathJaxConfig = document.createElement("script");
    mathJaxConfig.type = "text/x-mathjax-config";
    mathJaxConfig.text = `MathJax.Hub.Config({
      tex2jax: {
        inlineMath: [ ['$','$'], ],
        processEscapes: true
      }
    });`;
    document.body.appendChild(mathJaxConfig);
  }, []);

  return (
    <div className={styles.page}>
      {post ? (
        <Fragment>
          <Navbar fullyHideHeight={0} />
          <div className={`container ${styles.postContainer}`}>
            <div className={`row ${styles.titleRow}`}>
              <div className="col s12">
                <h4 className={styles.postTitle}>{post.title}</h4>
                <h5 className={styles.postSubtitle}>{post.subtitle}</h5>
                <p className={styles.publishDate}>
                  Published: {post.publishDate.toLocaleDateString()} | Tags:{" "}
                  {post.tags.map((tag) => (
                    <span key={tag} className={styles.postTag}>
                      {tag}
                    </span>
                  ))}
                </p>
              </div>
            </div>
            <div className={`row ${styles.titleRow}`}>
              <div className={`col s12 m12 l12 ${styles.imageContainer}`}>
                <img
                  className={styles.postImage}
                  src={post.titleImageUrl}
                  alt={post.titleImageDescription}
                />
              </div>
            </div>
            <div className={`row ${styles.titleRow}`}>
              <div
                className={`col s12 m12 l12 ${styles.postContentContainer}`}
                dangerouslySetInnerHTML={{ __html: marked(post.content) }}
              />
            </div>
          </div>
          <Footer acknowledgements={<div />} />
        </Fragment>
      ) : (
        <h3 style={{ textAlign: "center", marginTop: "3rem" }}>
          404 - Blog post not found - <a href="/blog">go back to blog page</a>.
        </h3>
      )}
    </div>
  );
};

export default PostPage;
