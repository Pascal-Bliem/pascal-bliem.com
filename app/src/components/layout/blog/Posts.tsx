import React, { useState } from "react";
import styles from "./Posts.module.scss";
import posts from "../../../assets/posts/posts";
import { TagsType } from "../../../assets/posts/postModel";
import marked from "marked";
import _ from "lodash";

const Posts = () => {
  const [tags, setTags] = useState<TagsType[]>([]);

  const tagHandler = (tag: TagsType) => {
    if (tags.includes(tag)) {
      setTags((prev) => prev.filter((el) => el !== tag));
    } else {
      setTags((prev) => [...prev, tag]);
    }
    const tagButton = document.getElementById(_.kebabCase(tag)) as HTMLElement;
    tagButton.classList.toggle(styles.tagActive);
  };

  return (
    <div className="container">
      <div className="row">
        <div className={`col s12 ${styles.tagContainer}`}>
          {/* Wrap the tag buttons in columns so that they align nicely on mobile */}
          Filter:
          <br className={styles.tagLineBreak} />
          <button
            id="data-science-ai-ml"
            onClick={() => tagHandler("Data Science & AI/ML")}
            className={styles.tagButton}
          >
            Data Science & AI/ML
          </button>
          <button
            id="web-development"
            onClick={() => tagHandler("Web Development")}
            className={styles.tagButton}
          >
            Web Development
          </button>
          <button
            id="learning"
            onClick={() => tagHandler("Learning")}
            className={styles.tagButton}
          >
            Learning
          </button>
          <button
            id="non-tech"
            onClick={() => tagHandler("Non-Tech")}
            className={styles.tagButton}
          >
            Non-Tech
          </button>
        </div>
      </div>

      {posts
        .filter((post) => {
          if (tags.length === 0) {
            return true;
          } else {
            return tags.every((tag: TagsType) => post.tags.includes(tag));
          }
        })
        .sort((a, b) => {
          return b.publishDate.getTime() - a.publishDate.getTime();
        })
        .map((post) => {
          return (
            <div
              key={_.kebabCase(post.title)}
              className={`row ${styles.postContainer}`}
            >
              <a
                href={`blog/${_.lowerCase(post.title)}`}
                style={{ display: "block" }}
              >
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

                <div className={`col s12 m12 l6 ${styles.imageContainer}`}>
                  <img
                    className={styles.postImage}
                    src={post.titleImageUrl}
                    alt={post.titleImageDescription}
                  />
                </div>
                <div
                  className={`col s12 m12 l6 ${styles.textPreviewContainer}`}
                  dangerouslySetInnerHTML={{ __html: marked(post.content) }}
                />
                <div className={`row ${styles.bottomRow}`}>
                  <div className={`col s12 `}>
                    Read more
                    <hr className={styles.bottomLine} />
                  </div>
                </div>
              </a>
            </div>
          );
        })}
    </div>
  );
};

export default Posts;
