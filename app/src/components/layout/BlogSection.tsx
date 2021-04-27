import React from "react";
import styles from "./BlogSection.module.scss";

const BlogSection = () => {
  return (
    <div className={styles.background}>
      <div className={`container ${styles.sectionContainer}`}>
        <hr id="blog" className={styles.sectionSeparator}></hr>
        <h4 className={styles.sectionHeader}>
          My <strong>Blog</strong>
        </h4>
        <div className={`row`}>This is gonna be the Blog section.</div>
        <p className={styles.sectionFooter}>
          <a href="#!">
            Visit the full Blog here <i className="fas fa-pencil-alt"></i>
          </a>
        </p>
      </div>
    </div>
  );
};

export default BlogSection;
