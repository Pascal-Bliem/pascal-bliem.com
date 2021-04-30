import React, { useEffect, Fragment } from "react";
import styles from "./Blog.module.scss";
import Navbar from "../layout/Navbar";
import TitleSection from "../layout/blog/TitleSection";
import Posts from "../layout/blog/Posts";
import Footer from "../layout/Footer";

const Blog = () => {
  useEffect(() => {
    document.title = "Blog - Pascal Bliem";
  }, []);

  return (
    <div className={styles.page}>
      <Navbar fullyHideHeight={200} initialFullyHide={true} />
      <TitleSection />
      <Posts />
      <Footer
        acknowledgements={
          <Fragment>
            <p>
              The banner images used on this side has kindly been provided by{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@martingarrido"
              >
                Martin Garrido
              </a>{" "}
              on{" "}
              <a className="highlight-link" href="https://unsplash.com/">
                unsplash.com
              </a>
              .
            </p>
          </Fragment>
        }
      />
    </div>
  );
};

export default Blog;
