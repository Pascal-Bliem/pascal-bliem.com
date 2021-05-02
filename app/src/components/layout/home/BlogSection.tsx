import React, { useEffect } from "react";
//@ts-ignore
import M from "materialize-css/dist/js/materialize.min.js";
import styles from "./BlogSection.module.scss";
import posts from "../../../assets/posts/posts";

const BlogSection = () => {
  useEffect(() => {
    document.addEventListener("DOMContentLoaded", function () {
      const elements = document.querySelectorAll(".carousel");
      M.Carousel.init(elements, {
        numVisible: 7,
        indicators: true,
      });
    });

    var carousel = document.getElementById("carousel") as HTMLElement;

    function moveNextCarousel() {
      var el = document.getElementById("carousel");
      var moveRight = M.Carousel.getInstance(el);
      moveRight.next(1);
    }

    const duration = 2500;
    let interval = setInterval(moveNextCarousel, duration);

    carousel.addEventListener("mouseenter", () => clearInterval(interval));
    carousel.addEventListener("mouseleave", () => {
      interval = setInterval(moveNextCarousel, duration);
    });

    return () => {
      clearInterval(interval);
      carousel.removeEventListener("mouseenter", () => clearInterval(interval));
      carousel.removeEventListener("mouseleave", () => {
        interval = setInterval(moveNextCarousel, duration);
      });
    };
  }, []);

  return (
    <div className={styles.background}>
      <div className={`container ${styles.sectionContainer}`}>
        <hr id="blog" className={styles.sectionSeparator}></hr>
        <h4 className={styles.sectionHeader}>
          My <strong>Blog</strong>
        </h4>
        <div className={`row`}>
          <div id="carousel" className={`carousel ${styles.carouselContainer}`}>
            {posts.map((post) => {
              return (
                <a
                  key={post.title}
                  className="carousel-item"
                  href={`/blog/${post.title.toLowerCase()}`}
                >
                  <div
                    className={`card small hoverable ${styles.cardContainer}`}
                  >
                    <div className={`card-image`}>
                      <div className={styles.cardImageGrad}>
                        <img
                          className={styles.cardImage}
                          src={post.titleImageUrl}
                          alt={post.title}
                        />
                      </div>
                      <span className={`card-title`}>
                        <strong className={styles.cardTitle}>
                          {post.title}
                        </strong>
                      </span>
                    </div>
                    <div className="card-content">
                      <p className={styles.cardText}>{post.subtitle}</p>
                      <p className={styles.cardPublishDate}>
                        Published: {post.publishDate.toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                </a>
              );
            })}
          </div>
        </div>
        <p className={styles.sectionFooter}>
          <a href="/blog">
            Visit the full Blog here <i className="fas fa-pencil-alt"></i>
          </a>
        </p>
      </div>
    </div>
  );
};

export default BlogSection;
