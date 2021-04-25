import React from "react";
import styles from "./TitleSection.module.scss";

const TitleSection = () => {
  return (
    <section className={styles.titleSection}>
      <div className="container">
        <div className="row">
          <div className={`col s12 ${styles.navLinkContainer}`}>
            <a className={styles.navLink} href="/#projects">
              Projects
            </a>
            <a className={styles.navLink} href="/#about-me">
              About me
            </a>
            <a className={styles.navLink} href="/#contact">
              Contact
            </a>
          </div>
        </div>
        <div className="row">
          <div className={"col s12 " + styles.titleContainer}>
            <h1 className={styles.titleHeading}>
              Hi, I'm <strong>Pascal</strong>
            </h1>
            <h3 className={styles.titleSubheading} id={styles.tsh1}>
              {"Data Scientist"}
            </h3>
            <h3 className={styles.titleSubheading} id={styles.tsh2}>
              {"ML Engineer"}
            </h3>
            <h3 className={styles.titleSubheading} id={styles.tsh3}>
              {"Web Developer"}
            </h3>
            <button
              className={styles.moreButton}
              onClick={() => (window.location.href = "/#about-me")}
            >
              More on me
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default TitleSection;
