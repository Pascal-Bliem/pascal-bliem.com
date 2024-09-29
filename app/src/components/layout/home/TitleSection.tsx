import React from "react";
import styles from "./TitleSection.module.scss";
import TitleSectionNavLinks from "../TitleSectionNavLinks";

const TitleSection = () => {
  return (
    <section className={styles.titleSection}>
      <TitleSectionNavLinks />
      <div className="row">
        <div className={"col s12 " + styles.titleContainer}>
          <h1 className={styles.titleHeading}>
            Hi, I'm <strong>Pascal</strong>
          </h1>
          <h3 className={styles.titleSubheading} id={styles.tsh1}>
            {"AI/ML & Data Science"}
          </h3>
          <h3 className={styles.titleSubheading} id={styles.tsh2}>
            {"Full Stack Web-Dev"}
          </h3>
          <h3 className={styles.titleSubheading} id={styles.tsh3}>
            {"Cloud Infrastructure"}
          </h3>
          <button
            className={styles.moreButton}
            onClick={() => (window.location.href = "/#about-me")}
          >
            More on me
          </button>
        </div>
      </div>
    </section>
  );
};

export default TitleSection;
