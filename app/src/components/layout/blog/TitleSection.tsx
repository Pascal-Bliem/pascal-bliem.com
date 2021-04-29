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
            My <strong>Blog</strong>
          </h1>
          <h3 className={styles.titleSubheading}>
            where I write about data science,
          </h3>
          <h3 className={styles.titleSubheading}>
            software development, and other topics.
          </h3>
        </div>
      </div>
    </section>
  );
};

export default TitleSection;
