import React from "react";
import styles from "./TitleSectionNavLinks.module.scss";

const TitleSectionNavLinks = () => {
  return (
    <div className="row">
      <div className={`col s12 ${styles.navLinkContainer}`}>
        <a className={styles.navLink} href="/#projects">
          Projects
        </a>
        <a className={styles.navLink} href="/#about-me">
          About me
        </a>
        <a className={styles.navLink} href="/#skills">
          Skills
        </a>
        <a className={styles.navLink} href="/blog">
          Blog
        </a>
        <a className={styles.navLink} href="/#contact">
          Contact
        </a>
        <a className={styles.navLink} href="/#bio">
          Bio
        </a>
      </div>
    </div>
  );
};

export default TitleSectionNavLinks;
