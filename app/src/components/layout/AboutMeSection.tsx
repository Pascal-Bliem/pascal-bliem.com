import React from "react";
import styles from "./AboutMeSection.module.scss";

const AboutMeSection = () => {
  return (
    <div className={styles.background}>
      <div className={`container ${styles.sectionContainer}`}>
        <div className={`row`}>
          <div
            className={`col s12 m6 l6 center-align ${styles.imageContainer}`}
          >
            <img
              className={`center-align ${styles.meImage}`}
              src="https://avatars.githubusercontent.com/u/51260327?s=400&u=a34db99cb85b5e3dcbeb563ab3faef26746e00f7&v=4"
              alt="Pascal Bliem"
            />
          </div>
          <div className="col s12 m6 l6">
            <hr id="about-me" className={styles.sectionSeparator}></hr>
            <h4 className={styles.sectionHeader}>
              About <strong>Me</strong>
            </h4>
            <p className={styles.paragraph}>
              I am a Data Scientist at HDI, one of Germany's leading insurance
              companies, where I work on customer value models, churn
              prevention, recommendation engines, marketing support, and data
              infrastructure improvements. I have a strong interest in providing
              data-driven and action-oriented solutions to high-impact business
              problems. Besides work I enjoy building cool tech things, doing
              web and mobile development, hosting a podcast, learning new
              languages (both natural and programming ones) and making music.
            </p>

            <p className={styles.iconContainer}>
              <div className="col s12 m6">
                <button className={styles.cvButton}>
                  <a
                    href="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/website/CV_202101_Pascal+Bliem.pdf"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <span style={{ color: "#fff" }}>Download CV</span>
                  </a>
                </button>
              </div>
              <div className="col s12 m6">
                <a href="https://github.com/Pascal-Bliem">
                  <i className={`fab fa-github ${styles.contactIcon}`}></i>
                </a>
                <a href="https://www.linkedin.com/in/pascal-bliem/">
                  <i className={`fab fa-linkedin ${styles.contactIcon}`}></i>
                </a>
                <a href="https://twitter.com/BliemPascal">
                  <i
                    className={`fab fa-twitter-square ${styles.contactIcon}`}
                  ></i>
                </a>
                <a href="mailto:bliem.pascal@gmail.com">
                  <i className={`fas fa-envelope ${styles.contactIcon}`}></i>
                </a>
              </div>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutMeSection;
