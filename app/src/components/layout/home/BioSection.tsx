import React from "react";
import styles from "./BioSection.module.scss";

const BioSection = () => {
  return (
    <div className={styles.background}>
      <div className={`container ${styles.sectionContainer}`}>
        <hr id="bio" className={styles.sectionSeparator}></hr>
        <h4 className={styles.sectionHeader}>
          My short <strong>Bio</strong>
        </h4>
        <div className={`row`}>
          <div className="col s12 m12 l6">
            <p className={styles.paragraph}>
              Recently I've pursued a personal goal of mine and invested time to
              learn Spanish and Chinese, and have also relocated to Taipei. I'm
              currently open to new professional opportunities in Taiwan.
            </p>
            <p className={styles.paragraph}>
              In my latest role I worked as a Machine Learning Engineer / Data
              Scientist and Full-Stack Engineer at{" "}
              <a className="highlight-link" href="https://www.peregrine.ai/">
                Peregrine Technologies
              </a>
              , a startup providing AI-powered traffic video analytics to make
              traffic safe and ecological. There, I work on computer vision
              tasks with deep machine learning, digital sensor data processing,
              scalable data processing pipelines in the cloud, and generally,
              full-stack software development. Besides my main job I also do web
              & mobile development used to co-host a podcast.
            </p>
            <p className={styles.paragraph}>
              Previously, I was a Data Scientist at{" "}
              <a className="highlight-link" href="https://www.hdi.de/">
                HDI
              </a>
              , one of Germany's leading insurance companies, where I worked on
              several customer-centric use cases such as customer value models,
              churn prevention, recommendation engines, marketing support, and
              data infrastructure improvements.
            </p>
            <p className={styles.paragraph}>
              Between my graduation and starting my first job outside academia,
              I spent over a year backpacking through 10 southeast Asian
              countries, staying significant portions of the time in Vietnam and
              Indonesia, where I volunteered as a teacher. During my travels, I
              got the chance to work on several{" "}
              <a
                className="highlight-link"
                href="https://github.com/Pascal-Bliem?tab=repositories"
              >
                personal projects
              </a>{" "}
              to sharpen my data science and coding skills, as I was
              transitioning to the field from a background of academic research.
            </p>
            <p className={styles.paragraph}>
              2018 I graduated summa cum laude with a master degree in materials
              engineering from RWTH Aachen University, where I had also been
              working in{" "}
              <a
                className="highlight-link"
                href="https://www.mch.rwth-aachen.de/"
              >
                materials chemistry
              </a>{" "}
              and{" "}
              <a
                className="highlight-link"
                href="https://www.institut-1a.physik.rwth-aachen.de/"
              >
                physics
              </a>{" "}
              research, supplemented by research stays at{" "}
              <a
                className="highlight-link"
                href="https://www.physics.uu.se/research/materials-physics+/"
              >
                Uppsala University
              </a>
              .
            </p>
            <p className={styles.paragraph}>
              My{" "}
              <a
                className="highlight-link"
                href="https://github.com/Pascal-Bliem/my-papers-and-theses#peer-reviewed-publications"
              >
                research
              </a>{" "}
              focused mostly on experimental and computational investigation of
              synthesis and characterization of nano-scaled materials and thin
              films for applications in energy generation, electronics, and wear
              resistance.
            </p>
          </div>
          <div className={`col s12 m12 l6 ${styles.bioListContainer}`}>
            <ul className={styles.bioList}>
              <li>
                <span className={styles.bioYear}>2024:</span>
                <br />
                Language Learning & Relocation to Taiwan
              </li>
              <li>
                <br />
                <span className={styles.bioYear}>2021 - 2023:</span>
                <br />
                Machine Learning Engineer at Peregrine Technologies
              </li>
              <li>
                <br />
                <span className={styles.bioYear}>2020 - 2021:</span>
                <br />
                Data Scientist at HDI Group
              </li>
              <li>
                <br />
                <span className={styles.bioYear}>2018 - 2019:</span>
                <br />
                Backpacking & Volunteer Teaching in South East Asia
              </li>
              <li>
                <br />
                <span className={styles.bioYear}>2013 - 2018:</span>
                <br />
                Materials Science Researcher at RWTH Aachen University{" "}
              </li>
              <li>
                <br />
                <span className={styles.bioYear}>2015 - 2018:</span>
                <br />
                Master in Materials Engineering at RWTH Aachen University
              </li>
              <li>
                <br />
                <span className={styles.bioYear}>2012 - 2015:</span>
                <br />
                Bachelor in Industrial Engineering at RWTH Aachen University
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BioSection;
