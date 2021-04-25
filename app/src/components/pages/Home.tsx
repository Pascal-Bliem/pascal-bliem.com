import React from "react";
import styles from "./Home.module.scss";
import Navbar from "../layout/Navbar";
import TitleSection from "../layout/TitleSection";
import ProjectSection from "../layout/ProjectSection";

const Home = () => {
  return (
    <div className={styles.page}>
      <Navbar />
      <TitleSection />
      <ProjectSection />
      <h1 id="about-me" style={{ marginTop: "10rem" }}>
        About me Section
      </h1>
      <h1 id="projects" style={{ marginTop: "10rem" }}>
        Portfolio Project Section
      </h1>
      <h1 id="skills" style={{ marginTop: "10rem" }}>
        Skills Section
      </h1>
      <h1 id="contact" style={{ marginTop: "10rem" }}>
        Contact Section
      </h1>
      <h1 id="bio" style={{ marginTop: "10rem" }}>
        Short Bio Section
      </h1>
    </div>
  );
};

export default Home;
