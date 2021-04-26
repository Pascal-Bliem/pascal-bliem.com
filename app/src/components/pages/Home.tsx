import React from "react";
import styles from "./Home.module.scss";
import Navbar from "../layout/Navbar";
import TitleSection from "../layout/TitleSection";
import ProjectSection from "../layout/ProjectSection";
import AboutMeSection from "../layout/AboutMeSection";
import SkillsSection from "../layout/SkillsSection";

const Home = () => {
  return (
    <div className={styles.page}>
      <Navbar />
      <TitleSection />
      <ProjectSection />
      <AboutMeSection />
      <SkillsSection />

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
