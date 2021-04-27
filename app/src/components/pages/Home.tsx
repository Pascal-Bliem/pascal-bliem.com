import React from "react";
import styles from "./Home.module.scss";
import Navbar from "../layout/Navbar";
import TitleSection from "../layout/TitleSection";
import ProjectSection from "../layout/ProjectSection";
import AboutMeSection from "../layout/AboutMeSection";
import SkillsSection from "../layout/SkillsSection";
import BlogSection from "../layout/BlogSection";
import ContactSection from "../layout/ContactSection";
import BioSection from "../layout/BioSection";
import Footer from "../layout/Footer";

const Home = () => {
  return (
    <div className={styles.page}>
      <Navbar />
      <TitleSection />
      <ProjectSection />
      <AboutMeSection />
      <SkillsSection />
      <BlogSection />
      <ContactSection />
      <BioSection />
      <Footer />
    </div>
  );
};

export default Home;
