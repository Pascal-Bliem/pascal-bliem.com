import React, { useEffect, Fragment } from "react";
import styles from "./Home.module.scss";
import Navbar from "../layout/Navbar";
import TitleSection from "../layout/home/TitleSection";
import ProjectSection from "../layout/home/ProjectSection";
import AboutMeSection from "../layout/home/AboutMeSection";
import SkillsSection from "../layout/home/SkillsSection";
import BlogSection from "../layout/home/BlogSection";
import ContactSection from "../layout/home/ContactSection";
import BioSection from "../layout/home/BioSection";
import Footer from "../layout/Footer";

const Home = () => {
  useEffect(() => {
    document.title = "Pascal Bliem";
  });

  return (
    <div className={styles.page}>
      <Navbar fullyHideHeight={250} initialFullyHide={true} />
      <TitleSection />
      <ProjectSection />
      <AboutMeSection />
      <SkillsSection />
      <BlogSection />
      <ContactSection />
      <BioSection />
      <Footer
        acknowledgements={
          <Fragment>
            <p>
              Images used on this side have kindly been provided by{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@markusspiske"
              >
                Markus Spiske
              </a>
              ,{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@brett_jordan"
              >
                Brett Jordan
              </a>
              ,{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@dre0316"
              >
                Andre Hunter
              </a>
              ,{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@dailykairos"
              >
                Peter Jones
              </a>
              ,{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@isaacmsmith"
              >
                Isaac Smith
              </a>
              ,{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@willfrancis"
              >
                Will Francis
              </a>
              ,{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@lukechesser"
              >
                Luke Chesser
              </a>
              ,{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@tetrakiss"
              >
                Arseny Togulev
              </a>
              ,{" "}
              <a className="highlight-link" href="https://unsplash.com/@tvick">
                Taylor Vick
              </a>
              ,{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@olafval"
              >
                Olaf Val
              </a>
              ,{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@neonbrand"
              >
                NeONBRAND
              </a>
              ,{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@hannahwrightdesigner"
              >
                Hannah Wright
              </a>
              ,{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@jefflssantos"
              >
                Jefferson Santos
              </a>
              , and{" "}
              <a
                className="highlight-link"
                href="https://unsplash.com/@thisisengineering"
              >
                ThisisEngineering RAEng
              </a>{" "}
              on{" "}
              <a className="highlight-link" href="https://unsplash.com/">
                unsplash.com
              </a>
              .
            </p>
            <p>
              Some of the icons are kindly provided by{" "}
              <a
                className="highlight-link"
                href="https://www.flaticon.com/authors/eucalyp"
                title="Eucalyp"
              >
                Eucalyp
              </a>{" "}
              from{" "}
              <a
                className="highlight-link"
                href="https://www.flaticon.com/"
                title="Flaticon"
              >
                flaticon.com
              </a>
            </p>
          </Fragment>
        }
      />
    </div>
  );
};

export default Home;
