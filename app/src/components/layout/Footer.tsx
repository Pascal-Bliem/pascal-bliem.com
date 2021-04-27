import React from "react";
import styles from "./Footer.module.scss";

const Footer = () => {
  return (
    <div className={styles.background}>
      <div className={`container ${styles.sectionContainer}`}>
        <div className={styles.sectionFooter}>
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
            <a className="highlight-link" href="https://unsplash.com/@dre0316">
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
            <a className="highlight-link" href="https://unsplash.com/@olafval">
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
          <p>© {new Date().getFullYear()} Pascal Bliem</p>
        </div>
      </div>
    </div>
  );
};

export default Footer;
