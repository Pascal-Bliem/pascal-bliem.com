import React, { useEffect } from "react";
import styles from "./Diarysta.module.scss";
import Navbar from "../../layout/Navbar";
import Footer from "../../layout/Footer";

const Diarysta = () => {
  useEffect(() => {
    document.title = "Diarysta - Pascal Bliem";
  }, []);

  return (
    <div className={styles.page}>
      <Navbar fullyHideHeight={0} initialFullyHide={false} />
      <div className={`container`}>
        <div className="row">
          <div className="col s12 m10 offset-m1 l8 offset-l2 center-align">
            <h1 className={styles.diarystaTitle}>Diarysta</h1>
          </div>
        </div>
        <div className="row">
          <div
            className={`col l8 offset-l2 m10 offset-m1 s12 ${styles.postContentContainer}`}
          >
            <p className="">
              Diarysta is a diary web application that let's you track your
              daily moods and activities and get a graphical summary of your
              personal diary-stats. You can create diary entries in which you
              specify your mood by selecting the corresponding emoji, pick from
              a selection of activities and write a personal note with the
              entry. You can then have a look at the stats page, where a couple
              of charts show you how your mood developed over a specified span
              of time, what activities you've done when, and how they correlate
              with your mood. You can select one of three languages in the menu:
              English, German, and Indonesian. Feel free to check out a demo{" "}
              <strong>
                <a href="https://diarysta.herokuapp.com/">here</a>
              </strong>
              . Note that the hosting instance hibernates after long inactivity,
              and it may take a few seconds for it to wake up and make the app
              available.
            </p>
            <div className={styles.diarystaImg}>
              <img
                src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/diarysta-frontend/entries.png"
                alt="Read and search your diary entries."
              />
            </div>
            <p>
              Diarysta is a MERN stack app, meaning that it is build with{" "}
              <a href="https://www.mongodb.com/">
                <strong>M</strong>ongoDB
              </a>
              ,{" "}
              <a href="https://expressjs.com/">
                <strong>E</strong>xpress
              </a>
              ,{" "}
              <a href="https://reactjs.org/">
                <strong>R</strong>eact
              </a>
              , and{" "}
              <a href="https://nodejs.org/">
                <strong>N</strong>ode.js
              </a>
              . It is also a single-page application that handles routing on the
              frontend and just fetches data on users and diary entries via API
              calls. This allows the backend to be simple,{" "}
              <a href="https://en.wikipedia.org/wiki/Representational_state_transfer">
                RESTful
              </a>
              , and to agnostically serve multiple different web or mobile
              frontends. I've described the design and functionality in more
              depth in two blog post on the backend and frontend of the app and
              it's fully open source, so you can have a look at its Github
              repository.
            </p>
            <ul>
              <li>
                <a href="https://github.com/Pascal-Bliem/diarysta">
                  {">"} The Diarysta Github repository
                </a>
              </li>
              <li>
                <a href="/blog/the%20diarysta%20backend">
                  {">"} A blog post on the Diarysta backend
                </a>
              </li>
              <li>
                <a href="/blog/the%20diarysta%20frontend">
                  {">"} A blog post on the Diarysta frontend
                </a>
              </li>
              <li>
                <a href="https://diarysta.herokuapp.com/">
                  {">"} A demo of Diarysta
                </a>
              </li>
            </ul>
          </div>
        </div>
      </div>
      <Footer acknowledgements={<div />} />
    </div>
  );
};

export default Diarysta;
