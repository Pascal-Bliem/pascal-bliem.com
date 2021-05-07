import React, { useEffect } from "react";
import styles from "./DoggoSnap.module.scss";
import Navbar from "../../layout/Navbar";
import Footer from "../../layout/Footer";
import initDoggoSnapForm from "../../../utils/doggoSnapForm";

export const DoggoSnap = () => {
  useEffect(() => {
    initDoggoSnapForm();
  }, []);

  useEffect(() => {
    document.title = "Doggo Snap - Pascal Bliem";
  }, []);

  return (
    <div className={styles.page}>
      <Navbar fullyHideHeight={0} initialFullyHide={false} />
      <div className={`container`}>
        <div className="row">
          <div className="col s12 m10 offset-m1 l8 offset-l2 center-align">
            <h1 className={styles.title}>Doggo Snap</h1>
          </div>
        </div>
        <div className="row">
          <div
            className={`col l8 offset-l2 m10 offset-m1 s12 ${styles.postContentContainer}`}
          >
            <p className={styles.paragraph}>
              Doggo Snap is a mobile app for Android and iOS which can be used
              for classifying a dog's breed from an image. It provides you with
              information on the breed and let's you save the dogs you've
              photographed so that you can have a look at them later and see
              where you've met them on a map. It was created using{" "}
              <a href="https://play.google.com/store/apps/details?id=com.pascalbliem.doggosnap">
                React Native
              </a>
              . You can download it from the{" "}
              <a href="https://play.google.com/store/apps/details?id=com.pascalbliem.doggosnap">
                Play Store
              </a>{" "}
              or{" "}
              <strong>
                <a href="#try">try out</a>
              </strong>{" "}
              the classification functionality below. Unfortunately, the app is
              currently not on the Apple App Store (becaus of their audacious
              fee for a developer account) but feel free to{" "}
              <a href="/#contact">contact</a> me if you want to have the iOS
              build. I previously published a couple of resources on the
              application:
            </p>

            <ul className={styles.paragraph}>
              <li>
                <a href="https://play.google.com/store/apps/details?id=com.pascalbliem.doggosnap">
                  {">"} The Doggo Snap app on Google Play Store
                </a>
              </li>
              <li>
                <a href="https://github.com/Pascal-Bliem/doggo-snap">
                  {">"} The Doggo Snap Github repository
                </a>
              </li>
              <li>
                <a href="https://github.com/Pascal-Bliem/doggo-snap-api">
                  {">"} The Doggo Snap Classification API Github repository
                </a>
              </li>
              <li>
                <a href="/blog/the%20doggo%20snap%20mobile%20app">
                  {">"} A blog post on the Doggo Snap mobile app
                </a>
              </li>
              <li>
                <a href="/blog/classifying%20dog%20breeds%20with%20deep%20learning">
                  {">"} A blog post on the neural network used for image
                  classification
                </a>
              </li>
              <li>
                <a href="/blog/transfer%20ml%20models%20easily%20with%20onnx">
                  {">"} A blog post on using ML models cross-platform with ONNX
                </a>
              </li>
            </ul>

            <h5 className={styles.paragraph}>How it works</h5>
            <p className={styles.paragraph}>
              The dog breed recognition from photos is done by a Deep Learning
              model, namely a slightly modified{" "}
              <a href="https://arxiv.org/abs/1801.04381">MobileNetV2</a>{" "}
              convolutional neural network build with{" "}
              <a href="https://pytorch.org/">PyTorch</a>, converted to and run
              by <a href="https://onnx.ai/">ONNX</a>. The client mobile app
              sends images via HTTP to the API and receives the classification
              as a response. No user data is saved on a server (except in logs),
              all saved dogs are persisted in an on-device{" "}
              <a href="https://www.sqlight.org/">SQLite</a> database, and the
              app state is managed with{" "}
              <a href="https://redux.js.org/">Redux</a>. You can find a
              graphical overview of the app's functionality below. Please also
              check out the blog posts linked above for a more detailed
              description on the app, the machine learning, and their interplay.
            </p>

            <div className={styles.imageContainer}>
              <img
                src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/doggo-snap/diagramm.png"
                alt="An overview of the Doggo Snap App."
              />
            </div>

            <h5 id="try" className={styles.paragraph}>
              Try it out
            </h5>
            <p className={styles.paragraph}>
              Upload an image of a dog below, post it to the Doggo Snap API, and
              get back the dog's breed! The API goes into hibernation after
              being inactive for a while, so it may take a few seconds to wake
              up again.
            </p>

            <p className={styles.paragraph}>
              API status:
              <span
                id="doggosnap-app-status"
                className={styles.doggosnapAppStatus}
              >
                {" "}
                waking up
              </span>
            </p>
            <p className={styles.paragraph}>
              <label htmlFor="upload-img">Upload an image:</label>
              <br />
              <br />
              <input
                type="file"
                id="upload-img"
                name="upload-img"
                accept="image/*"
              />
              <br />
              <small
                id="upload-img-error"
                className={`${styles.error} ${styles.hidden}`}
              >
                ERROR MESSAGE
              </small>
            </p>
            <img
              className={`${styles.dogImg} ${styles.hidden}`}
              id="dog-img"
              src="#"
              alt="selected dog"
              crossOrigin="anonymous"
            />

            <div className={`${styles.hidden}`} id="classification-result">
              <p>Your dog is a:</p>
            </div>
            <small
              id="classification-error"
              className={`${styles.error} ${styles.hidden}`}
            >
              ERROR MESSAGE
            </small>
          </div>
        </div>
      </div>
      <Footer acknowledgements={<div />} />
    </div>
  );
};

export default DoggoSnap;
