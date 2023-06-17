import React, { useEffect } from "react";
import styles from "./ToxBlock.module.scss";
import Navbar from "../../layout/Navbar";
import Footer from "../../layout/Footer";
import initToxblockForm from "../../../utils/toxblockForm";

export interface ChartBarProps {
  barLabel: string;
  barId: string;
  barNumberId: string;
}

const ChartBar = ({ barLabel, barId, barNumberId }: ChartBarProps) => {
  return (
    <div className="row">
      <div className="col l2 s12">
        <p className={styles.toxblockBarLabel}>{barLabel}</p>
      </div>
      <div className="col l10 s12">
        <div className={styles.toxblockBar} id={barId}></div>
        <p className={styles.toxblockBarNumber} id={barNumberId}>
          0%
        </p>
      </div>
    </div>
  );
};

const ToxBlock = () => {
  useEffect(() => {
    initToxblockForm();
  }, []);

  useEffect(() => {
    document.title = "ToxBlock - Pascal Bliem";
  }, []);

  return (
    <div className={styles.page}>
      <Navbar fullyHideHeight={0} initialFullyHide={false} />
      <div className={`container`}>
        <div className="row">
          <div className="col s12 m10 offset-m1 l8 offset-l2 center-align">
            <h1 className={styles.toxblockTitle}>
              <span className={styles.toxSpan}>Tox</span>
              <span className={styles.blockSpan}>Block </span>
            </h1>
          </div>
        </div>
        <div className="row">
          <div
            className={`col l8 offset-l2 m10 offset-m1 s12 ${styles.postContentContainer}`}
          >
            <p className="">
              ToxBlock is a machine learning application for recognizing toxic
              language in text. It can potentially be employed for automatically
              screening text in articles, posts, and comments on social media,
              digital news, online forums etc. and blocking it or flagging it
              for further review by human intelligence. It can predict
              probabilities for classifying English text into six categories of
              verbal toxicity: toxic, severe toxic, obscene, threat, insult, and
              identity hate. You can{" "}
              <strong>
                <a href="#try">try it out</a>
              </strong>{" "}
              below.
            </p>
            <p>
              The underlying ToxBlock application is a machine learning package
              written in Python, which is being served as a containerized REST
              API to which HTTP requests with potentially toxic text can be send
              over the internet. I previously published a couple of resources on
              the application:
            </p>
            <ul>
              <li>
                <a href="https://github.com/Pascal-Bliem/tox-block">
                  {">"} The ToxBlock Github repository
                </a>
              </li>
              <li>
                <a href="https://github.com/Pascal-Bliem/tox-block-api">
                  {">"} The ToxBlock API Github repository
                </a>
              </li>
              <li>
                <a href="https://pypi.org/project/tox-block/">
                  {">"} ToxBlock on the Python package index (PyPI)
                </a>
              </li>
              <li>
                <a href="/blog/tox%20block%20using%20ai%20to%20keep%20discussions%20clean">
                  {">"} A blog post on the ToxBlock package
                </a>
              </li>
              <li>
                <a href="/blog/the%20tox%20block%20api%20bring%20ml%20models%20into%20action%20by%20serving%20them%20on%20the%20web">
                  {">"} A blog post on the ToxBlock API
                </a>
              </li>
            </ul>

            <h3>How it works</h3>
            <p>
              I got into deeper detail in the afore mentioned blog posts, but to
              sum it up shortly, the workflow is as follows: The input data is
              send in JSON format via a HTTP POST request to the API app, which
              is hosted on a cloud instance at{" "}
              <a href="https://www.render.com/">Render</a>. The app is deployed
              as a <a href="https://www.docker.com/">Docker</a> container, which
              contains a webserver running the{" "}
              <a href="https://flask.palletsprojects.com/en/1.1.x/">Flask</a>{" "}
              micro web framework. Flask provides the routes to the API
              endpoints, through which the actual functionality of ToxBlock is
              served.
            </p>
            <img
              className={styles.toxblockImg}
              src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/tox-block/workflow.png"
              alt="ToxBlock workflow"
            />
            <p>
              Inside of the ToxBlock package is where the real action happens:
              the text input is encoded numerically and embedded using{" "}
              <a href="https://nlp.stanford.edu/projects/glove/">pretrained</a>{" "}
              word vectors, and then fed into a recurrent neural network, which
              has been trained on about 250000{" "}
              <a href="https://www.kaggle.com/c/jigsaw-toxic-comment-classNameification-challenge/data">
                text samples
              </a>
              . The neural network predicts the probabilities of a given input
              text belonging to any of the six categories of verbal toxicity,
              and finally, the prediction is returned in JSON format via the
              HTTP response.
              <div id="try" />
            </p>

            <h3>Try it out</h3>
            <p>
              Type an input text into the text field below, post it to the
              ToxBlock API, and get back the toxicity classification. The API
              goes into hibernation after being inactive for a while, so it may
              take a few seconds to wake up again.
            </p>
            <form id="toxblock-form" action="" method="POST">
              <p>
                API status:
                <span
                  id="toxblock-app-status"
                  className={styles.toxblockAppStatus}
                >
                  {" "}
                  waking up
                </span>
              </p>
              <div className={styles.toxblockFormControl}>
                <label htmlFor="toxblock-input">Input Text</label>
                <textarea
                  name="toxblock-input"
                  className={styles.toxblockInput}
                  id="toxblock-input"
                  cols={30}
                  rows={5}
                  placeholder="Type something nasty..."
                ></textarea>
                <small>ERROR MESSAGE</small>
              </div>
              <div className={styles.toxblockPostButtonDiv}>
                <button
                  type="submit"
                  className={`btn waves-effect ${styles.toxblockPostButton}`}
                >
                  POST
                </button>
              </div>
            </form>
            <div id="toxblock-chart" className={styles.toxblockChart}>
              <ChartBar
                barLabel="Toxic"
                barId={styles.toxicBar}
                barNumberId="toxic-bar-number"
              />
              <ChartBar
                barLabel="Severe toxic"
                barId={styles.severeToxicBar}
                barNumberId="severe-toxic-bar-number"
              />
              <ChartBar
                barLabel="Obscene"
                barId={styles.obsceneBar}
                barNumberId="obscene-bar-number"
              />
              <ChartBar
                barLabel="Insult"
                barId={styles.insultBar}
                barNumberId="insult-bar-number"
              />
              <ChartBar
                barLabel="Threat"
                barId={styles.threatBar}
                barNumberId="threat-bar-number"
              />
              <ChartBar
                barLabel="Identity hate"
                barId={styles.identityHateBar}
                barNumberId="identity-hate-bar-number"
              />
            </div>
          </div>
          <script src="../scripts/toxblockForm.js"></script>
        </div>
      </div>
      <Footer acknowledgements={<div />} />
    </div>
  );
};

export default ToxBlock;
