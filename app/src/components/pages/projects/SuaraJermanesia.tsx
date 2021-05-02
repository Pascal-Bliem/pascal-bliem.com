import React from "react";
import styles from "./Diarysta.module.scss";
import Navbar from "../../layout/Navbar";
import Footer from "../../layout/Footer";

const SuaraJermanesia = () => {
  return (
    <div className={styles.page}>
      <Navbar fullyHideHeight={0} initialFullyHide={false} />
      <div className={`container`}>
        <div className="row">
          <div className="col s12 m10 offset-m1 l8 offset-l2 center-align">
            <h1 className={styles.title}>Suara Jermanesia Podcast</h1>
          </div>
        </div>
        <div className="row">
          <div
            className={`col l8 offset-l2 m10 offset-m1 s12 ${styles.postContentContainer}`}
          >
            <p className="">
              Together with my co-host Regita Aditia Mahardika, I host the Suara
              Jermanesia Podcast (in Bahasa Indonesia) in which we discuss
              aspects of living in Germany as an Indonesian and vice versa. We
              talk about culture, language, options for traveling, studying, and
              working, as well as our personal stories and experiences from
              living in each other's home country. Occasionally, we also
              interview guest (usually other Indonesians living in Germany) to
              share their piece of wisdom on the topics we're discussing.
              <br />
              <br />
              You can listen to Suara Jermanesia on{" "}
              <strong>
                <a href="https://anchor.fm/suara-jermanesia">Anchor</a>
              </strong>
              ,{" "}
              <strong>
                <a href="https://open.spotify.com/show/6wMcqnayIeh89n0oVGX1hy">
                  Spotify
                </a>
              </strong>
              , or{" "}
              <strong>
                <a href="https://www.youtube.com/channel/UCHdFvB3GLTQp4Aynr9H6-Vg">
                  YouTube
                </a>
              </strong>
              . Selamat dengarkan!
            </p>
            <div className={styles.diarystaImg}>
              <img
                src="https://lh3.googleusercontent.com/a-/AOh14GgbSeBIzMXFyWjSaclK9dvGerO6_n0UaE_yK5c1=s600-k-no-rp-mo"
                alt="Your Suara Jermanesia hosts."
              />
            </div>
          </div>
        </div>
      </div>
      <Footer acknowledgements={<div />} />
    </div>
  );
};

export default SuaraJermanesia;
