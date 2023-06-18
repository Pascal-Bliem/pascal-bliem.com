import React, { Fragment } from "react";
import styles from "./SkillsSection.module.scss";
import SkillsSectionCard from "./SkillsSectionCard";

const SkillsSection = () => {
  return (
    <div className={`container ${styles.sectionContainer}`}>
      <hr id="skills" className={styles.sectionSeparator}></hr>
      <h4 className={styles.sectionHeader}>
        <strong>Skills</strong> and <strong>Interests</strong>
      </h4>
      <div className={`row`}>
        {/* Unsplash Luke Chesser https://unsplash.com/photos/JKUTrJ4vK00 */}
        <SkillsSectionCard
          imageUrl="https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1350&q=80"
          title="Data Science & Engineering"
          iconName="analysis"
          description="Understanding, analyzing, and visualizing phenomena with (big) data and statistics, building on a modern data pipeline and database/warehouse infrastructure."
          longDescription={
            <Fragment>
              <p className={styles.modalParagraph}>
                Every business is now aiming at becoming data-driven and
                supporting decision making with data insights. Often there is
                value hidden in data, which cannot be discovered by simple
                reporting. To reveal hidden patterns, and hence, business value
                in data, I apply techniques from a variety of disciplines.
              </p>
              <p className={styles.modalParagraph}>
                Data Science is an interdisciplinary field in which we try to
                extract knowledge and insights from data by using methods from
                computer science, maths, and statistics. Data Engineering is the
                process of getting all the necessary infrastructure in place to
                make this process of knowledge extraction from data technically
                possible.
              </p>
              <p className={styles.modalParagraph}>
                I first got into data analysis during my work as a materials
                scientist and am now applying it in an enterprise setting, in
                which I facilitate the use of big data with statistical
                analysis, machine learning, visualization, data processing
                pipelines, data bases (SQL & NoSQL), and cloud computing
                solutions.
              </p>
            </Fragment>
          }
        />
        {/* Unsplash Arseny Togulev https://unsplash.com/photos/MECKPoKJYjM */}
        <SkillsSectionCard
          imageUrl="https://images.unsplash.com/photo-1555255707-c07966088b7b?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1490&q=80"
          title="AI & Machine Learning"
          iconName="machine-learning"
          description="Employing state-of-the-art algorithms, such as neural networks and gradient-boosted decision trees, for predictive analytics and classification."
          longDescription={
            <Fragment>
              <p className={styles.modalParagraph}>
                The term AI is used to describe a lot of things, but in an
                applied settings, it usually refers to machine learning (ML); a
                technique used for all sorts of predictive analytics,
                classification, language processing, computer vision, and much
                more.
              </p>
              <p className={styles.modalParagraph}>
                Machine learning models are mathematical/statistical constructs
                that can be algorithmically trained to recognize patterns in
                data. The data is a crucial part of machine learning, as the
                models' internal parameters are mathematically optimized while
                training data is being passed through them.
              </p>
              <p className={styles.modalParagraph}>
                I train a variety of ML algorithms, such as gradient boosting
                machines, neural networks, support vector machines, clustering
                etc. on data to predict if customers are likely to cancel their
                contract or to buy a new product, to estimate future usage of a
                service, to recognize handwriting, to classify sentiment in
                text, understand traffic scenes and localize road assets, and
                much more.
              </p>
            </Fragment>
          }
        />
        {/* Unsplash Taylor Vick https://unsplash.com/photos/M5tzZtFCOfs */}
        <SkillsSectionCard
          imageUrl="https://images.unsplash.com/photo-1558494949-ef010cbdcc31?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1491&q=80"
          title="Cloud Computing"
          iconName="cloud"
          description="Building highly scalable applications and infrastructure for computation, data storage, and machine learning in the cloud on Amazon Web Services (AWS)."
          longDescription={
            <Fragment>
              <p className={styles.modalParagraph}>
                With computer infrastructure and platform providers, it has
                become possible to set up any required hardware "for rent" in
                the cloud, in an easily scalable way. This hands off the
                physical purchase and maintenance of servers, sometimes even
                their administration, to the cloud data centers.
              </p>
              <p className={styles.modalParagraph}>
                Providers with a large service catalog, such as Amazon Web
                Services (AWS), allow entire solution stacks to be constructed,
                containing object storage, data bases, servers, routing,
                auto-scaling, load-balancers, or serverless & container
                services.
              </p>
              <p className={styles.modalParagraph}>
                I use cloud platforms such as AWS or Render to scalably host web
                servers (e.g. this website here) and data bases, store resources
                for my blog in S3 object store, train neural networks on GPU
                processors in AWS or Google Colab, serve machine learning models
                via REST APIs, and build complex and scalable data processing
                pipelines using various AWS services or orchestrating different
                pipeline components on kubernetes clusters.
              </p>
            </Fragment>
          }
        />
        {/* Unsplash Olaf Val https://unsplash.com/photos/UTk9cXzYWAg  */}
        <SkillsSectionCard
          imageUrl="https://images.unsplash.com/photo-1618761714954-0b8cd0026356?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80"
          title="Web & Mobile Development"
          iconName="webdev"
          description="Creating front-end and back-end parts of responsive web and mobile applications and APIs which harness the potential of build-in data science solutions."
          longDescription={
            <Fragment>
              <p className={styles.modalParagraph}>
                The internet connects everything; from consumers searching
                information, over web services interacting with each other via
                APIs, to smart devices in the internet of things (IOT). The
                networking technologies and protocols used in web technologies
                provide a unified way of interaction between a variety of
                different technologies.
              </p>
              <p className={styles.modalParagraph}>
                Full-stack web development encompasses responsive design of
                websites and user interfaces with technologies such as HTML,
                CSS, and JavaScript, development of server-side code to handle
                serving of sites, routing, and data bases, as well as design of
                API services that can be integrated into other applications,
                e.g. for serving machine learning predictions.
              </p>
              <p className={styles.modalParagraph}>
                I build sites (such as this one here) or mobile apps with the
                aforementioned technologies or frameworks such as React.js /
                React Native, server-side code with the Express web application
                framework on Node.js and MongoDB or SQL data bases, as well as
                serving machine learning models in Python via Flask or FastAPI
                as a Docker container.
              </p>
            </Fragment>
          }
        />
        <div
          className={`col s12 m12 l6 xl4 show-on-large ${styles.fillerContainer}`}
        >
          <i className={`fas fa-atom ${styles.fillerIcon}`}></i>
        </div>
        {/* Unsplash NeONBRAND https://unsplash.com/photos/zFSo6bnZJTw  */}
        <SkillsSectionCard
          imageUrl="https://images.unsplash.com/photo-1509062522246-3755977927d7?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1404&q=80"
          title="Teaching & Volunteering"
          iconName="teaching"
          description="I used to teach English, German, maths, science, and coding, e.g. at ILEC Hanoi, Jolly English Club, Jogjakarta Community School, or Project Child Indonesia."
          longDescription={
            <Fragment>
              <p className={styles.modalParagraph}>
                I have always enjoyed helping others to understand complex
                phenomena and help them on their way to achieve their personal
                and professional goals. While still working at RWTH Aachen
                University, I got the chance to help preparing classes or labs,
                and assist in supervision and review of thesis works. After my
                time in academia, I traveled through southeast Asia and had the
                great privilege to volunteer as a teacher in several places.
              </p>
              <p className={styles.modalParagraph}>
                During my time in Vietnam I taught English and German, first at{" "}
                <a href="https://www.volunteersbase.com/asia/vietnam/jolly-club_i1273">
                  Jolly Club
                </a>{" "}
                in the remote mountain town of Ha Giang, then at{" "}
                <a href="http://ilec.edu.vn/">ILEC</a> in the capital Hanoi.
                After moving to Jogjakarta, Indonesia, I first taught maths,
                physics, and chemistry at{" "}
                <a href="https://jogjacommunityschool.org/">
                  Jogjakarta Community School
                </a>
                , then about health, hygiene, and environmental sustainability
                at{" "}
                <a href="https://projectchild.ngo/">Project Child Indonesia</a>.
              </p>
              <p className={styles.modalParagraph}>
                Currently, I try to learn and teach as much as I can from and to
                my coworkers, support students who do their thesis work in our
                company, participate in meet-ups, and just generally exchange
                ideas on tech and science whenever I get the chance. Please
                don't hesitate to get <a href="#contact">in touch</a> with me!
              </p>
            </Fragment>
          }
        />
        {/* Unsplash Hannah Wright https://unsplash.com/photos/ZzWsHbu2y80  */}
        <SkillsSectionCard
          imageUrl="https://images.unsplash.com/photo-1571498664957-fde285d79857?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=564&q=80"
          title="Language & Cultural Exchange"
          iconName="language"
          description="I love to learn about culture and languages. I co-hosted a cultural exchange podcast in Indonesian and also trying to improve my Spanish and Chinese."
          longDescription={
            <Fragment>
              <p className={styles.modalParagraph}>
                I am fascinated by the new worlds that open up when being
                immersed into a foreign culture and picking up bits of its
                language. I've learned English as a second language during my
                early teenage years and I'm still amazed by how much knowledge
                it has unlocked for me ever since, ranging from guitar tutorial
                videos to basically anything that has to do with programming.
              </p>
              <p className={styles.modalParagraph}>
                While I lived in Indonesia, I got fairly fluent in Indonesian,
                which added so much more quality to my stay there as I was able
                to deeply connect with my neighbors and learn about their lives.
                It also helped me with my Malaysian, every time I had to go
                there for a visa run. I used to co-host the{" "}
                <a href="http://www.pascal-bliem.com/suarajermanesia">
                  Suara Jermanesia Podcast
                </a>{" "}
                in which we talk (in Indonesian) about the experiences of an
                Indonesian living in Germany and vice versa. Living in Taiwan
                during a research internship also sparked my interest in the
                Chinese language, which I find very difficult though; hence, I'm
                still far from having elaborate conversations. Since I recently
                also visited Spain a couple of times, I've picked up Spanish as
                well and have become quite fluent.
              </p>
              <p className={styles.modalParagraph}>
                I've visited almost 30 countries and I'm always able to gain
                some new perspective on life and the world from those visits.
                Whenever possible, I try to get into contact with locals as much
                as I can, sometimes through couch surfing or hitchhiking. I
                think that understanding different cultures allows one to pick
                the best parts for oneself and become a better human. In case
                you need a place to stay, contact me on{" "}
                <a href="https://www.couchsurfing.com/people/pascal-bliem">
                  Couchsurfing
                </a>{" "}
                :)
              </p>
            </Fragment>
          }
        />
        {/* Unsplash Jefferson Santos https://unsplash.com/photos/fCEJGBzAkrU  */}
        <SkillsSectionCard
          imageUrl="https://images.unsplash.com/photo-1510915361894-db8b60106cb1?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80"
          title="Making Music"
          iconName="guitar"
          description="I enjoy listening to and making music, mostly playing blues or rock on acoustic or electric guitar or base. I also have a sizeable collection of flutes :D"
          longDescription={
            <Fragment>
              <p className={styles.modalParagraph}>
                I've been hooked ever since I got my first cheap Stratocaster
                knock-off. Sure, I had always loved listening to music, but
                having my first guitar in my hands added a whole new dimension.
                There is something magical about being able to put your heart
                and soul into sound, created by whatever instrument is between
                your fingers.
              </p>
              <p className={styles.modalParagraph}>
                For me that first love was an electric guitar and blues music.
                Until today, plain blues is most of what I play, occasionally
                mixed with rock, hard rock, or metal. I also love the crafty
                aspect of musical instruments. I've build my favorite guitar
                myself, even winding the pickup coils and soldering the
                electronics.
              </p>
              <p className={styles.modalParagraph}>
                Besides electric and acoustic guitar, I've also been fooling
                around on other instruments, such as base, piano, violin, erhu,
                and wind or brass instruments. I've made it a habit to bring
                flutes as souvenirs from my travels, so by now, I have flutes
                from about five different countries.
              </p>
            </Fragment>
          }
        />
        {/* Unsplash ThisisEngineering RAEng https://unsplash.com/photos/mF6gB6hV5OU */}
        <SkillsSectionCard
          imageUrl="https://images.unsplash.com/photo-1581093450021-4a7360e9a6b5?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80"
          title="Materials Science"
          iconName="chemistry"
          description="During my studies, I did a lot of experimental and computational research in materials science, chemistry, and physics at RWTH Aachen, Uppsala, and National Tsing Hua Universities."
          longDescription={
            <Fragment>
              <p className={styles.modalParagraph}>
                The urge to understand what the world around me is made of and
                how it works has led me to study materials science and
                engineering, and more generally, physics. I realized that I
                needed to actively participate in research to really deepen my
                understand, which is why I picked up a research assistant
                position already in the second year of my Bachelors.
              </p>
              <p className={styles.modalParagraph}>
                During my research in experimental and computational{" "}
                <a href="https://www.institut-1a.physik.rwth-aachen.de/">
                  materials physics
                </a>{" "}
                and{" "}
                <a href="https://www.mch.rwth-aachen.de/">
                  materials chemistry
                </a>{" "}
                , I've developed the faculty of scientific inquiry that still
                serves me today. I had the chance to conduct research
                internships at{" "}
                <a href="http://m102.nthu.edu.tw/~s102011510/index.html">
                  National Tsing Hua University
                </a>{" "}
                and{" "}
                <a href="https://www.physics.uu.se/research/materials-physics+/">
                  Uppsala University
                </a>
                , and overall, author or coauthor 9{" "}
                <a href="https://github.com/Pascal-Bliem/my-papers-and-theses#peer-reviewed-publications">
                  peer-reviewed publications
                </a>{" "}
                on synthesis and characterization of nano-scaled materials and
                thin films for applications in energy generation, electronics,
                and wear resistance.
              </p>
              <p className={styles.modalParagraph}>
                After this period in academic research, in which I gained the
                understanding I was looking for, I felt that it was time to
                shift my professional focus towards more applied tasks. Starting
                from the programming and statistics skills I acquired in
                computational research, I improved my software engineering
                abilities and worked on several machine learning projects, which
                eventually led up to my current profession as a Data Scientist.
              </p>
            </Fragment>
          }
        />
      </div>
    </div>
  );
};

export default SkillsSection;
