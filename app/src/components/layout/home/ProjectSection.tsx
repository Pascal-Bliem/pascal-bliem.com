import React from "react";
import styles from "./ProjectSection.module.scss";
import ProjectSectionCard from "./ProjectSectionCard";

const ProjectSection = () => {
  return (
    <div className={`container ${styles.sectionContainer}`}>
      <hr id="projects" className={styles.sectionSeparator}></hr>
      <h4 className={styles.sectionHeader}>
        Key <strong>Projects</strong>
      </h4>
      <div className={`row`}>
        {/* Unsplash Growtika https://unsplash.com/@growtika  https://unsplash.com/photos/an-abstract-image-of-a-sphere-with-dots-and-lines-nGoCBxiaRO0*/}
        <ProjectSectionCard
          linkUrl="/vec-brain"
          imageUrl="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/website/vec-brain-card.jpg"
          title="Vec Brain LLM"
          description="Vec Brain is a personal knowledge base app using a large language model & RAG to let you store knowledge and create answers and summaries based on your personal notes."
        />
        {/* Unsplash Brad Jordan https://unsplash.com/photos/U32jeOdkgfA */}
        <ProjectSectionCard
          linkUrl="/doggo-snap"
          imageUrl="https://images.unsplash.com/photo-1559715541-d4fc97b8d6dd?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1267&q=80"
          title="Doggo Snap"
          description="Doggo Snap is a mobile app that lets you classify over 120 dog breeds from photos by employing neural networks for image recognition."
        />
        {/* Unsplash Andre Hunter https://unsplash.com/photos/5otlbgWJlLs */}
        <ProjectSectionCard
          linkUrl="/tox-block"
          imageUrl="https://images.unsplash.com/photo-1503525148566-ef5c2b9c93bd?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1350&q=80"
          title="Tox Block"
          description="ToxBlock is a machine learning application for recognizing toxic language in text. It can be employed for automatically screening text in articles, posts, etc."
        />
        {/* Unsplash Peter Jones https://unsplash.com/photos/UIQHGm8XyFU */}
        <ProjectSectionCard
          linkUrl="/diarysta"
          imageUrl="https://images.unsplash.com/photo-1607544835807-d79eefc44ee4?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80"
          title="Diarysta"
          description="Diarysta is a diary web application that let's you track your daily moods and activities and get a graphical summary of your personal diary-stats."
        />
        {/*I should download this https://www.sciencespo.fr/sites/default/files/shutterstock_309509678_6.jpg*/}
        <ProjectSectionCard
          linkUrl="/blog/insights%20from%20the%20european%20social%20survey%208"
          imageUrl="https://www.sciencespo.fr/sites/default/files/shutterstock_309509678_6.jpg"
          title="EU Social Survey"
          description="A full-stack data science project, which analyzes the 8th European Social Survey. How do Europeans think about about social, political, and economic issues?"
        />
        {/* Unsplash Isaac Smith https://unsplash.com/photos/6EnTPvPPL6I */}
        <ProjectSectionCard
          linkUrl="/blog/error%20rate%20control%20in%20statistical%20significance%20testing"
          imageUrl="https://images.unsplash.com/photo-1543286386-713bdd548da4?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1350&q=80"
          title="Error Rate Control"
          description="A discussion and simulation study on proper error rate control in statistical hypothesis significance testing, exploring corrections for multiple tests and optional stopping."
        />
      </div>
      <p className={styles.sectionFooter}>
        <a href="https://github.com/Pascal-Bliem">
          Find more projects on my Github <i className="fab fa-github"></i>
        </a>
      </p>
    </div>
  );
};

export default ProjectSection;
