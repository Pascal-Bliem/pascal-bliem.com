import React from "react";
import styles from "./ProjectSectionCard.module.scss";

export interface ProjectSectionCardProps {
  linkUrl: string;
  imageUrl: string;
  title: string;
  description: string;
}

const ProjectSectionCard = ({
  linkUrl,
  imageUrl,
  title,
  description,
}: ProjectSectionCardProps) => {
  return (
    <div className="col s12 s12 l6 xl4">
      <a className={styles.cardLink} href={linkUrl}>
        <div className="card medium">
          <div className={`card-image`}>
            <div className={styles.cardImageGrad}>
              <img className={styles.cardImage} src={imageUrl} alt={title} />
            </div>
            <span className={`card-title`}>
              <strong>{title}</strong>
            </span>
          </div>
          <div className="card-content">
            <p>{description}</p>
          </div>
        </div>
      </a>
    </div>
  );
};

export default ProjectSectionCard;
