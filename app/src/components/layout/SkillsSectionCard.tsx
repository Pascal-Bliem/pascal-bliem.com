import React, { Fragment } from "react";
import styles from "./SkillsSectionCard.module.scss";

export interface SkillsSectionCardProps {
  imageUrl: string;
  title: string;
  description: string;
  longDescription: JSX.Element;
  iconName: string;
}

const SkillsSectionCard = ({
  imageUrl,
  title,
  description,
  longDescription,
  iconName,
}: SkillsSectionCardProps) => {
  return (
    <Fragment>
      <div className="col s12 m12 l6 xl4">
        <a
          className={`modal-trigger ${styles.cardLink}`}
          href={`#modal_${btoa(title)}`}
        >
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
              <p className={styles.cardText}>{description}</p>
            </div>
          </div>
        </a>
      </div>

      {/* Modal that is only visible on clicking the card */}
      <div
        style={{ maxWidth: "700px" }}
        id={`modal_${btoa(title)}`}
        className={`modal ${styles.modalContainer}`}
      >
        <div className="modal-content">
          <img
            src={require(`../../assets/images/${iconName}.svg`).default}
            className={styles.skillIcon}
            alt={iconName}
          />
          <h4 className={styles.modalTitle}>{title}</h4>
          {longDescription}
        </div>
        <div className="modal-footer">
          <a
            href="#!"
            className="modal-close waves-effect waves-green btn-flat"
          >
            Close
          </a>
        </div>
      </div>
    </Fragment>
  );
};

export default SkillsSectionCard;
