import React from "react";
import styles from "./ContactSection.module.scss";

export interface ContactIconProps {
  icon: string;
  link: string;
  text: string;
}

const ContactIcon = ({ icon, link, text }: ContactIconProps) => {
  return (
    <div className="col s12 m6 l6">
      <a href={link}>
        <div className={styles.contactIconLink}>
          <i
            className={`${icon} ${styles.contactIcon} hoverable`}
            style={contactIconStyle}
          ></i>
          <span>{text}</span>
        </div>
      </a>
    </div>
  );
};

// a global config for the a element sets the color,
// so we have to locally overwrite the color in-line
const contactIconStyle = {
  color: "#00adb5",
};

const ContactSection = () => {
  return (
    <div className={`container ${styles.sectionContainer}`}>
      <hr id="contact" className={styles.sectionSeparator}></hr>
      <h4 className={styles.sectionHeader}>
        <strong>Contact</strong> Details
      </h4>
      <div className={`row`}>
        <ContactIcon
          icon="fab fa-github"
          link="https://github.com/Pascal-Bliem"
          text="Github"
        />
        <ContactIcon
          icon="fab fa-linkedin"
          link="https://www.linkedin.com/in/pascal-bliem/"
          text="LinkedIn"
        />
        <ContactIcon
          icon="fas fa-envelope"
          link="mailto:pascal@pascal-bliem.com"
          text="pascal@pascal-bliem.com"
        />
        <ContactIcon
          icon="fa-brands fa-line"
          link="tel:+886919140587"
          text="+886 919140587"
        />
      </div>
      <p className={styles.sectionFooter}>
        Let's exchange ideas on data, tech, science, and culture{" "}
        <i className="far fa-lightbulb"></i>
      </p>
    </div>
  );
};

export default ContactSection;
