import React from "react";
import styles from "./Footer.module.scss";

// TODO: Add contact icons in footer

export interface FooterProps {
  acknowledgements: JSX.Element;
}

const Footer = ({ acknowledgements }: FooterProps) => {
  return (
    <div className={styles.background}>
      <div className={`container ${styles.sectionContainer}`}>
        <div className={styles.sectionFooter}>
          {acknowledgements}
          <div className="col s12 m6">
            <a href="https://github.com/Pascal-Bliem">
              <i className={`fab fa-github ${styles.contactIcon}`}></i>
            </a>
            <a href="https://www.linkedin.com/in/pascal-bliem/">
              <i className={`fab fa-linkedin ${styles.contactIcon}`}></i>
            </a>
            <a href="mailto:pascal@bliem.de">
              <i className={`fas fa-envelope ${styles.contactIcon}`}></i>
            </a>
          </div>
          <p>Copyright © {new Date().getFullYear()} Pascal Bliem</p>
        </div>
      </div>
    </div>
  );
};

export default Footer;
