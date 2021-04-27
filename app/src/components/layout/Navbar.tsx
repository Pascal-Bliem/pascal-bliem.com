import React, { Fragment, useEffect } from "react";
import "./Navbar.module.scss";
import styles from "./Navbar.module.scss";

const Navbar = () => {
  useEffect(() => {
    const navbar = document.getElementById("navbar") as HTMLElement;
    // new class for hidden navbar
    const navbarHide = styles.navbarHide;
    const navbarHideFully = styles.navbarHideFully;

    // hide the navbar when scrolling down and show it again
    // when scrolling up
    let lastScrollPosition = 0;
    const scrollHandler = (event: Event) => {
      if (window.scrollY < 300 && window.innerWidth > 1024) {
        navbar.classList.add(navbarHideFully);
        lastScrollPosition = window.scrollY;
      } else if (window.scrollY > lastScrollPosition) {
        navbar.classList.remove(navbarHideFully);
        navbar.classList.add(navbarHide);
        lastScrollPosition = window.scrollY;
      } else if (window.scrollY < lastScrollPosition) {
        navbar.classList.remove(navbarHideFully);
        navbar.classList.remove(navbarHide);
        lastScrollPosition = window.scrollY;
      }
    };

    const mouseEnterHandler = () => navbar.classList.remove(navbarHide);

    window.addEventListener("scroll", scrollHandler);

    // show the navbar when hovering over it
    navbar.addEventListener("mouseenter", mouseEnterHandler);

    return () => {
      window.removeEventListener("scroll", scrollHandler);
      window.removeEventListener("mouseenter", mouseEnterHandler);
    };
  });

  return (
    <Fragment>
      {/* <div id="navbar" className={"navbar-fixed " + styles.navbarContainer}> */}
      <nav id="navbar" className={`${styles.navbar} ${styles.navbarHideFully}`}>
        <div className="nav-wrapper">
          <a href="#!" data-target="mobile-sidenav" className="sidenav-trigger">
            <i className={"material-icons " + styles.text}>menu</i>
          </a>
          <ul className={"hide-on-med-and-down " + styles.navUl}>
            <li className={styles.navUlLi}>
              <a
                className={styles.text + " " + styles.navUlLiA}
                href="/#projects"
              >
                Projects
              </a>
            </li>
            <li className={styles.navUlLi}>
              <a
                className={styles.text + " " + styles.navUlLiA}
                href="/#about-me"
              >
                About me
              </a>
            </li>
            <li className={styles.navUlLi}>
              <a
                className={styles.text + " " + styles.navUlLiA}
                href="/#skills"
              >
                Skills
              </a>
            </li>
            <li className={styles.navUlLi}>
              <a className={styles.text + " " + styles.navUlLiA} href="/#blog">
                Blog
              </a>
            </li>
            <li className={styles.navUlLi}>
              <a
                className={styles.text + " " + styles.navUlLiA}
                href="/#contact"
              >
                Contact
              </a>
            </li>
          </ul>
        </div>
      </nav>
      {/* </div> */}
      {/* This is the side nav for mobile view */}
      <ul className="sidenav teal lighten-5" id="mobile-sidenav">
        <li>
          <a className={styles.text} href="/#projects">
            Projects
          </a>
        </li>
        <li>
          <a className={styles.text} href="/#about-me">
            About me
          </a>
        </li>
        <li>
          <a className={styles.text} href="/#skills">
            Skills
          </a>
        </li>
        <li>
          <a className={styles.text} href="/#blog">
            Blog
          </a>
        </li>
        <li>
          <a className={styles.text} href="/#contact">
            Contact
          </a>
        </li>
      </ul>
    </Fragment>
  );
};

export default Navbar;
