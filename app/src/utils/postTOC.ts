import styles from "../components/pages/PostPage.module.scss";

export default () => {
  const htmlBody = document.querySelector("body") as HTMLBodyElement;
  const postBody = document.querySelector(
    `.${styles.postContentContainer}`
  ) as HTMLDivElement;
  const headings = Array.from(
    postBody.querySelectorAll("h1, h2, h3, h4, h5, h6")
  ) as HTMLElement[];

  // create the TOC element
  const toc = document.createElement("div");
  toc.classList.add(styles.postPageToc);
  toc.classList.add(styles.tocHide);
  const tocHeading = document.createElement("h5");
  tocHeading.innerText = "Content";
  toc.appendChild(tocHeading);

  // make sure it doesn't overlap with the post body
  function calcMaxWidth() {
    let shrink = 0.8;
    if (window.innerWidth > 1500) {
      shrink = 0.6;
    }
    return `${Math.floor(postBody.offsetLeft * shrink)}px`;
  }

  const maxWidthHandler = () => (toc.style.maxWidth = calcMaxWidth());
  maxWidthHandler();

  // also when the window size changes
  window.addEventListener("resize", maxWidthHandler);

  // append to HTML body
  htmlBody.appendChild(toc);

  // TOC is hidden at the top of the page and only becomes
  // visible when scrolling down to the post body
  const scrollHideHandler = (event: Event) => {
    if (window.scrollY < postBody.offsetTop - toc.offsetHeight) {
      toc.classList.add(styles.tocHide);
    } else if (window.scrollY > postBody.offsetTop - toc.offsetHeight) {
      toc.classList.remove(styles.tocHide);
    }
  };
  window.addEventListener("scroll", scrollHideHandler);

  // create the links to the headings in the TOC
  const tocLinks: HTMLAnchorElement[] = [];
  headings.forEach((h) => {
    const heading = document.createElement("p");
    const link = document.createElement("a");
    link.innerText = h.innerText;
    link.href = `#${h.id}`;
    heading.appendChild(link);
    toc.appendChild(heading);
    tocLinks.push(link);
  });

  // check if a heading is already within the scroll pane,
  // if yes add class "current" to the corresponding TOC link which
  // will highlight it, remove class "current" from all other links
  const scrollCurrentHandler = (event: Event) => {
    for (let i = 0; i < headings.length; i++) {
      // get the offset of the headings w.r.t. the document
      const offset_i =
        (window.pageYOffset || document.documentElement.scrollTop) +
        headings[i].getBoundingClientRect().top;
      let offset_i1: number;
      if (i < headings.length - 1) {
        offset_i1 =
          (window.pageYOffset || document.documentElement.scrollTop) +
          headings[i + 1].getBoundingClientRect().top;
      } else {
        offset_i1 = 1000000;
      }

      if (i === headings.length - 1 && window.scrollY >= offset_i - 10) {
        tocLinks[i].classList.add(styles.current);
      } else if (
        window.scrollY >= offset_i - 10 &&
        window.scrollY < offset_i1 - 10
      ) {
        tocLinks[i].classList.add(styles.current);
      } else {
        tocLinks[i].classList.remove(styles.current);
      }
    }
  };
  window.addEventListener("scroll", scrollCurrentHandler);

  return { maxWidthHandler, scrollHideHandler, scrollCurrentHandler };
};
