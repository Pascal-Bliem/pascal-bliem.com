import styles from "../components/pages/projects/ToxBlock.module.scss";

//  This file is form validation and API calls for the ToxBlock page

export default () => {
  // callback function receiving the health status form the ToxBlock REST API
  // (which was called via the server to avoid CORS trouble) and updates the
  // status display on the website accordingly
  const checkHealthStatus = (responseText: string) => {
    const status = document.getElementById(
      "toxblock-app-status"
    ) as HTMLSpanElement;
    if (responseText === "ok") {
      status.innerText = " online";
      status.style.color = "#2ecc71";
    } else {
      status.innerText = " offline";
      status.style.color = "#e74c3c";
    }
  };

  // check if the ToxBlock API is online and healthy
  fetch("https://tox-block-api.herokuapp.com/health")
    .then((response) => response.text())
    .then((text) => checkHealthStatus(text));

  // Everything below here is form validation for the input text
  // and handling submission of the form
  const form = document.getElementById("toxblock-form") as HTMLFormElement;
  const input = document.getElementById(
    "toxblock-input"
  ) as HTMLTextAreaElement;

  // if validation error occurs, display it
  const showError = (input: HTMLTextAreaElement, message: string) => {
    const formControl = input.parentElement as HTMLElement;
    formControl.className = `${styles.toxblockFormControl} ${styles.error}`;
    const small = formControl.querySelector("small") as HTMLElement;
    small.innerText = message;
  };

  // if validation error was corrected, remove error display
  const removeError = (input: HTMLTextAreaElement) => {
    const formControl = input.parentElement as HTMLElement;
    formControl.className = styles.toxblockFormControl;
  };

  // validation function to check for empty string input
  const noEmptyString = (input: HTMLTextAreaElement) => {
    if (input.value === "") {
      showError(input, "The input must not be empty.");
      return false;
    } else {
      return true;
    }
  };

  // validation function to check for string input
  // without latin letters using regex
  const noLetters = (input: HTMLTextAreaElement) => {
    const regex = /[A-Za-z]*/;
    let matches = input.value.match(regex);
    matches = matches ? matches : [""];

    if (matches[0] === "") {
      showError(input, "The input must contain latin letters.");
      return false;
    } else {
      return true;
    }
  };

  // callback function that receives the predictions returned by the POST
  // request to the ToxBlock REST API and updates the bars and
  // bar number labels in the bar chart accordingly
  const updateBarChart = (responseText: string) => {
    // predictions returned in request response
    interface Predictions {
      identity_hate: number;
      insult: number;
      obscene: number;
      severe_toxic: number;
      text: string;
      threat: number;
      toxic: number;
    }
    const predictions: Predictions = JSON.parse(responseText)["predictions"];
    const predKeys = [
      "toxic",
      "severe_toxic",
      "obscene",
      "insult",
      "threat",
      "identity_hate",
    ] as Array<keyof Predictions>;

    // bar elements
    const toxic = document.getElementById(styles.toxicBar) as HTMLElement;
    const severeToxic = document.getElementById(
      styles.severeToxicBar
    ) as HTMLElement;
    const obscene = document.getElementById(styles.obsceneBar) as HTMLElement;
    const insult = document.getElementById(styles.insultBar) as HTMLElement;
    const threat = document.getElementById(styles.threatBar) as HTMLElement;
    const identityHate = document.getElementById(
      styles.identityHateBar
    ) as HTMLElement;

    const bars = [toxic, severeToxic, obscene, insult, threat, identityHate];

    // number label elements
    const toxicNum = document.getElementById("toxic-bar-number") as HTMLElement;
    const severeToxicNum = document.getElementById(
      "severe-toxic-bar-number"
    ) as HTMLElement;
    const obsceneNum = document.getElementById(
      "obscene-bar-number"
    ) as HTMLElement;
    const insultNum = document.getElementById(
      "insult-bar-number"
    ) as HTMLElement;
    const threatNum = document.getElementById(
      "threat-bar-number"
    ) as HTMLElement;
    const identityHateNum = document.getElementById(
      "identity-hate-bar-number"
    ) as HTMLElement;

    const numbers = [
      toxicNum,
      severeToxicNum,
      obsceneNum,
      insultNum,
      threatNum,
      identityHateNum,
    ];

    // for all six categories, update the bar widths and number labels
    for (let i = 0; i < 6; i++) {
      const newWidth = Math.round(+predictions[predKeys[i]] * 100);
      bars[i].style.width = `${(newWidth / 100) * 81 + 1}%`;
      numbers[i].innerText = `${newWidth}%`;
    }
  };

  // event listener that prevents default form submission,
  // performs input data validation, and if input is correct,
  // posts the input (via the server to avoid CORS issues) to
  // the ToxBlock REST API
  form.addEventListener("submit", (event) => {
    // form input validation
    event.preventDefault();
    const letters = noLetters(input);
    const empty = noEmptyString(input);

    // if input correct, POST to ToxBlock API
    if (empty && letters) {
      removeError(input);
      fetch("https://tox-block-api.herokuapp.com/v1/make_single_prediction", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input_data: input.value }),
      })
        .then((response) => response.text())
        .then((text) => updateBarChart(text));
    }
  });
};
