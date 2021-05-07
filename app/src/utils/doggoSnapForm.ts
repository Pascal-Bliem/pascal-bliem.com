import styles from "../components/pages/projects/DoggoSnap.module.scss";
import _ from "lodash";

/* Utility function to convert a canvas to a BLOB */
const dataURLToBlob = (dataURL: string) => {
  const BASE64_MARKER = ";base64,";
  if (dataURL.indexOf(BASE64_MARKER) === -1) {
    const parts = dataURL.split(",");
    const contentType = parts[0].split(":")[1];
    const raw = parts[1];

    return new Blob([raw], { type: contentType });
  }

  const parts = dataURL.split(BASE64_MARKER);
  const contentType = parts[0].split(":")[1];
  const raw = window.atob(parts[1]);
  const rawLength = raw.length;

  var uInt8Array = new Uint8Array(rawLength);

  for (let i = 0; i < rawLength; ++i) {
    uInt8Array[i] = raw.charCodeAt(i);
  }

  return new Blob([uInt8Array], { type: contentType });
};

//  This file is form validation and API calls for the ToxBlock page
const initDoggoSnapForm = () => {
  // callback function receiving the health status form the ToxBlock REST API
  // (which was called via the server to avoid CORS trouble) and updates the
  // status display on the website accordingly
  const checkHealthStatus = (responseText: string) => {
    const status = document.getElementById(
      "doggosnap-app-status"
    ) as HTMLSpanElement;
    if (responseText === '{"status":"ok"}') {
      status.innerText = " online";
      status.style.color = "#2ecc71";
    } else {
      status.innerText = " offline";
      status.style.color = "#e74c3c";
    }
  };

  // check if the Doggo Snap API is online and healthy
  fetch("https://doggo-snap-api.herokuapp.com/health")
    .then((response) => response.text())
    .then((text) => checkHealthStatus(text));

  //   file input
  // URL input field

  const uploadImgInput = document.getElementById(
    "upload-img"
  ) as HTMLInputElement;
  const dogImg = document.getElementById("dog-img") as HTMLImageElement;
  const classificationResult = document.getElementById(
    "classification-result"
  ) as HTMLDivElement;

  // when the image is loaded, resize it, append the
  // image data to a form and send it to the API
  dogImg.onload = (e: Event) => {
    // Resize the image
    const canvas = document.createElement("canvas");
    const newSize = 224;

    canvas.width = newSize;
    canvas.height = newSize;
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    ctx.drawImage(dogImg, 0, 0, newSize, newSize);

    const dataUrl = canvas.toDataURL("image/jpeg");
    const resizedImage = dataURLToBlob(dataUrl);

    const formData = new FormData();

    if (resizedImage && dataUrl) {
      formData.append("image", resizedImage);
      formData.append("topN", "3");

      // send post request with the image attached
      fetch("https://doggo-snap-api.herokuapp.com/v1/classify_dog_breed", {
        method: "POST",
        body: formData,
      })
        .then((res) => res.json())
        .then((data) => setClassification(data.predictions));
    }
  };

  interface Classification {
    breed: string;
    probability: number;
  }

  const setClassification = (data: Classification[]) => {
    removeError("classification-error");
    classificationResult.innerHTML = "";

    const dog1 = document.createElement("h5");
    dog1.innerText = `${_.startCase(_.toLower(data[0].breed))} (${(
      data[0].probability * 100
    ).toFixed(1)}%)`;
    classificationResult.append(dog1);

    if (data[0].probability < 0.95) {
      const p = document.createElement("p");
      p.innerText = "Other likely breeds:";
      const dog2 = document.createElement("h6");
      const dog3 = document.createElement("h6");
      dog2.innerText = `${_.startCase(_.toLower(data[1].breed))} (${(
        data[1].probability * 100
      ).toFixed(1)}%)`;
      dog3.innerText = `${_.startCase(_.toLower(data[2].breed))} (${(
        data[2].probability * 100
      ).toFixed(1)}%)`;
      classificationResult.append(p, dog2, dog3);
    }

    if (data[0].probability < 0.7) {
      showError(
        "classification-error",
        "The classification certainties are very low! Are you sure this is a pure breed dog?"
      );
    }

    classificationResult.classList.remove(styles.hidden);
  };

  // if validation error occurs, display it
  const showError = (id: string, message: string) => {
    const errorText = document.getElementById(id) as HTMLElement;
    errorText.className = `${styles.error}`;
    errorText.innerText = message;
  };

  // if validation error was corrected, remove error display
  const removeError = (id: string) => {
    const errorText = document.getElementById(id) as HTMLElement;
    errorText.className = `${styles.error} ${styles.hidden}`;
  };

  // validation function to check if uploaded file has an image extension
  const noImageFile = (input: HTMLInputElement) => {
    const filePath = input.value;
    var allowedExtensions = /(\.jpg|\.jpeg|\.png)$/i;

    if (!allowedExtensions.exec(filePath)) {
      showError(
        "upload-img-error",
        "The uploaded file must me a valid image of type JPG or PNG (no GIF)!"
      );
      input.value = "";
    } else {
      return true;
    }
  };

  uploadImgInput.addEventListener("change", (event) => {
    removeError("upload-img-error");

    if (
      noImageFile(uploadImgInput) &&
      uploadImgInput.files &&
      uploadImgInput.files[0]
    ) {
      const reader = new FileReader();
      reader.onload = (e: Event) => {
        dogImg.src = reader.result as string;
        dogImg.classList.remove(styles.hidden);
      };
      reader.readAsDataURL(uploadImgInput.files[0]);
    }
  });
};

export default initDoggoSnapForm;
