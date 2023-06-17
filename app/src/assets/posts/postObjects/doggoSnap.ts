import Post from "../postModel";

export default new Post(
  // title
  "The Doggo Snap Mobile App",
  // subtitle
  "Classify dogs' breeds whenever you meet them",
  // publishDate
  new Date("2021-04-19"),
  // titleImageUrl
  "https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/doggo-snap/title.png",
  // titleImageDescription
  "What breed is it? Find out with Doggo Snap!",
  // tags
  ["Data Science", "Web Development"],
  // content
  `**TL;DR:** I built a mobile app called Doggo Snap which can be used for classifying a dog's breed from an image. It provides you with information on the breed and let's you save the dogs you've photographed so that you can have a look at them later and see where you've met them on a map. I created the app using React Native and built it for [Android](https://play.google.com/store/apps/details?id=com.pascalbliem.doggosnap) and iOS (though the iOS version is not available in the app store as of now). For classifying a dog's breed with computer vision, the app calls an API which hosts a Deep Learning model (a convolutional neural network) in the cloud. 
   
### Find out what dog breed it is!
    
A lot of people like dogs - of course they do, humans have been domesticating their "best friends" for aeons, which resulted in a diversity of breeds originating from all over the globe. Certainly, there are some real dog enthusiasts who know every breed and can tell a breed's typical size, weight, or temperament. But most people, even if they like dogs, probably only know a handful of breeds by name. So what do you do when you meet a cute dog outside and would like to know what breed it is? Well, you could ask the dog's owner, they usually know. Or you download Doggo Snap, take a picture of the dog, and get all the info you wanna have on your new furry friend. With the [Doggo Snap](https://play.google.com/store/apps/details?id=com.pascalbliem.doggosnap) app, you can take a photo or upload one from your gallery and classify it fast. You'll be shown the breed name, how certain the algorithm is with its decision, and info on a breeds typical temperament, size, weight, lifespan, and region of origin.
    
The app was build in [React Native](http://www.reactnative.com/), an open-source mobile application framework in which you can write an application in JavaScript (as you would in [React](https://reactjs.org/) for web applications), and compile it to true native apps for Android or iOS. For a smoother development workflow, I used the [Expo](https://expo.io/) framework. The app is designed to make people happy who care about privacy; no user data is stored on a server. The only contact the app has to a backend is sending the dog image via a POST request for classification, which is not saved to a backend database. Instead, the dogs that are saved by the user are handled in app state with [Redux](https://redux.js.org/) and persisted to an on-device [SQLite](https://www.sqlite.org/index.html) database. The neural network doing the image classification, which I described in [an earlier post](http://www.pascal-bliem.com/blog/classifying%20dog%20breeds%20with%20deep%20learning), is a slightly modified [MobileNetV2](https://arxiv.org/abs/1801.04381) pretrained on [ImageNet](http://www.image-net.org/) and fine tuned on 121k pictures of 121 dog breeds. The model is being hosted in [Renders's](https://www.render.com/) cloud with the help of [FastAPI](https://fastapi.tiangolo.com/). I won't go into any code here, as the app's code isn't really arrangeable in a nice linear fashion and it is just too much stuff to bore you with here. If you want to see details though, please have a look at the [Github repo](https://github.com/Pascal-Bliem/doggo-snap).
    
### Thinking about functionality, state, and navigation
    
What does an app for dog breed classification need to do and how do we let all components of the application know whats going on? React apps are generally composed of components, which are independent, reusable, and conceptually isolated pieces of the UI. Some of these components higher up in the component hierarchy will be the screens we eventually see in the app. To get around from one screen to another, I use the stack navigator provided by the [React Navigation](https://reactnavigation.org/) library. Individual components can hold state to keep track of whats going on inside of them, and they can also pass down this information as properties (or just props) to their child components. However, if there are many components at different levels of the UI hierarchy that need to access the same state information, this can get messy pretty quickly. It would be much better if there was a single source of truth that is easily accessible from all components. As mentioned above, in this app I use Redux for state management, which will alow us to tap into the app state and also dispatch actions that modify the state, from anywhere within the application. To make sure that the user doesn't lose their saved dogs every time they restart the app, the state is loaded from and mirrored to an on-device SQLite database.
    
![An overview of Doggo Snap's functionality](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/doggo-snap/diagramm.png)
    
We don't have any user data being saved on a backend, so no user creation or authentication is required. Pretty much everything stays on the user's device. Starting at the home screen, we want to be able to take a new photo or select an old one from the gallery and classify it.  

<img src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/doggo-snap/HomeScreen.jpg" style="width: 50%;" alt="The Home Screen">  

The classification happens by sending the image via a HTTP request to the machine learning API, getting the classification data back in the response, and displaying it on a classification screen, to which we automatically navigate from the home screen. On the classification screen, the taken image along with the dog's breed, and the certainty of the classification will be displayed. In addition some info on the breed is given, such as typical temperament, height, weight, lifespan, and region of origin.
    
<img src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/doggo-snap/ClassificationScreen.jpg" style="width: 50%;" alt="The Classification Screen">  

Up to this point, no state is needed. But the user should be able to save her classified dogs as well. This is obviously were state management comes in. When the user choses to save a dog, she navigates to the save screen, where the dog can be given a name. Furthermore the user can specify where she met the dog by either determining the device's current location or by picking a location on the map manually. For the map component, I use [React Native Maps](https://github.com/react-native-maps/react-native-maps), which will use Google Maps as a map provider if you're on Android or Apple Maps if you're on iOS. Besides getting the raw latitude and longitude of the dogs location, I also use the [Nominatim](https://wiki.openstreetmap.org/wiki/Nominatim) reverse geocoding API to generate a human-readable address from the location data. All this data, the image, breed, name, location, and address, is used to create a dog instance which is saved to the local database and appended to the app state's list of saved dogs.
    
<img src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/doggo-snap/SaveScreen.jpg" style="width: 50%;" alt="The Save Screen">  

After saving, the user will be navigated to a screen with a list of her saved dogs. The dog entries can be tapped to get to a details screen which is essentially the same (using mostly the same components) as the classification screen, except that it now show's the dog's name in addition to its breed, as well as a map view showing the dog's location. The dog entry can also be deleted form this screen, which would trigger a deletion of the dog instance from the database and the app state.
    
<img src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/doggo-snap/DogListScreen.jpg" style="width: 50%;" alt="The Dog List Screen">  

To explore where exactly the user met which saved dog or to explore where the dog-encounter-density is particularly high, she can navigate from the the home screen to the map screen. All the dogs are shown as markers in the form of the dogs' images at their respective locations. Tapping such a marker will navigate to the same details page as would tapping a dog's list entry.
    
<img src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/doggo-snap/MapScreen.jpg" style="width: 50%;" alt="The Map Screen">

Finally, the user may run into the problem that she's interested in learning about a lot of dog breeds but does not necessarily encounter all of them in her usual surroundings. No problem! Doggo Snap supports 121 different dog breeds and all the info on these breeds comes bundled with the app. From the home screen, the user can navigate to the Explore-all-Breeds-Screen which displays a list of all supported dog breeds. Tapping on any entry navigates you to the same kind of details screen as the classification screen.
    
<img src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/doggo-snap/AllBreedsScreen.jpg" style="width: 50%;" alt="The All-Breeds Screen">  

### Conclusion

That's it! A React Native app that let's you classify dog breeds from images, save your favorite dog, and revisit them, even on a map! In previous posts, I've already elaborated on how to [build a Deep Learning model](http://www.pascal-bliem.com/blog/classifying%20dog%20breeds%20with%20deep%20learning) in PyTorch for image classification on dog photos, and how to [transfer it to the ONNX format](http://www.pascal-bliem.com/blog/transfer%20ml%20models%20easily%20with%20onnx) (which is how the API uses the model). In this post, I've explained how I make these machine learning capabilities accessible to users on their smartphones. Using React Native, I've coded React in JavaScript that was turned to native applications for Android and iOS. If you're interested to look at it in detail, check out the [Github repo](https://github.com/Pascal-Bliem/doggo-snap), or download the app from the [Play Store](https://play.google.com/store/apps/details?id=com.pascalbliem.doggosnap). Thanks a lot for reading!
    `
);
