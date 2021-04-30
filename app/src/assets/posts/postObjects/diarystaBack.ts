import Post from "../postModel";

export default new Post(
  // title
  "The Diarysta Backend",
  // subtitle
  "Building the backend and REST-API for the Diarysta diary app with Node.js, Express, and MongoDB",
  // publishDate
  new Date("2020-09-24"),
  // titleImageUrl
  "https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/diarysta-backend/Blog-Article-MERN-Stack.jpg",
  // titleImageDescription
  "Let's build a MERN stack app!",
  // tags
  ["Web Development"],
  // content
  `**TL;DR**: I built a diary app called Diarysta in which you can track your daily moods and activities and get a graphical summary of your personal diary-stats. It is a MERN stack (MongoDB, Express, React, Node.js) application and in this blog post, I will discuss the design and implementation of the back end. There will be another post on the frontend/user interface soon. You can visit a demo of the app [here](https://diarysta.herokuapp.com/) and also have a look at its [Github repo](https://github.com/Pascal-Bliem/diarysta).

### Keep track of your mood

Do you know how your mood fluctuates over time or how it correlates with your every-day activities? Traditionally, that's what people have diaries for. But hey, wouldn't it be cooler to use an app that let's you track those things in a way that is neatly organized, updatable, searchable, and gives you a graphic summary of how you've felt and what you've done? Yes, it would! So that's what Diarysta is all about. A web app that let's you create diary entries in which you specify your mood and select the activities you've done, all in your browser. You can find Diarysta's source code on [Github](https://github.com/Pascal-Bliem/diarysta). If you're not interested in how it's technically implemented, you may stop reading right here and just check out the actual project [here](https://diarysta.herokuapp.com/). Note that the hosting instance hibernates after long inactivity, and it may take a few seconds for it to wake up and make the app available.

The app is build with the MERN stack, which stands for [**M**ongoDB](https://www.mongodb.com/), [**E**xpress](https://expressjs.com/), [**R**eact](https://reactjs.org/), and [**N**ode.js](https://nodejs.org/). MongoDB is a NoSQL document-based database, Node.js is a JavaScript runtime environment that executes JavaScript code outside a web browser, and Express is minimalist web framework that runs on Node. Those are the components I'll talk about in this post. In an upcoming post, I'll discuss the frontend of the app, which is written in React, a JavaScript library for building user interfaces or UI components. In the following, I will discuss the technical details of this project, how its designed, and in some cases how it is actually implemented. Since the code base is quite large, I will not discuss every bit of it down to the source code, but I'll provide code snippets wherever I find them useful.

### Defining the data models

Let's think for a moment about what we actually want to do in the app, what will be the core functionality? We want users be able to register or login and compose, read, update, and delete diary entries. Clearly, we're dealing with two principle entities: the user and the (diary) entry. For these entities we have to create database models. MongoDB is a document-based database (in contrast to e.g. relational SQL database), so the data model reminds a lot of a Javascript object. This has the advantage of being well suited for unstructured data (such as text), having a flexible and easily extendible schemas and high scalability. It is, however, less suitable for complex queries and analytics. The documents or objects in MongoDB cannot be joined like tables in a SQL database but we can refer to a document within another document with its unique ID; that way we can still relate users to their entries. For a user, we'll need her name, email (which will serve as a login), password, and date of creation. For an entry, we'll need a reference to the user that created it, date of creation, the user's mood, a selection of activities which the user carried out that day, as well as a filed for free text notes. Using [Mongoose](https://mongoosejs.com/), a fantastic object modeling library for MongoDB, the schemas would look like this:

\`\`\`javascript
const mongoose = require("mongoose");

const UserSchema = mongoose.Schema({
  name: {
    type: String,
    required: true,
  },  email: {
    type: String,
    required: true,
    unique: true,
  },  password: {
    type: String,
    required: true,
  },  date: {
    type: Date,
    default: Date.now,
  },
});

const EntrySchema = mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "user",
  },  date: {
    type: Date,
    default: Date.now,
  },  mood: {
    type: Number,
    required: true,
  },  activities: {
    type: [String],
    default: [],
  },  note: {
    type: String,
    default: "",
  },
});

module.exports = {
  user: mongoose.model("user", UserSchema),
  entry: mongoose.model("entry", EntrySchema),
};
\`\`\`

### API endpoints & authentication

After we've defined the data models and connected the database to the Node.js server, we can think about how we'll be able to get data in and out of the database. We'll use [Express](https://expressjs.com/), a minimal and flexible Node.js web application framework, to build the API that our frontend will later contact. For every "bit" of functionality, everything that we want to do with the data, we'll create an API endpoint, which means that we'll have to create routes to these endpoints in Express. I won't go through all the source code of the routes as that would get way to verbose, but you can imagine that in each route, we process some data and put it into or out of the database. You can still check out the full coe on [Github](https://github.com/Pascal-Bliem/diarysta). So, what exactly is it that we need to do? First of all, we need to be able to create a user so we'll need a \`POST api/users\` route. Then we need to be able to login the user (lets call the route \`POST api/auth\`) and get the logged in user back (\`GET api/auth\`). For that last route, we need a way to authenticate the user. I chose to use [JSON web tokens](https://jwt.io/). When the user logs in successfully, the server sends back the token in the response:

\`\`\`javascript
const jwt = require("jsonwebtoken");

// some code to authenticate the user ...
// if authentication was successful:

jwt.sign(user.id, config.get("jwtsecret"), {  expiresIn: 36000  }, (err, token) => {
    if (err) throw err;
    res.json({ token });
});
\`\`\`

The token will be stored in local storage on the client and will be send (if present) in a custom \`x-auth-token\` header with each request. On the server-side, each request to a non-public route (such as \`GET api/auth\`), will go through authentication middleware that looks something like this:

\`\`\`javascript
const jwt = require("jsonwebtoken");
const config = require("config");

module.exports = (req, res, next) => {
  // Get token from header
  const token = req.header("x-auth-token");

  // check if there's no token
  if (!token) {
    return res.status(401).json({ message: "No token, authorization failed." });
  }

  try {
    // verify the token
    const decoded = jwt.verify(token, config.get("jwtsecret"));
    req.user = decoded.user;
    next();
  } catch (error) {
    return res.status(401).json({ message: "Token is not valid." });
  }
};
\`\`\`

Now that we got the user-related routes and authentication taken care of, we can have a look at the entries. We'll need to be able to create an entry (\`POST api/entries\`), update it (\`PUT api/entries/:id\`), delete it (\`DELETE api/entries/:id\`), and get all of a user's entries (\`GET api/entries\`). The route parameter \`:id\` in the URLs is the entry's unique ID in the database; this way we can access an individual existing entry. Whenever we create, update, or receive entries, the body of the request or response will contain entry objects in JSON format:

\`\`\`javascript
// An example for a diary entry object
{
  "user_id": "5f91be79f323e992seh534ze8534640cb",
  "date": "2020-10-22T17:16:22.599+00:00",
  "mood": 4,
  "activities": ["sports", "languages", "date"],
  "note": "What a great day do be alive and cook some nice food!"
}
\`\`\`

### A few last tweaks

Regarding the actual API, that's actually all we have to do. The frontend will now have all the endpoints it needs to work with data on users and their diary entries. There are still a few more things to add to make the backend production ready. First of all, I want to add a simple health check route to be able to check if the app is up and running as expected:

\`\`\`javascript
// health endpoint
app.get("/health", (req, res) => {
  res.status(200).send("ok");
});
\`\`\`

This is going to be particularly useful as I'm planning to host the app on [Heroku's](https://www.heroku.com/) free tier, on which instances go into hibernation after some time of inactivity. Last but not least, I need to serve the actual frontend/user-interface to the client. Traditionally, that would mean that I'd have to do a whole lot of additional routing to serve the client different files while she's navigating through the pages of my app. However, I'm using React to build a single-page application, which "interacts with the user by dynamically rewriting the current web page with new data from the web server, instead of the default method of the browser loading entire new pages" ([Wikipedia](https://en.wikipedia.org/wiki/Single-page_application)). That means all of the routing will actually happen within the frontend and the server will only have to serve this one single-page app:

\`\`\`javascript
// assuming the app's production build is in client/build/index.html
app.use(express.static("client/build"));
app.get("*", (req, res) => {
  res.sendFile(path.resolve(__dirname, "client", "build", "index.html"));
});
\`\`\`

Now, that's really all. We've set up the API with basic CRUD (create, read, update, and delete) functionality and the rest of the action is going to happen on the frontend. This should demonstrate how the use of single-page apps that handle routing on the frontend allow for the backend to be simple, [RESTful](https://en.wikipedia.org/wiki/Representational_state_transfer), and able to agnostically serve multiple different web or mobile frontends. I know, I've skipped a lot of details here for the sake of an improved reading experience; please have a look at Diarysta's [Github repo](https://github.com/Pascal-Bliem/diarysta) if you want to have a more in-depth look at the code. In the next blog post I will discuss the Diarysta frontend in depth, so stay tuned, and thanks a lot for reading!
`
);
