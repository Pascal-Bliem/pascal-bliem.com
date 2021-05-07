import React, { useEffect } from "react";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import "materialize-css/dist/css/materialize.min.css";
//@ts-ignore
import M from "materialize-css/dist/js/materialize.min.js";
import "./App.scss";
import "./highlightjs.scss";
import Home from "./components/pages/Home";
import Blog from "./components/pages/Blog";
import PostPage from "./components/pages/PostPage";
import ToxBlock from "./components/pages/projects/ToxBlock";
import Diarysta from "./components/pages/projects/Diarysta";
import SuaraJermanesia from "./components/pages/projects/SuaraJermanesia";
import DoggoSnap from "./components/pages/projects/DoggoSnap";

function App() {
  useEffect(() => {
    // initializes Materialize JavaScript
    M.AutoInit();
    //  eslint-ignore-next-line
  }, []);

  return (
    <Router>
      <div>
        <Switch>
          <Route exact path="/blog" component={Blog} />
          <Route exact path="/blog/:postName" component={PostPage} />
          <Route exact path="/tox-block" component={ToxBlock} />
          <Route exact path="/diarysta" component={Diarysta} />
          <Route exact path="/suara-jermanesia" component={SuaraJermanesia} />
          <Route exact path="/doggo-snap" component={DoggoSnap} />
          <Route exact path="/" component={Home} />
        </Switch>
      </div>
    </Router>
  );
}

export default App;
