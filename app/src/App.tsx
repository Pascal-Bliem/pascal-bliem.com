import React, { useEffect } from "react";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import "materialize-css/dist/css/materialize.min.css";
//@ts-ignore
import M from "materialize-css/dist/js/materialize.min.js";
import "./App.scss";
import Home from "./components/pages/Home";
import Blog from "./components/pages/Blog";
import PostPage from "./components/pages/PostPage";

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
          <Route exact path="/" component={Home} />
        </Switch>
      </div>
    </Router>
  );
}

export default App;
