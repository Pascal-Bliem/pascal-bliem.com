import Post from "../postModel";

export default new Post(
  // title
  "The ToxBlock API: bring ML models into action by serving them on the web",
  // subtitle
  "Wrapping ToxBlock's functionality into a REST API, containerize it with Docker, and deploy it to the cloud",
  // publishDate
  new Date("2020-07-12"),
  // titleImageUrl
  "https://www.cloudways.com/blog/wp-content/uploads/Rest-API-introduction.jpg",
  // titleImageDescription
  "Bringing machine learning from research to production with containers and APIs.",
  // tags
  ["Data Science", "Web Development"],
  // content
  `**TL;DR:** I built the open-source [ToxBlock API](https://github.com/Pascal-Bliem/tox-block-api), which serves [ToxBlock](https://github.com/Pascal-Bliem/tox-block), a Python deep learning package for recognizing toxic language in text, as a web service. If you're only interested in the usage of the API, feel free to jump to the [demo section](#usage). If you're interested in why it's a good idea to serve machine learning models via a REST API and how it can be done, just continue reading the rest of this post.

*Disclaimer: Since this post discusses an application for classification of toxic language, it inevitably contains examples of toxic language that may be considered profane, vulgar, or offensive. If you do not wish to be exposed to toxic language, DO NOT proceed to read any further.*

In my [last post](http://www.pascal-bliem.com/blog/tox%20block%20using%20ai%20to%20keep%20discussions%20clean), I introduced ToxBlock, a Python machine learning application for recognizing toxic language in text. This time I want to talk about how the predictive functionality of this app can be put to use and integrated into other applications in the easiest way by serving it as a REST API, containerize the app with Docker, and deploy it to a cloud platform. Actually making use of machine learning models is something that seemed to have entered data science education only quite recently. By now, there are bunch of blog posts and tutorials and even some online courses on the deployment of machine learning models, but I remember that when I did a [data science specialization](https://www.coursera.org/specializations/data-science-python) on Coursera, they didn't go beyond Jupyter notebooks. But at some point, every data scientist is gonna ask herself how to deploy the model from the notebook and integrate it into other production systems.

The best strategy for deployment will of course depend on the individual case. For very low latency in predictions it might be a good idea to use streaming via a message queue or directly code the model into the app that will use it. For large batch predictions, it might be a better idea to store them in a shared data base which other applications can then query. In terms of flexibility, it is always a good compromise to serve a model via a web service in the form of a REST API (**re**presentative **s**tate **t**ransfer **a**pplication **p**rogramming **i**nterface). All predictions can be handled on-the-fly by sending HTTP requests. Pretty much every language has integrations for web technology and can communicate via the hyper text transfer protocol (HTTP), which makes it very easy to integrate the machine learning model into any other application. Let's elaborate on this in the next section. 

### Why using a REST API

As just mentioned above, it makes the integration with other applications very easy. Imagine the alternatives: You could just throw the code over to the engineer who develops an app that needs your model, which is probably gonna turn into a mess, trying to glue your code into the rest of the codebase. It may work a lot better if you wrap up your model in a self-contained package (like [\`tox-block\`](https://pypi.org/project/tox-block/)) that exposes the needed functionality to the outside. But what if that other app isn't written in Python? Maybe it's a Java Android app or a NodeJS web app. You'd need some Frankenstein wrapper code to integrate your package. 

It would be much easier if you'd have some kind of contract between applications that specifies how an input should look like and what kind of output you'll get back - that's the API. Implementing it as a web service, accessible by HTTP requests, allows it to be accessed by almost any application from anywhere, as long as there is internet. The prefix REST, meaning representational state transfer, refers to the architecture style of the API. REST imposes 6 constraints on how the API has to look like to be truly RESTful, such as a uniform interface and a non-dependency and statelessness in the client-server-interaction. You can read a more detailed discussion [here](https://restfulapi.net/rest-architectural-constraints/).

### Building the API with Flask and gunicorn

Setting up the API is actually a fairly easy thing to do. We already have the predictive capabilities from the [\`tox-block\` package](https://github.com/Pascal-Bliem/tox-block) and can use the functions \`tox_block.prediction.make_single_prediction\` or \`tox_block.prediction.make_predictions\` to make predictions. Now we just have to find a way to make those available via HTTP requests. I decided to go with Flask, a micro web framework written in Python, because it is very easy and fast to set it up and it has all the functionality we need. For every Flask application, you'd set up some config, write some routes, and create the app from it (which you can all find in Flask's [quick start guide](https://flask.palletsprojects.com/en/1.1.x/quickstart/)). The interesting part for us are the routes that will wrap the prediction functions. Let's take \`make_predictions\`:

\`\`\`python
import os
from flask import Blueprint, request, jsonify, abort
from tox_block.prediction import make_predictions
from tox_block_api.validation import validate_multiple_inputs
from tox_block import __version__ as _version
from tox_block_api import __version__ as api_version

tox_block_app = Blueprint("tox_block_app", __name__)

@tox_block_app.route("/v1/make_predictions", methods=["POST"])
def api_make_predictions():
    if request.method == "POST":
        # Step 1: Extract POST data from request body as JSON
        input_json = request.get_json()
        
        # Step 2: Validate the input
        input_data, errors = validate_multiple_inputs(input=input_json)
        if not errors is None:
            abort(400, f"Errors occurred when validating the input data: {errors}")

        # Step 3: Model prediction
        prediction = make_predictions(input_texts=input_data)

        # Step 5: Return the response as JSON
        return jsonify({"predictions": prediction,
                        "model_version": _version,
                        "api_version": api_version,
                        })
\`\`\`
We create a route to \`/v1/make_predictions\` that will accept HTTP POST requests with the Flask decorator \`@tox_block_app.route("/v1/make_predictions", methods=["POST"])\`. In the function under the decorator, we read the JSON-formatted input data from the request body, which looks something like this:
\`\`\`python
{"input_data": ["Some texts", "to be classified"]}
\`\`\`
These inputs are passed to the prediction function \`make_predictions\`, and the predictions, together with model and API versions, are returned as JSON in the body of the HTTP response. 

Before we call the prediction function, we should validate the input data which is done by \`validate_multiple_inputs\`:
\`\`\`python
def validate_multiple_inputs(input: List[str]) -> Tuple[List[str], str]:
    
    errors = None
    
    try:
        # check if JSON contains the key "input_texts"
        input = input.get("input_data", None)
        if input is None:
            raise KeyError("The key 'input_data' was not found in the received JSON.") 
        # check if input is list
        if isinstance(input,list):
            # check if list is empty
            if len(input) == 0:
                raise ValueError("Passed an empty list.")
            # check if all list items are non-empty strings
            for i, item in enumerate(input):
                if not isinstance(item,str):
                    raise TypeError(f"The list item at position {i} is not a string.")
                if item == "":
                    raise ValueError(f"The list item at position {i} is an empty string.")
        else:
            raise TypeError("The passed object is not a list of strings.")  
    except (ValueError, TypeError, KeyError) as exc:
        errors = str(exc)   
        
    return input, errors
\`\`\`
This function checks if the input is a list with only non-empty string elements. Any validation errors will be returned and caught by
\`\`\`python
if not errors is None:
            abort(400, f"Errors occurred when validating the input data: {errors}")
\`\`\`
which will cause a HTTP response of type \`400 Bad request\` with the error message in its body.

That's basically all there is to it. You can find all the source code and some usage examples in the \`tox-block-api\` [repo on Github](https://github.com/Pascal-Bliem/tox-block-api).

Now, Flask itself does come with a very simple built-in web server, but it should only be used for development purpose. You can instead run the Flask app on [gunicorn](https://gunicorn.org/), which is a Python WSGI HTTP Server for UNIX. Gunicorn will allow to handle a high volume of traffic and can handle requests in parallel. It internally handles calling of the Flask code by potentially having parallel workers ready to handle requests, whereas the build-in Flask server only handles requests sequentially. Having gunicorn installed and the entry point to the Flask app in some kind of \`run.py\` file, you can run Flask on gunicorn in a production setting with just two command lines:
\`\`\`bash
#!/usr/bin/env bash
export IS_DEBUG=\${DEBUG:-false}
exec gunicorn --bind 0.0.0.0:$PORT --access-logfile - --error-logfile - run:application
\`\`\`

### Packing everything up in a Docker container

What is a container and why bother using it? Probably everyone who has worked with Python knows how annoying dependencies can be, even if you use virtual environment management tools like [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/). Sometimes not all dependencies are available on one package index, you'll start using both \`conda\` and \`pip\` install and they don't keep track of each other, some version requirements are conflicting an you're unsure if it's going to work anyway, yadda yadda yadda. Even if you get all dependencies installed with their right versions, sometimes unexpected things happen when you switch from one operating system to another. 

This is where [Docker](https://www.docker.com/) containers come in really handy. They wrap up the entire application, with everything it needs to run, into one entity that can be executed on a runtime (the Docker runtime in this case). No matter where you want to run the application, as long as you have Docker, you can run the container because it already includes everything it needs to run. Containers are more light weight than virtual machines (VM). They don't each need a hypervisor-controlled guest operating system that requires a fixed amount of resources, but run on the Docker engine which itself runs on the host operating system (as sketched in the figure below). Hence, containers are more flexible in resource allocation, they share a single kernel and can share application libraries. They generally have a lower system overhead compared to VMs, resulting in better performance and faster launch times. Finally, containers can be orchestrated to perform optimally together by systems like [Kubernetes](https://kubernetes.io/) or [Docker Swarm](https://docs.docker.com/engine/swarm/).

![A comparison of Docker containers and virtual machines.](https://i1.wp.com/www.docker.com/blog/wp-content/uploads/Blog.-Are-containers-..VM-Image-1-1024x435.png?ssl=1)

Once Docker is [installed](https://docs.docker.com/get-docker/), creating a container image is fairly easy. All commands need to be included in a [\`Dockerfile\`](https://docs.docker.com/engine/reference/builder/). We'll start building on a Python image that is available on [Docker hub](https://hub.docker.com/), a container image registry:
\`\`\`dockerfile
# inside of Dockerfile
FROM python:3.7.0
\`\`\`
Then we'll create a user to run the app, set the working directory fo the app to run in, environment variables, and copy everything from the local directory into the containers working directory:
\`\`\`dockerfile
RUN adduser --disabled-password --gecos '' tox-block-api-user
WORKDIR /opt/tox_block_api
ENV FLASK_APP run.py
ADD ./ /opt/tox_block_api/
\`\`\`
Then we install all the requirements we need:
\`\`\`dockerfile
RUN pip install --upgrade pip
RUN pip install -r /opt/tox_block_api/requirements.txt
\`\`\`
Make the shell script that'll start the Flask app on gunicorn executable and set ownership for our user:
\`\`\`dockerfile
RUN chmod +x /opt/tox_block_api/run.sh
RUN chown -R tox-block-api-user:tox-block-api-user ./
USER tox-block-api-user
\`\`\`
And finally, expose a port and run the application:
\`\`\`dockerfile
EXPOSE 5000
CMD ["bash", "./run.sh"]
\`\`\`
That's the whole \`Dockerfile\`. Now we can build the Docker image and, from it, launch a docker container:
\`\`\`bash
$ docker build -t tox_block_api:latest
$ docker run --named tox_block_api -d -p 5000:5000 -rm tox_block_api:latest
\`\`\`

Most cloud providers have their own container image registries to which one can push Docker images. I deployed this project on [Render](https://www.render.com/). With Render, it's very easy to deploy Docker containers by pushing the image to their registry and then releasing it as a web app.

### Trying out the ToxBlock API
<a id="usage"></a>

Okay, demo time. Feel free to follow along and test the ToxBlock API yourself, but please keep in mind: the app is running on one of Render's free instances, which is for development purpose only. It won't be able to handle a lot of traffic and if I notice that it is being used heavily, I'll be forced to take it down or introduce authentication. If you want to integrate it into one of your apps, you can easily set it up yourself. It is fully open-source and you can find the usage instructions in the \`README.md\` of its [repo](https://github.com/Pascal-Bliem/tox-block-api). Now, first of all we'll need something that can send HTTP requests. For API testing, I usually use [Postman](https://www.postman.com/), but if you don't want to install anything locally or sign up anywhere, you can use web-based API testing services, e.g. [reqbin.com](https://reqbin.com/). 

Let's check if the web service is up and running by sending a GET request to [\`tox-block-api.onrender.com/health\`](https://tox-block-api.onrender.com/health). You should get get back response saying 200 ok. Render's free instances go into a sleep mode if they're idle, so it may take a few seconds to wake it up after sending the request. You can also check the version of the deep learning model and the API with a GET request to [\`tox-block-api.onrender.com/version\`](https://tox-block-api.onrender.com/version). At the time of writing this post, this should return a JSON like this:
\`\`\`python
{
"api_version": "0.1.0",
"model_version": "0.1.2"
}
\`\`\`
Okay, if that has worked, we can now try to classify some potentially toxic text. Predictions for single strings of text can me made via sending a POST request to the endpoint [\`tox-block-api.onrender.com/v1/make_single_prediction\`]. The request's body should contain JSON data with the key \`input_data\` and string as value:
\`\`\`python
{
    "input_data": "I will kill you, you f***king idiot!"
}
\`\`\`
You should get back status 200 and a JSON looking like
\`\`\`python
{
    "api_version": "0.1.0",
    "model_version": "0.1.2",
    "predictions": {
        "identity_hate": 0.1710592806339264,
        "insult": 0.9883397221565247,
        "obscene": 0.9885633587837219,
        "severe_toxic": 0.7870364189147949,
        "text": "I will kill you, you f***king idiot!",
        "threat": 0.8483908176422119,
        "toxic": 0.9998680353164673
    }
}
\`\`\`
Similarly, predictions for multiple strings of text can me made via sending a POST request to the endpoint [\`tox-block-api.onrender.com/v1/make_predictions\`]. The request's body should contain JSON data with the key \`input_data\` and a list of strings as value:
\`\`\`python
{
    "input_data": ["Good morning my friend, I hope you're having a fantastic day!",
                  "I will kill you, you f***king idiot!"]
}
\`\`\`
The response will contain a JSON with the machine learning model and API version, and for each input element, the original text, and the predicted probabilities for each category of toxicity:
\`\`\`python
{
    "api_version": "0.1.0",
    "model_version": "0.1.2",
    "predictions": {
        "0": {
            "identity_hate": 0.0021067343186587095,
            "insult": 0.00757843442261219,
            "obscene": 0.004466842859983444,
            "severe_toxic": 0.0006274481420405209,
            "text": "Good morning my friend, I hope you're having a fantastic day!",
            "threat": 0.009578478522598743,
            "toxic": 0.05347811430692673
        },
        "1": {
            "identity_hate": 0.17105941474437714,
            "insult": 0.9883397221565247,
            "obscene": 0.9885633587837219,
            "severe_toxic": 0.7870364785194397,
            "text": "I will kill you, you f***king idiot!",
            "threat": 0.8483907580375671,
            "toxic": 0.9998679757118225
        }
    }
}
\`\`\`
In case your input doesn't have the right format you should get a response saying 400 Bad request, along with an error message. For example, if your JSON doesn't contain the key \`"input_data"\`, you'll get 
\`\`\`html
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<title>400 Bad Request</title>
<h1>Bad Request</h1>
<p>Errors occurred when validating the input data: &quot;The key 'input_data' was not found in the received JSON.&quot;</p>
\`\`\`
or if you pass something that cannot be interpreted a string, e.g. \`{"input_data": 123456}\`, you will get
\`\`\`html
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<title>400 Bad Request</title>
<h1>Bad Request</h1>
<p>Errors occurred when validating the input data: The passed object is not a string.</p>
\`\`\`

And that's about it. I hope I was able to bring across why it's a good idea to serve machine learning models as a containerized application via a REST API and how to do so. Please tell me if you end up using **ToxBlock** or the **ToxBlock API** within one of your own projects. Thanks for reading!
`
);
