---
title: "A simple deep neural network text classifier API"
date: 2019-04-18T17:30:40+06:00
image: images/blog/tensorflow-0.jpg
author: CEO
---

### How to build a simple sentiment analysis classifier API

---

In this blog post, we would like to show you a minimal example on how to build a **simple deep neural network API** for sentiment analysis. From designing the model and training of the data to the deployment of the API.

A very **common use case for A.I. is to classify text**, which is not an easy task even for advanced models. This type of tool is particularly useful when we need to classify comments, such as reviews in review sites, comments on social media and so on.

With this type of API, you can **save a lot of time** when looking at reviews of your products and services by just focusing on the reviews you care about, automatically moderating negative comments until approval, and so on. The applications are endless.

A huge amount of companies are already using this type of technologies to tackle bad reviews and comments from their customers and figure out what's wrong before it's too late. Without having to manually read thousands of reviews, something that would cost a lot of resources.

#### Step 1 - Setting up our Machine Learning model

First of all, we need to design our Machine Learning Model for this task. And for that we will need a bunch of dependencies.

We start by importing everything we are going to need into a new file called `DNNClassifier.py`

We will be using `Tensorflow` and `Pandas`

``` python
import re
import os
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from custom_log import create_logger
from tensorflow.contrib.learn.python.learn.estimators import run_config
from dnn_loader import MODEL_DIR
```

Note the `crate_logger` and `dnn_loader` imports are from different files in the project that we will get back to later.

Now that we have everything we need, we are going to download and load the training datasets.

``` python

# Load all files from a directory in a Data Frame.


def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(
                re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.Data Frame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.


def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.


def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                         "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                        "aclImdb", "test"))

    return train_df, test_df

# train_df and test_df are 25 thousand rows of sentence, sentiment and polarity
# sentence: the sentence we are analyzing
# sentiment: score of the movie review, i.e. 0-10
# polarity: negative or positive


train_df, test_df = download_and_load_datasets()

```

Those 3 python functions will take care of downloading and loading the data for training our Machine Learning Model.

First function `load_directory_data` with `directory` parameter returns a `Pandas` data frame that will contain the training data.

Second function `load_dataset`, also with a `directory` parameter will call the previously defined function `load_directory_data` with positive and negative reviews for training and return a data frame.

Third function `download_and_load_datasets` downloads the dataset we will be using from the Tensorflow hub and loads the training and test data with `load_dataset`

Finally, we call the downloading and loading of the train and test data with `download_and_load_datasets`

---


Now that we have all the data we need, that is the training and test datasets loaded, we have to define our input functions for Tensorflow, the feature column, the configuration and the estimator, which would be our `DNNClassifier`

``` python

train_df.head()

# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], num_epochs=None, shuffle=True)


# # Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], shuffle=False)


# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df["polarity"], shuffle=False)


embedded_text_feature_column = hub.text_embedding_column(
    key="sentence",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")


config = run_config.RunConfig(
    model_dir=MODEL_DIR,
    save_checkpoints_steps=100,
    keep_checkpoint_max=10
)

# Make Deep Neural Network
estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
    config=config)
```

First, `train_df.head()` prints the data frame's top 5 rows, useful when looking at the output from the terminal.

Then, we define our `train_input_fn` which as the name implies is the train input function, which will be a tensorflow estimator pandas input function, and we will give it the train data frame. This returns our train input function into our `train_input_fn`

Same goes for the `predict_train_input_fn`, which will be assigned the return function for predicting returned by `tf.estimator.inputs.pandas_input_fn`

And also for `predict_test_input_fn`, which is our predict function but on the tet dataset. As returned from `tf.estimator.inputs.pandas_input_fn` but passing the test data frame.

Finally, the `embedded_text_feature_column` is assigned a column by a module spec, in this case `https://tfhub.dev/google/nnlm-en-dim128/1` is the module spec and `sentence` is the key.


Then we set up the configuration. We are importing the model directory from a different file, but basically it looks like this

``` python
MODEL_DIR = os.path.dirname(
    os.getcwd() + "/../data/models/DNNClassifier_model")
```

Now that we have everything set up, we can define our model.

We use the `tf.estimator.DNNClassifier`, which is a deep neural network calssifier.

We set the hidden units to `500` and `100`, which are an iterable number of hidden units per layer. All layers are fully connected, for example, 500 and 100 would mean that the first layer has 500 nodes and the second one has 100.


#### Step 2 - Training our model

``` python

# Training for 1,000 steps means 128,000 training examples with the default
# batch size. This is roughly equivalent to 5 epochs since the training dataset
# contains 25,000 examples.
estimator.train(input_fn=train_input_fn, steps=1000)

# evaluate
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
# Show results
logger.info("Training accuracy: {accuracy}".format(**train_eval_result))
logger.info("Test accuracy: {accuracy}".format(**test_eval_result))


```

We train our model with `estimator.train` and we set 1000 steps. In this case we can benefit from using our GPU for faster computation.

After the 1000 steps are completed, we evaluate the results of training our model and print them on out terminal


``` bash
Training accuracy: 92%
Test accuracy: 85%
```

Now we have a working, trained Machine Learning model we can use for predict on any new text.

---


#### Step 3 - The API

We just want to illustrate a very easy to set up API that can use our trained model to predict on new data. But we don't want to make things too complicated, so we'll use Flask


``` python

# -*- coding: utf-8 -*-
from flask import Flask
import json
from flask_cors import CORS
from flask import request
import tensorflow as tf

from dnn_loader import embedded_text_feature_column, config

app = Flask(__name__)
CORS(app)

estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
    config=config)


@app.route('/api', methods=['POST'])
def api():
    try:
        if request.method == 'POST':
            text = request.form["text"]

            def input_eval_set():
                td = {}
                td["sentence"] = []
                td["sentence"].append(text)
                dataset = tf.data.Dataset.from_tensor_slices(td)
                dataset = dataset.batch(1)
                return dataset.make_one_shot_iterator().get_next()
            pred = estimator.predict(
                input_fn=input_eval_set, predict_keys="classes")
            r = next(pred)["classes"][0]
            if r == b'1':
                return"Positive"
            return "Negative"
    except Exception:
        return "Invalid request"

```

That's the whole API handler code.

First we define an estimator that loads the model we already trained.

Then we define a flask route `/api` with method `POST`.

On it, when a new request is made, we set up an input for our model with the data from the POST request. Then we predict and return the results.

The prediction is made here `r = next(pred)["classes"][0]` which can be `Positive` of it's a 1 or `Negative` if it's a 0.

All the code is available for use on github: https://github.com/octohedron/sclapi which also contains the dnn_loader file that I omitted because it's not so important.


#### Step 4 - Local deployment

Now that we have everything set up, we can run our Flask app in localhost to test that everything is working.

+ 1. Export flask app, `export FLASK_APP=$(pwd)/api.py`
+ 2. Run it with `flask run`
+ 3. Make a `POST` request to `http://127.0.0.1:5000/api` with `form-data` field `text` and some text, i.e. `Today was a wonderful day`, should return `Positive`.

---

Although this is not a production set up, it works for development and illustration of a working deployment.

