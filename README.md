# Full example of LSTM training in TensorFlow, and REST API serving & calling

## Instructions:

The first notebook will create a model, save it, then delete the local variable, reload the model completely, and serve its predictions as a REST API on localhost ([http://127.0.0.1:5000/](http://127.0.0.1:5000/)).

1. Read `1_train_and_save_LSTM.ipynb` and execute all the code. Once fully executed, the notebook will have trained and serialized the neural net to a local `./cache;`, folder which is reused at the end of the notebook to launch the REST API. The last two cells of this notebook that loads the model and serve it could as well be executed as a `.py` file instead of as a notebook.
2. Read `2_call_rest_api_and_eval.ipynb` and while the first notebook is running and has reached the end of the code, execute all the code of the second notebook. The code of the second notebook reads the test data on disks, serializes it to JSON, and calls the REST API on the localhost of the machine. Ideally you'd open your ports to the world so as to use a public IP address, or an URL with a DNS pointing to the machine.
3. If you want more information, dive into the local `.py` files for some more reading on how things works.

Note: you may install all the requirements.txt before starting.

## Future improvements

Note that in the `requirements.txt`, an URL to a yet-unreleased version of Neuraxle is used as we needed to apply some changes for the example to fully work with TensorFlow. We will soon release those changes as `neuraxle==0.2.2` on PyPI. This means that soon, it will be possible to update this `requirements.txt` to use an official version hosted on the Python Package Index (PyPI), installable with `pip install neuraxle==0.2.2`. Once released, the entry for Neuraxle in the `requirements.txt` will look like `neuraxle==0.2.2` as it'll finally be an officially released version.

You may keep up to date by cloning a new version of the project later. Visit the current [LSTM REST API serving demo project's page](https://github.com/Neuraxio/LSTM-Human-Activity-Recognition) to get updates as they are coded.
