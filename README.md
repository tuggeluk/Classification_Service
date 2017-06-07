# ShapeClassifier

## What is ShapeClassifier
ShapeClassifier is a webservice that let you classify musical notation

## What is it made of ?
ShapeClassifier is based on the microframework `Flask`. All of it's code is in the `main.py` file, alongside with the classifier module which is in the `classifier` directory.  
The main route is `/classify`, it lets you upload an image tries to classify it and then returns a string containing the name of what has been recognized.  
Actual requirements are the followings :
- images must be 64px wide and 128px high
- the classifier works better if the symbol to recognize is centered in the image and provide some additional context (surrounding of the symbol)  

Only one configuration is supported right now. You can choose whether to save the images uploaded in the directory specified by `app.config['UPLOAD_FOLDER']` by setting `app.config['SAVE_PATCHES_ON_DISK']` to `True` or `False`

## Running ShapeClassifier
To run the classifier using docker just use `./install_server.sh`  
To run it without docker you can use `./start_server.sh` but this will then require you to have `Flask` installed  
Basically `./start_server.sh` just runs `Flask` with the right application (here `main.py`)
