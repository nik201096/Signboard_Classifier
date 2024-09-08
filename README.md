# Signboard_Classifier

# Traffic Sign Classifier Web Application

## Overview

This project is a **Flask-based web application** that allows users to upload an image of a traffic sign and get it classified using a pre-trained deep learning model (LeNet). The application is designed to recognize **43 different traffic sign categories**. It uses a Convolutional Neural Network (CNN) model to predict the category of the traffic sign.

The project demonstrates the integration of **Flask**, **TensorFlow/Keras**, **Docker**, and **Computer Vision** techniques to create a fully functional web application.

### Classes of Traffic Signs
The dataset includes 43 traffic sign classes such as speed limits, no entry signs, pedestrian crossings, and more.

Example classes:
- Speed Limit (20km/h)
- No Entry
- Stop Sign
- Pedestrian Crossing
- Priority Road

## Features

- Upload an image of a traffic sign and classify it.
- Use a pre-trained CNN model based on LeNet.
- Display the uploaded image along with the predicted traffic sign category.
- Clean up uploaded files after classification to avoid unnecessary storage.

## Technologies Used

- **Python**: Programming language used to build the app and train the model.
- **Flask**: Web framework to serve the application.
- **TensorFlow/Keras**: Deep learning framework used to build and load the pre-trained model.
- **HTML/CSS**: Frontend for user interaction.
- **Docker**: For containerizing the application.
- **Jinja2**: Templating engine for rendering HTML pages in Flask.

