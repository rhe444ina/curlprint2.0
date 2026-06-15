# CurlPrint
A machine learning project that uses a neural network to design personalized 3D-printable hair tools based on an individual's hair profile.

Given features such as curl pattern, hair density, strand thickness, porosity, and styling goals, the model predicts the optimal tool geometry: including tooth spacing, tooth length, handle angle, and tip shape, and generates a custom 3D-printable design. 

## CURRENT STATUS

THIS IS AN EARLY PROTOTYPE. The current version demonstrates the core pipeline: encoding a hair profile, predicting tool dimensions, and generating a simple STL comb model. I am currently refactoring the codebase/improving model evaluation and expanding the dataset.

## Motivation
Many hair tools are designed as one-size-fits-all products despite significant differences in curl patterns and strand thickness. CurlPrint explores whether ML can generate personalized, 3D-printable hair tools tailored to an individual's hair characteristics. The idea was inspired by my own experiences with naturally textured 3A hair and the lack of customizable hair tools. It's the reason I went straight natural.

## Features
- Predicts comb dimensions from abovementioned hair characteristics
- Generates a 3D-printable STL model
- Modular Python pipeline
- Easily extendable with additional training data, taken from WOC @ Georgia Tech

## Tech Stack
- Python
- NumPy
- scikit-learn
- trimesh
- numpy-stl
