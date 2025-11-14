NYC Taxi Fare Modeling Script
=============================

This project processes NYC Yellow Taxi trip data and trains a linear regression
model to estimate trip fares using engineered features.

Files included:
- requirements.txt  : required Python packages
- code.txt          : the full program (rename to a .py file when running)
- description.txt   : overview of the workflow
- readme.txt        : usage instructions

Instructions:
1. Place the dataset file `2017_Yellow_Taxi_Trip_Data.csv` in the same directory.
2. Install dependencies:
       pip install -r requirements.txt
3. Rename code.txt to something like:
       taxi_model.py
4. Run the program:
       python taxi_model.py

The script prints progress information, evaluates the model, and displays
diagnostic plots for predictions and residuals.
