

# Disaster Prediction Flask App

This Flask application is designed to predict disaster impacts using various environmental and geographical parameters. It utilizes machine learning models for its predictions.

## Features

- Predictions based on user-input data such as wind speed, rainfall, sea level, and other factors.
- Options to select disaster types, development levels, and more for a tailored analysis.
- Visual and informative results page to display the prediction outcome.

## Installation

To run this application, you'll need Python installed on your machine, along with the Flask framework and other dependencies like Pandas, NumPy, and Scikit-Learn.

Install required packages:
   ```bash
   pip install flask pandas numpy scikit-learn
   ```

## Usage

1. Run the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. Input the required data and submit it to receive the disaster impact predictions.

## Data and Models

- The application uses `test.csv` for data analysis (ensure this file is present in the app directory).
- Machine learning models (`out.pkl` and `elct.pkl`) are used for predictions. These should be located in the app directory.

## Templates

- `disaster.html`: The main page where users input data.
- `result.html`: Displays the prediction results.

## Contributions

Contributions to this project are welcome. You can suggest improvements or report issues via GitHub issues.

