# House_Price_Prediction

#  ExploreHomes – Bangalore House Price Prediction (XGBoost + Flask + Frontend)

ExploreHomes is a complete **end-to-end real estate price prediction platform** that estimates Bangalore house prices using a trained **XGBoost Machine Learning model**.  
The project includes:

- A trained ML pipeline (XGBoost + OneHotEncoder + StandardScaler)
- Flask backend API for prediction
- Fully designed frontend (HTML/CSS/JS)
- Real-time price estimation UI
- Sample listings, hero slider, search filter, and more

## Features

###  **Frontend**
- Modern and clean UI design  
- Hero image slider  
- Real-estate sample cards  
- Search and filter built-in  
- Price prediction page with:
  - Location input
  - Sqft, BHK, Bathroom fields  
  - AI price estimation  
  - Similar property comparison
  - 
### **Backend**
- Flask-based REST API (`/api/predict`)
- Loads trained XGBoost pipeline
- Cleans and normalizes locations before prediction
- Handles unknown locations gracefully (“Other” category)

### **Machine Learning**
- XGBoost Regression Model (R² ≈ 0.82)
- Automatic preprocessing pipeline:
  - OneHotEncoding for location
  - StandardScaler for numeric fields
  - Custom location cleaning
- Outlier handling and feature engineering

## How the ML Model Works

The model is trained using the Bangalore House Dataset.  
The pipeline includes:

- **Feature Engineering**
  - Convert sqft ranges
  - Extract BHK from description
  - Remove unrealistic sqft/BHK ratios
  - Price per sqft analysis

- **Encoding**
  - OneHotEncoder for 120+ Bangalore locations  
  - Unknown locations mapped to “Other”

- **Model**
  - XGBoost Regressor  
  - Tuned hyperparameters  
  - ~87% prediction accuracy

##  Application Preview

### Home Page – Premium Listings & AI Estimates
![ExploreHomes Home](screenshots/home.png)

### AI Price Estimator Page
![ExploreHomes Predict](screenshots/predict.png)

> The above screenshots showcase the current UI and overall user experience of the application.

## Future Improvements

- Auto-suggest locations dropdown
- Google Places API integration
- Interactive map view
- Deployment on Render/Railway/Heroku


