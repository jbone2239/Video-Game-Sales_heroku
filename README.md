# This project analyzes 15,000+ structured video game sales records to predict top-selling games by region. Using Random forest, the model identifies which features (genre, platform, publisher, etc.) best predict regional success. The trained model is deployed on Heroku for interactive predictions.

- This repository contains everything needed to deploy the Video Game Sales prediction app on Heroku.
- The core files include app.py, which runs the Flask web application, and Procfile, which tells Heroku how to start the app.
- The requirements.txt file lists all Python dependencies required for deployment.
- A saved label_encoder.pkl and model_columns.csv ensure the trained Random Forest model can handle categorical variables and maintain consistent feature inputs during prediction.
- The templates/ folder stores the HTML files for the web interface.
- The Jupyter notebook VGSales.ipynb documents the full analysis, model training, and evaluation steps.
- Finally, the repo includes .gitignore for version control hygiene and PNG screenshots (docker image to heroku.PNG, docker push to heroku.PNG, heroku app test.PNG) showing Docker/Heroku deployment steps and example predictions.
