from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load('vgsales_model.pkl')
le = joblib.load('label_encoder.pkl')

# Load model columns 
model_columns = pd.read_csv('model_columns.csv', header=None).squeeze().tolist()

# Dynamically extract values for dropdowns
platforms = sorted([col.replace('Platform_', '') for col in model_columns if col.startswith('Platform_')])
genres = sorted([col.replace('Genre_', '') for col in model_columns if col.startswith('Genre_')])
publishers = sorted([col.replace('Publisher_', '') for col in model_columns if col.startswith('Publisher_')])

@app.route('/')
def home():
    return render_template('index.html',
                           platforms=platforms,
                           genres=genres,
                           publishers=publishers)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        platform = request.form['platform']
        genre = request.form['genre']
        publisher = request.form['publisher']
        year = int(request.form['year'])

        # Build input feature dictionary with all model columns set to 0
        input_dict = {col: 0 for col in model_columns}
        input_dict['Year'] = year

        # Enable one-hot encoded columns based on user input
        plat_col = f'Platform_{platform}'
        genre_col = f'Genre_{genre}'
        pub_col = f'Publisher_{publisher}'

        if plat_col in input_dict:
            input_dict[plat_col] = 1
        if genre_col in input_dict:
            input_dict[genre_col] = 1
        if pub_col in input_dict:
            input_dict[pub_col] = 1

        # Build input DataFrame with correct column order
        input_df = pd.DataFrame([input_dict], columns=model_columns)

        # DEBUG PRINT (optional)
        # print(input_df.columns)

        # Make prediction
        prediction = model.predict(input_df)
        region = le.inverse_transform(prediction)[0]

        return render_template('index.html',
                               prediction_text=f"Predicted Top Region: {region}",
                               platforms=platforms,
                               genres=genres,
                               publishers=publishers)
    
    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}",
                               platforms=platforms,
                               genres=genres,
                               publishers=publishers)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


