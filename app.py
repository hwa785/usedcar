from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result.html', methods=['GET','POST'])
def result():
    if request.method=='GET':
        brand = request.args.get('brand')
        year = request.args.get('year')
        mileage = request.args.get('mileage')
        mpg = request.args.get('mpg')
        enginesize = request.args.get('enginesize')

        df = pd.read_csv('./usedcar.csv')

        train, test = train_test_split(df, test_size=0.2, random_state=2)
        train, val = train_test_split(train, test_size=0.2, random_state=2)

        target = 'price'
        features = ['year','mileage','mpg','engineSize','brand'] 

        X_train = train[features]
        y_train = train[target]  # fit

        X_val = val[features]
        y_val = val[target]  # 정확도 판별, 예측 오류 평가

        X_test = test[features]
        y_test = test[target]  # 일반화 오류 평가

        pipe_3 = make_pipeline(
            OneHotEncoder(use_cat_names=True), 
            SimpleImputer(),
            XGBRegressor()
        )

        tt = TransformedTargetRegressor(regressor=pipe_3, func=np.log1p, inverse_func=np.expm1)

        tt.fit(X_train, y_train)

        row = pd.DataFrame(columns=['year','mileage','mpg','engineSize','brand'])
        row.loc[1] = [int(year),int(mileage),float(mpg),int(enginesize),brand]

        price = float(tt.predict(row))
        price = round(price,2)


        return render_template('result.html',price=price,brand=brand)
    else: return render_template('result.html')

if __name__ == "__main__":
    app.run(debug=True)