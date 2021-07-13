import pickle
from flask import Flask, request
import pandas
from pandas import DataFrame
from app.format import *

app = Flask(__name__)

model = pickle.load(open('/Users/provitchordia/PycharmProjects/predictProject/app/lr_model.sav', 'rb'))

@app.route('/predict',methods=['GET'])
def my_app():
    dict = request.get_json()
    if dict is not None :
        vals =DataFrame(list(dict.values())).T
        data = formatData(vals)
        score = model.predict(data)

        if score[0] == 1:
            out = 'score for model = {}'.format(score) + " Loan Approved"
        else:
            out = 'score for model = {}'.format(score) + " Loan Denied"
    else:
        out = "pass"
    print(out)
    return out

# @app.route('/')
# def my_app_test():
#     print("hello world")
#     return {'hello': 'world'}

if __name__ == '__main__':
    app.run(debug=True)
