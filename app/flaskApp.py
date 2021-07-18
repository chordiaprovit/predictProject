import pickle
from flask import Flask, request, render_template
import pandas
from pandas import DataFrame
from app.format import *

app = Flask(__name__)

model = pickle.load(open('/Users/provitchordia/PycharmProjects/predictProject/app/lr_model.sav', 'rb'))

@app.route('/predict',methods=['POST','GET'])
def my_app():
    Gender =  request.form.get("Gender")
    Married =  request.form.get("Married")
    Dependents =  request.form.get("Dependents")
    Education =  request.form.get("Education")
    Self_Employed =  request.form.get("Self_Employed")
    ApplicantIncome = request.form.get("ApplicantIncome")
    CoapplicantIncome = request.form.get("CoapplicantIncome")
    LoanAmount = request.form.get("LoanAmount")
    Loan_Amount_Term = request.form.get("Loan_Amount_Term")
    Credit_History = request.form.get("Credit_History")
    Property_Area =  request.form.get("Property_Area")

    vals = [Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]

    data = formatData(DataFrame(vals).T)
    print(data.to_numpy())
    score = model.predict(data)
    # dict = request.get_json()
    # if dict is not None :
    #     vals =DataFrame(list(dict.values())).T
    if score[0] == 1:
       out = 'score for model = {}'.format(score) + " Loan Approved"
    else:
       out = 'score for model = {}'.format(score) + " Loan Denied"

    print(out)
    return render_template("index.html", out=out)

@app.route('/')
def my_app_test():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
