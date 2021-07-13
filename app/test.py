import unittest
import app.flaskApp
import requests

class TestFlaskApiUsingRequests(unittest.TestCase):
    def test_predict(self):
        response = requests.get('http://127.0.0.0:8081/predict')
        self.assertEqual(response.json(), {"Gender": "Male","Married": "No","Dependents": "0","Education": "Graduate",
                                           "Self_Employed": "No","ApplicantIncome": 5000,"CoapplicantIncome": 0,
                                           "LoanAmount": 500,"Loan_Amount_Term": 360,"Credit_History": 1,
                                           "Property_Area": "Urban"})
        print(response)

class TestFlaskApi(unittest.TestCase):
    def setUp(self):
        self.app = app.flaskApp.app.test_client()

    def test_hello_world(self):
        response = self.app.get('/predict')
        print(response)

if __name__ == "__main__":
    unittest.main()

