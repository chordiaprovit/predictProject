import numpy as np

np.seterr(divide='ignore')


def formatData(data):
    print(data)
    '''convert male and female to 0 and 1'''
    data[0] = data[0].map({'Male': 0, 'Female': 1})

    '''converting string fields into Boolean and assuming na is not married'''
    data[1] = data[1].map({'Yes': True, 'No': False})

    '''converting dependents from string to int assuming 3+ is 3 and missing is 0'''

    data[2] = data[2].map({'0': int(0), '1': int(1), '2': int(2), '3+': int(3)})

    '''covert education to Boolean and assuming blanks are False'''
    data[3] = data[3].map({'Graduate': True, 'Not Graduate': False})

    ''' covert self employed to Boolean and assuming blanks are False'''
    data[4] = data[4].map(dict(Yes=True, No=False))

    '''assuming that loan term is 360 if not filled in'''
    data[7] = data[7].astype(np.float64)

    '''assuming that loan term is 360 if not filled in'''
    data[8] = 0

    ''' converting Property from string to int'''
    data[10] = data[10].map({'Urban': 1, 'Semiurban': 2, 'Rural': 3})

    data = data.drop([0, 1], axis=1)
    return data


def formatDataOHE(data):
    '''assuming that loan term is 360 if not filled in'''
    data.Loan_Amount_Term = data.Loan_Amount_Term.fillna(360, inplace=True)

    '''converting string fields into Boolean and assuming na is not married'''
    # data.Married = data.Married.map({'Yes': True, 'No': False})
    # data.Married.fillna('No', inplace=True)

    '''converting dependents from string to int assuming 3+ is 3 and missing is 0'''
    data.Dependents = data.Dependents.map({'0': int(0), '1': int(1), '2': int(2), '3+': int(3)})
    # data.Dependents.fillna(int(0), inplace=True)

    '''covert education to Boolean and assuming blanks are False'''
    data.Education = data.Education.map({'Graduate': True, 'Not Graduate': False})
    # data.Education.fillna('Not Graduate', inplace=True)

    '''convert male and female to 0 and 1'''
    # data.Gender = data.Gender.map({'Male': 0, 'Female': 1})
    # data.Gender.fillna('Female', inplace=True)

    ''' covert self employed to Boolean and assuming blanks are False'''
    data.Self_Employed = data.Self_Employed.map(dict(Yes=True, No=False))
    # data.Self_Employed.fillna('No', inplace=True)

    # '''assuming that loan term is 360 if not filled in'''
    # data['Loan_Amount_Term'] = data['Loan_Amount_Term'].astype(np.float64)
    print("OHE", data)
    return data