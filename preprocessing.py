import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting 
    important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """
    #Defining the map function
    # def binary_map(feature):
        # return feature.map({'Yes':1, 'No':0})
    
    # These columns are not useful as the feature us used for identification of employees. 
    # df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'],axis=1,inplace = True)

    # Encode binary categorical features
    # binary_list = ['SeniorCitizen','Dependents', 'PhoneService', 'PaperlessBilling']
    # df[binary_list] = df[binary_list].apply(binary_map)
    
    # Encoding gender category
    df['Sex'] = df['Sex'].map({'male':1, 'female':0})

    
    #Drop values based on operational options
    if (option == "Online"):
        columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked_Q', 'Embarked_S']

        #Encoding the other categorical categoric features with more than two categories
        # df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
        df = pd.get_dummies(df, drop_first=True).reindex(columns=columns, fill_value=0)
        
        
        
    elif (option == "Batch"):
        pass
        df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
        columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked_Q', 'Embarked_S']
        
        #Encoding the other categorical categoric features with more than two categories
        df = pd.get_dummies(df, drop_first=True).reindex(columns=columns, fill_value=0)
    else:
        print("Incorrect operational options")


    #feature scaling
    sc = MinMaxScaler()
    df['Age'] = sc.fit_transform(df[['Age']])
    # df['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']])
    # df['TotalCharges'] = sc.fit_transform(df[['TotalCharges']])
    return df
        




