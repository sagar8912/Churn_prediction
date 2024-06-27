import pandas as pd

def preprocess_input(input_data):
    """
    Preprocess the input data to match the format used during training.
    """
    # Ensure the columns are in the same order and format as during training
    required_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France',
                        'Geography_Germany', 'Geography_Spain']
    
    # Convert categorical columns
    input_data['Gender'] = input_data['Gender'].map({'Male': 0, 'Female': 1})
    input_data = pd.get_dummies(input_data, columns=['Geography'], drop_first=True)
    
    # Ensure all necessary columns are present
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match training data
    input_data = input_data[required_columns]
    
    return input_data



