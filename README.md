# Churn Prediction Model Report

#### Overview
This project aims to predict customer churn using various machine learning techniques, primarily focusing on XGBoost and Logistic Regression. By leveraging customer service data, the model aims to identify potential churners, enabling proactive retention strategies.

#### Data

Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

The dataset has many features  but only the most important were used for the deployment such as:

**InternetService:** Type of internet service used by the customer.

**Contract:** Type of customer contract (month-to-month, yearly, etc.).

**OnlineSecurity:** Whether the customer has opted for online security services.

**PhoneService:** Whether the customer has a phone service.

**TechSupport:** Whether the customer has opted for technical support services.

#### Data Preprocessing

- Categorical features were encoded using one-hot encoding.
- Imbalanced data was addressed using the scale_pos_weight function to create synthetic samples for the minority class.
  
#### Model Evaluation

The XGBoost model was evaluated using accuracy, precision, recall, and F1-score as key metrics. The results are as follows:

**Accuracy Score**

Accuracy: 77.00%


**Classification Report**

                           Precision	Recall	F1-Score	Support
           0  (Not Churn) 	0.88	    0.80	   0.84	    1036
           1  (Churn)	      0.55	    0.69	   0.62	    373
 
**Macro Average:**  Precision: 0.72, Recall: 0.75, F1-Score: 0.73

**Weighted Average:** Precision: 0.79, Recall: 0.77, F1-Score: 0.78

#### Hyperparameter Tuning with GridSearch

To optimize model performance, GridSearchCV was used to find the best hyperparameters:

_**Best Parameters:**

**max_depth:** 1
**n_estimators:** 100
**Best Score:** 80.12%

#### Deployment

The model has been deployed on Streamlit, where users can interact with the churn prediction model by inputting relevant customer details to obtain predictions. The web app allows for user-friendly access to the model's insights and predictions, and can be accessed here https://share.streamlit.io/.

#### Conclusion

This project showcases a machine learning approach to predicting customer churn. The XGBoost model was able to achieve a reasonably high accuracy of 77%. The deployment of this model via Streamlit ensures that business users and decision-makers can interact with the model in real-time.
This process ensured that the model was fine-tuned for optimal performance, leading to improved accuracy in predicting customer churn.

#### Future Work

- Exploring additional features that could improve model performance.
- implementing ensemble methods to combine predictions from different models.
- Continuously updating the model with new customer data to refine predictions.

