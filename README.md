
# Churn Prediction Model Using Machine Learning

Churn (loss of customers to competition) is a problem 
for companies because it is more expensive to acquire a 
new customer than to keep your existing one from leaving. 
Therefore, it is important to understand which of the factors 
contribute most to customer churn and to predict which customers 
will potentially churn based on service-related factors. 
In this paper, we propose Churn Prediction Model to deal with the 
aforementioned problem. 


## About My Project
### EDA and Data Cleaning
The dataset contains 3,333 rows and 21 columns, in which the last
column indicating whether a customer churned or not.
- State
- Account Length
- Area Code
- Phone Number
- International Plan
- Voice Mail Plan
- Number Vmail messages
- Total Day Minutes
- Total Day Call
- Total Day Charge
- Total Eve Minutes
- Total Eve Call
- Total Eve Charge
- Total Night Minutes
- Total Night Call
- Total Night Charge
- Total Intl Minutes
- Total Intl Call
- Total Intl Charge
- Customer Service Call
- Churn or not

Acknowledging that not all the data is crucial for predicting whether the
customer is churned or not, some feature is not used for learning. After splitting the dataset into feature and label, 
this model use train_test_split by Scikit-learn to divide the feature and label into the training set and testing set. 
Then since the data range is varied, I rescale the data with StandardScaler by Scikit-learn and turn String-type data into 
Numeric data with LabelEncoder. One prevalent problem when we deal with real-time data is that it is likely to be imbalanced. 
Hence, I use SMOTE (Synthetic Minority Oversampling Technique) to upsample the data. 

![Data Before Upscaling](https://lh3.googleusercontent.com/cjtt7o4vgZ9NmGywl_x3JqYm4wV09jbV7W4e1TOl70hZHg-YY7dX9XseJTym24jgEsfk-CFLKVCGnOiKPvKNHYLJtIxahHZurYFLd42_wbwBFgTMF34cyf2l5lDWBu7f7SKx-_PgBA=w2400)
![Data After Upscaling](https://lh3.googleusercontent.com/C47VRTYOn0obzAkzKfT_vxXqMx5jp9HMui7ogoi_5YOCTmy3tTulAS24S-lWcRodCyd_JlewHHjU5fh5FbdmBeO5qLnydWqha43Wd4sNCkoyGGkrJrWJZXGQ7K48priiKASDkYRt3g=w2400)

### Model Description
This model applied neural networks and supervised learning to learn from the labeled data. Using Keras Sequential Model to bring out the best performance since Keras is a high-level API of Tensorflow-a popular library for multiple machine learning tasks. The model consists of 4 main Compartments:
1. An input layer with the shape to fit all the cleaned features.
2. An output layer with 'sigmoid' activation.
3. Three fully-connected layers with 'relu' activation. 
4. Between layers is dropout layers to prevent overfitting.

    Since this is a binary classification problem, this model uses a binary cross entropy loss function and the Adam optimizer of Keras. The metrics used are binary accuracy, precision, recall, and AUC. Finally, we calculate f1-score and plot the graph of these metrics and the confusion matrix graph. 
### Training Result

![Data Before Upscaling](https://lh3.googleusercontent.com/KuEqHozzUO0u5bZD6DNXFWkRI4ZrTNLLO0rN15P2_xqTyr5uR8_ec8rGWvW5ewbVSFvCKRjzfliDa0BjXjoAe17BjbvgH7nBsAd0hV62JOhovzvyzUjJjj_xy1BdRpu0TruXxxcCMg=w2400)
![Data After Upscaling](https://lh3.googleusercontent.com/jr1Qr3-fVox5Y_3i3NnQ932h3COnfRJXnflfUEfaySMdspImXmZr30PIoJxoexLi93RwHKivThICCsnhEImEq1mqnulRrMcXHVQNisRoVlVSg18Xyb5nFMli0Vi5_Asol234hoo7SA=w2400)

### Known Issues
- Given that this Churn Prediction Model uses SMOTE to deal with imbalanced data, there could be a probability that the data might be duplicated and cause overfitting, which overall greatly impacts the output.

### Usage Value
- Overall, companies can apply their pre-existed dataset to this model to learn, and then it will make the prediction on the new unlabeled dataset. 
- From the result predicted by the model, companies can provide the customer service that can bring out the highest efficiency and limit the cost in terms of taking care of staying customers and finding a way to attract prominent leaving customers.