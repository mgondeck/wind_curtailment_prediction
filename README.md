# **Predict energy curtailment of wind power plants**
Marie Gondeck, Shashank Kumar 

--- 

**Background**

The increasing integration of renewable energy sources into the grid poses significant challenges for transmission operators striving to maintain its stability. Due to the inherently fluctuating nature of renewable energy generation, unexpected oversupply of electricity is increasingly posing a risk of grid congestion<sup>1</sup>. 

To mitigate instabilities in grid frequency and voltage caused by sudden changes in electricity production or demand, transmission operators employ redispatch measures. **_Redispatch_** involves adjusting the initial dispatch schedule, which outlines the power generation plans of various power plants based on their reported capacity. Thereby, transmission operators can regulate the output of power plants, reducing congestion upstream and downstream, albeit at a cost in form of compensation payments for the plant operators. In Europe, these expenses are typically passed on to consumers through higher grid tariffs, negatively impacting overall economic welfare<sup>2</sup>.

The models trained in this project focuses on **_Curtailment_**, a crucial aspect of redispatch efforts that involves reducing power plant output lower to current availability<sup>3</sup>. The excess energy that remains unused is termed curtailed energy. Curtailment thus results in wasted renewable energy capacity and places significant financial strain on power plant operators and potentially hindering progress toward renewable energy targets. Alternative approaches, such as storage or conversion of surplus energy, could offer more economically viable solutions. However, this requires knowing whether curtailment will take place - predicting this is the aim of this work. 

---

**Results and Outlook**

Our best-performing XGBoost model achieved a precision of 42% in predicting the occurrence of curtailment. To achieve greater precision, the integration of additional features is a key objective. For instance, incorporating grid demand and supply metrics, such as frequency and voltage fluctuations, could indicate stability issues requiring curtailment to maintain grid integrity. Understanding the times of peak electricity demand and including generation data from other renewable sources, such as solar, hydropower, and biomass, could help contextualise the overall energy landscape and its impact on wind energy utilisation. Furthermore, market-related features, including energy prices and carbon credit prices, could also influence curtailment decisions, as economic considerations often dictate energy production strategies.

---

**How to use it & what it is**

The preferred way is to work with Google Colab and Drive, as we store the files and models in a Google Drive folder. However, you can also fork the repository and work with Jupyterlab, in which case you will need to adjust the file paths. 

- **_Data_**: you can navigate through the notebooks in the data folder and download the relevant csv files from the corresponding websites.
- **_Baseline Model_**: the baseline folder contains a very simple prediction model, which acts as a performance benchmark against which our advanced models have been compared. The model predicts curtailment when the maximum wind gust exceeds a threshold of 9 m/s.
- **_Advanced Models_**: we trained two supervised machine learning models, namely XGBoost (Advanced Gradient Boosting) and Extra Trees (Extremely Randomized Trees) and combined them into a Soft Voting Classifier. In addition, we trained a Deep Learning Model. More detailed descriptions of the models can be found below.
- **_EDA/Feature Engineering_**: these notebooks contain in-depth analyses to understand our dataset. In particular, we identified a significant imbalance in the class distribution, which must be addressed to ensure accurate prediction of the minority class. Given that no valid patterns could be identified across the year, month, or day, it was determined that incorporating date-time features is not viable. However, partial autocorrelation plots indicated the potential benefit of creating new features by lagging existing ones (see below for further description). In the case of the ExtraTree Classifier, the number of features was reduced by three (before the lagging), as the Pearson-Correlation and Pairwise Scatterplots indicated the presence of multicollinearity. The features were removed because the ExtraTree randomly selects the split point for each feature selected at a node. In contrast, XGBoost models inherently perform feature selection by evaluating splits that improve the calculated gain, which reflects the improvement in accuracy brought by a feature. Likewise, DL models automatically learn relevant features by adjusting the weights. 
- **_Presentation & Webapp_**: the main folder contains a presentation we created for a general audience that presents our findings. Furthermore, we have developed a web application with Streamlit for our EDA component. This can be used as follows: 

```bash
# for Linux/MacOs users
# create virtual environment, activate it and install streamlit 
pyenv local 3.11.3
python -m venv .streamlit_env
source .streamlit_env/bin/activate
pip install -r requirements.txt

# run the app 
streamlit run app.py
```

---

**Further explanations**

**_Class Imbalance_**: Due to our highly imbalanced dataset (about 10% have a positive target value, i.e. curtailed), the models tend to overfit the behaviour of the overrepresented majority class and ignore the behaviour of the minority class. To combat this, a technique called SMOTE (Synthetic Minority Oversampling Technique) is used here. SMOTE increases the number of samples in the minority class with the hypothesis that the interpolated points from a close neighbour of minority instances can be considered to belong to that minority class<sup>4</sup>.

**_Lagging_**: Lagging is a feature engineering technique used in time series analysis to incorporate past information of a series as input features for predictive models. By shifting the time series data by one or more time steps, known as 'lags', new features are created that represent previous values of the dataset<sup>9</sup>. To determine the optimal number of lags to include as features, one can utilize Partial Autocorrelation Function (PACF) plots. The PACF measures the correlation of a time series with its own lag, excluding correlations from intermediate lags. This provides a view of how many past periods significantly influence the current value. 

**_XGBoost_**: XGBoost is an advanced decision-tree-based ensemble algorithm that employs a gradient boosting framework. Predominantly recognized for its execution speed and model accuracy, XGBoost builds decision trees sequentially, correcting residuals in a level-wise manner where all nodes at a particular depth are expanded simultaneously. This parallel expansion facilitates efficient computation and enhances scalability. The algorithm optimizes the objective function (here 'binary:logistic' for binary classification) through gradient-based optimization, integrating regularization terms (L1 and L2) to mitigate overfitting and ensure model robustness. By calculating gradients for each data instance, XGBoost assesses potential gains from various feature splits, enabling precise feature selection. Each successive tree in the ensemble focuses on reducing the residual errors from its predecessors, culminating in a final prediction that aggregates the weighted contributions of all trees <sup>6</sup>.

**_ExtraTree_**: ExtraTrees is also an ensemble learning method that constructs a multitude of decision trees. However, it introduces additional randomness into the model as it randomly selects cut-points for each feature rather than searching for the best possible thresholds, as is typical in other tree methods. This randomness helps in reducing the variance of the model by sacrificing a slight increase in bias. The final prediction is made by averaging the predictions of all trees, smoothing out predictions and improving generalization to new data <sup>7,11</sup>.

**_Soft Voting_**: A soft voting classifier is an ensemble learning technique that combines multiple classification models to form a meta-model. Unlike hard voting, which counts the votes of classifiers for the most frequent label, soft voting takes into account the probability estimates for all possible outcomes provided by each individual classifier. It calculates the weighted average of these probabilities and then selects the class with the highest average probability as the final prediction. This method allows the ensemble to leverage the confidence levels of the individual classifiers, making it generally more flexible and robust and enhancing both predictive performance and decision confidence <sup>10</sup>.

**_Deep Learning_**: Deep learning is a subset of machine learning that utilizes neural networks with multiple layers to model complex patterns and relationships in data. CNN (Convolutional Neural Network) models are traditionally used for image processing tasks. However, they have been adapted to handle sequential time series data by applying convolutions across the temporal dimension. In a CNN for time series, the input data is treated as a 1D signal, with each time step corresponding to a feature dimension. The convolutional layers consist of filters (kernels) that slide over the input data and extract local patterns or features. By applying convolutions across the temporal dimension, the model can capture patterns and relationships within the time series data. ReLU activation functions are applied to introduce non-linearity, crucial for capturing complex temporal dynamics. Pooling layers follow, reducing feature map dimensions while retaining key temporal features. The model uses the Adam optimization algorithm, known for efficiently handling sparse gradients and aiding in faster convergence. To enhance model training, techniques like adjusting the learning rate upon plateaus in performance improvement, and early stopping prevents overfitting by terminating training when monitored performance metrics cease to improve <sup>8</sup>.


----

References:\
[1]: [Renewable energy curtailment: A case study on today's and tomorrow's congestion management](https://www.sciencedirect.com/science/article/abs/pii/S0301421517307115)\
[2]: [Integration of day-ahead market and redispatch to increase cross-border exchanges in the European electricity market](https://www.sciencedirect.com/science/article/pii/S030626192031165X)\
[3]: [A review and analysis of renewable energy curtailment schemes and Principles of Access: Transitioning towards business as usual](https://www.sciencedirect.com/science/article/abs/pii/S0301421517307115)\
[4]: [Imbalanced Data ML: SMOTE and its variants](https://medium.com/totalenergies-digital-factory/imbalanced-data-ml-smote-and-its-variants-c69a4b32f7e7)\
[5]: [How to Fix k-Fold Cross-Validation for Imbalanced Classification](https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/)\
[6]: [What is XGBoost?](https://www.nvidia.com/en-us/glossary/xgboost/#:~:text=XGBoost%2C%20which%20stands%20for%20Extreme,%2C%20classification%2C%20and%20ranking%20problems.)\
[7]:[ExtraTreesClassifier](https://medium.com/@namanbhandari/extratreesclassifier-8e7fc0502c7)\
[8]: [CNN Model for Time-Series Analysis](https://medium.com/@yashakash.singh7/cnn-model-for-time-series-analysis-3b58b4254790)\
[9]: [Introduction to feature engineering for time series forecasting](https://medium.com/data-science-at-microsoft/introduction-to-feature-engineering-for-time-series-forecasting-620aa55fcab0)\
[10]: [What is Hard and Soft Voting in Machine Learning?](https://ilyasbinsalih.medium.com/what-is-hard-and-soft-voting-in-machine-learning-2652676b6a32)\
[11]: [ML | Extra Tree Classifier for Feature Selection](https://www.geeksforgeeks.org/ml-extra-tree-classifier-for-feature-selection/)



