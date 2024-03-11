**Predict energy curtailment of wind power plants**\
Marie Gondeck, Shashank Kumar

--- 

**Background**

The increasing integration of renewable energy sources into the grid poses significant challenges for transmission operators striving to maintain its stability. Due to the inherently fluctuating nature of renewable energy generation, unexpected oversupply of electricity is increasingly posing a risk of grid congestion<sup>1</sup>. 

To mitigate instabilities in grid frequency and voltage caused by sudden changes in electricity production or demand, transmission operators employ redispatch measures. **_Redispatch_** involves adjusting the initial dispatch schedule, which outlines the power generation plans of various power plants based on their reported capacity. Thereby, transmission operators can regulate the output of power plants, reducing congestion upstream and downstream, albeit at a cost in form of compensation payments for the plant operators. In Europe, these expenses are typically passed on to consumers through higher grid tariffs, negatively impacting overall economic welfare<sup>2</sup>.

This model trained in this project focuses on **_Curtailment_**, a crucial aspect of redispatch efforts that involves reducing power plant output lower to current availability<sup>3</sup>. The excess energy that remains unused is termed curtailed energy. Curtailment thus results in wasted renewable energy capacity and places significant financial strain on power plant operators and potentially hindering progress toward renewable energy targets. Alternative approaches, such as storing or converting surplus energy, could offer more economically viable solutions.

**Data / EDA**

tbd.

**Class Imbalance**

SMOTE<sup>4</sup>:\
Due to our highly imbalanced dataset (about 10% have a positive target value, i.e. curtailed), the ML model tends to overfit the behaviour of the overrepresented majority class and ignore the behaviour of the minority class. To combat this, a technique called SMOTE (Synthetic Minority Oversampling Technique) is used here, as the nature of the rare event of curtailment makes it impossible to get more minority samples. SMOTE increases the number of samples in the minority class with the hypothesis that the interpolated points from a close neighbour of minority instances can be considered to belong to that minority class.

Stratified k-fold cross-validation<sup>5</sup>:\
When estimating the performance of our models, we were faced with the challenge of highly imbalanced dataset. If randomly splitting our train/test set, it is likely that most folds will have few or no examples of the minority class. This means that some or perhaps many of the model scores will be misleading, as the model only needs to correctly predict the majority class.
Instead of splitting the datset randomly we thus maintained the same class distribution in each test subset by incorporating statification. 
Timeseries split because: tbd.


**Baseline Model**

• not lagged\
• no preprocessing\
• only feature wind gust max


**Streamlit**

For MacOs/Linux users:
```bash
# Create virtual environment, activate it and install streamlit 
pyenv local 3.11.3
python -m venv .streamlit_env
source .streamlit_env/bin/activate
pip install -r requirements.txt
```
```bash
# run the app 
treamlit run app.py
```


----

Sources:\
[1]: [Renewable energy curtailment: A case study on today's and tomorrow's congestion management](https://www.sciencedirect.com/science/article/abs/pii/S0301421517307115)\
[2]: [Integration of day-ahead market and redispatch to increase cross-border exchanges in the European electricity market](https://www.sciencedirect.com/science/article/pii/S030626192031165X)\
[3]: [A review and analysis of renewable energy curtailment schemes and Principles of Access: Transitioning towards business as usual](https://www.sciencedirect.com/science/article/abs/pii/S0301421517307115)\
[4]: [Imbalanced Data ML: SMOTE and its variants](https://medium.com/totalenergies-digital-factory/imbalanced-data-ml-smote-and-its-variants-c69a4b32f7e7)\
[5]: [How to Fix k-Fold Cross-Validation for Imbalanced Classification](https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/)



