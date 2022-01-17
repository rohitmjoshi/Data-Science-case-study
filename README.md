# Data-Science-case-study

# Predicting-age-of-abalone-using-regression

## Introduction
  - Abalone is a shellfish considered a delicacy in many parts of the world. An excellent source of iron and pantothenic acid, and a nutritious food resource and farming in Australia, America and East Asia. 100 grams of abalone yields more than 20% recommended daily intake of these nutrients. The economic value of abalone is positively correlated with its age. Therefore, to detect the age of abalone accurately is important for both farmers and customers to determine its price. However, the current technology to decide the age is quite costly and inefficient. Farmers usually cut the shells and count the rings through microscopes to estimate the abalones age. Telling the age of abalone is therefore difficult mainly because their size depends not only on their age, but on the availability of food as well. Moreover, abalone sometimes form the so-called 'stunted' populations which have their growth characteristics very different from other abalone populations This complex method increases the cost and limits its popularity. Our goal in this report is to find out the best indicators to forecast the rings, then the age of abalones.
  
### Dataset
#### Background
 - This dataset comes from an original (non-machine-learning) study and received in December 1995:
    - Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and Wes B Ford (1994)
    - "The Population Biology of Abalone (_Haliotis_ species) in Tasmania. I. Blacklip Abalone (_H. rubra_) from the            North Coast and Islands of Bass Strait",
    - Sea Fisheries Division, Technical Report No. 48 (ISSN 1034-3288).
    - Dataset can be found on [UIC Machine learning repository site](https://archive.ics.uci.edu/ml/datasets/Abalone)
    - **Citation:**
        - There are more than 30 papers that cites this data set. Please find the full list at [UIC Machine learning repository site](https://archive.ics.uci.edu/ml/datasets/Abalone) 
 
#### Description
 - From the original data examples with missing values were removed (the majority having the predicted value missing),    and the ranges of the continuous values have been scaled for use with an ANN (by dividing by 200). For the purpose    of this analysis, we will scale those variables back to its original form by multiplying by 200.
 
 - Total number of observations in dataset: **4176**
 - Total number of variables in dataset : **8**
 
 - Metadata and attribute information:
    - Given is the attribute name, attribute type, the measurement unit and a brief description.  The number of rings is      the value to predict as a continuous value.
   
#### Variable List
   | Name   |      Data Type      |  Measurement | Description |
   |----------|:-------------|:------| :-----------|
   | Sex |  categorical (factor) |  |  M, F, and I (Infant)  |
   | Length |  continuous	 | mm |  Longest shell measurement  |
   | Diameter |  continuous	 | mm | perpendicular to length  |
   | Height |  continuous	 | mm |  with meat in shell  |
   | Whole weight |  continuous	 | grams	 |  whole abalone  |
   | Shucked weight |  continuous	 | grams	 |  weight of meat  |
   | Viscera weight	 |  continuous	 | grams	 |  gut weight (after bleeding)  |
   | Shell weight |  continuous	 | grams	 |  after being dried  |
   | Rings |  continuous	 |  | +1.5 gives the age in years  |

#### Interest

- **Predicting the age of abalone from physical measurements:**  
    - The age of abalone is determined by cutting the shell       through the cone, staining it, and counting the number of rings through a microscope -- a boring and
      time-consuming task.  Other measurements, which are easier to obtain, are used to predict the age.  Further           information, such as weather patterns and location (hence food availability) may be required to solve the             problem.
- **Apply different regression techniques:**
    - We are interested in performing various regression techniques such as additive models, interactions, polynomial transformations of the variables etc to be able to predict and assess the accuracy of our prediction.
- **Beyond regression models:**
    - Is there any other type of machine learning methodology which can predict the age more accurately than using regression model ?

- **Implementation in real application:**
    - Understand whether the data set & the regression models are sufficient to predict the age of abalone accurately enough so that it can be used in real application. 


The goal is to predict the age of abalones using machine learning models. Instead of predicting raw age in years, the final goal will be to predict "young","medium","old"-defined as grouping ring classes 1-8, 9-10, and 11 on.

Classification of Abalones Data Goal: Predicting age of abalones Input variables: Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight, Rings • Sex is either Male ("M"), Female ("F") or Infant ("I"), this is not suitable for regression algorithms, so I created a numeric feature: 1:Male, 2: Female, 3:Infant • There are no missing/null values in dataset. At 2 places height is 0mm , but considering it in mm we can ignore it considering 4000 rows • Added age column in the dataframe for classification. Age is classified based on the number of rings. Rings 1-8 -->Age 1, denoting young Rings 9-10 -->Age 2, denoting middle Rings 11-29 -->Age 3, denoting old

I used 4 classification algorithms and their respective accuracy is as follows:

Models

Neural Network Accuracy : 64.6889952153 Root mean square error(RMSE): 0.6777389936698861 Mean Absolute error(MAE): 0.388516746411

Logistic Regression Accuracy : 65.2631578947 Root mean square error(RMSE): 0.6903296487356203 Mean Absolute error(MAE): 0.39043062201

Random Forest Accuracy : 62.5837320574 Root mean square error(RMSE): 0.7254762501100116 Mean Absolute error(MAE): 0.424880382775

KNN Classifier Accuracy : 61.8181818182 Root mean square error(RMSE): 0.7228333405962345 Mean Absolute error(MAE): 0.428708133971

Conclusion: Neural network gives highest accuracy for classification of age data among all the 4 models, depending on rmse value of 0.677 which is lowest.
