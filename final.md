---
title: "Final Project"
author: "Christian"
output: html_document 
---

# Final project for Applied Machine Learning
## Predicting exercise quality using Fitbit data
### Christian de Azevedo
This is my final project for the Johns Hopkins / Coursera machine learning course
Data are from http://groupware.les.inf.puc-rio.br/har

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
Read more: http://groupware.les.inf.puc-rio.br/har#ixzz48nBLYBRP

Much thanks to them.

A description of the project is as follows:

> Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).


The goal of this project is to find a machine learning algorithm that can predict how an exercise was performed.


```
## [1] 19622   160
```

```
## [1]  20 160
```

#### Project Steps

1. remove irrelevant variables (time, user identification, etc)
2. remove variables that are missing data
3. subset training data into training/testing for cross-validation
   + 70% training
   + 30% testing
   + 3-fold cross validation
4. Remove variables with near zero variance   
5. Compare some algorithms
   + Random Forest Model
   + glm
6. make predictions on remaining test data using best algorithm



####Remove variables with NA 


```r
trainingMod <- trainingOrig[, colSums(is.na(trainingOrig)) == 0]
testingMod <- testingOrig[, colSums(is.na(testingOrig)) == 0]
dim(trainingMod)
```

```
## [1] 19622    60
```

```r
dim(testingMod)
```

```
## [1] 20 60
```

####Remove variables with near zero variance 

```r
nzv <- nearZeroVar(trainingMod)
trainingMod <- trainingMod[, -nzv]
testingMod <- testingMod[, -nzv]
dim(trainingMod)
```

```
## [1] 19622    59
```

```r
dim(testingMod)
```

```
## [1] 20 59
```


####Remove irrelevant variables

```r
trainingMod <- trainingMod[, -(1:7)]
testingMod <- testingMod[, -(1:7)]
dim(trainingMod)
```

```
## [1] 19622    52
```

```r
dim(testingMod)
```

```
## [1] 20 52
```


####Setup Training and Testing sub data (70-30 split) for Cross Validation


```r
inTrain <- createDataPartition(y = trainingMod$classe, p = 0.7,
                                list = FALSE)
training <- trainingMod[inTrain, ]; testing <- trainingMod[-inTrain, ]
train_control <- trainControl(method="cv", number=3)
```

###Prediction Model Training

####Random Forest Model

```r
rfMod <- train(classe ~ ., method = "rf", data = trainingMod, 
 trControl = train_control)

print(rfMod)
```

```
## Random Forest 
## 
## 19622 samples
##    51 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 13081, 13081, 13082 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9922536  0.9902001
##   26    0.9927632  0.9908449
##   51    0.9891957  0.9863318
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 26.
```

#### Classification Tree

```r
cTreeMod <- train(classe ~ ., method = "rpart", data = trainingMod,
 trControl = train_control)
 
print(cTreeMod)
```

```
## CART 
## 
## 19622 samples
##    51 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 13080, 13082, 13082 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa    
##   0.02983905  0.5142703  0.3736228
##   0.03567868  0.4992901  0.3533663
##   0.06318544  0.4028070  0.1971550
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.02983905.
```


###Prediction Time

```r
finalPrediction <- predict(rfMod, testingMod) 
write.csv(finalPrediction, file = "predictions.csv")
```
