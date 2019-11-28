#Importing the dataset
setwd("C:\\Users\\Saptarshi Datta\\Desktop\\R PROGRAMMING")
data=read.csv("Cars_edited.csv",header=TRUE)
str(data)

#Taking backup of numeric data 
data.numeric=data
data=na.omit(data)

# Target variable
data$Transport=ifelse(data$Transport=="Car",1,0)
length(which(data$Transport=="1"))*100/nrow(data)
data$Gender=ifelse(data$Gender=="Male",1,0)

# Correlation check
library(corrplot)
correlations = cor(data) 
corrplot(correlations, type="lower", diag = FALSE)


#converting to factor variables
summary(data)
str(data)
data$Engineer=as.factor(data$Engineer)
data$MBA=as.factor(data$MBA)
data$license=as.factor(data$license)
data$Transport=as.factor(data$Transport)
data$Gender=as.factor(data$Gender)
str(data)


###Basic EDA
library(funModeling)
library(tidyverse) 
library(Hmisc)

basic_eda <- function(data)
{
  summary(data)
  df_status(data)
  freq(data) 
  profiling_num(data)
  hist(data)
  describe(data)
  
}
basic_eda(data)

# Checking null data
sapply(data,function(x) sum(is.na(x)))


# Checking # of unique values in each column
sapply(data,function(x) length(unique(x)))
str(data)


attach(data)


# For continuous variables

boxplot(Age~Transport)
summary(Age)
ggplot(data, aes(x = Age)) + 
  geom_density(aes(fill = Transport), alpha = 0.3) + 
  scale_color_manual(values = c("#868686FF", "#EFC000FF")) + 
  scale_fill_manual(values = c("darkturquoise", "lightcoral")) + xlim(15,50)

boxplot(Work.Exp~Transport)
summary(Work.Exp)
ggplot(data, aes(x = Work.Exp)) + 
  geom_density(aes(fill = Transport), alpha = 0.3) + 
  scale_color_manual(values = c("#868686FF", "#EFC000FF")) + 
  scale_fill_manual(values = c("darkturquoise", "lightcoral")) + xlim(-10,30)

boxplot(Salary~Transport)
summary(Salary)
ggplot(data, aes(x = Salary)) + 
  geom_density(aes(fill = Transport), alpha = 0.3) + 
  scale_color_manual(values = c("#868686FF", "#EFC000FF")) + 
  scale_fill_manual(values = c("darkturquoise", "lightcoral")) + xlim(-5,70)

boxplot(Distance~Transport)
summary(Distance)
ggplot(data, aes(x = Distance)) + 
  geom_density(aes(fill = Transport), alpha = 0.3) + 
  scale_color_manual(values = c("#868686FF", "#EFC000FF")) + 
  scale_fill_manual(values = c("darkturquoise", "lightcoral")) + xlim(-5,30)

# For categorical features

ggplot(data, aes(x = Gender, fill = Transport)) + 
  geom_bar(width = 0.25, alpha=0.5) + 
  scale_fill_manual(values = c('darkturquoise', 'lightcoral'))

prop.table(table(Gender,Transport),1)*100

ggplot(data, aes(x = Engineer, fill = Transport)) + 
  geom_bar(width = 0.25, alpha=0.5) + 
  scale_fill_manual(values = c('darkturquoise', 'lightcoral'))

prop.table(table(Engineer,Transport),1)*100

ggplot(data, aes(x = MBA, fill = Transport)) + 
  geom_bar(width = 0.25, alpha=0.5) + 
  scale_fill_manual(values = c('darkturquoise', 'lightcoral'))

prop.table(table(MBA,Transport),1)*100

ggplot(data, aes(x = license, fill = Transport)) + 
  geom_bar(width = 0.25, alpha=0.5) + 
  scale_fill_manual(values = c('darkturquoise', 'lightcoral'))

prop.table(table(license,Transport),1)*100


library(caret)

#Data Slicing for unbalanced dataset
set.seed(1234)
library(caTools)
sample = sample.split(data$Transport, SplitRatio = .70)
train = subset(data, sample == TRUE)
test  = subset(data, sample == FALSE)


prop.table(table(data$Transport))
prop.table(table(train$Transport))
prop.table(table(test$Transport))



#working with SMOTE
library(DMwR)
balanced.data.train <- SMOTE(Transport ~., train, perc.over = 4700, k = 5, perc.under = 200)
#in SMOTE we have to define our equation
#perc.over means that 1 minority class will be added for every value of perc.over
prop.table(table(balanced.data.train$Transport))
str(balanced.data.train)
summary(balanced.data.train)
summary(train)
#now we have increased the minority class. We are adding 48 for every minority class sample. - perc.over
#We are subtracting 10 for every 100 - perc.under. We are taking out of the majority class as well.


options(scipen=999)


#Logistic Regression
logit_model1 = glm(Transport ~ ., data = balanced.data.train, 
                   family = binomial(link="logit"))
summary(logit_model1)

#Checking Multicollinearity
library(car)
vif(logit_model1)

#Removing Multicollinearity 1426
logit_model2 = glm(Transport ~ Age+Gender+Salary+Distance+license, data = balanced.data.train, 
                   family = binomial(link="logit"))
summary(logit_model2)
vif(logit_model2)
#We observe,  model 2 is better than Model 1

# Likelihood ratio test
library(lmtest)
lrtest(logit_model2)


# Pseudo R-square
library(pscl)
pR2(logit_model2)


# Odds Ratio
exp(coef(logit_model2))


# Probability
exp(coef(logit_model2))/(1+exp(coef(logit_model2)))


# Performance metrics (balanced data set train)
library(caret)
balanced.data.train$pred = predict(logit_model2, data=balanced.data.train, type="response")
balanced.data.train$pred = ifelse(balanced.data.train$pred>0.5,1,0)
confusionMatrix(table(balanced.data.train$Transport,balanced.data.train$pred))


# Performance metrics (data set - test)
library(caret)
test$pred = predict(logit_model2, newdata=test, type="response")
test$pred = ifelse(test$pred>0.5,1,0)
confusionMatrix(table(test$Transport,test$pred))


# ROC plot
library(ROCR)
balanced.train.roc = prediction(balanced.data.train$pred, balanced.data.train$Transport)
plot(performance(balanced.train.roc, "tpr", "fpr"), 
     col = "red", main = "ROC Curve for train data")
abline(0, 1, lty = 8, col = "blue")

# AUC
balanced.train.auc = performance(balanced.train.roc, "auc")
balanced.train.auc = as.numeric(slot(balanced.train.auc, "y.values"))
balanced.train.auc

# KS
ks.balanced.train <- performance(balanced.train.roc, "tpr", "fpr")
balanced.train.ks <- max(attr(ks.balanced.train, "y.values")[[1]] - (attr(ks.balanced.train, "x.values")[[1]]))
balanced.train.ks

# Gini
balanced.train.gini = (2 * balanced.train.auc) - 1
balanced.train.gini


### Model Building - KNN

#Use KNN Classifier 
#normalize the test & train data
norm=function(x){(x-min(x))/(max(x)-min(x))}
norm.balanced.data=as.data.frame(lapply(balanced.data.train[,c(1,5,6,7)],norm))
norm.balanced.data=cbind(balanced.data.train[,c(2,3,4,8,9)],norm.balanced.data)
test.knn=as.data.frame(lapply(test[,c(1,5,6,7)],norm))
test.knn=cbind(test[,c(2,3,4,8,9)],test.knn)

str(balanced.data.train)

#KNN Algorithm
library(class)
knn.pred = knn(norm.balanced.data[,-c(5)], test.knn[,-c(5)], norm.balanced.data[,5], k =5) 
confusionMatrix( table(test.knn$Transport, knn.pred))

### Naive Bayes

library(e1071)
NB = naiveBayes(Transport ~ Age+Gender+license, data = balanced.data.train[,-10])
predNB = predict(NB, test, type = "class")
confusionMatrix(table(test[,9], predNB))

#Bagging and Boosting

#loading a few libraries
library(gbm)          # basic implementation using AdaBoost
library(xgboost)      # a faster implementation of a gbm
library(caret)        # an aggregator package for performing many machine learning models

#Let's start using bagging



library(ipred)
library(rpart)

#we can modify the maxdepth and minsplit if needed
#r doc, https://www.rdocumentation.org/packages/iprebaggingd/versions/0.4-0/topics/
data.bagging <- bagging(Transport ~.,data=train,control=rpart.control(maxdepth=5, minsplit=4))
test$pred.class <- predict(data.bagging, test)
confusionMatrix(table(test$Transport,test$pred.class))
#we are comapring our class with our predicted values
#Bagging can help us only so much when we are using a data set that is such imbalanced.

#Boosting

features.train=data.matrix(train[,1:8])
target.train=data.matrix(train[,9])
features.test=data.matrix(test[,1:8])

#in this code chunk we will playing around with all the values untill we find the best fit
#let's play with shrinkage, known as eta in xbg
tp_xgb<-vector()
lr <- c(0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1)
md<-c(1,3,5,7,9,15)
nr<-c(2, 50, 100, 1000, 10000)
for (i in nr) {
  
  xgb.fit <- xgboost(
    data = features.train,
    label = target.train,
    eta = 0.1,
    max_depth = 3,
    nrounds = 2,
    nfold = i,
    objective = "binary:logistic",  # for regression models
    verbose = 1,               # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  

  test$xgb.pred.class <- predict(xgb.fit, features.test)

  tp_xgb<-cbind(tp_xgb,sum(test$Transport==1 & test$xgb.pred.class>=0.5))
  #if your class=1 and our prediction=0.5, we are going to display it with the next line compare the same algorithm     for different values
  
}



xgb.fit <- xgboost(
  data = features.train,
  label = target.train,
  eta = 0.1,
  max_depth = 7,
  nrounds = 2,
  nfold = 5,
  objective = "binary:logistic",  # for regression models
  verbose = 1,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)
test$xgb.pred.class=ifelse(test$xgb.pred.class>=0.5,1,0)
confusionMatrix(table(test$Transport,test$xgb.pred.class))
#here there is significant imporvement over all the models that we have done so far

