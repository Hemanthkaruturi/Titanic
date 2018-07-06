#Importing Data
train <- read.csv(file.choose(), na.strings = c(""," ",NA))

#Finding missing values
sort(colSums(is.na(train)), decreasing = TRUE)

#Finding total number of values in cabin
NROW(train$Cabin)

#Since cabin has more than 50% of missing values
#we can Remove cabin variable
train$Cabin <- NULL

#variables in trainset
names(train)

#Imputing missing values using missForst
#missForest builds a random forest model for each variable.
#Then it uses the model to predict missing values in the variable with the help of observed values
#install.packages('missForest')
library('missForest')

train.imp <- missForest(train[,c(1,2,3,6,11)])
train[,c(1,2,3,6,11)] <- train.imp$ximp


#Error in missing values
train.imp$OOBerror

#converting catogorical variables in to numeric
train$Name <- factor(as.numeric(train$Name))
train$Sex <- factor(train$Sex, labels = c(0,1), levels = c('female','male'))
train$Embarked <- factor(train$Embarked, labels = c(1,2,3), levels = c('C','Q','S'))
train$Ticket <- factor(as.numeric(train$Ticket))

#Converting character variables to numeric
train$Name <- as.integer(train$Name)
train$Sex <- as.integer(train$Sex)
train$Ticket <- as.integer(train$Ticket)
train$Embarked <- as.integer(train$Embarked)



#Feature Scaling
library(caret)
preObj <- preProcess(train[,c(1,3,4,5,6,7,8,9,10,11)], method=c("center", "scale"))
training <- predict(preObj, train[,c(1,3,4,5,6,7,8,9,10,11)])
training$Survived <- train$Survived



#Finding Outliers
plot(train)

#Splitting train data in to two datasets for validation
library(caTools)
set.seed(123)
split <- sample.split(training$Survived, SplitRatio = 0.80)
trainset <- subset(training, split == TRUE)
testset <- subset(training, split == FALSE)

#Taking dependent and indpendent variables
x_train <- trainset[-11]
y_train <- trainset[,11]

x_test <- testset[-11]
y_test <- testset[,11]

#Feature Selection
f_train <- trainset[,c(2,4,5,6,7,9,10,11)]

#Fitting training data to KNN
library(class)
knn_classifier <- knn(train =x_train , test = x_test, cl = y_train, k=5, prob = TRUE)

#Fitting training data to SVM
library(e1071)
svm_classifier = svm(formula = Survived ~ .,
                     data = f_train,
                     type = 'C-classification',
                     kernel = 'sigmoid')


#Predicting the test results
svm_pred <- predict(svm_classifier, newdata = x_test)


#confusion matrix                                                           normal/pca
cm_knn <- table(y_test,knn_classifier)                  #64% before scalling, After scaling 79%
cm_svm <- table(y_test, svm_pred)                       #61% before scalling, After Scaling 73%, After feature selection 72%
