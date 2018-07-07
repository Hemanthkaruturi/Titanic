#Importing Data
train <- read.csv(file.choose(), na.strings = c(""," ",NA))

#Finding missing values
sort(colSums(is.na(train)), decreasing = TRUE)

#Data Exploration
library(ggplot2)
library(dplyr)
data <- train
# Total Number of Passengers: 891
NROW(data$PassengerId)

#Total Number of Passengers - Dead:549 Live:342
table(data$Survived)
ggplot(data , aes(x=Survived, fill = Sex)) + geom_bar() + labs(x = 'Dead and Survived') + geom_text(stat='count', aes(label=..count..), vjust=1.50, size=4, color="white",fontface = "bold")


#Total Number of 1st,2nd,3rd class people:216,184,491
table(data$Pclass)


#Total Number of 1st, 2nd, 3rd class people that are survived:136,87,119
count(data[which(data$Survived==1 & data$Pclass==1),])
count(data[which(data$Survived==1 & data$Pclass==2),])
count(data[which(data$Survived==1 & data$Pclass==3),])
ggplot(data, aes(x=Pclass, fill = as.character(Survived))) + geom_bar() + labs(x = 'Passengers class') + geom_text(stat='count', aes(label=..count..), vjust=-0.25, size=4)


# Number of Males:577 Females:314
table(data$Sex)

# Number of Males:109/468 Females:233/81 (survived/dead)
table(data$Sex,data$Survived)
ggplot(data, aes(x=Sex, fill = as.character(Survived))) + geom_bar() + geom_text(stat='count', aes(label=..count..), vjust=-0.25, size=4)


# Number of people under Age 0-10:64 10-20:115 20-30:229 30-40:153 40-50:84 50-60:42 60-70:17 70-80:4 80-90:0 Unknown:177

count(data[which(data$Age>=0 & data$Age<=10),])
count(data[which(data$Age>=11 & data$Age<=20),])
count(data[which(data$Age>=21 & data$Age<=30),])
count(data[which(data$Age>=31 & data$Age<=40),])
count(data[which(data$Age>=41 & data$Age<=50),])
count(data[which(data$Age>=51 & data$Age<=60),])
count(data[which(data$Age>=61 & data$Age<=70),])
count(data[which(data$Age>=71 & data$Age<=80),])


count(data[which(data$Age==999),])

ggplot(data, aes(x=Age, fill = as.character(Survived))) + geom_histogram(binwidth = 10) + geom_text(stat='count', aes(label=..count..), vjust=-0.25, size=4)


# Number of people that are survived under Age 0-10:38 10-20:44 20-30:84 30-40:69 40-50:33 50-60:17 60-70:4 70-80:1 unknown:52
count(data[which((data$Age>=0 & data$Age<=10) & data$Survived==1),])
count(data[which((data$Age>=11 & data$Age<=20) & data$Survived==1),])
count(data[which((data$Age>=21 & data$Age<=30) & data$Survived==1),])
count(data[which((data$Age>=31 & data$Age<=40) & data$Survived==1),])
count(data[which((data$Age>=41 & data$Age<=50) & data$Survived==1),])
count(data[which((data$Age>=51 & data$Age<=60) & data$Survived==1),])
count(data[which((data$Age>=61 & data$Age<=70) & data$Survived==1),])
count(data[which((data$Age>=71 & data$Age<=80) & data$Survived==1),])
count(data[which((data$Age>=81 & data$Age<=90) & data$Survived==1),])

count(data[which(data$Age==999 & data$Survived==1),])

# Total Number of a,b,c,d,e,f,g cabins:15, 47, 59, 33, 32, 13, 4 Missing:687
NROW(grep("^A", data$Cabin, value = TRUE))
NROW(grep("^B", data$Cabin, value = TRUE))
NROW(grep("^C", data$Cabin, value = TRUE))
NROW(grep("^D", data$Cabin, value = TRUE))
NROW(grep("^E", data$Cabin, value = TRUE))
NROW(grep("^F", data$Cabin, value = TRUE))
NROW(grep("^G", data$Cabin, value = TRUE))


# Number of people embarked in Cherbourg, Qeenstown, Southampton: 168, 77, 644
Summary(data)


# Number of people survived embarked in Cherbourg, Qeenstown, Southampton: 93, 30, 217
count(data[which(data$Embarked=='C' & data$Survived==1),])
count(data[which(data$Embarked=='Q' & data$Survived==1),])
count(data[which(data$Embarked=='S' & data$Survived==1),])
count(data[which(is.na(data$Embarked) & data$Survived==1),])
ggplot(data, aes(x=Embarked, fill = as.character(Survived))) + geom_bar() + geom_text(stat='count', aes(label=..count..), vjust=-0.25, size=4)


# Number of people survived with sibblings/spouses: __
#    0   1
# 0 398 210
# 1  97 112
# 2  15  13
# 3  12   4
# 4  15   3
# 5   5   0
# 8   7   0
table(data$SibSp,data$Survived)
ggplot(data, aes(x=SibSp, fill = as.character(Survived))) + geom_bar() + geom_text(stat='count', aes(label=..count..), vjust=-0.25, size=4)


# Number of people survived with Parents/children: __
#   0    1
# 0 445 233
# 1  53  65
# 2  40  40
# 3   2   3
# 4   4   0
# 5   4   1
# 6   1   0
table(data$Parch,data$Survived)
ggplot(data, aes(x=Parch, fill = as.character(Survived))) + geom_bar() + geom_text(stat='count', aes(label=..count..), vjust=-0.25, size=4)


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

#Fitting training data to naive bayes
library(e1071)
nb_classifier = naiveBayes(Survived ~ .,
                           data = trainset)

#Fitting training data to decision tree
library(rpart)
dt_classifier <- rpart(formula = Survived ~ ., data = trainset, method = "class")

#Fitting training data to Random Forest
library(randomForest)
set.seed(123)
rf_classifier <- randomForest(x = x_train, y = y_train, ntree = 500)

#Fitting training data to XGBoost Model
library(xgboost)
xg_classifier <- xgboost(data = as.matrix(x_train), label = y_train, nrounds = 10)

#Fitting training data to Gradient boost model
library(gbm)
gb_classifier <- gbm(Survived ~ ., data = trainset,distribution = "gaussian",n.trees = 10000, interaction.depth = 4, shrinkage = 0.01)

#cross validation
library(caret)
train_control<- trainControl(method="cv", number=10, savePredictions = TRUE)
model<- train(Survived~., data=trainset, trControl=train_control, method="rpart")


#Predicting the test results
svm_pred <- predict(svm_classifier, newdata = x_test)
nb_pred <-  predict(nb_classifier, newdata = x_test)
dt_pred <-  predict(dt_classifier, newdata = x_test, type='class')
rf_pred <-  predict(rf_classifier, newdata = x_test)
xg_pred <-  predict(xg_classifier, newdata = as.matrix(x_test))
xg_pred <- ifelse(xg_pred >= 0.5, 1,0)
n.trees = seq(from=100 ,to=10000, by=100)
gb_pred <- predict(gb_classifier, newdata = x_test, n.trees = n.trees,na.action = na.pass)
kf_pred <- predict(model, newdata = x_test)

#confusion matrix                                                         
cm_knn <- table(y_test,knn_classifier)    #64% before scalling, After scaling 79%

cm_svm <- table(y_test, svm_pred)         #61% before scalling, After Scaling 73%, After feature selection 72%

# nb_log <- ifelse(nb_pred > 0.50, "YES", "NO")
# cm_nb <- table(y_test, nb_log)           


cm_dt <- table(y_test, dt_pred)            #79% accuracy before feature selection

rf_log <- ifelse(rf_pred > 0.50, "YES", "NO")
cm_rf <- table(y_test, rf_log)            #84% accuracy before feature selection

kf_log <- ifelse(kf_pred > 0.50, "YES", "NO")
cm_kf <- table(y_test, kf_log)            #76% accuracy before feature selection

cm_xg <- table(y_test, xg_pred)           #82.5 accuracy before feature selection    

# gb_log <- ifelse(gb_pred > 0.50, "YES", "NO")
# cm_gb <- table(y_test, gb_log)           

