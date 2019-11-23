################################################################################
## COMPARISON STUDY OF MACHINE LEARNING: ANALYSIS OF HEART DISEASE PREDICTION ##
################################################################################

# Load libraries
library(caret)
library(pROC)
library(randomForest)
library(e1071)
library(Metrics)

# Load the dataset
data <- read.csv("processed.cleveland.data.csv", header = FALSE)

# Attach data
attach(data)

# Name the columns
names(data) <-  c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","diagnosis")

# The number of healthy people and sick people in the dataset
sum(data$diagnosis == 0)
sum(data$diagnosis >= 1)

# View the dataset
View(data)

# Show rows and columns in the data
dim(data)

# Calculate the value of missing values in the dataset
sum(is.na(data))

# Ignore NA
data <- na.omit(data)
sum(is.na(data))

# Barplot of diagnosis of heart disease
par(mfrow=c(1,1))
data$diagnosis[data$diagnosis  > 0] <- 1
barplot(table(data$diagnosis), names = c("Not present", "Present"),main = "Diagnosis of Heart Disease", xlab = "Heart Disease", ylab = "Numbers")

# Histogram of Raw data for each variables
par(mfrow=c(2,2))
hist(data$age)
hist(data$sex)
hist(data$cp)
hist(data$trestbps)
hist(data$chol)
hist(data$fbs)
hist(data$restecg)
hist(data$thalach)
hist(data$exang)
hist(data$oldpeak)
hist(data$slope)
hist(data$ca)
hist(data$thal)
hist(data$diagnosis)

# Preprocess data
data$diagnosis <- as.character(data$diagnosis)
data$diagnosis[which(data$diagnosis == "0")] <- "healthy"
data$diagnosis[which(data$diagnosis == "1")] <- "sick"

# Training and Testing data for Validation
set.seed(10)
inTrainRows <- createDataPartition(as.factor(data$diagnosis), p = 0.7, list = FALSE)
trainData <- data[inTrainRows,]
testData <-  data[-inTrainRows,]
nrow(trainData)/(nrow(testData) + nrow(trainData)) # Checking whether really 70%
fitControl <- trainControl(method = "repeatedcv", number = 10,  repeats = 3, classProbs = TRUE, summaryFunction = twoClassSummary)
metric <- "Accuracy"

# View training data and testing data rows and columns
dim(trainData);dim(testData);

# Accuracy
Accuracy = list()

# Logistic regression
set.seed(10)
logRegModel <- train(as.factor(diagnosis) ~., data = trainData, method = 'glm', family = 'binomial', trControl = fitControl, preProcess = c("center","scale"), tuneLength = 10, metric = metric)
logRegPrediction <- predict(logRegModel, testData)
logRegConfMat <- confusionMatrix(logRegPrediction, factor(testData$diagnosis))
Accuracy$logReg <- logRegConfMat$overall['Accuracy']                                 

# Random Forest
set.seed(10)
RFModel <- randomForest(as.factor(diagnosis) ~., data = trainData, importance = TRUE, ntree = 2000, trControl = fitControl, preProcess = c("center","scale"), tuneLength = 10, metric = metric)
RFPrediction <- predict(RFModel, testData)
RFConfMat <- confusionMatrix(RFPrediction, factor(testData$diagnosis))
Accuracy$RF <- RFConfMat$overall['Accuracy']

# Support Vector Machine
set.seed(10)
svmModel <- svm(as.factor(diagnosis) ~., data = trainData, method = "svmRadial", trControl  =  fitControl, preProcess = c("center","scale"), tuneLength = 10, metric = metric)
svmPrediction <- predict(svmModel, testData)
svmConfMat <- confusionMatrix(svmPrediction, as.factor(testData$diagnosis))
Accuracy$svm <- svmConfMat$overall['Accuracy']

# K-Nearest Neighbors
set.seed(10)
knnModel <- train(as.factor(diagnosis) ~., data = trainData, method = "knn", trControl = fitControl, preProcess = c("center","scale"), tuneLength = 10, metric = metric)
knnPrediction <- predict(knnModel, testData)
knnConfMat <- confusionMatrix(knnPrediction, as.factor(testData$diagnosis))
Accuracy$knn <- knnConfMat$overall['Accuracy']

# Accuracy Table
row.names <- names(Accuracy)
col.names <- c("Accuracy")
AccuTable <- as.data.frame(matrix(Accuracy, nrow = 4, ncol = 1))
dimnames(AccuTable) <- list(row.names, col.names)
AccuTable

# Plot the Accuracy Barplot
par(mfrow=c(1,1))
Accu <- c(0.7865169, 0.7752809, 0.8202247, 0.7640449)
names(Accu) <- c("logreg", "rf", "svm", "knn")
plot <- barplot(Accu, main = "Accuracy", xlab = "Machine Learning Algorithms", ylab = "Accuracy", ylim = range(0:1))
text(x = plot, y = Accu, labels = Accu, pos = 3)

# Confusion Matrix and Statistics
logRegConfMat
RFConfMat
svmConfMat
knnConfMat

# Detach data
detach(data)
