#Building a model for weight lifting exercises

###Background
####Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 
####The accuracy of received model -- built by using PCA with threshold 95% and Random Forest -- is about 96.8% on training dataset and 98 % for validation set, and 95% on 19/20 testing set.

##Loading required libraries and preparing data

library(caret)
library(VIM)
library(gridExtra)
library(rpart)
library(MASS)
library(randomForest)


### Getting data
set.seed(12345)
Training_data <- read.csv("pml-training.csv")
Test_data <- read.csv("pml-testing.csv")

##Exploratory analysis
### Remove unneeded variables
####Many given variables are aggregated statistical parameters for summary of given sensors data within time window. We need to keep only raw data and remove them from further analysis.
Variables_to_remove <- "min|max|stddev|total|var|avg|ampl|kurtosis|skewness"
####We can also remove some other variables:
####X, which is numerical order of all records; user_name - does not matter for our analysis;
####cvtd_timestamp which is timestamp in a date format; new_window, 
####raw_timestamp_part1 big values that are corresponded with num_window.

Data <- Training_data[,grep(Variables_to_remove,names(Training_data),invert=T)]
Data<-Data[,grep("X|user_name|raw_timestamp_part_1|cvtd_timestamp|new_window",names(Data),invert=T)]

## Setting training and validation sets
inTraining <- createDataPartition(y=Data$classe,p=.75,list=F) 
training <- Data[inTraining,]
validation <- Data[-inTraining,]
train_predictors <- training[,-c(51)]
train_outcome <- training[,c(51)]
validation_pred <- validation[,-c(51)]
validation_outcome <- validation[,c(51)]

## PCA analysis
####Now exploring how classes are separated by the first plot
train_predictors_scaled <- scale(train_predictors[,c(-1)],center=T,scale=T)
pc <- prcomp(train_predictors_scaled) 
cumsum((pc$sdev^2 / sum(pc$sdev^2))) 
training_predictors_pc <- as.matrix(train_predictors_scaled) %*% pc$rotation[,c(1,2)]
training_predictors_pc <- data.frame(x=training_predictors_pc[,1],y=training_predictors_pc[,2])
q1<-qplot(data=training_predictors_pc,x=x,y=y,col=train_predictors[,c(1)])
q2<-qplot(data=training_predictors_pc,x=x,y=y,col=train_outcome)
grid.arrange(q1,q2,ncol=1)

####Looks like the first 2 layers of PCA only separate those 6 people quite well, but it is not clear how to separate their performance.

## Machine learning
####Cross-validation with allowing parallel calculation to accelerate the process
myCtrl <- trainControl(method="cv",number=3,allowParallel=T)
####Now we can use tree to classify

### Classification tree
Tree <- train(train_outcome~.,method="rpart",data= train_predictors,trControl=myCtrl)
Tree$results 
plot(Tree)

####The accuracy is not good enough, so need more powerful model.Let's try LDA.
## LDA
Lda <- train(train_outcome~.,method="lda",data= train_predictors,trControl=myCtrl)
Lda$results

####Accuracy is around 70% and Kappa - 62,5%.
####We can try to use random forest


## Random Forest
RF <- train(train_outcome~.,data= train_predictors,method="rf",trControl=myCtrl,allowParalell=T)
RF
confusionMatrix(predict(RF, validation_pred),validation_outcome)

####Random forest performance on training set is 99% and on validation is 99%. 
####The model looks very good.

## Testing on test set
test_predictors <- Test_data[,names(train_predictors)]
predict(RF,test_predictors)