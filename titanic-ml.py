#Import Dataset train.csv
library(readr)
train <- read_csv("../input/train.csv")

#Explore Raw Data
class(train) 
head(train)
str(train)
names(train)


summary(train)

#Visualize variables Age and Fare
hist(train$Age)
hist(train$Fare)
plot(train$Age, train$Fare)

