
setwd("C:/Users/amolpawal/Downloads/gender-voice-recognition-master/gender-voice-recognition-master")
library(caret)
#install.packages('lattice')
library(lattice)
#install.packages('ggplot2')
library(ggplot2)
set <- read.csv('voice.csv')
set_size <- nrow(set)
training_size <- round(set_size * 0.8)
test_size <- set_size - training_size
trials <- 01
output <- 'output.txt'
accuracies <- list()
fastMethods <- c('rpart', 'knn')
allMethods <- c('rpart', 'knn', 'glm', 'C5.0', 'ctree', 'lda', 'svmLinear', 'rf')
for (trial in 1:trials) {
  print(paste0('Trial ', trial))
  training_set_indices <- sample(set_size, training_size)
  test_set_indices <- setdiff(1:set_size, training_set_indices)
  for (method in allMethods) {
    print(paste0('Using method ', method))
    model <- train(label ~ ., data = set[training_set_indices,], method = method)
    set.test <- set[test_set_indices,]
    prediction <- predict(model, set.test)
    accuracy <- confusionMatrix(prediction, set.test$label)$overall['Accuracy']
    print(accuracy)
    accuracies[[method]][[trial]] <- accuracy
  }
}

 for (method in names(accuracies)) {
  accuracies[[method]] <- mean(accuracies[[method]])
}

sink(output)
print(accuracies)
sink()

### Voice Recognition for Data Science ###
##########################################

### Load Packages
#################
rm(list=ls())
library(readr)
library(MASS)
library(caret)
voice <- read_csv("voice.csv")
summary(voice)

### bring data in shape
#######################
voice$sex <- voice$label
voice$sex[voice$label == "female"] <- 0
voice$sex[voice$label == "male"] <- 1 
voice$sex <- as.factor(voice$sex)
voice$label <- NULL

### Compute Logit Model on all attributes
#########################################
logit1 = glm(sex ~ ., family = binomial(link = "logit"), data = voice)
summary(logit1)
# lets find the ideal attributes for Logit Model.
step <- stepAIC(logit1, direction="both")
### Display ideal attributes
step$anova
### Compute suggested Logit Model
logit2 = glm(sex ~ Q25 + Q75 + kurt + sp.ent + sfm + meanfun + minfun + 
               modindx, family = binomial(link = "logit"), data = voice)
summary(logit2)
# Intercept seems not necessary so, the model is computed without
logit3 = glm(sex ~ 0 + Q25 + Q75 + kurt + sp.ent + sfm + meanfun + minfun + 
               modindx, family = binomial(link = "logit"), data = voice)
summary(logit3)
# The AIC of model3 has decrease a little compared to model2

### Model Evaluation
####################
# split data into subsets -> trainingset(.8) and testset(.2)
x <- createDataPartition(voice$sex, p=0.80, list=FALSE)
training <- voice[x,]
testing <- voice[-x,]
mod_fit <- train(sex ~ 
                   0 + Q25 + Q75 + kurt + sp.ent + sfm + meanfun + minfun + modindx,
                 data=training, method="glm", family="binomial")
### Get an idear aabout accuracy of prediction
predict(mod_fit, newdata=testing)
#predict(mod_fit, newdata=testing, type = "prob")
pred = predict(mod_fit, newdata=testing)
confusionMatrix(data=pred, testing$sex)

### K-Fold Cross Validation
###########################
ctrl <- trainControl(method = "repeatedcv", number = 15, savePredictions = TRUE)
mod_fit <- train(sex ~ 
                   0 + Q25 + Q75 + kurt + sp.ent + sfm + meanfun + minfun + modindx,
                 data=voice, method="glm", family="binomial",
                 trControl = ctrl, tuneLength = 8)
pred = predict(mod_fit, newdata=testing)
confusionMatrix(data=pred, testing$sex)


library(fftw)
library(seewave)
#install.packages('readr')
library(tuneR)
   #install.packages('ggplot2')
library(ggplot2)
library(randomForest)
install.packages('tuneR')
library(tuneR)
humanFrequency <- 280

analyzeWav <- function(file, start = 0, end = Inf,log=FALSE) {
  wave <- file
  tuneWave <- readWave(file.path(getwd(), wave), from = start, to = end, units = "seconds")
  waveSpec <- spec(tuneWave, f = tuneWave@samp.rate, plot = F)
  analysis <- specprop(waveSpec, f = tuneWave@samp.rate, flim = c(0, humanFrequency / 1000), plot = F)
  
  meanfreq <- analysis$mean / 1000
  sd <- analysis$sd / 1000
  median <- analysis$median / 1000
  Q25 <- analysis$Q25 / 1000
  Q75 <- analysis$Q75 / 1000
  IQR <- analysis$IQR / 1000
  skew <- analysis$skewness
  kurt <- analysis$kurtosis
  sp.ent <- analysis$sh
  sfm <- analysis$sfm
  mode <- analysis$mode / 1000
  centroid <- analysis$cent / 1000
  
  fundamental <- fund(tuneWave, f = tuneWave@samp.rate, ovlp = 50, threshold = 5, wl = 2048,
                      ylim = c(0, humanFrequency / 1000), fmax = humanFrequency, plot = F)
  
  meanfun <- mean(fundamental[, 'y'], na.rm = T)
  minfun <- min(fundamental[, 'y'], na.rm = T)
  maxfun <- max(fundamental[, 'y'], na.rm = T)
  
  b <- c(0, 22)
  dom <- dfreq(tuneWave, f = tuneWave@samp.rate, wl = 2048, ylim = c(0, humanFrequency / 1000),
               ovlp = 0, threshold = 5, bandpass = b * 1000, fftw = T, plot = F)[, 2]
  
  meandom <- mean(dom, na.rm = TRUE)
  mindom <- min(dom, na.rm = TRUE)
  maxdom <- max(dom, na.rm = TRUE)
  dfrange <- (maxdom - mindom)
  duration <- (end - start)
  
  changes <- vector()
  for(d in which(!is.na(dom))) {
    change <- abs(dom[d] - dom[d + 1])
    changes <- append(changes, change)
  }
  if(mindom == maxdom) modindx <- 0 else modindx <- mean(changes, na.rm = T) / dfrange
  
  obj <- data.frame(duration, meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp.ent, sfm, mode, 
                    centroid, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx)
  names(obj) <- c("duration", "meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt", "sp.ent", 
                  "sfm","mode", "centroid", "meanfun", "minfun", "maxfun", "meandom", "mindom", "maxdom",
                  "dfrange", "modindx")
  obj
}



new<-data.frame("duration"=0.05, "meanfreq"=0.05, "sd"=0.06, "median"=0.03, "Q25"=0.01, "Q75"=0.09, "IQR"=0.07, "skew"=12.86, "kurt"=274.40, "sp.ent"=0.80, 
                "sfm"=0.49,"mode"=0, "centroid"=0.05, "meanfun"=0.08, "minfun"=0.01, "maxfun"=0.27, "meandom"=0.007, "mindom"=0.007, "maxdom"=0,
                "dfrange"=0, "modindx"=0)


prediction <- predict(model, new)
print(prediction)
cat(prediction, file = 'prediction.txt')

library("randomForest")
analyzedVoice <- readWave("C:/Users/amolpawal/Downloads/gender-voice-recognition-master/gender-voice-recognition-master/voices/angela.wav", from = 1, to = 5, units = "seconds")
#model.forest <- randomForest(label ~ ., data = file)
modelPath <- 'model.forest.rds'
file.path(getwd(), modelPath)
#model.forest <- readRDS('model.forest.rds')
prediction <- predict(model.forest, analyzedVoice)
print(prediction)



