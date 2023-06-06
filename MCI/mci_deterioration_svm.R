install.packages('../AdniDeterioration/', repos = NULL, type="source")

library(AdniDeterioration)
library(caret)
library(performanceEstimation)
library(doParallel)
library(pROC)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

pre <- Sys.time()

dat <- read.csv('data/mci_progress.csv')
dat$last_DX <- as.factor(dat$last_DX)
dat$X <- NULL

mcPerf <- data.frame(ROC = numeric(), Sens = numeric(), Spec = numeric(),
                     Accuracy = numeric(), Kappa = numeric())
mcRep <- 1

ctrl <- trainControl(method = 'cv', number = 5, classProbs = T, 
                     summaryFunction = twoClassSummary,
                     verboseIter = F)

grid <- expand.grid(degree = 1, scale = seq(0.1, 1, 0.01), C = c(0.25, 0.5))

for (j in 1:mcRep) {
  # create nrfolds folds and start outer CV
  print(j)
  nrfolds = nrow(dat)/3 
  
  folds <- createFolds(dat$last_DX, k = nrfolds) 
  
  totalnewPrediction <- c(NA)
  length(totalnewPrediction) <- nrow(dat)
  
  totalprobabilities <- c(NA)
  length(totalprobabilities) <- nrow(dat)
  
  for (n in 1:nrfolds){
    
    trained <- dat[-folds[[n]],]
    training <- bootstrapping(training = trained, m = 250, group = 'MCI')
    test <- dat[folds[[n]],]
    
    impute_train <- preProcess(training, method = "knnImpute")
    training <- predict(impute_train, training)
    
    impute_test <- preProcess(rbind(training[,-1], test[,-1]),
                              method = "knnImpute")
    
    test[,-1] <- predict(impute_test, test[,-1])
    
    # tuning
    model <- train(last_DX ~ ., training, method = 'svmPoly', 
                   metric = "ROC",
                   # preProc = c("center", "scale"),
                   tuneGrid = grid,
                   trControl = ctrl)
    
    
    
    ### post processing cross evaluation
    
    # ROC 
    evalResults <- data.frame(last_DX = test$last_DX)
    evalResults$rf <- predict(model, test, type = "prob")[, 1]
    evalResults$newPrediction <- predict(model, test)
    
    
    totalnewPrediction[folds[[n]]] <- evalResults$newPrediction
    totalprobabilities[folds[[n]]] <- evalResults$rf
  }
  totalnewPrediction <- ifelse(totalnewPrediction == 1, 'CN_MCI',
                               ifelse(totalnewPrediction == 2,
                                      'Dementia', totalnewPrediction))
  totalnewPrediction <- factor(totalnewPrediction, levels = c('CN_MCI',
                                                              'Dementia'))
  
  # confusion matrix all dataset
  
  cm <- confusionMatrix(totalnewPrediction, dat$last_DX, positive = 'Dementia')
  cm
  
  # perf
  rfROCfull <- roc(dat$last_DX, totalprobabilities, levels = c('CN_MCI',
                                                               'Dementia'))
  
  v <- c(ROC = auc(rfROCfull), cm$byClass[c(1, 2)], cm$overall[c(1, 2)])
  names(v) <- c('ROC', 'Sens', 'Spec', 'Accuracy', 'Kappa')
  v <- data.frame(t(v))
  
  mcPerf <- rbind(mcPerf, v)
}

post <- Sys.time()
write.csv(mcPerf, 'data/mci_svm_boot_inner_mcperf.csv')

