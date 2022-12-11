install.packages('../AdniDeterioration/', repos = NULL, type="source")

library(AdniDeterioration)
library(caret)
library(performanceEstimation)
library(doParallel)
library(pROC)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

dat <- read.csv('data/mci_progress.csv')
dat$X <- NULL

boot_dat <- bootstrapping(training = dat, m = 250, group = 'MCI')

#table(boot_dat$last_DX)

mcPerf <- data.frame(ROC = numeric(), Sens = numeric(), Spec = numeric(),
                     Accuracy = numeric(), Kappa = numeric())
mcRep <- 1

ctrl <- trainControl(method = 'cv', number = 5, classProbs = T, 
                     summaryFunction = twoClassSummary,# sampling='smote',
                     verboseIter = F)

grid <- expand.grid(lambda = seq(0, 1, 0.1), alpha = seq(0, 1, 0.1))

for (j in 1:mcRep) {
  # create nrfolds folds and start outer CV
  print(j)
  nrfolds = nrow(boot_dat)/3 
  
  folds <- createFolds(boot_dat$last_DX, k = nrfolds) 
  
  totalnewPrediction <- c(NA)
  length(totalnewPrediction) <- nrow(boot_dat)
  
  totalprobabilities <- c(NA)
  length(totalprobabilities) <- nrow(boot_dat)
  
  for (n in 1:nrfolds){
    
    training <- boot_dat[-folds[[n]],]
    test <- boot_dat[folds[[n]],]
    
    # training$last_DX <- factor(training$last_DX)
    # test$last_DX <- factor(test$last_DX)
    # # missing values imputation
    
    impute_train <- preProcess(training, method = "knnImpute")
    training <- predict(impute_train, training)
    
    impute_test <- preProcess(rbind(training[,-1], test[,-1]),
                              method = "knnImpute")
    
    test[,-1] <- predict(impute_test, test[,-1])
    
    # tuning
    model <- train(last_DX ~ ., training, method = "glmnet", 
                      metric = "ROC",
                      # preProc = c("center", "scale"),
                      tuneGrid = grid,
                      trControl = ctrl)
    
    
    
    ### post processing cross evaluation
    
    # ROC 
    evalResults <- data.frame(last_DX = test$last_DX)
    evalResults$rf <- predict(model, test, type = "prob")[, 1]
    evalResults$newPrediction <- predict(model, test)
    
    # partial ROCs
    # nrow_test <- nrow(test)
    # newPrediction<-c(NA)
    # length(newPrediction) <- nrow_test
    # 
    # for (i in 1:nrow_test){
    #   
    #   rfROC <- roc(evalResults[-i, 'last_DX'], evalResults[-i, 'rf'],
    #                levels = c('CN', 'MCI_AD'))
    #   #rfROC
    #   #plot(rfROC, legacy.axes=T)
    #   
    #   # alternative cutoff
    #   rfThresh<- coords(rfROC, x = 'best', best.method = 'youden')
    #   #rfThresh <- coords(rfROC, x = 'best', best.method = 'closest.topleft')
    #   # rfThresh
    #   
    #   # new predictions
    #   newPrediction[i] <- ifelse(evalResults[i, 'rf'] >= rfThresh[1],
    #                              'MCI_AD', 'CN')
    #   
    # }
    
    totalnewPrediction[folds[[n]]] <- evalResults$newPrediction
    totalprobabilities[folds[[n]]] <- evalResults$rf
  }
  totalnewPrediction <- ifelse(totalnewPrediction == 1, 'CN_MCI',
                               ifelse(totalnewPrediction == 2,
                                      'Dementia', totalnewPrediction))
  totalnewPrediction <- factor(totalnewPrediction, levels = c('CN_MCI',
                                                              'Dementia'))
  
  # confusion matrix all dataset
  cm <- confusionMatrix(totalnewPrediction, boot_dat$last_DX, positive = 'Dementia')
  cm
  
  # perf
  rfROCfull <- roc(boot_dat$last_DX, totalprobabilities, levels = c('CN_MCI',
                                                               'Dementia'))
  
  v <- c(ROC = auc(rfROCfull), cm$byClass[c(1, 2)], cm$overall[c(1, 2)])
  names(v) <- c('ROC', 'Sens', 'Spec', 'Accuracy', 'Kappa')
  v <- data.frame(t(v))
  
  mcPerf <- rbind(mcPerf, v)
}


