library(tidyverse)
library(caret)

adni_slim <- read.csv('data/adni_slim.csv')

missing.perc <- apply(adni_slim, 2, function(x) sum(is.na(x))) / nrow(adni_slim)

# adni_slim <- adni_slim[, which(missing.perc <= 0.5)]
adni_slim <- adni_slim[adni_slim$PTMARRY != 'Unknown',]
adni_slim <- adni_slim[adni_slim$last_visit > 0,]

dummies <- dummyVars(last_DX ~., data = adni_slim)
data_numeric <- predict(dummies, newdata= adni_slim)
data_numeric <- as.data.frame(data_numeric)
data_numeric <-data.frame(adni_slim$last_DX, data_numeric)

names(data_numeric)[1] <- 'last_DX'

data_numeric$X <- NULL

y <- data_numeric %>% 
  mutate_all(funs(ifelse(is.na(.), 1, 0)))

names(y) <- paste0(names(y), '_na')

data_numeric <- cbind(data_numeric, y)

mci_progress <- data_numeric[data_numeric$DXMCI == 1,]
mci_progress$last_DX <- factor(ifelse(mci_progress$last_DX == 'Dementia',
                                     'Dementia', 'CN_MCI'),
                              levels = c('CN_MCI', 'Dementia'))

mci_progress$DXCN <- NULL
mci_progress$DXDementia <- NULL
mci_progress$DXMCI <- NULL

write.csv(mci_progress, 'data/mci_progress.csv')
