install.packages('../AdniDeterioration/', repos = NULL, type="source")

library(AdniDeterioration)

dat <- read.csv('data/mci_progress.csv')
dat$X <- NULL

boot_dat <- bootstrapping(training = dat, m = 10000, group = 'MCI')

#table(boot_dat$last_DX)

dat_split <- data_split(dat = boot_dat, y = 'last_DX', size = 0.8)
