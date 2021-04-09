
### This file computes the intercept and slope for the love dataset, but assuming
### that we have the groups (0,0), (1,0), (0,1) encoded by 2 binary variables!
### This is GCM with known groups 

library(lavaan)

dataset <- read.csv("playground_data/lovedata.csv")
temporal_measure <- dataset[,1:4]
names(temporal_measure)[1] <- "y_i0"
names(temporal_measure)[2] <- "y_i1"
names(temporal_measure)[3] <- "y_i2"
names(temporal_measure)[4] <- "y_i3"
groups <- dataset[6:7]
names(groups)[1] <- 'x1'
names(groups)[2] <- 'x2'
dataset2 <- cbind(temporal_measure, groups)

# We'll not write the covariances etc explicitly, because:
#
# "Technically, the growth() function is almost identical to the sem() function. 
# But a mean structure is automatically assumed, and the observed intercepts are
# fixed to zero by default, while the latent variable intercepts/means are 
# freely estimated." -- https://lavaan.ugent.be/tutorial/growth.html

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ -3.*y_i0 + 3.*y_i1 + 9.*y_i2 + 36.*y_i3
eta0 ~ x1 + x2
eta1 ~ x1 + x2
'

fit <- growth(model, data=dataset2)
summary(fit)
#                Estimate  Std.Err  z-value  P(>|z|)
# eta0             75.427    0.752  100.347    0.000
# eta1             -0.114    0.023   -5.060    0.000
# Regressions:
# eta0 ~ x1        -4.082    1.740   -2.345    0.019
# eta0 ~ x2         1.170    1.766    0.662    0.508
# eta1 ~ x1        -0.117    0.052   -2.226    0.026
# eta1 ~ x2         0.044    0.053    0.826    0.409
