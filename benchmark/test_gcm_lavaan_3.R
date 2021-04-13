# Now, we'll add a covarince structure in y_i

library(lavaan)

dataset <- read.csv("playground_data/lovedata.csv")
dataset2 <- dataset[,1:4]
names(dataset2)[1] <- "y_i0"
names(dataset2)[2] <- "y_i1"
names(dataset2)[3] <- "y_i2"
names(dataset2)[4] <- "y_i3"

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ -3.*y_i0 + 3.*y_i1 + 9.*y_i2 + 36.*y_i3
y_i0 ~~ y_i1
y_i0 ~~ y_i2
y_i0 ~~ y_i3
y_i1 ~~ y_i2
y_i1 ~~ y_i3
y_i2 ~~ y_i3
'

fit <- growth(model, data=dataset2)
# lavaan WARNING:
# Could not compute standard errors! The information matrix could
# not be inverted. This may be a symptom that the model is not
# identified.
summary(fit)
#                Estimate  Std.Err  z-value  P(>|z|)
# eta0             75.175       NA  
# eta1             -0.115       NA   


#### let's try to build the model more directly:
same_model <- '
# main equation:
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ -3.*y_i0 + 3.*y_i1 + 9.*y_i2 + 36.*y_i3
# compute intercept and slope for eta
eta0 ~ 1
eta1 ~ 1
# covariance in eta
eta0 ~~ eta0
eta1 ~~ eta1
eta0 ~~ eta1
y_i0 ~~ y_i0
y_i1 ~~ y_i1
y_i2 ~~ y_i2
y_i3 ~~ y_i3
y_i0 ~~ y_i1
y_i0 ~~ y_i2
y_i0 ~~ y_i3
y_i1 ~~ y_i2
y_i1 ~~ y_i3
y_i2 ~~ y_i3
y_i0 ~ 0*1
y_i1 ~ 0*1
y_i2 ~ 0*1
y_i3 ~ 0*1
'

sem_fit <- sem(same_model, data=dataset2)
summary(sem_fit)
# Exactly same result as the expression above