library(lavaan)

dataset <- read.csv("playground_data/lovedata.csv")
dataset2 <- dataset[,1:4]
names(dataset2)[1] <- "y_i0"
names(dataset2)[2] <- "y_i1"
names(dataset2)[3] <- "y_i2"
names(dataset2)[4] <- "y_i3"

# We'll not write the covariances etc explicitly, because:
#
# "Technically, the growth() function is almost identical to the sem() function. 
# But a mean structure is automatically assumed, and the observed intercepts are
# fixed to zero by default, while the latent variable intercepts/means are 
# freely estimated." -- https://lavaan.ugent.be/tutorial/growth.html

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ -3.*y_i0 + 3.*y_i1 + 9.*y_i2 + 36.*y_i3
'

fit <- growth(model, data=dataset2)
summary(fit)
#                Estimate  Std.Err  z-value  P(>|z|)
# eta0             75.427    0.752  100.347    0.000
# eta1             -0.114    0.023   -5.060    0.000

##### let's try with ordinary sem fitter

wrong_fit <- sem(model, data=dataset2)
summary(wrong_fit)
# no intercept for eta0 and eta1

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
# disturbance
# eps0 =~ 1.*y_i0
# eps1 =~ 1.*y_i1
# eps2 =~ 1.*y_i2
# eps3 =~ 1.*y_i3
# eps0 ~~ eps0
# eps1 ~~ eps1
# eps2 ~~ eps2
# eps3 ~~ eps3
y_i0 ~~ y_i0
y_i1 ~~ y_i1
y_i2 ~~ y_i2
y_i3 ~~ y_i3
y_i0 ~ 0*1
y_i1 ~ 0*1
y_i2 ~ 0*1
y_i3 ~ 0*1
'

sem_fit <- sem(same_model, data=dataset2)
summary(sem_fit)
# same result as with simple syntax and growth() fitting
#                Estimate  Std.Err  z-value  P(>|z|)
# eta0             75.427    0.752  100.347    0.000
# eta1             -0.114    0.023   -5.060    0.000