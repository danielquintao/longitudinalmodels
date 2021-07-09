library(flexmix)
dataset <- read.csv("benchmark9_to_df.csv", header=TRUE)
m = stepFlexmix(observation ~ (time+I(time^2)|individual), data=dataset, k = 3, nrep=10,
                control=list(classify="weighted", iter.max=500))
parameters(m)
parameters(m)["sigma",]^2
prior(m)
posterior(m)
clusters(m) 

       