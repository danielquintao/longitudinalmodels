library(flexmix)
dataset <- read.csv("benchmark7_to_df.csv", header=TRUE)
m = stepFlexmix(observation ~ time|individual, data=dataset, k = 3, nrep=10,
                control=list(classify="weighted"))
parameters(m)
parameters(m)["sigma",]^2
prior(m)
posterior(m)
clusters(m) 

       