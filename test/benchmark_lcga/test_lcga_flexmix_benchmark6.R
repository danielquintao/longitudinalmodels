library(flexmix)
dataset <- read.csv("benchmark6_to_df.csv", header=TRUE)
m = stepFlexmix(observation ~ time|individual, data=dataset, k = 2, nrep=10,
                control=list(classify="weighted"))
parameters(m)
parameters(m)["sigma",]^2
prior(m)
posterior(m)
clusters(m) 

### try lcmm
# library(lcmm)
# m2 = hlme(fixed=observation~time, random=~-1,
#          subject='individual', mixture=~time,
#          classmb=~-1, ng=2, data=dataset)
# summary(m2)


       