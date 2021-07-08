library(lcmm)

###### PAQUID DATASET #########
# normalize the outxome MMSE
library(NormPsy)
paquid$normMMSE <- normMMSE(paquid$MMSE)
# center and reduce age to avoid numerical issues etc (proust-lima)
paquid$age65 <- (paquid$age-65)/10
head(paquid)
##############################

summary(data_hlme)

head(data_hlme, 30)

m = hlme(fixed=Y~poly(Time, degree=2, raw=TRUE), random=~-1, # raw=T does not work!
     subject='ID', mixture=~poly(Time, degree=2, raw=TRUE),
     classmb=~-1, ng=2, data=data_hlme)
summary(m)
