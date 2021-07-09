library(flexmix)
data(NPreg)
m1 = flexmix(yn ~ x + I(x^2), data = NPreg, k = 2)
m1
parameters(m1, component = 1)
parameters(m1, component = 2)

# run 10 times and get best results
stepFlexmix(yn ~ x + I(x^2), data = NPreg, k = 2, nrep=10)

# control = list(classify="auto") or control = list(classify="weighted")?
