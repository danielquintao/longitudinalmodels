import semopy
from semopy import Model, ModelMeans
from semopy.means import estimate_means
from semopy import gather_statistics
import pandas as pd
import numpy as np
from gcm_plot import plot

### GCM model with time-variant disturbances (residual variance, i.e. "eps")
# import data of love in mariage after birth of 1st child
total_data = pd.read_csv("playground_data/lovedata.csv")
data = total_data.iloc[:,0:4].rename(columns={
    'M1':'y_i0','M2':'y_i1','M3':'y_i2','M4':'y_i3'
    }) # love scores 
time = np.array([-3,3,9,36])
datahack = data
datahack['hack'] = 1 # let's try to model the constant 1 explicitly as a manifest variable (to assign variance 0)

# string-description of the model:
# P.S. I think we should define the factors 1 EXPLICITLY so that they are not considered params
#      to be fit
# P.S.S. 1. =~ eta0 + eta1 gives us smth but it seems that the interpreter understood that 1 is
#        a variable name and not a constant
desc = '''
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ -3.*y_i0 + 3.*y_i1 + 9.*y_i2 + 36.*y_i3
eta0,eta1 ~~ eta0 + eta1
# eta0, eta1 ~ 1
eta0,eta1 ~ NA*hack
hack ~~ 0*hack
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
# y_i0 ~ 0*1
# y_i1 ~ 0*1
# y_i2 ~ 0*1
# y_i3 ~ 0*1
'''
mod = ModelMeans(desc)
res = mod.fit(datahack)
print("INFORMATION ON FIT:\n{}\n".format(res)) 
ins = mod.inspect()
print("VARIANCE INSPECTION:\n{}\n".format(ins))
g = semopy.semplot(mod, "model_gcm3.png", plot_covs=False)

### plot
np_data = data.to_numpy()
vector_eta = np.array([
    ins.loc[ins.lval == 'eta0' and ins.op == '~' and ins.rval == 'hack', 'Estimate'].values[0],
    ins.loc[ins.lval == 'eta1' and ins.op == '~' and ins.rval == 'hack', 'Estimate'].values[0]])
print(vector_eta.shape)
print(np_data.shape)
plot(vector_eta, time, np_data)