
require("reticulate")

source_python("pickle_reader.py")
AA, BB <- read_pickle_file('mimic_df_np1.pkl')


Ttmts <- AA
Yout <- BB
T <- ncol(AA)

lte <- rep(NA, T-H)
gte <- rep(NA, T-H)
gte_pred <- rep(NA, T-H)

Anodes <- c(sprintf("A%d", T-H:T))
for (t in T-H:T) {
  Anodes <- c(sprintf("A%d", t-H:t))
  glb0 <- ltmle(data, Anodes=Anodes, Lnodes="L", Ynodes=sprintf("Y%d", t), abar=c(rep(0,H))
  glb1 <- ltmle(data, Anodes=Anodes, Lnodes="L", Ynodes=sprintf("Y%d", t), abar=c(rep(1,H))
  llb0 <- ltmle(data, Anodes=sprintf("A%d", t), Lnodes="L", Ynodes=sprintf("Y%d", t), abar=c(0))
  llb1 <- ltmle(data, Anodes=sprintf("A%d", t), Lnodes="L", Ynodes=sprintf("Y%d", t), abar=c(1))
  lte[t] = llb1 - llb0
  gte[t] = glb1 - glb0
}

myts_lte <- ts(lte, start=T-H, end=T, frequency=1)
myts_gte <- ts(gte, start=T-H, end=T, frequency=1)

library(forecast)
library(dse)
for (t in T-H:T) {
  model <- estVARXls (TSdata (input = window(myts_lte, start=t-H,end=t-1), output = window(myts_gte, start=t-H,end=t-1) ))
  z <- forecast(TSmodel(estVARXls(window(myts_lte, start=t-H, end=t)))
  gte_pred[t] <- z[t]*myts_lte[t] 
