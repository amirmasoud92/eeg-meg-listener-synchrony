# One-sided JZS Bayes factors (Rouder et al., 2009; Cauchy prior scale sqrt(2)/2 on delta).
#   Rscript one_sided_bayes_factors.R tests.csv out.json
# tests.csv: test, subject, value
suppressPackageStartupMessages({library(BayesFactor); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
out <- list(jzs = list())

tests <- read.csv(args[1])
for (tt in unique(tests$test)) {
  x <- tests$value[tests$test == tt]
  bf <- ttestBF(x, mu = 0, nullInterval = c(0, Inf), rscale = "medium")
  out$jzs[[tt]] <- list(bf10_positive = exp(bf@bayesFactor$bf[1]), n = length(x))
}
write(toJSON(out, auto_unbox = TRUE, digits = 8), args[2])
