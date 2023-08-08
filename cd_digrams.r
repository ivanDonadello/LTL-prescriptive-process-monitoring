library(scmamp)
data <- read.table("media/output/all_results.csv", header=TRUE, sep=",", strip.white = TRUE)
data <- data[, 2:ncol(data)]
colnames(data)[2] <- "Choice + Existence"
colnames(data)[3] <- "Positive rel. + Existence"
colnames(data)[4] <- "Negative rel. + Existence"

plotCD(data, alpha=0.01)