# Nettoyer les données 

kepler <- read.csv("kepler.csv")

cleanDataSet <- function(data.frame.arg, pourcentage) {
  variables.guarder <- c()
  for(variable.index in 1:length(names(data.frame.arg))){
    nombre.nas <- sum(is.na(data.frame.arg[variable.index]))
    if(nombre.nas <= pourcentage * nrow(data.frame.arg)){
      nom.character <- names(data.frame.arg)[variable.index]
      variables.guarder <- c(variables.guarder, nom.character)
    }
  }
  kepler.clean <- subset(data.frame.arg, select = variables.guarder)
  kepler.clean <- na.omit(kepler.clean)
  return(kepler.clean)
}

POURCENTAGE <- 1/10
kepler.clean <- cleanDataSet(kepler, POURCENTAGE)
sum(is.na(kepler.clean))

write.csv(kepler.clean, file = "kepler_clean.csv")
