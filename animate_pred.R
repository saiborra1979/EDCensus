
needed <- c('cowplot','data.table', 'gganimate')
pckgs <- data.frame(installed.packages())
pckgs$Package <- as.character(pckgs$Package)

for (need in needed) {
  if (need %in% pckgs$Package) {
    library(need,character.only = T)
  } else {
    install.packages(need)
  }
}

dir_base <- getwd()
dir_flow <- file.path(dir_base,'..','output','flow')

################################
# --- (1) LOAD IN THE DATA --- #




