######################################################################
#                       _         _            _           _        #
#    _   _   __ _  ___ (_) _ __  | | __ _   _ | |_  _   _ | | __    #
#   | | | | / _  |/ __|| || '_ \ | |/ /| | | || __|| | | || |/ /    #
#   | |_| || (_| |\__ \| || | | ||   < | |_| || |_ | |_| ||   <     #
#    \__, | \__,_||___/|_||_| |_||_|\_\ \__,_| \__| \__,_||_|\_\    #
#    |___/                                                          #
#    ____                            _  _                           #
#   / __ \   __ _  _ __ ___    __ _ (_)| |    ___  ___   _ __ ___   #
#  / / _  | / _  || '_   _ \  / _  || || |   / __|/ _ \ | '_   _ \  #
# | | (_| || (_| || | | | | || (_| || || | _| (__| (_) || | | | | | #
#  \ \__,_| \__, ||_| |_| |_| \__,_||_||_|(_)\___|\___/ |_| |_| |_| #
#   \____/  |___/                                                   #
#####################################################################
#@author: Yasin KÜTÜK          ######################################
#@web   : yasinkutuk.com       ######################################
#@email : yasinkutuk@gmail.com ######################################
#####################################################################



#Initials#####
options(digits = 4)
if(.Platform$OS.type=="windows"){
  path="d://Dropbox/_My_Research/Coursera-Chapter/03.Data/"
  respath="d://Dropbox/_My_Research/Coursera-Chapter/05.Res/"
  print("Hocam Windows'dasın!")
} else {
  path = "/media/DRIVE/Dropbox/_My_Research/Coursera-Chapter/03.Data/"
  respath = "/media/DRIVE/Dropbox/_My_Research/Coursera-Chapter/05.Res/"
  print("Abi Linux bu!")
}


# Check to see if packages are installed. Install them if they are not, then load them into the R session.
# Package Check Function ####
check.packages <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

# Pack Installations ####
packages<-c('openxlsx','xlsx','dplyr','data.table', 'pander', 'sjPlot', 'stargazer', 'kableExtra', 'summarytools', 'DT', 'htmltools')
check.packages(packages)


# Read Data ####
courseinfo <- openxlsx::read.xlsx(paste0(path,'04.CourseInfo.xlsx'), sheet = 1)


# How many courses were scraped ####
nrow(courseinfo)

# How many Specializations are there?
uniqcourse <- unique(courseinfo$sched2)

# Courses for each specializations
spectable <- data.frame(table(courseinfo$sched2))
spectable <- spectable[spectable$Freq<20,] # Tek ders içinde olanları çıkarıyorum
spectable <- spectable[order(-spectable$Freq),]


# Specialization and the number courses #
view(descr(spectable$Freq))
view(dfSummary(spectable))

view(spectable)

kbl(spectable[1:10, 1:2]) %>%
  kable_styling(fixed_thead = T)

courseinfo$score_count
summary(spectable$Courses)

x <- data.frame(unclass(summary(spectable$Courses)), check.names = FALSE, stringsAsFactors = FALSE)

datatable(head(spectable,20))

pander(head(spectable,10))

courseinfo$courseId
