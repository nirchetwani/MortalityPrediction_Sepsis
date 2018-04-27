library("sqldf")
library("anytime")
library("dplyr")
setwd("E:/cse_6242_spring2018/project")

#################################### patients/admissions data ######################################
admissions = read.csv(file = "ADMISSIONS.csv", header = T)
patients = read.csv(file = "PATIENTS.csv", header = T)
icu_stays = read.csv(file = "ICUSTAYS.csv", header = T)

names(admissions) <- tolower(names(admissions))
names(patients) <- tolower(names(patients))
names(icu_stays) <- tolower(names(icu_stays))

# query to merge and admissions and patients table - call table 1
query1 = "select a.row_id, a.subject_id, a.admittime, a.dischtime, a.deathtime, a.admission_type, 
          a.admission_location, a.discharge_location, a.religion, a.marital_status, a.ethnicity, a.diagnosis, 
          a.hospital_expire_flag, b.gender, b.dob, b.dod, b.dod_hosp, b.expire_flag from admissions as a 
          left join patients as b on a.subject_id = b.subject_id"

table1 = sqldf(query1)
table1$admit_time_num = as.numeric(as.POSIXct(table1$admittime))
table1$disch_time_num = as.numeric(as.POSIXct(table1$dischtime))

# icu stays time keys
icu_stays$intime_num = as.numeric(as.POSIXct(strptime(icu_stays$intime, "%m/%d/%Y %H:%M")))
icu_stays$outtime_num = as.numeric(as.POSIXct(strptime(icu_stays$outtime, "%m/%d/%Y %H:%M")))


# trying full outer join in dplyr
temp1 = table1 %>%
  full_join(icu_stays, by = c("subject_id"))
temp1$flag = ifelse((temp1$intime_num > temp1$admit_time_num) & (temp1$outtime_num > temp1$disch_time_num), 1, 0)
temp1 = subset(temp1, flag == 1)
colnames(temp1)[1] = "row_id"
temp = ifelse(temp1$deathtime == " ", (anytime(temp1$dod) - anytime(temp1$dob))/(60*60*24*365), 
              (anytime(temp1$dischtime) - anytime(temp1$dob))/(60*60*24*365))

temp1$age = temp
temp1$age = ifelse(temp1$age > 89, 91.4, temp1$age)


# merging the previous table with ICU stays
query2 = "select a.*, b.intime, b.outtime, b.los from table1 as a left join icu_stays as b on a.subject_id = b.subject_id 
          and a.admit_time_num < b.intime_num and a.disch_time_num > b.outtime_num"

table2 = sqldf(query2)
drops <- c("admit_time_num", "disch_time_num")
table2 = table2[ , !(names(table2) %in% drops)]


# creating age variable
temp = ifelse(table2$deathtime == " ", (anytime(table2$dod) - anytime(table2$dob))/(60*60*24*365), 
              (anytime(table2$dischtime) - anytime(table2$dob))/(60*60*24*365))

table2$age = temp
table2$age = ifelse(table2$age > 89, 91.4, table2$age)
temp2 = subset(table2, is.na(table2$intime))
names_filter = colnames(temp2)

final_data = rbind(temp1[, names_filter], temp2)


#################################### lab-events data ######################################
labevents = read.csv(file = "LABEVENTS.csv", header = T)
labitems = read.csv(file = "D_LABITEMS.csv", header = T)
names(labevents) <- tolower(names(labevents))
names(labitems) <- tolower(names(labitems))


query3 = "select a.itemid, a.subject_id, a.charttime, a.value, a.valuenum, a.valueuom, a.flag, 
          b.label, b.fluid, b.category 
          from labevents as a left join labitems as b on a.itemid = b.itemid"
table3 = sqldf(query3)
filter_data = subset(table3, itemid %in% c(50820, 50813, 51222))
write.csv(file = "lab_data_ph_lactate_hb.csv", filter_data, row.names = F)
