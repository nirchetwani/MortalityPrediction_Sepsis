library("sqldf")
library("anytime")
library("dplyr")
setwd("E:/cse_6242_spring2018/project/data_raw/data_csvs")

fun_closest <- function(data, subjectid, time_val) {
  temp = subset(data, subject_id == subjectid)
  if(nrow(temp) > 0) {
    min_diff_temp = min(abs(temp$datetime - time_val))
  }
  index = which(colnames(data) == "valuenum")
  result = temp[which(abs(temp$datetime - time_val) == min_diff_temp), index]
  return(ifelse(result[1]> 0, result[1], 0))
}

labdata = read.csv(file = "lab_data_ph_lactate_hb.csv", header = T)
temp_vals = read.csv(file = "temp_vals.csv", header = T)
scvo2_vals = read.csv(file = "scvo2_vals.csv", header = T)
spo2_vals = read.csv(file = "spo2_vals.csv", header = T)
colnames(temp_vals) = tolower(colnames(temp_vals))
colnames(spo2_vals) = tolower(colnames(spo2_vals))
colnames(scvo2_vals) = tolower(colnames(scvo2_vals))

septic_patients = read.csv(file = "explicit_septic_patients.csv", header = T)
colnames(septic_patients) = tolower(colnames(septic_patients))
unique_septic = unique(septic_patients$subject_id)

#chartevents = read.csv(file = "chartevents_scvo2data1-7.csv", header = T)

scvo2_vals = subset(scvo2_vals, subject_id %in% unique_septic)
labdata = subset(labdata, subject_id %in% unique_septic)
spo2_vals = subset(spo2_vals, subject_id %in% unique_septic)
temp_vals = subset(temp_vals, subject_id %in% unique_septic)


temp_vals$datetime = anytime(temp_vals$charttime)
spo2_vals$datetime = anytime(spo2_vals$charttime)
scvo2_vals$datetime = anytime(scvo2_vals$charttime)

#unique_patients = unique(scvo2_vals$subject_id)

scvo2_vals$time_num = as.numeric(as.POSIXct(scvo2_vals$charttime))

spo2_vals$time_num = as.numeric(as.POSIXct(spo2_vals$charttime))

temp_vals$time_num = as.numeric(as.POSIXct(temp_vals$charttime))

scvo2_vals$spo2_agg = 0
scvo2_vals$temp = 0
scvo2_vals$ph = 0
scvo2_vals$lactate = 0
scvo2_vals$hb = 0
# scvo2_vals$temp_measure = 0

temp_lactate = subset(labdata, itemid %in% c(50813) & subject_id %in% unique_septic)
temp_lactate$datetime = anytime(temp_lactate$charttime)
temp_hb = subset(labdata, itemid %in% c(51222) & subject_id %in% unique_septic)
temp_hb$datetime = anytime(temp_hb$charttime)
temp_ph = subset(labdata, itemid %in% c(50820) & subject_id %in% unique_septic)
temp_ph$datetime = anytime(temp_ph$charttime)

# 0.006944444 is the value for 10 mins

check_spo2 = 0

for(i in 1:nrow(scvo2_vals)) {
  spo2 = subset(spo2_vals, subject_id == scvo2_vals[i, 2] & abs(scvo2_vals[i, 16] - datetime) < 0.006944444)
  if (nrow(spo2) > 0 ) {
    scvo2_vals$spo2_agg[i] = mean(spo2$valuenum)
    check = check + 1
  }
  tryCatch(
    {
      scvo2_vals$temp[i] = fun_closest(temp_vals, scvo2_vals[i, 2], scvo2_vals[i, 16])    
    }, 
    error = function(e) {
      scvo2_vals$temp[i] = -9999
    }
  )
  tryCatch(
    {
      scvo2_vals$ph[i] = fun_closest(temp_ph, scvo2_vals[i, 2], scvo2_vals[i, 16])    
    }, 
    error = function(e) {
      scvo2_vals$ph[i] = -9999
    }
  )
  tryCatch(
    {
      scvo2_vals$hb[i] = fun_closest(temp_hb, scvo2_vals[i, 2], scvo2_vals[i, 16])
    }, 
    error = function(e) {
      scvo2_vals$hb[i] = -9999
    }
  )
  tryCatch(
    {
      scvo2_vals$lactate[i] = fun_closest(temp_lactate, scvo2_vals[i, 2], scvo2_vals[i, 16])
    }, 
    error = function(e) {
      scvo2_vals$lactate[i] = -9999
    }
  )
  if(i %% 500 == 0) {
    print(i)
  }
}
write.csv(file = "training_data.csv", scvo2_vals, row.names = F)

