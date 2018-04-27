# query for scvo2 values
select *
from [chartevents.chart_events]
where ITEMID in (823, 6024, 5896, 3776, 2396, 2398,
                 2574, 6212, 7063, 7293, 227549, 225674, 227685, 227686, 226541)
and VALUENUM is not null


# query for temperature
select *
from [chartevents.chart_events]
where ITEMID in (646, 220277)
and VALUENUM is not null
and SUBJECT_ID in
(
  select SUBJECT_ID from [chartevents.table_scvo2_vals] group by SUBJECT_ID
)

# query for spo2 values
select *
from [chartevents.chart_events]
where ITEMID in (676, 677, 678, 679, 223761,
                 223762)
and VALUENUM is not null
and SUBJECT_ID in
(
  select SUBJECT_ID from [chartevents.table_scvo2_vals] group by SUBJECT_ID
)
