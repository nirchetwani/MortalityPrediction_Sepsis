BACKEND:

    Codes (present in "\Codes") and files description:

        Data Manipulation and handling:
            1. data_manipulation1.R:

                what it does: gets appropriate lab-results for all patients that might be important for our analysis
                input datasets [source -- MIMIC3, handled on local]: admissions, d_labitems, icustays, labevents, patients [all these files present in folder "\Data\Input"]
                output: lab_data_ph_lactate_hb -- containing the lab records with ph, lactate and hb values, and patient-death dates [located in "\Data\Output"]

            2. data_manipulation2.R:
                what it does: creates the final data with important features using definitions provided by doctors. the output of this code is consumed for predictive-modeling.
                input datasets [source -- MIMIC3]:
                    Original data, handled on Google Cloud/Big-Query [size 34 GB] - chartevents.csv [file not uploaded because of large size]
                    Files generated from Google BigQuery manipulation - temp_vals, scvo2_vals, spo2_vals*
                    Explicit-septic patients data - SQL, to identify patients with sepsis, agnus.csv
                    File generated from manipulation1: lab_data_ph_lactate_hb
                output:
                    training_data: used for modeling
                *queries used in bigquery: queries_data_manipulation.sql


        Modeling:
            1. training_code.py:
                what it does: takes the data generated from code data_manipulation2.R, and evaluates, stores different models to predict mortality
                input: training_data
                output:
                    model files: xgboost.dat, svm.dat, lr.dat, randomforest.dat
                    scaler: the scaler used to scale the test data
                    Both these set of files are present in ["\ModelFiles"]

            2. testing_code.py:
                what it does: given a doctor enters data in a certain format, gives out the list of patients that are moderately-critical, critical and very-critical
                input: sample_test_data.csv [input by doctor, present in folder "\Test"], model-files and scaler created above
                output: critical_patients_records.csv and prob_predictions_model-level [records of all patients/time-stamps that are critical, contained in "\Data\Output"]
