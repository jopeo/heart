#!/usr/bin/env python

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction",
                   page_icon='../Images/heart.png')

df_name = "df.h5"


states = (
		"Alabama",
		"Alaska",
        "Arizona",
        "Arkansas",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "District of Columbia",
        "Florida",
		"Georgia",
		"Hawaii",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
		"Maine",
		"Maryland",
        "Massachusetts",
        "Michigan",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
		"New Hampshire",
		"New Jersey",
		"New Mexico",
		"New York",
		"North Carolina",
		"North Dakota",
		"Ohio",
		"Oklahoma",
		"Oregon",
		"Pennsylvania",
		"Rhode Island",
		"South Carolina",
		"South Dakota",
		"Tennessee",
		"Texas",
		"Utah",
		"Vermont",
		"Virginia",
		"Washington",
		"West Virginia",
		"Wisconsin",
		"Wyoming",
		"Guam",
		"Puerto Rico",
		)

metstats = ("Metropolitan",
            "Nonmetropolitan")

urbstats = ("Urban",
            "Rural")

sexes = ("Male", "Female")

rfhealths = ("Excellent",
             "Very good",
             "Good",
             "Fair",
             "Poor")

hcvu651s = ("Yes", "No")

totindas = ("Yes", "No")

asthms1s = ("Currently have asthma", "Previously had asthma", "Never been diagnosed with asthma")

drdxar2s = ("Yes", "No")

exteth3s = ("No", "Yes")

denvst3s = ("Yes", "No")

races = ("White only",
         "Black only",
         "American Indian or Alaskan Native",
         "Asian only",
         "Native Hawaiian or other Pacific Islander",
         "Other (only one)",
         "Multiracial",
         "Hispanic"
         )

educags = ("Have not graduated high school",
           "Graduated high school",
           "Some college",
           "Graduated college"
           )

incomgs = ("Less than $15,000",
           "$15,000 to less than $25,000",
           "$25,000 to less than $35,000",
           "$35,000 to less than $50,000",
           "$50,000 or more"
           )


smoker3s = ("Everyday smoker",
            "Smoke less than every day",
            "Former smoker"
            "Never smoker")

drnkany5s = ("Yes",
             "No")

rfbing5s = ("Yes",
            "No")

rfdrhv7s = ("No",
            "Yes")

pneumo3s = ("Yes",
            "No")

rfseat3s = ("Always",
            "Not always")

drnkdrvs = ("Yes",
            "No")

rfmam22s = ("Yes",
            "No")

flshot7s = ("Yes",
            "No")

rfpap35s = ("Yes",
            "No")

rfpsa23s = ("Yes",
            "No")

crcrec1s = ("Yes",
            "Yes but not within time",
            "Never")

aidtst4s = ("Yes",
            "No")

persdoc2s = ("Yes",
             "More than one",
             "No")

chcscncrs = ("Yes",
             "No")

chcocncrs = ("Yes",
             "No")

chccopd2s = ("Yes",
             "No")

qstlangs = ("English",
            "Spanish")

addepev3s = ("Yes",
             "No")

chckdny2s = ("Yes",
             "No")

diabete4s = ("Yes",
             "No")

maritals = ("Married",
            "Divorced",
            "Widowed",
            "Separated",
            "Never married",
            "Member of unmarried couple"
            )

chldcnts = ("No children",
            "One child in household",
            "Two children in household",
            "Three children in household",
            "Four children in household",
            "Five or more children in household")



def show_predict_page():
	
	X = pd.read_hdf(df_name)
	
	new_entry = pd.DataFrame(0, index=range(1), columns=X.columns)
	
	st.title("Heart Disease Prediction")
	st.subheader("A deep neural network for predicting heart disease")
	
	state = st.selectbox("In which state do you reside?", states)
	# todo: check state entries compared with model numbers
	new_entry.iloc[0]._STATE = states.index(state) + 1
	
	metstat = st.radio("Do you live in a metropolitan county?", metstats)
	new_entry.iloc[0]._METSTAT = metstats.index(metstat) + 1

	urbstat = st.radio("Do you live in a urban or rural county?", urbstats)
	new_entry.iloc[0]._URBSTAT = urbstats.index(urbstat) + 1
	
	sex = st.radio("What is your biological sex?", sexes)
	new_entry.iloc[0].SEXVAR = sexes.index(sex) + 1
	
	age = st.slider("What is your age?", 0, 100)
	new_entry.iloc[0]._AGE80 = 80 if age > 80 else age
	
	rfhealth = st.radio("How would you describe your health?", rfhealths)
	new_entry.iloc[0]._RFHLTH = rfhealths.index(rfhealth)
	new_entry.iloc[0]._RFHLTH = 2 if new_entry.iloc[0]._RFHLTH >= 3 else 1
	
	phys14d = st.slider("During the past month, how many days were you physically ill or injured?", 0, 30)
	if phys14d == 0:
		new_entry.iloc[0]._PHYS14D = 1
	elif 1 <= phys14d <= 13:
		new_entry.iloc[0]._PHYS14D = 2
	else:
		new_entry.iloc[0]._PHYS14D = 3
	
	ment14d = st.slider("During the past month, how many days were you either stressed, depressed, or not well emotionally?", 0, 30)
	if ment14d == 0:
		new_entry.iloc[0]._MENT14D = 1
	elif 1 <= ment14d <= 13:
		new_entry.iloc[0]._MENT14D = 2
	else:
		new_entry.iloc[0]._MENT14D = 3
	
	hcvu651 = st.radio("Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans such as Medicare, or Indian Health Service?", hcvu651s)
	new_entry.iloc[0]._HCVU651 = hcvu651s.index(hcvu651) + 1
	
	totinda = st.radio("During the past month, other than your regular job, did you participate in any physical activities or exercises such as running, calisthenics, golf, gardening, or walking for exercise?", totindas)
	new_entry.iloc[0]._TOTINDA = totindas.index(totinda) + 1
	
	asthms1 = st.radio("Have you ever had a doctor diagnose you with asthma?", asthms1s)
	new_entry.iloc[0]._ASTHMS1 = asthms1s.index(asthms1) + 1
	
	drdxar2 = st.radio("Have you ever had a doctor diagnose you with any form of arthritis?", drdxar2s)
	new_entry.iloc[0]._DRDXAR2 = drdxar2s.index(drdxar2) + 1
	
	exteth3 = st.radio("Have you ever had a permanent tooth extracted?", exteth3s)
	new_entry.iloc[0]._EXTETH3 = exteth3s.index(exteth3) + 1
	
	denvst3 = st.radio("Over the past year, have you visited the dentist?", denvst3s)
	new_entry.iloc[0]._DENVST3 = denvst3s.index(denvst3) + 1
	
	st.write(f"variable: {denvst3} \n"
	         f"df: {new_entry.iloc[0]._DENVST3}")
	
	
	# new_entry.iloc[0]._RACE = st.selectbox("With which race(s) do you identify?", races)
	# new_entry.iloc[0]._EDUCAG = st.radio("What is your highest level of education?", educags)
	# new_entry.iloc[0]._INCOMG = st.radio("What is your annual houshold income from all sources?", incomgs)
	# new_entry.iloc[0]._SMOKER3 = st.radio("Do you smoke cigarettes?", smoker3s)
	# new_entry.iloc[0].DRNKANY5 = st.radio("During the past month, have you had at least one drink of alcohol? ", drnkany5s)
	# new_entry.iloc[0]._RFBING5 = st.radio("During the past month, have you had at least one binge-drinking occasion"
	#                    "(five or more drinks for males, or four or more drinks for females)?", rfbing5s)
	# new_entry.iloc[0]._RFDRHV7 = st.radio("Are you a heavy drinker (14 or more drinks per week for males, or 7 or more drinks per week for"
	#                    " females)?", rfdrhv7s)
	# new_entry.iloc[0]._PNEUMO3 = st.radio("Have you ever has a pneumonia vaccination?", pneumo3s)
	# new_entry.iloc[0]._RFSEAT3 = st.radio("Do you always wear a seatbelt when in a vehicle?", rfseat3s)
	# new_entry.iloc[0]._DRNKDRV = st.radio("Do you always wear a seatbelt when in a vehicle?", drnkdrvs)
	# new_entry.iloc[0]._DRNKDRV = st.radio("Have you ever driven at least once when perhaps you have had too much to drink?", drnkdrvs)
	# new_entry.iloc[0]._RFMAM22 = st.radio("Have you had a mammogram in the past 2 years?", rfmam22s)
	# new_entry.iloc[0]._FLSHOT7 = st.radio("Have you had a flu shot within the past year?", flshot7s)
	# new_entry.iloc[0]._RFPAP35 = st.radio("Have you had a Pap test in the past 3 years?", rfpap35s)
	# new_entry.iloc[0]._RFPSA23 = st.radio("Have you had a PSA (prostate specific antigen) test in the past 2 years?", rfpsa23s)
	# new_entry.iloc[0]._CRCREC1 = st.radio("Have you full met the USPSTF recommendations for colorectal cancer screening "
	#                    "(this includes a sigmoidoscopy within the past 10 years, a blood stool test within the past year,"
	#                    "a stool DNA test within the past 3 years, or a colonoscopy within the past 10 years)?", crcrec1s)
	# new_entry.iloc[0]._AIDTST4 = st.radio("Have you ever been tested for HIV?", aidtst4s)
	# new_entry.iloc[0].PERSDOC2 = st.radio("Do you have one person you think of as your personal doctor or health care provider?", persdoc2s)
	# new_entry.iloc[0].CHCSCNCR = st.radio("Have you ever been told that you had skin cancer?", chcscncrs)
	# new_entry.iloc[0].CHCOCNCR = st.radio("Have you ever been told that you had any other types of cancer?", chcocncrs)
	# new_entry.iloc[0].CHCCOPD2 = st.radio("Have you ever been told that you had chronic obstructive pulmonary disease, C.O.P.D.,"
	#                     " emphysema or chronic bronchitis?", chccopd2s)
	# new_entry.iloc[0].QSTLANG = st.radio("What language are you using to complete this questionnaire?", qstlangs)
	# new_entry.iloc[0].ADDEPEV3 = st.radio("Have you ever been told that you had a depressive disorder (including depression, major "
	#                     "depression, dysthymia, or minor depression)?", addepev3s)
	# new_entry.iloc[0].CHCKDNY2 = st.radio("Not including kidney stones, bladder infection or incontinence, were you ever told you had "
	#                     "kidney disease?", chckdny2s)
	# new_entry.iloc[0].DIABETE4 = st.radio("Have you ever been told that you have diabetes?", diabete4s)
	# new_entry.iloc[0].MARITAL = st.radio("What is your marital status?", maritals)

	# HTM4 = st.slider("What is your height in centimeters (5 feet 5 inches is about 165 centimeters)?", 100, 210)
	# WTKG3 = st.slider("What is your weight in kilograms (150 pounds is about 68 kilograms)?", 35, 160)
	# new_entry.iloc[0]._BMI5 = WTKG3 / (HTM4 ^ 2)
	# new_entry.iloc[0]._CHLDCNT = st.radio("How many children do you have in your household?", chldcnts)
	# new_entry.iloc[0]._DRNKWK1 = st.slider("How many alcholic drinks do you haver per week in total?", 0, 200)
	# new_entry.iloc[0].SLEPTIM1 = st.slider("How many hours of sleep do you get in a 24-hour period?", 0, 24)
	
	st.button("Calculate Probability of Heart Disease")
	
	return


if __name__ ==  "__main__":
	show_predict_page()