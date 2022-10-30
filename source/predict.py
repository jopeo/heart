#!/usr/bin/env python

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Heart Disease Prediction",
                   page_icon='../Images/heart.png')

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

exteth3s = ("Yes", "No")

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

metstats = ("Metropolitan",
            "Nonmetropolitan")

urbstats = ("Urban",
            "Rural")

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

	st.title("Heart Disease Prediction")
	st.subheader("A deep neural network for predicting heart disease")
	
	state = st.selectbox("In which state do you reside?", states)
	sex = st.radio("What is your biological sex?", sexes)
	age = st.slider("What is your age?", 0, 100)
	rfhealth = st.radio("How would you describe your health?", rfhealths)
	phys14d = st.slider("During the past month, how many days were you physically ill or injured?", 0, 30)
	ment14d = st.slider("During the past month, how many days were you either stressed, depressed, or not well"
	                    " emotionally?", 0, 30)
	hcvu651 = st.radio("Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs,"
	                   " or government plans such as Medicare, or Indian Health Service?", hcvu651s)
	totinda = st.radio("During the past month, other than your regular job, did you participate in any physical activities or"
	                   " exercises such as running, calisthenics, golf, gardening, or walking for exercise?", totindas)
	asthms1 = st.radio("Have you ever had a doctor diagnose you with asthma?", asthms1s)
	drdxar2 = st.radio("Have you ever had a doctor diagnose you with any form of arthritis?", drdxar2s)
	exteth3 = st.radio("Have you ever had a permanent tooth extracted?", exteth3s)
	denvst3 = st.radio("Over the past year, have you visited the dentist?", denvst3s)
	race = st.selectbox("With which race(s) do you identify?", races)
	educag = st.radio("What is your highest level of education?", educags)
	incomg = st.radio("What is your annual houshold income from all sources?", incomgs)
	metstat = st.radio("Do you live in a metropolitan county?", metstats)
	urbstat = st.radio("Do you live in a urban or rural county?", urbstats)
	smoker3 = st.radio("Do you smoke cigarettes?", smoker3s)
	drnkany5 = st.radio("During the past month, have you had at least one drink of alcohol? ", drnkany5s)
	rfbing5 = st.radio("During the past month, have you had at least one binge-drinking occasion"
	                   "(five or more drinks for males, or four or more drinks for females)?", rfbing5s)
	rfdrhv7 = st.radio("Are you a heavy drinker (14 or more drinks per week for males, or 7 or more drinks per week for"
	                   " females)?", rfdrhv7s)
	pneumo3 = st.radio("Have you ever has a pneumonia vaccination?", pneumo3s)
	rfseat3 = st.radio("Do you always wear a seatbelt when in a vehicle?", rfseat3s)
	drnkdrv = st.radio("Do you always wear a seatbelt when in a vehicle?", drnkdrvs)
	drnkdrv = st.radio("Have you ever driven at least once when perhaps you have had too much to drink?", drnkdrvs)
	rfmam22 = st.radio("Have you had a mammogram in the past 2 years?", rfmam22s)
	flshot7 = st.radio("Have you had a flu shot within the past year?", flshot7s)
	rfpap35 = st.radio("Have you had a Pap test in the past 3 years?", rfpap35s)
	rfpsa23 = st.radio("Have you had a PSA (prostate specific antigen) test in the past 2 years?", rfpsa23s)
	crcrec1 = st.radio("Have you full met the USPSTF recommendations for colorectal cancer screening "
	                   "(this includes a sigmoidoscopy within the past 10 years, a blood stool test within the past year,"
	                   "a stool DNA test within the past 3 years, or a colonoscopy within the past 10 years)?", crcrec1s)
	aidtst4 = st.radio("Have you ever been tested for HIV?", aidtst4s)
	persdoc2 = st.radio("Do you have one person you think of as your personal doctor or health care provider?", persdoc2s)
	chcscncr = st.radio("Have you ever been told that you had skin cancer?", chcscncrs)
	chcocncr = st.radio("Have you ever been told that you had any other types of cancer?", chcocncrs)
	chccopd2 = st.radio("Have you ever been told that you had chronic obstructive pulmonary disease, C.O.P.D.,"
	                    " emphysema or chronic bronchitis?", chccopd2s)
	qstlang = st.radio("What language are you using to complete this questionnaire?", qstlangs)
	addepev3 = st.radio("Have you ever been told that you had a depressive disorder (including depression, major "
	                    "depression, dysthymia, or minor depression)?", addepev3s)
	chckdny2 = st.radio("Not including kidney stones, bladder infection or incontinence, were you ever told you had "
	                    "kidney disease?", chckdny2s)
	diabete4 = st.radio("Have you ever been told that you have diabetes?", diabete4s)
	marital = st.radio("What is your marital status?", maritals)
	age80 = 80 if age > 80 else age
	htm4 = st.slider("What is your height in centimeters (5 feet 5 inches is about 165 centimeters)?", 100, 210)
	wtkg3 = st.slider("What is your weight in kilograms (150 pounds is about 68 kilograms)?", 35, 160)
	bmi5 = wtkg3 / (htm4 ^ 2)
	chldcnt = st.radio("How many children do you have in your household?", chldcnts)
	drnkwk1 = st.slider("How many alcholic drinks do you haver per week in total?", 0, 200)
	sleptim1 = st.slider("How many hours of sleep do you get in a 24-hour period?", 0, 24)
	
	return


if __name__ ==  "__main__":
	show_predict_page()