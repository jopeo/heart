#!/usr/bin/env python

import streamlit as st
from pandas import DataFrame, concat, read_hdf
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import load

st.set_page_config(page_title="Heart Disease Prediction",
                   page_icon='./res/heart.png')

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.image('./res/heart_section.gif')

df_name = "df.h5"
model_name = "model10.joblib"


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

educags = ("Did not graduate high school",
           "Graduated high school",
           "Attended college or technical school",
           "Graduated college or technical school"
           )

incomgs = ("Less than $15,000",
           "$15,000 to less than $25,000",
           "$25,000 to less than $35,000",
           "$35,000 to less than $50,000",
           "$50,000 or more"
           )


smoker3s = ("Everyday smoker",
            "Smoke less than every day",
            "Former smoker",
            "Never smoked")

drnkany5s = ("Yes",
             "No")

rfbing5s = ("No", "Yes")

rfdrhv7s = ("No",
            "Yes")

pneumo3s = ("Yes",
            "No")

rfseat3s = ("Always",
            "Nearly always",
            "Sometimes",
            "Seldom",
            "Never wear a seatbelt")

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
            "Never had any of the recommended CRC tests")

aidtst4s = ("Yes",
            "No")

persdoc2s = ("Yes, only one",
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

features_cat = ['_STATE',       # geographical state]
                'SEXVAR',       # Sex of Respondent 1 MALE, 2 FEMALE
                '_RFHLTH',      # Health Status  1 Good or Better Health 2 Fair or Poor Health
                                    # 9 Don’t know/ Not Sure Or Refused/ Missing
                '_PHYS14D',     # Healthy Days 1 Zero days when physical health not good
                                    #  2 1-13 days when physical health not good
                                    # 3 14+ days when physical health not good
                                    # 9 Don’t know/ Refused/Missing
                '_MENT14D',     # SAME AS PHYS
                '_HCVU651',     # Health Care Access  1 Have health care coverage 2 Do not have health care coverage 9 Don’t know/ Not Sure, Refused or Missing
                '_TOTINDA',     # Exercise 1 Had physical activity or exercise 2 No physical activity or exercise in last 30 days 9 Don’t know/ Refused/ Missing
                '_ASTHMS1',     # asthma? 1 current 2 former 3 never
                '_DRDXAR2',     # ever arthritis? 1 Diagnosed with arthritis 2 Not diagnosed with arthritis
                '_EXTETH3',     # ever had teeth extracted? 1 no 2 yes 9 dont know
                '_DENVST3',     # dentist in past year? 1 yes 2 no 9 don't know
                '_RACE',        # 1 White only, nonHispanic, 2 Black only, nonHispanic, 3 American Indian or Alaskan Native only,Non-Hispanic 4 Asian only, nonHispanic  5 Native Hawaiian or other Pacific Islander only, Non-Hispanic 6 Other race only, nonHispanic 7 Multiracial, nonHispanic 8 Hispanic Respondents who reported they are of Hispanic origin. ( _HISPANC=1) 9 Don’t know/ Not sure/ Refused
                '_EDUCAG',      # level of education completed 1 no grad high school, 2 high school, 3 some college, 4 graduated college, 9 don't know
                '_INCOMG',      # Income categories (1 Less than $15,000, 2 $15,000 to less than $25,000, 3 $25,000 to less than $35,000, 4 $35,000 to less than $50,000, 5 $50,000 or more, 9 dont know
                '_METSTAT',     # metropolitan status 1 yes, 2 no
                '_URBSTAT',     # urban rural status 1 urban 2 rural
                '_SMOKER3',     # four-level smoker status: everyday smoker, someday smoker, former smoker, non-smoker
                'DRNKANY5',     # had at least one drink of alcohol in the past 30 days
                '_RFBING5',     # binge drinkers (males having five or more drinks on one occasion, females having four or more drinks on one occasion 1 no 2 yes
                '_RFDRHV7',     # heavy drinkers 14 drinks per week or less, or Female Respondents who reported having 7 drinks per week or less 1 no 2 yes
                '_PNEUMO3',     # ever had a pneumonia vaccination
                '_RFSEAT3',     # always wear seat belts 1 yes 2 no
                '_DRNKDRV',     # drinking and driving 1 yes 2 no
                '_RFMAM22',     # mammogram in the past two years 1 yes 2 no
                '_FLSHOT7',     # flu shot within the past year 1 yes 2 no
                '_RFPAP35',     # Pap test in the past three years 1 yes 2 no
                '_RFPSA23',     # PSA test in the past 2 years
                '_CRCREC1',     # fully met the USPSTF recommendations for rectal cancer screening 1 yes, 2 yes but not within time, 3 never
                '_AIDTST4',     # ever been tested for HIV
                'PERSDOC2',     # personal doctor yes = 1, more = 2, no = 3 Do you have one person you think of as your personal doctor or health care provider? (If ´No´ ask ´Is there more than one or is there no person who you think of as your personal doctor or health care provider?´.)
                'CHCSCNCR',     # (Ever told) (you had) skin cancer? 1 yes 2 no
                'CHCOCNCR',     # (Ever told) (you had) any other types of cancer? 1 yes 2 no
                'CHCCOPD2',     #  (Ever told) (you had) chronic obstructive pulmonary disease, C.O.P.D., emphysema or chronic bronchitis? 1 yes 2 no
                'QSTLANG',     # 1 english 2 spanish
                'ADDEPEV3',     # (Ever told) (you had) a depressive disorder (including depression, major depression, dysthymia, or minor depression)? 1 yes 2 no
                'CHCKDNY2',     # Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?  1 yes 2 no
                'DIABETE4',     # (Ever told) (you had) diabetes? 1 yes 2 no
                'MARITAL'      #  (marital status) 1 married 2 divorced 3 widowed 4 separated 5 never married 6 member of unmarried couple
                ]

features_num = ['_AGE80',       #  imputed age value collapsed above 80
                'HTM4',  # height in centimeters
                'WTKG3',  # weight in kilograms, implied 2 decimal places
                '_BMI5',  # body mass index
                '_CHLDCNT',  # number of children in household.
                '_DRNKWK1',  # total number of alcoholic beverages consumed per week.
                'SLEPTIM1',  # how many hours of sleep do you get in a 24-hour period?
                ]


def process(prediction_data, X):
	# rows_to_keep = q.shape[0]
	rows_to_keep = prediction_data.shape[0]
	
	# inputs = pd.concat([X, z])
	inputs = concat([X, prediction_data])
	# inputs.shape
	
	# todo: replace NaNs with most frequent (mode) (X_mode)
	
	processed = DataFrame()
	
	for cat in features_cat:
		# print(cat)
		one_hots = OneHotEncoder()
		cat_encoded = one_hots.fit_transform(inputs[[cat]])
		cat_encoded_names = one_hots.get_feature_names_out([cat])
		cat_encoded = DataFrame(cat_encoded.todense(), columns=cat_encoded_names)
		# print(cat_encoded_names)
		# print(len(cat_encoded_names))
		processed = concat([processed, cat_encoded], axis=1)
	
	for num in features_num:
		num_scaled = StandardScaler().fit_transform(inputs[[num]])
		num_scaled = DataFrame(num_scaled, columns=[num])
		processed = concat([processed, num_scaled], axis=1)
	
	to_model = processed.iloc[processed.shape[0] - rows_to_keep:].copy()
	# to_model.shape
	
	return to_model


def show_predict_page():
	
	X = read_hdf(df_name)
	new_entry = DataFrame(0, index=range(1), columns=X.columns)
	
	st.title("Heart Disease Prediction")
	st.subheader("A machine learning algorithm for predicting heart disease")
	
	state = st.selectbox("In which state do you reside?", states)
	# todo: check state entries compared with model numbers
	new_entry.iloc[0]._STATE = states.index(state) + 1
	
	metstat = st.radio("Do you live in a metropolitan county?", metstats)
	st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
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
	
	race = st.radio("With which race(s) do you identify?", races)
	new_entry.iloc[0]._RACE = races.index(race) + 1
	
	educag = st.radio("What is your highest level of education?", educags)
	new_entry.iloc[0]._EDUCAG = educags.index(educag) + 1
	
	incomg = st.radio("What is your annual houshold income from all sources?", incomgs)
	new_entry.iloc[0]._INCOMG = incomgs.index(incomg) + 1
	
	smoker3 = st.radio("Do you smoke cigarettes?", smoker3s)
	new_entry.iloc[0]._SMOKER3 = smoker3s.index(smoker3) + 1
	
	drnkany5 = st.radio("During the past month, have you had at least one drink of alcohol? ", drnkany5s)
	new_entry.iloc[0].DRNKANY5 = drnkany5s.index(drnkany5) + 1
	
	rfbing5 = st.radio("During the past month, have you had at least one binge-drinking occasion (five or more drinks for males, or four or more drinks for females)?", rfbing5s)
	new_entry.iloc[0]._RFBING5 = rfbing5s.index(rfbing5) + 1
	
	rfdrhv7 = st.radio("Are you a heavy drinker (14 or more drinks per week for males, or 7 or more drinks per week for females)?", rfdrhv7s)
	new_entry.iloc[0]._RFDRHV7 = rfdrhv7s.index(rfdrhv7) + 1
	
	pneumo3 = st.radio("Have you ever has a pneumonia vaccination?", pneumo3s)
	new_entry.iloc[0]._PNEUMO3 = pneumo3s.index(pneumo3) + 1
	
	rfseat3 = st.radio("How often do you wear a seatbelt when in a vehicle?", rfseat3s)
	new_entry.iloc[0]._RFSEAT3 = 1 if rfseat3s.index(rfseat3) < 1 else 2
	
	drnkdrv = st.radio("Have you ever driven at least once when perhaps you have had too much to drink?", drnkdrvs)
	new_entry.iloc[0]._DRNKDRV = drnkdrvs.index(drnkdrv) + 1
	
	rfmam22 = st.radio("Have you had a mammogram in the past 2 years?", rfmam22s)
	new_entry.iloc[0]._RFMAM22 = rfmam22s.index(rfmam22) + 1
	
	flshot7 = st.radio("Have you had a flu shot within the past year?", flshot7s)
	new_entry.iloc[0]._FLSHOT7 = flshot7s.index(flshot7) + 1
	
	rfpap35 = st.radio("Have you had a Pap test in the past 3 years?", rfpap35s)
	new_entry.iloc[0]._RFPAP35 = rfpap35s.index(rfpap35) + 1
	
	rfpsa23 = st.radio("Have you had a PSA (prostate specific antigen) test in the past 2 years?", rfpsa23s)
	new_entry.iloc[0]._RFPSA23 = rfpsa23s.index(rfpsa23) + 1
	
	crcrec1 = st.radio("Have you full met the USPSTF recommendations for colorectal cancer (CRC) screening (this includes a sigmoidoscopy within the past 10 years, a blood stool test within the past year, a stool DNA test within the past 3 years, or a colonoscopy within the past 10 years)?", crcrec1s)
	new_entry.iloc[0]._CRCREC1 = crcrec1s.index(crcrec1) + 1
	
	aidtst4 = st.radio("Have you ever been tested for HIV?", aidtst4s)
	new_entry.iloc[0]._AIDTST4 = aidtst4s.index(aidtst4) + 1
	
	persdoc2 = st.radio("Do you have one person you think of as your personal doctor or health care provider?", persdoc2s)
	new_entry.iloc[0].PERSDOC2 = persdoc2s.index(persdoc2) + 1
	
	chcscncr = st.radio("Have you ever been told that you had skin cancer?", chcscncrs)
	new_entry.iloc[0].CHCSCNCR = chcscncrs.index(chcscncr) + 1
	
	chcocncr = st.radio("Have you ever been told that you had any other types of cancer?", chcocncrs)
	new_entry.iloc[0].CHCOCNCR = chcocncrs.index(chcocncr) + 1
	
	chccopd2 = st.radio("Have you ever been told that you had chronic obstructive pulmonary disease, C.O.P.D., emphysema or chronic bronchitis?", chccopd2s)
	new_entry.iloc[0].CHCCOPD2 = chccopd2s.index(chccopd2) + 1
	
	qstlang = st.radio("What language are you using to complete this questionnaire?", qstlangs)
	new_entry.iloc[0].QSTLANG = qstlangs.index(qstlang) + 1
	
	addepev3 = st.radio("Have you ever been told that you had a depressive disorder (including depression, major depression, dysthymia, or minor depression)?", addepev3s)
	new_entry.iloc[0].ADDEPEV3 = addepev3s.index(addepev3) + 1
	
	chckdny2 = st.radio("Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?", chckdny2s)
	new_entry.iloc[0].CHCKDNY2 =  chckdny2s.index(chckdny2) + 1
	
	diabete4 = st.radio("Have you ever been told that you have diabetes?", diabete4s)
	new_entry.iloc[0].DIABETE4 = diabete4s.index(diabete4) + 1
	
	marital = st.radio("What is your marital status?", maritals)
	new_entry.iloc[0].MARITAL = maritals.index(marital) + 1
	
	htm4 = st.slider("What is your height in centimeters (5 feet 5 inches is about 165 centimeters)?", 100, 210)
	wtkg3 = st.slider("What is your weight in kilograms (150 pounds is about 68 kilograms)?", 35, 160)
	bmi = wtkg3 / (htm4/100*htm4/100) * 100
	new_entry.iloc[0]._BMI5 = int(bmi)
	
	chldcnt = st.radio("How many children do you have in your household?", chldcnts)
	new_entry.iloc[0]._CHLDCNT = chldcnts.index(chldcnt) + 1
	
	new_entry.iloc[0]._DRNKWK1 = st.slider("How many alcholic drinks do you haver per week in total?", 0, 200)
	
	new_entry.iloc[0].SLEPTIM1 = st.slider("How many hours of sleep do you get in a 24-hour period?", 0, 24)
	
	
	clicked = st.button("Calculate Probability of Heart Disease")
	
	if clicked:
		# calculate and show
		to_predict = process(new_entry, X)
		input_shape = [to_predict.shape[1]]
		
		model = load(model_name)
		
		y_new = model.predict_proba(to_predict)
		
		# st.write(f"Your calculated probability of heart disease is: {100*clicked:.2f}% \n")
		# st.subheader(f"Your calculated probability of heart disease is: {y_new}% \n {type(y_new)}")
		st.subheader(f"Your calculated probability of heart disease is:")
		st.header(f" {100*y_new[0][1]:.2f}%")
		# st.header(f" {100*float(y_new[[0]]):.2f}% \n")
		clicked = False
		
	
	return


if __name__ ==  "__main__":
	show_predict_page()