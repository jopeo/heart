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


def show_predict_page():

	st.title("Heart Disease Prediction")
	st.subheader("A deep neural network for predicting heart disease")
	
	state = st.selectbox("In which state do you reside?", states)
	sex = st.radio("What is your biological sex?", sexes)
	age = st.slider("What is your age?", 0, 100)
	rfhealth = st.radio("How would you describe your health?", rfhealths)
	phys14d = st.slider("During the past month, how many days were you physically ill or injured?", 0, 30)
	ment14d = st.slider("During the past month, how many days were you either stressed, depressed, or not well emotionally?", 0, 30)
	hcvu651 = st.radio("Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs,"
	                   " or government plans such as Medicare, or Indian Health Service?", hcvu651s)
	totinda = st.radio("During the past month, other than your regular job, did you participate in any physical activities or"
	                   " exercises such as running, calisthenics, golf, gardening, or walking for exercise?", totindas)
	


	pass


if __name__ ==  "__main__":
	show_predict_page()