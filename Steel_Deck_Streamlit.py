import streamlit as st
import pandas as pd
from pickle import load
import joblib
import pickle
import numpy as np
import math
from PIL import Image
import os
from config.definitions import ROOT_DIR


#Load models and scalers
kl_stiff_SVR=joblib.load(os.path.join(ROOT_DIR,'Deck_Prelim_kl_stiff_SVR.joblib'))
kl_stiff_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Deck_Prelim_kl_stiff_SVR.pkl'),'rb'))

kl_unstiff_SVR=joblib.load(os.path.join(ROOT_DIR,'Deck_Prelim_kl_unstiff_SVR.joblib'))
kl_unstiff_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Deck_Prelim_kl_unstiff_SVR.pkl'),'rb'))

kd_flange_SVR=joblib.load(os.path.join(ROOT_DIR,'Deck_Prelim_kd_flange_SVR.joblib'))
kd_flange_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Deck_Prelim_kd_flange_SVR.pkl'),'rb'))

Mcrl_stiff_SVR=joblib.load(os.path.join(ROOT_DIR,'Deck_Prelim_Mcrl_stiff_SVR.joblib'))
Mcrl_stiff_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Deck_Prelim_Mcrl_stiff_SVR.pkl'),'rb'))

Mcrl_unstiff_SVR=joblib.load(os.path.join(ROOT_DIR,'Deck_Prelim_Mcrl_unstiff_SVR.joblib'))
Mcrl_unstiff_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Deck_Prelim_Mcrl_unstiff_SVR.pkl'),'rb'))

Mcrd_flange_SVR=joblib.load(os.path.join(ROOT_DIR,'Deck_Prelim_Mcrd_flange_SVR.joblib'))
Mcrd_flange_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Deck_Prelim_Mcrd_flange_SVR.pkl'),'rb'))

Mcrd_web_SVR=joblib.load(os.path.join(ROOT_DIR,'Deck_Prelim_Mcrd_web_SVR.joblib'))
Mcrd_web_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Deck_Prelim_Mcrd_web_SVR.pkl'),'rb'))

Mu_SVR=joblib.load(os.path.join(ROOT_DIR,'Deck_Prelim_Mu_SVR.joblib'))
Mu_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Deck_Prelim_Mu_SVR.pkl'),'rb'))

st.header('Properties of Steel Deck in Bending Predicted by Support Vector Machine (SVM)')

st.sidebar.header('User Input Parameters')

def user_input_features():
    deck_type=st.sidebar.selectbox("Deck Type",('1F','1.5B','1.5BST1','1.5BST2','1.5BST3','3N','2CST1','2CST2','2CST3','3CST1','3CST2','3CST3'))
    if deck_type=='1F': deck_gauge=st.sidebar.slider("Deck Gauge",min_value=20, max_value=26, step=2) 
    else: deck_gauge=st.sidebar.slider("Deck Gauge",min_value=16, max_value=22, step=1)    

    yield_strength = st.sidebar.radio("Yield Strength (ksi)",('33','40','50','60'))
	
    if deck_type=='1F': deck_span=st.sidebar.slider("Deck Span (ft)",min_value=3.0, max_value=5.0, step=0.25) 
    elif deck_type=='1.5B' or deck_type=='1.5BST1' or deck_type=='1.5BST2' or deck_type=='1.5BST3': 
        deck_span=st.sidebar.slider("Deck Span (ft)",min_value=4.0, max_value=8.0, step=0.25)
    elif deck_type=='2CST1' or deck_type=='2CST2' or deck_type=='2CST3': 
        deck_span=st.sidebar.slider("Deck Span (ft)",min_value=6.0, max_value=10.0, step=0.25)
    elif deck_type=='3CST1' or deck_type=='3CST2' or deck_type=='3CST3' or deck_type=='3N': 
        deck_span=st.sidebar.slider("Deck Span (ft)",min_value=10.0, max_value=14.0, step=0.25)
    
    bending_orientation=st.sidebar.radio("Bending Orientation",('Normal','Inverted'))
	
    data = {'Deck Type': deck_type,
	        'Deck Gauge': deck_gauge,
			'Fy (ksi)': yield_strength,
        	'L (ft)': deck_span,
            'BO': bending_orientation}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

L=df['L (ft)'].values.item()
Fy=df['Fy (ksi)'].values.item()
Deck_Ga=df['Deck Gauge'].values.item()
Deck_Type=df['Deck Type'].to_string(index=False)
Bending_Orientation=df['BO'].to_string(index=False)

image_deck_1F = Image.open(os.path.join(ROOT_DIR,'1F.png'))
image_deck_15B = Image.open(os.path.join(ROOT_DIR,'15B.png'))
image_deck_3N = Image.open(os.path.join(ROOT_DIR,'3N.png'))
image_deck_15BST1 = Image.open(os.path.join(ROOT_DIR,'15BST1.png'))
image_deck_15BST2 = Image.open(os.path.join(ROOT_DIR,'15BST2.png'))
image_deck_15BST3 = Image.open(os.path.join(ROOT_DIR,'15BST3.png'))
image_deck_2CST1 = Image.open(os.path.join(ROOT_DIR,'2CST1.png'))
image_deck_2CST2 = Image.open(os.path.join(ROOT_DIR,'2CST2.png'))
image_deck_2CST3 = Image.open(os.path.join(ROOT_DIR,'2CST3.png'))
image_deck_3CST1 = Image.open(os.path.join(ROOT_DIR,'3CST1.png'))
image_deck_3CST2 = Image.open(os.path.join(ROOT_DIR,'3CST2.png'))
image_deck_3CST3 = Image.open(os.path.join(ROOT_DIR,'3CST3.png'))

if Deck_Type=='1F': st.image(image_deck_1F)
elif Deck_Type=='1.5B': st.image(image_deck_15B)
elif Deck_Type=='3N': st.image(image_deck_3N)
elif Deck_Type=='1.5BST1': st.image(image_deck_15BST1)
elif Deck_Type=='1.5BST2': st.image(image_deck_15BST2)
elif Deck_Type=='1.5BST3': st.image(image_deck_15BST3)
elif Deck_Type=='2CST1': st.image(image_deck_2CST1)
elif Deck_Type=='2CST2': st.image(image_deck_2CST2)
elif Deck_Type=='2CST3': st.image(image_deck_2CST3)
elif Deck_Type=='3CST1': st.image(image_deck_3CST1)
elif Deck_Type=='3CST2': st.image(image_deck_3CST2)
elif Deck_Type=='3CST3': st.image(image_deck_3CST3)

if Deck_Ga==26: t_d=0.0179
elif Deck_Ga==24: t_d=0.0238
elif Deck_Ga==22: t_d=0.0295
elif Deck_Ga==21: t_d=0.0329
elif Deck_Ga==20: t_d=0.0358
elif Deck_Ga==19: t_d=0.0418
elif Deck_Ga==18: t_d=0.0474
elif Deck_Ga==17: t_d=0.0538
elif Deck_Ga==16: t_d=0.0598

if Deck_Type=='1F': R_ins=0.125
else: R_ins=0.188

if Deck_Type=='1F': N_hats=9
elif Deck_Type=='1.5B' or Deck_Type=='1.5BST1' or Deck_Type=='1.5BST2' or Deck_Type=='1.5BST3': N_hats=6
elif Deck_Type=='2CST1' or Deck_Type=='2CST2' or Deck_Type=='2CST3' or Deck_Type=='3CST1' or Deck_Type=='3CST2' or Deck_Type=='3CST3' or Deck_Type=='3N': 
    N_hats=3

if Deck_Type=='1F': ang_web=46.39
elif Deck_Type=='1.5B' or Deck_Type=='1.5BST1' or Deck_Type=='1.5BST2' or Deck_Type=='1.5BST3': ang_web=72.5
elif Deck_Type=='3N': ang_web=83
elif Deck_Type=='2CST1' or Deck_Type=='2CST2' or Deck_Type=='2CST3': ang_web=64.106
elif Deck_Type=='3CST1' or Deck_Type=='3CST2' or Deck_Type=='3CST3': ang_web=67.38

if Deck_Type=='1F': w_tf=0.786
elif Deck_Type=='1.5B' or Deck_Type=='1.5BST1' or Deck_Type=='1.5BST2' or Deck_Type=='1.5BST3': w_tf=3.144
elif Deck_Type=='3N': w_tf=5.000
elif Deck_Type=='2CST1' or Deck_Type=='2CST2' or Deck_Type=='2CST3': w_tf=4.765
elif Deck_Type=='3CST1' or Deck_Type=='3CST2' or Deck_Type=='3CST3': w_tf=4.499

if Deck_Type=='1F': w_bf=0.786
elif Deck_Type=='1.5B' or Deck_Type=='1.5BST1' or Deck_Type=='1.5BST2' or Deck_Type=='1.5BST3': w_bf=1.377
elif Deck_Type=='3N': w_bf=1.5625
elif Deck_Type=='2CST1' or Deck_Type=='2CST2' or Deck_Type=='2CST3': w_bf=4.765
elif Deck_Type=='3CST1' or Deck_Type=='3CST2' or Deck_Type=='3CST3': w_bf=4.499

if Deck_Type=='1F': w_w=1.244
elif Deck_Type=='1.5B' or Deck_Type=='1.5BST1' or Deck_Type=='1.5BST2' or Deck_Type=='1.5BST3': w_w=1.275
elif Deck_Type=='3N': w_w=2.6875
elif Deck_Type=='2CST1' or Deck_Type=='2CST2' or Deck_Type=='2CST3': w_w=2.05
elif Deck_Type=='3CST1' or Deck_Type=='3CST2' or Deck_Type=='3CST3': w_w=3.000

if Deck_Type=='1F': w_l=0.33
elif Deck_Type=='1.5B' or Deck_Type=='1.5BST1' or Deck_Type=='1.5BST2' or Deck_Type=='1.5BST3': w_l=0.689
elif Deck_Type=='3N': w_l=0.75
elif Deck_Type=='2CST1' or Deck_Type=='2CST2' or Deck_Type=='2CST3': w_l=2.382
elif Deck_Type=='3CST1' or Deck_Type=='3CST2' or Deck_Type=='3CST3': w_l=2.25

if Deck_Type=='1F' or Deck_Type=='1.5B' or Deck_Type=='3N': ang_st=0.0
elif Deck_Type=='1.5BST1' or Deck_Type=='1.5BST2' or Deck_Type=='1.5BST3' or Deck_Type=='2CST1' or Deck_Type=='2CST2' or Deck_Type=='2CST3': ang_st=30.625
elif Deck_Type=='3CST1' or Deck_Type=='3CST2' or Deck_Type=='3CST3': ang_st=36.501

#Not sure about bp=0 for flat flanges (double-check later)
if Deck_Type=='1F' or Deck_Type=='1.5B' or Deck_Type=='3N': b_p=0.0
elif Deck_Type=='1.5BST1': b_p=0.895
elif Deck_Type=='1.5BST2': b_p=1.25
elif Deck_Type=='1.5BST3': b_p=1.0725
elif Deck_Type=='2CST1': b_p=1.706
elif Deck_Type=='2CST2': b_p=2.0625
elif Deck_Type=='2CST3': b_p=1.3495
elif Deck_Type=='3CST1': b_p=1.687
elif Deck_Type=='3CST2': b_p=1.98
elif Deck_Type=='3CST3': b_p=1.394

ang_web_rad=ang_web*math.pi/180
r_cl=R_ins+0.5*t_d
add_web=(math.sin(0.5*ang_web_rad)/math.sin(0.5*math.pi-0.5*ang_web_rad))*r_cl
b_tfo=w_tf+2*add_web
b_bfo=w_bf+2*add_web
b_wo=w_w+2*add_web
b_lo=w_l+add_web
h_d=b_wo*math.sin(ang_web_rad)
P_d=2*b_wo*math.cos(ang_web_rad)+b_tfo+b_bfo
CW_d=N_hats*b_tfo+(N_hats-1)*b_bfo+N_hats*2*h_d/math.tan(ang_web_rad)+2*b_lo

ang_st_rad=ang_st*math.pi/180
add_st=(math.sin(0.5*ang_st_rad)/math.sin(0.5*math.pi-0.5*ang_st_rad))*r_cl

if ang_st==0: b_sto=0
else: b_sto=w_tf-2*(b_p+add_st)

h_sto=math.tan(ang_st_rad)*0.5*b_sto

st.subheader('Input Parameters')



input_parameters={'Deck Type': Deck_Type,
	              'Deck Gauge': "{:.0f}".format(Deck_Ga),
			      'Yield Strength (ksi)': Fy,
        	      'Deck Span (ft)':"{:.2f}".format(L),
                  'Bending Orientation': Bending_Orientation}
input_parameters_df=pd.DataFrame(input_parameters, index=[0])
st.dataframe(input_parameters_df)

st.subheader('Profile Dimensions')

profile_dimensions1={'t (in.)': "{:.4f}".format(t_d),
                     'h (in.)': "{:.3f}".format(h_d),
	                 'P (in.)': "{:.3f}".format(P_d),
			         'CW (in)': "{:.3f}".format(CW_d),
        	         'btfo (in.)':"{:.3f}".format(b_tfo),
				     'bbfo (in.)':"{:.3f}".format(b_bfo),
				     'blo (in.)':"{:.3f}".format(b_lo)}
profile_dimensions1_df=pd.DataFrame(profile_dimensions1, index=[0])
st.dataframe(profile_dimensions1_df)

profile_dimensions2={'alf_w (deg)':"{:.3f}".format(ang_web),
				     'r (in.)':"{:.3f}".format(r_cl),
				     'bsto (in.)':"{:.3f}".format(b_sto),
				     'hsto (in.)':"{:.3f}".format(h_sto),
				     'alf_s (deg)':"{:.3f}".format(ang_st)}
profile_dimensions2_df=pd.DataFrame(profile_dimensions2, index=[0])
st.dataframe(profile_dimensions2_df)


if Bending_Orientation=='Normal': BO=1
else: BO=2

X_Mcrl_stiff=np.array([[L,h_d,t_d,ang_web,r_cl,b_tfo,b_bfo,b_lo,ang_st,h_sto,N_hats,P_d, BO]])
X_Mcrl_unstiff=np.array([[L,h_d,t_d,ang_web,r_cl,b_tfo,b_bfo,b_lo,ang_st,h_sto,N_hats,P_d]])
X_Mcrd_web=np.array([[L,h_d,t_d,ang_web,r_cl,b_tfo,b_bfo,b_lo,ang_st,h_sto,N_hats,P_d]])
X_Mcrd_flange=np.array([[L,h_d,t_d,ang_web,r_cl,b_tfo,b_bfo,b_lo,ang_st,h_sto,N_hats,P_d, BO]])
X_Mu=np.array([[L,h_d,t_d,ang_web,r_cl,b_tfo,b_bfo,b_lo,ang_st,h_sto,N_hats,P_d, Fy, BO]])

X_kl_stiff=np.array([[L,h_d,t_d,ang_web,r_cl,b_tfo,b_bfo,b_lo,ang_st,h_sto,N_hats,P_d, BO]])
X_kl_unstiff=np.array([[L,h_d,t_d,ang_web,r_cl,b_tfo,b_bfo,b_lo,ang_st,h_sto,N_hats,P_d]])
X_kd_flange=np.array([[L,h_d,t_d,ang_web,r_cl,b_tfo,b_bfo,b_lo,ang_st,h_sto,N_hats,P_d, BO]])

X_Mcrl_stiff_sc=Mcrl_stiff_scaler.transform(X_Mcrl_stiff)
X_Mcrl_unstiff_sc=Mcrl_unstiff_scaler.transform(X_Mcrl_unstiff)
X_Mcrd_web_sc=Mcrd_web_scaler.transform(X_Mcrd_web)
X_Mcrd_flange_sc=Mcrd_flange_scaler.transform(X_Mcrd_flange)
X_Mu_sc=Mu_scaler.transform(X_Mu)

X_kl_stiff_sc=kl_stiff_scaler.transform(X_kl_stiff)
X_kl_unstiff_sc=kl_unstiff_scaler.transform(X_kl_unstiff)
X_kd_flange_sc=kd_flange_scaler.transform(X_kd_flange)


Mcrl_stiff=Mcrl_stiff_SVR.predict(X_Mcrl_stiff_sc).item()/(CW_d/12)

if Bending_Orientation=='Inverted': Mcrl_unstiff=Mcrl_unstiff_SVR.predict(X_Mcrl_unstiff_sc).item()/(CW_d/12)
else: Mcrl_unstiff=0.0

if Bending_Orientation=='Inverted': Mcrd_web=Mcrd_web_SVR.predict(X_Mcrd_web_sc).item()/(CW_d/12)
else: Mcrd_web=0.0

if Deck_Type=='1F' or Deck_Type=='1.5B' or Deck_Type=='3N': Mcrd_flange=0.0
elif Deck_Type=='1.5BST1' and Bending_Orientation=='Inverted': Mcrd_flange=0.0
elif Deck_Type=='1.5BST2' and Bending_Orientation=='Inverted': Mcrd_flange=0.0
elif Deck_Type=='1.5BST3' and Bending_Orientation=='Inverted': Mcrd_flange=0.0
else: Mcrd_flange=Mcrd_flange_SVR.predict(X_Mcrd_flange_sc).item()/(CW_d/12)

Mu=Mu_SVR.predict(X_Mu_sc).item()/(CW_d/12)

kl_stiff=kl_stiff_SVR.predict(X_kl_stiff_sc).item()

if Bending_Orientation=='Inverted':kl_unstiff=kl_unstiff_SVR.predict(X_kl_unstiff_sc).item()
else: kl_unstiff=0.0

if Deck_Type=='1F' or Deck_Type=='1.5B' or Deck_Type=='3N': kd_flange=0.0
elif Deck_Type=='1.5BST1' and Bending_Orientation=='Inverted': kd_flange=0.0
elif Deck_Type=='1.5BST2' and Bending_Orientation=='Inverted': kd_flange=0.0
elif Deck_Type=='1.5BST3' and Bending_Orientation=='Inverted': kd_flange=0.0
else: kd_flange=kd_flange_SVR.predict(X_kd_flange_sc).item()

st.subheader('SVM Predictions')

st.write('Deck Moments (k-in/ft)')

predictions1={'Mcrl,s': "{:.3f}".format(Mcrl_stiff),
             'Mcrl,u': "{:.3f}".format(Mcrl_unstiff),
             'Mcrd,w': "{:.3f}".format(Mcrd_web),
             'Mcrd,f': "{:.3f}".format(Mcrd_flange),
             'Mu': "{:.3f}".format(Mu)}
predictions1_df=pd.DataFrame(predictions1, index=[0])
st.dataframe(predictions1_df)
st.caption('Note: 0.000 stands for not applicable')

st.write('Buckling Coefficients')

predictions2={'kl,s': "{:.2f}".format(kl_stiff),
             'kl,u': "{:.2f}".format(kl_unstiff),
			 'kd,f': "{:.2f}".format(kd_flange)}
predictions2_df=pd.DataFrame(predictions2, index=[0])
st.dataframe(predictions2_df)
st.caption('Note: 0.00 stands for not applicable')

st.subheader('Nomenclature')
st.write('Mcrl,s is local elastic buckling moment of stiffened flanges;')
st.write('Mcrl,u is local elastic buckling moment of unstiffened flanges;')
st.write('Mcrd,w is distortional elastic buckling moment of the web-edge flange junction;')
st.write('Mcrd,f is distortional elastic buckling moment of the flange-stiffener junction;')
st.write('Mu is ultimate moment;')
st.write('kl,s is plate buckling coefficient of stiffened flanges;')
st.write('kl,u is plate buckling coefficient of unstiffened flanges;')
st.write('kd,f is plate buckling coefficient for distortional buckling of deck flanges with a longitudinal stiffener.')

st.subheader('Typical Buckling Modes')

st.write('Elastic Buckling of Stiffened Flanges')

image1 = Image.open(os.path.join(ROOT_DIR,'Local_Stiff_1.png'))
st.image(image1)

image2 = Image.open(os.path.join(ROOT_DIR,'Local_Stiff_2.png'))
st.image(image2)

st.write('Elastic Buckling of Unstiffened Flanges')

image3 = Image.open(os.path.join(ROOT_DIR,'Local_Unstiff.png'))
st.image(image3)

st.write('Distortional Elastic Buckling of The Web-Edge Flange Junction')

image4 = Image.open(os.path.join(ROOT_DIR,'Distort_Web.png'))
st.image(image4)

st.write('Distortional Elastic Buckling of The Flange-Stiffener Junction')

image5 = Image.open(os.path.join(ROOT_DIR,'Distort_Flange.png'))
st.image(image5)

st.subheader('Reference')
st.write('Degtyarev, V.V. (2022). "Exploring machine learning for predicting elastic buckling and ultimate moments of steel decks in bending." Proceedings of the Annual Stability Conference. Structural Stability Research Council (SSRC), Denver, Colorado. (in press)')

st.subheader('Source code')
st.markdown('[GitHub](https://github.com/vitdegtyarev/Steel_Deck-Streamlit)', unsafe_allow_html=True)

