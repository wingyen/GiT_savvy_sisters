import streamlit as st
import pandas as pd
import numpy as np

st.title('Savvy Sisters Green Buildings')

if 'peptide_input' not in st.session_state:
  st.session_state.peptide_input = ''

# Input peptide
st.sidebar.subheader('Input peptide sequence')

def insert_active_peptide_example():
    st.session_state.peptide_input = 'LLNQELLLNPTHQIYPVA'

def insert_inactive_peptide_example():
    st.session_state.peptide_input = 'KSAGYDVGLAGNIGNSLALQVAETPHEYYV'

def clear_peptide():
    st.session_state.peptide_input = ''

peptide_seq = st.sidebar.text_input('Enter peptide sequence', st.session_state.peptide_input, key='peptide_input', help='Be sure to enter a valid sequence')
st.sidebar.button('Example of an active AMP', on_click=insert_active_peptide_example)
st.sidebar.button('Example of an inactive peptide', on_click=insert_inactive_peptide_example)
st.sidebar.button('Clear input', on_click=clear_peptide)

if st.session_state.peptide_input == '':
  st.subheader('Welcome to the app!')
  st.info('Enter peptide sequence in the sidebar to proceed', icon='👈')
else:
  st.subheader('⚛️ Input peptide:')
  st.info(peptide_seq)

col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")