import streamlit as st
import base64
import pandas as pd

st.set_page_config(page_title="CNN Code", page_icon="")

st.markdown("# CNN Code")
st.sidebar.header("CNN Code")
st.write(
    """Preprocessing:
The model was trained using a modified version of the FMA_Medium dataset. Initially, the full dataset was sorted based on their genres. After this, data was cleaned (corrupted files removed as per FMA github), and certain genres were removed. This decision was taken based on their popularity (ex: historic genre was ignored) and the amount of songs present in each genre. Additionally, to avoid having an unbalanced dataset, certain songs were also removed. Lastly, to increase the dataset size and to allow the model to detect the genre of a 10 second snippet song, all smaples were split in three. 

Below is a table of the final modified dataset used to train the model:"""
)

data = {
    'Genre': ['Classical', 'Folk','Hiphop','Jazz','Pop','Rock','Total'],
    'Snippets':  ['1857', '1857','1857','1152','1857','1857','10,437']
}

df = pd.DataFrame(data)
st.table(df)

def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

show_pdf('Resources/CNN Code.pdf')
      
