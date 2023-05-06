import streamlit as st
import base64
import pandas as pd

st.set_page_config(page_title="CNN Code", page_icon="")

st.markdown("# CNN Code")
st.sidebar.header("CNN Code")
st.write(
    """This is the code used to create the model. It was trained using a balanced set of music samples obtained from FMA_Medium. The 30 second long songs found were snipped in 3, which meant the model learned on 10 second music snippets. Below is a table created to demonstrate the data:"""
)

data = {
    'Genre': ['Folk', 'Pop','Rock','Country','Classical','Jazz','Hiphop','Total'],
    'Snippets':  ['300', '300','300','300','300','300','300','2100']
}

df = pd.DataFrame(data)
st.table(df)

def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

show_pdf('Resources/CNN Code.pdf')
      