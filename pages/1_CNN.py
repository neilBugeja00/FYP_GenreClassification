import streamlit as st
import base64

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

st.markdown("# CNN Code")
st.sidebar.header("CNN Code")
st.write(
    """This is Eric Zacharia's code used for the CNN in his 'Convolutional Neural Network Classifier' open source project."""
)

def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

show_pdf('Resources/CNN Code.pdf')
      