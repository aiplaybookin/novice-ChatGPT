import streamlit as st
import requests
from news import fetch_news
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import base64
import os
import openai
import nltk

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "<OPENAI API KEY>"

# Set up the company logo and name
company_logo = './company_logo.png'  # Replace with the path to your company logo
company_name = 'AIPlabook'  # Replace with your company name

nltk.download('punkt')


# Function to fetch news based on keyword
def generate_newsletter(news_items):
    # Display the news items in a 4x3 grid layout
    for i in range(0, len(news_items), 3):
        row = st.columns(3)
        for j in range(i, min(i+3, len(news_items))):
            with row[j-i]:
                st.subheader(news_items[j]['title'])
                st.write(news_items[j]['summary'])
                #summary = news_items[j]['summary']
                #st.write(f"<p style='color:blue;'>{summary}</p>", unsafe_allow_html=True)


def generate_news_pdf(news_items):
    doc = SimpleDocTemplate("news_report.pdf", pagesize=letter)
    elements = []

    # Define a style for the news titles
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.spaceBefore = 14
    title_style.spaceAfter = 14

    # Define a style for the article summaries
    summary_style = styles["BodyText"]

    # Add news titles, article summaries, and URLs as paragraphs to the PDF
    for item in news_items:
        title = Paragraph(item['title'], title_style)
        summary = Paragraph(item['summary'], summary_style)

        # Create a hyperlink by adding the 'href' attribute to the Paragraph
        url = f"<a href='{item['url']}' color='blue' underline='true'>{item['url']}</a>"
        url_paragraph = Paragraph(url, styles["BodyText"])
        elements.append(title)
        elements.append(url_paragraph)
        elements.append(summary)
        elements.append(Spacer(1, 12))

    doc.build(elements)


def main():
    # Set page title
    st.set_page_config(page_title="", page_icon=company_logo, layout='wide')

    # Load company logo
    #company_logo = './company_logo.png'  # Update with the path to your company logo

    # Add company logo and name to app header
    col_logo, col_name, col_ext = st.columns([1, 3, 1])
    col_logo.image(company_logo, width=100)
    col_name.title('🤖Generate your own 🗞NewsLetters📰')  # Replace 'Company Name' with your actual company name
    col_ext.write("")
    # Add content to the rest of the app
    st.write('  AI PLAYBOOK ')
    # Add more widgets, text, or images as needed
    
    #st.markdown("---")

    # Search for keyword
    col1, col2, col3 = st.columns([6, 7, 6])
    col1.write("")
    keyword = col2.text_input("Search for a keyword", max_chars=20)
    #col3.write("")
    if col2.button("Search"):
        if keyword:
            articles = fetch_news(keyword)
            if len(articles) > 0:
                st.subheader(f"Latest News on '{keyword}':")
                # Generate and display the newsletter
                generate_newsletter(articles)

                st.write("---")
                col1, col2 = st.columns([1, 2])
                col1.write("")  # Empty column to align button to the right
                download_button = col2.button("Download Articles as PDF")
                
                if download_button:
                    pdf_buffer = generate_news_pdf(articles)
                    st.write("Downloading articles as PDF...")
                    st.success("News Letter is ready to be downloaded as a PDF file.")
                    # Encode pdf_buffer.getvalue() to base64 and then decode to string
                    pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                    st.markdown(
                        f'<a href="data:application/pdf;base64,{pdf_base64}" download="sample_text.pdf">Click here to download</a>',
                        unsafe_allow_html=True,
                    )



if __name__ == '__main__':
    main()

