import gradio as gr
from huggingface_hub import InferenceClient
import io
from docx import Document
import os
import pymupdf
# For PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Initialize Hugging Face Inference Client with Mistral-7B
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct",
    token=os.getenv("HF_TOKEN"))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_document = pymupdf.open(pdf_file)
        text = "".join(page.get_text() for page in pdf_document)
        return text.strip() or "No text could be extracted from the PDF."
    except Exception as e:
        return f"Error reading PDF: {e}"

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    try:
        doc = Document(docx_file)
        text = "\n".join(para.text for para in doc.paragraphs)
        return text.strip() or "No text could be extracted from the DOCX file."
    except Exception as e:
        return f"Error reading DOCX: {e}"

# Function to analyze CV and generate report
def parse_cv(file, job_description):
    if file is None:
        return "Please upload a CV file.", "", None
    try:
        file_path = file.name
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif file_ext == ".docx":
            extracted_text = extract_text_from_docx(file_path)
        else:
            return "Unsupported file format. Please upload a PDF or DOCX file.", "", None
    except Exception as e:
        return f"Error reading file: {e}", "", None
    if extracted_text.startswith("Error"):
        return extracted_text, "Error during text extraction. Please check the file.", None
    prompt = (
        f"Analyze the CV against the job description. Provide a summary, assessment, "
        f"and a score 0-10.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate CV:\n{extracted_text}\n"
    )
    try:
        analysis = client.text_generation(prompt, max_new_tokens=512)
        report_text = f"--- Analysis Report ---\n{analysis}"
        pdf_path = create_pdf_report(report_text)
        return extracted_text, report_text, pdf_path
    except Exception as e:
        return extracted_text, f"Analysis Error: {e}", None

# Function to create PDF report
def create_pdf_report(report_text):
    if not report_text.strip():
        report_text = "No analysis report to convert."

    pdf_path = "analysis_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    title = Paragraph("<b>Analysis Report</b>", styles['Title'])
    Story.append(title)
    Story.append(Spacer(1, 12))

    report_paragraph = Paragraph(report_text.replace("\n", "<br/>"), styles['BodyText'])
    Story.append(report_paragraph)

    doc.build(Story)
    return pdf_path

# Function to process and optimize resume
def process_resume(resume_file, job_title):
    if resume_file is None:
        return "Please upload a resume file."
    try:
        file_path = resume_file.name
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == ".pdf":
            resume_text = extract_text_from_pdf(file_path)
        elif file_ext == ".docx":
            resume_text = extract_text_from_docx(file_path)
        else:
            return "Unsupported file format. Please upload a PDF or DOCX file."
        if resume_text.startswith("Error"):
            return resume_text
        prompt = (
            f"Rewrite and optimize the following resume for the job title: {job_title}.\n"
            f"Ensure the resume is ATS-friendly, highlights relevant skills, experience, and includes industry keywords.\n\n"
            f"Resume:\n{resume_text}\n"
        )
        optimized_resume = client.text_generation(prompt, max_new_tokens=1024)
        return optimized_resume
    except Exception as e:
        return f"Error processing resume: {e}"

# Build the Gradio UI
demo = gr.Blocks()
with demo:
    gr.Markdown("## AI-powered CV Analyzer and Optimizer")
    with gr.Tab("CV Analyzer"):
        gr.Markdown("### Upload your CV and provide the job description")
        file_input = gr.File(label="Upload CV", file_types=[".pdf", ".docx"])
        job_desc_input = gr.Textbox(label="Job Description", lines=5)
        extracted_text = gr.Textbox(label="Extracted CV Content", lines=10, interactive=False)
        analysis_output = gr.Textbox(label="Analysis Report", lines=10, interactive=False)
        analyze_button = gr.Button("Analyze CV")
        download_pdf_button = gr.File(label="Download Analysis Report PDF", interactive=False)
        analyze_button.click(parse_cv, [file_input, job_desc_input], [extracted_text, analysis_output, download_pdf_button])

    with gr.Tab("CV Optimizer"):
        gr.Markdown("### Upload your Resume and Enter Job Title")
        resume_file = gr.File(label="Upload Resume (PDF or Word)", file_types=[".pdf", ".docx"])
        job_title_input = gr.Textbox(label="Job Title", lines=1)
        optimized_resume_output = gr.Textbox(label="Optimized Resume", lines=20)
        optimize_button = gr.Button("Optimize Resume")
        optimize_button.click(process_resume, [resume_file, job_title_input], [optimized_resume_output])

if __name__ == "__main__":
    demo.queue().launch()
