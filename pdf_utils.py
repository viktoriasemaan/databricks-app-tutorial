import os
import io
import base64
from typing import List, Dict, Any
import streamlit as st
import PyPDF2
from PIL import Image

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def get_pdf_info(pdf_file) -> Dict[str, Any]:
    """Get PDF file information."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Get file size
        pdf_file.seek(0, 2)  # Seek to end
        file_size = pdf_file.tell()
        pdf_file.seek(0)  # Reset to beginning
        
        return {
            "page_count": len(pdf_reader.pages),
            "file_size": file_size,
            "filename": pdf_file.name
        }
    except Exception as e:
        st.error(f"Error getting PDF info: {e}")
        return {"page_count": 0, "file_size": 0, "filename": "unknown"}

def validate_pdf_file(pdf_file) -> bool:
    """Validate that the uploaded file is a valid PDF."""
    try:
        # Check file extension
        if not pdf_file.name.lower().endswith('.pdf'):
            return False
        
        # Try to read the PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        if len(pdf_reader.pages) == 0:
            return False
        
        return True
    except Exception:
        return False

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
