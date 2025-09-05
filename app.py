import logging
import os
import streamlit as st
from model_serving_utils import query_endpoint, is_endpoint_supported
from PIL import Image
import io
import json
import time
from dotenv import load_dotenv
from database_utils import (
    get_holiday_requests, update_holiday_request, add_holiday_request,
    save_pdf, save_questions, get_all_pdfs, get_questions_by_pdf, 
    get_random_questions, save_test_result, get_test_history,
    create_quiz, get_all_quizzes
)
from pdf_utils import extract_text_from_pdf, get_pdf_info, validate_pdf_file, format_file_size
from question_utils import generate_questions_from_text, validate_question_format, format_questions_for_display
import pandas as pd
from datetime import datetime, date

# Load environment variables from .env file
load_dotenv()

# Disable git info warning
os.environ['STREAMLIT_GIT_ENABLED'] = 'false'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variable is set correctly
SERVING_ENDPOINT = os.getenv('SERVING_ENDPOINT')

# For local development, provide a helpful error message
if not SERVING_ENDPOINT:
    st.error("🚫 Missing SERVING_ENDPOINT Environment Variable")
    st.markdown("""
    **To fix this error, you need to set the SERVING_ENDPOINT environment variable:**
    
    **Option 1: Export in terminal**
    ```bash
    export SERVING_ENDPOINT="your-serving-endpoint-name"
    streamlit run app.py
    ```
    
    **Option 2: Create a .env file**
    Create a `.env` file in your project root with:
    ```
    SERVING_ENDPOINT=your-serving-endpoint-name
    ```
    
    **Option 3: Set inline when running**
    ```bash
    SERVING_ENDPOINT="your-serving-endpoint-name" streamlit run app.py
    ```
    
    Replace `your-serving-endpoint-name` with your actual Databricks serving endpoint name.
    """)
    st.stop()

# Check if the endpoint is supported
endpoint_supported = is_endpoint_supported(SERVING_ENDPOINT)

def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )

user_info = get_user_info()

# Streamlit app
# Initialize all session state variables at the top
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Knowledge Testing App")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Knowledge Testing", "Question Generation", "AI Chatbot", "Holiday Manager"])


with tab1:
    st.header("Knowledge Testing")
    st.markdown("Test your knowledge with randomly selected questions from available quizzes.")
    
    # Quiz Selection Section
    st.subheader("Select Quiz")
    
    # Get available quizzes
    quizzes_df = get_all_quizzes()
    
    if quizzes_df.empty:
        st.warning("⚠️ No quizzes available. Please create some quizzes first in the Question Generation tab.")
    else:
        # Quiz selection
        quiz_options = {}
        for _, quiz in quizzes_df.iterrows():
            quiz_options[f"{quiz['quiz_name']} ({quiz['total_questions']} questions) - {quiz['created_date'].strftime('%Y-%m-%d')}"] = quiz['quiz_id']
        
        selected_quiz_display = st.selectbox(
            "Choose a quiz to take:",
            options=list(quiz_options.keys()),
            key="quiz_selection"
        )
        
        selected_quiz_id = quiz_options[selected_quiz_display] if selected_quiz_display else None
        selected_quiz_name = selected_quiz_display.split(' (')[0] if selected_quiz_display else None
        
        # Initialize session state for test
        if "test_questions" not in st.session_state:
            st.session_state.test_questions = None
        if "test_answers" not in st.session_state:
            st.session_state.test_answers = {}
        if "test_started" not in st.session_state:
            st.session_state.test_started = False
        if "test_completed" not in st.session_state:
            st.session_state.test_completed = False
        if "test_start_time" not in st.session_state:
            st.session_state.test_start_time = None
        
        # Start new test
        if not st.session_state.test_started:
            st.subheader("🎯 Start New Test")
            
            col1, col2 = st.columns(2)
            with col1:
                num_test_questions = st.slider("Number of Questions", min_value=3, max_value=20, value=5, key="test_questions_slider")
            with col2:
                time_limit = st.selectbox("Time Limit", ["No Limit", "5 minutes", "10 minutes", "15 minutes"], index=0, key="test_time_limit")
            
            if st.button("Start Test", type="primary", key="start_test_btn"):
                with st.spinner("🎲 Selecting random questions..."):
                    st.session_state.test_questions = get_random_questions(num_test_questions, selected_quiz_id)
                    st.session_state.test_answers = {}
                    st.session_state.test_started = True
                    st.session_state.test_completed = False
                    st.session_state.test_start_time = time.time()
                    st.rerun()
        
        # Take the test
        elif st.session_state.test_started and not st.session_state.test_completed:
            st.subheader("📝 Taking Test")
            
            # Show progress
            progress = len(st.session_state.test_answers) / len(st.session_state.test_questions)
            st.progress(progress)
            st.write(f"Progress: {len(st.session_state.test_answers)}/{len(st.session_state.test_questions)} questions answered")
            
            # Show current question
            current_q_idx = len(st.session_state.test_answers)
            if current_q_idx < len(st.session_state.test_questions):
                current_question = st.session_state.test_questions.iloc[current_q_idx]
                
                st.write(f"**Question {current_q_idx + 1}:** {current_question['question_text']}")
                
                # Answer options
                answer = st.radio(
                    "Select your answer:",
                    options=["A", "B", "C", "D"],
                    format_func=lambda x: f"{x}) {current_question[f'option_{x.lower()}']}",
                    key=f"q_{current_question['question_id']}"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Next Question", type="primary"):
                        st.session_state.test_answers[current_question['question_id']] = answer
                        st.rerun()
                
                with col2:
                    if st.button("Finish Test", type="primary"):
                        st.session_state.test_answers[current_question['question_id']] = answer
                        st.session_state.test_completed = True
                        st.rerun()
            
            # Finish test button
            if len(st.session_state.test_answers) == len(st.session_state.test_questions):
                if st.button("Finish Test", type="primary"):
                    st.session_state.test_completed = True
                    st.rerun()
        
        # Show results
        elif st.session_state.test_completed:
            st.subheader("📊 Test Results")
            
            # Calculate results
            correct_answers = 0
            total_questions = len(st.session_state.test_questions)
            time_taken = int(time.time() - st.session_state.test_start_time) if st.session_state.test_start_time else 0
            
            results = []
            for _, question in st.session_state.test_questions.iterrows():
                user_answer = st.session_state.test_answers.get(question['question_id'], '')
                is_correct = user_answer == question['correct_answer']
                if is_correct:
                    correct_answers += 1
                
                results.append({
                    'question': question['question_text'],
                    'user_answer': user_answer,
                    'correct_answer': question['correct_answer'],
                    'is_correct': is_correct
                })
            
            score_percentage = (correct_answers / total_questions) * 100
            
            # Display score
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Score", f"{correct_answers}/{total_questions}")
            with col2:
                st.metric("Percentage", f"{score_percentage:.1f}%")
            with col3:
                st.metric("Time Taken", f"{time_taken // 60}:{time_taken % 60:02d}")
            with col4:
                grade = "A" if score_percentage >= 90 else "B" if score_percentage >= 80 else "C" if score_percentage >= 70 else "D" if score_percentage >= 60 else "F"
                st.metric("Grade", grade)
            
            # Save results to database
            question_ids = st.session_state.test_questions['question_id'].tolist()
            save_test_result(total_questions, correct_answers, score_percentage, time_taken, question_ids, selected_quiz_id, selected_quiz_name)
            
            # Show detailed results
            st.subheader("📋 Detailed Results")
            for i, result in enumerate(results, 1):
                with st.expander(f"Question {i}: {'✅' if result['is_correct'] else '❌'}", expanded=False):
                    st.write(f"**Question:** {result['question']}")
                    st.write(f"**Your Answer:** {result['user_answer']}")
                    st.write(f"**Correct Answer:** {result['correct_answer']}")
                    if result['is_correct']:
                        st.success("✅ Correct!")
                    else:
                        st.error("❌ Incorrect")
            
            # Reset test
            if st.button("Take Another Test", type="primary"):
                st.session_state.test_questions = None
                st.session_state.test_answers = {}
                st.session_state.test_started = False
                st.session_state.test_completed = False
                st.session_state.test_start_time = None
                st.rerun()
        
        # Show test history
        st.subheader("📈 Test History")
        history_df = get_test_history()
        
        if not history_df.empty:
            # Display test history with clickable entries
            for _, test in history_df.iterrows():
                with st.expander(f"📅 {test['test_date'].strftime('%Y-%m-%d %H:%M')} - Score: {test['score_percentage']:.1f}%", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Questions", test['total_questions'])
                    with col2:
                        st.metric("Correct", test['correct_answers'])
                    with col3:
                        st.metric("Score %", f"{test['score_percentage']:.1f}%")
                    with col4:
                        st.metric("Time", f"{test['time_taken_seconds'] // 60}:{test['time_taken_seconds'] % 60:02d}")
                    
                    # Calculate and display grade
                    grade = "A" if test['score_percentage'] >= 90 else "B" if test['score_percentage'] >= 80 else "C" if test['score_percentage'] >= 70 else "D" if test['score_percentage'] >= 60 else "F"
                    st.write(f"**Grade:** {grade}")
                    
                    # Show quiz name if available
                    if 'quiz_name' in test and test['quiz_name']:
                        st.write(f"**Quiz:** {test['quiz_name']}")
                    
                    # Performance feedback
                    if test['score_percentage'] >= 90:
                        st.success("🎉 Excellent performance!")
                    elif test['score_percentage'] >= 80:
                        st.info("👍 Good job!")
                    elif test['score_percentage'] >= 70:
                        st.warning("📚 Keep practicing!")
                    else:
                        st.error("💪 Don't give up! Review the material and try again.")
        else:
            st.info("No test history available yet.")


with tab2:
    st.header("Question Generation")
    st.markdown("Upload PDFs and generate multiple choice questions using AI. Create reusable quizzes for testing.")
    
    # PDF Upload Section
    st.subheader("Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to generate questions from its content"
    )
    
    if uploaded_file is not None:
        # Validate PDF
        if not validate_pdf_file(uploaded_file):
            st.error("❌ Invalid PDF file. Please upload a valid PDF document.")
        else:
            # Get PDF info
            pdf_info = get_pdf_info(uploaded_file)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", format_file_size(pdf_info["file_size"]))
            with col2:
                st.metric("Pages", pdf_info["page_count"])
            with col3:
                st.metric("Filename", pdf_info["filename"][:20] + "..." if len(pdf_info["filename"]) > 20 else pdf_info["filename"])
            
            # Extract text
            with st.spinner("📖 Extracting text from PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
            
            if pdf_text:
                st.success("✅ PDF text extracted successfully")
                
                # Quiz creation settings
                st.subheader("Create Quiz")
                
                col1, col2 = st.columns(2)
                with col1:
                    quiz_name = st.text_input("Quiz Name", value=f"Quiz - {pdf_info['filename'][:30]}", key="quiz_name_input")
                with col2:
                    quiz_description = st.text_input("Quiz Description (Optional)", key="quiz_description_input")
                
                # Question generation settings
                st.subheader("Question Generation Settings")
                
                col1, col2 = st.columns(2)
                with col1:
                    num_questions = st.slider("Number of Questions", min_value=3, max_value=20, value=5, key="question_gen_slider")
                with col2:
                    difficulty = st.selectbox("Difficulty Level", ["easy", "medium", "hard"], index=1, key="question_gen_difficulty")
                
                # System prompt for better question generation
                st.subheader("AI System Prompt")
                st.markdown("Customize how the AI generates questions. This helps create better, more targeted questions.")
                
                default_prompt = """You are a question extraction specialist. Your PRIMARY task is to extract existing questions from documents and format them properly.

PROCESS:
1. EXTRACT FIRST: Find all existing questions in the document (questions with ?, multiple choice, sample questions, etc.)
2. FORMAT: Structure each question with Question, Answer choices (A,B,C,D), Correct Answer, and Difficulty
3. ONLY IF NO QUESTIONS FOUND: Generate new questions in the same format

OUTPUT FORMAT for each question:
Question: [exact question text from document]
A) [option A text]  
B) [option B text]
C) [option C text]
D) [option D text]
Correct Answer: [letter]
Difficulty: [easy/medium/hard]

CRITICAL RULES:
- Always extract existing questions first before generating new ones
- Copy question text exactly as it appears in the document
- If original questions lack answer choices, create appropriate ones
- Only generate new questions if zero questions are found in the document"""

                system_prompt = st.text_area(
                    "System Prompt (Advanced)",
                    value=default_prompt,
                    height=200,
                    help="This prompt guides the AI on how to generate questions. Modify it to change the AI's behavior.",
                    key="system_prompt_textarea"
                )
                
                # Generate questions button
                if st.button("Generate Questions", type="primary", key="generate_questions_btn"):
                    with st.spinner("🤖 Generating questions with AI..."):
                        try:
                            # Generate questions using the provided system prompt
                            generated_questions = generate_questions_from_text(
                                pdf_text, 
                                num_questions, 
                                serving_endpoint=SERVING_ENDPOINT,
                                system_prompt=system_prompt
                            )
                            
                            if generated_questions:
                                # Store in session state for persistence
                                st.session_state.generated_questions = generated_questions
                                st.session_state.pdf_info = pdf_info
                                st.session_state.pdf_text = pdf_text
                                st.session_state.quiz_name = quiz_name
                                st.session_state.quiz_description = quiz_description
                                
                                st.success(f"✅ Generated {len(generated_questions)} questions successfully!")
                                
                                # Display generated questions
                                st.subheader("Generated Questions (Ready to Save)")
                                
                                # Display questions in expandable format like the image
                                for i, question in enumerate(generated_questions):
                                    with st.expander(f"Question {i + 1}", expanded=True):
                                        st.markdown(f"**Question:** {question['question']}")
                                        st.markdown(f"**A)** {question['option_a']}")
                                        st.markdown(f"**B)** {question['option_b']}")
                                        st.markdown(f"**C)** {question['option_c']}")
                                        st.markdown(f"**D)** {question['option_d']}")
                                        st.markdown(f"**Correct Answer:** {question['correct_answer']}")
                                        st.markdown(f"**Difficulty:** {question.get('difficulty', 'medium')}")
                                
                                # Save to database button
                                if st.button("Save Quiz to Database", type="primary", key="save_quiz_btn"):
                                    # Save PDF first
                                    pdf_id = save_pdf(pdf_info["filename"], pdf_text, pdf_info["file_size"], pdf_info["page_count"])
                                    
                                    if pdf_id:
                                        # Create quiz
                                        quiz_id = create_quiz(quiz_name, quiz_description, pdf_id, len(generated_questions))
                                        
                                        if quiz_id:
                                            # Save questions
                                            if save_questions(generated_questions, quiz_id):
                                                st.success("✅ Quiz saved to database successfully!")
                                                # Clear session state after successful save
                                                st.session_state.generated_questions = None
                                                st.session_state.pdf_info = None
                                                st.session_state.pdf_text = None
                                                st.session_state.quiz_name = None
                                                st.session_state.quiz_description = None
                                            else:
                                                st.error("❌ Failed to save questions to database.")
                                        else:
                                            st.error("❌ Failed to create quiz in database.")
                                    else:
                                        st.error("❌ Failed to save PDF to database.")
                            else:
                                st.error("❌ Failed to generate questions. Please try again.")
                        except Exception as e:
                            st.error(f"❌ Error generating questions: {str(e)}")
                
            else:
                st.error("❌ Failed to extract text from PDF. Please try a different file.")
    
    # Show existing quizzes
    st.subheader("📚 Your Quizzes")
    
    # Get all quizzes
    quizzes_df = get_all_quizzes()
    if not quizzes_df.empty:
        st.dataframe(
            quizzes_df[['quiz_name', 'description', 'total_questions', 'created_date']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "quiz_name": "Quiz Name",
                "description": "Description",
                "total_questions": "Questions",
                "created_date": "Created Date"
            }
        )
    else:
        st.info("No quizzes created yet.")


with tab3:
    st.header("AI Chatbot")
    
    # Image upload and display section
    st.subheader("Image Upload & View")
    st.markdown("Drop or upload an image file to view it below.")

    # File uploader for images
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
        help="Supported formats: PNG, JPG, JPEG, GIF, BMP, WEBP. Max size: 10MB"
    )

    # Show size warning if file is selected
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 10:
            st.warning(f"⚠️ **Large Image Detected**: {file_size_mb:.1f}MB. This image will be automatically compressed to fit within the 10MB limit.")

    # Display uploaded image
    if uploaded_file is not None:
        try:
            # Read and display the image
            image = Image.open(uploaded_file)
            
            # Always process images for optimal performance
            file_size = uploaded_file.size
            max_size = 10 * 1024 * 1024  # 10MB in bytes
            
            # Check if image needs compression
            needs_compression = file_size > max_size
            if needs_compression:
                st.warning(f"⚠️ Image is {file_size / (1024*1024):.1f}MB. Processing to fit within 10MB limit...")
            
            # Always optimize images for better performance
            if image.mode in ('RGBA', 'LA'):
                # Convert RGBA to RGB for better compression
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            
            # Always resize images to reasonable dimensions for better performance
            max_dimension = 1024
            original_size = image.size
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                st.info(f"📏 Image resized from {original_size} to {new_size} pixels")
            
            # Save with compression
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
            img_byte_arr.seek(0)
            
            # Create new PIL Image from processed bytes
            image = Image.open(img_byte_arr)
            
            final_size_mb = len(img_byte_arr.getvalue()) / (1024*1024)
            st.success(f"✅ Image processed successfully! Final size: {final_size_mb:.1f}MB")
            
            # Show compression ratio if original was large
            if needs_compression:
                compression_ratio = (1 - final_size_mb / (file_size / (1024*1024))) * 100
                st.info(f"📊 Compression achieved: {compression_ratio:.1f}% size reduction")
            
            # Store the image in session state for chat use
            st.session_state.current_image = image

            # Show image info
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Image Details:**")
                st.write(f"**Format:** {image.format}")
                st.write(f"**Mode:** {image.mode}")
                st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")

            with col2:
                st.write("**Image Preview:**")
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

            # Show instructions for using the image in chat
            st.success("✅ Image uploaded successfully! You can now ask questions about it in the chat below.")
            st.info("💡 Try asking: 'What do you see in this image?' or 'Describe this image'")
            
            # Add a button to clear the current image
            if st.button("Clear Image", type="primary"):
                st.session_state.current_image = None
                st.rerun()

            # Option to download the image
            if st.button("Download Image", type="primary"):
                # Convert image to bytes for download
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format or 'PNG')
                img_byte_arr = img_byte_arr.getvalue()

                st.download_button(
                    label="Click to Download",
                    data=img_byte_arr,
                    file_name=uploaded_file.name,
                    mime=f"image/{image.format.lower() if image.format else 'png'}"
                )

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please try uploading a different image file.")

    # Add a separator
    st.divider()

    # Show current image status
    if st.session_state.current_image is not None:
        st.info("🖼️ **Image Ready for Chat**: You can now ask questions about the uploaded image in the chat below!")

    # Check if endpoint is supported and show appropriate UI
    if not endpoint_supported:
        st.error("⚠️ Unsupported Endpoint Type")
        st.markdown(
            f"The endpoint `{SERVING_ENDPOINT}` is not compatible with this basic chatbot template.\n\n"
            "This template only supports chat completions-compatible endpoints.\n\n"
            "👉 **For a richer chatbot template** that supports all conversational endpoints on Databricks, "
            "please see the [Databricks documentation](https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app)."
        )
    else:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask me about the image or anything else..."):
            # Prepare user message
            user_message = {"role": "user", "content": prompt}
            
            # If there's a current image, include it in the message (only for the first question)
            if st.session_state.current_image is not None:
                user_message["image"] = st.session_state.current_image
                # Clear the current image after using it so it won't be included in follow-ups
                st.session_state.current_image = None
            
            # Add user message to chat history
            st.session_state.messages.append(user_message)
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                # Show loading indicator
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Query the Databricks serving endpoint
                        assistant_response = query_endpoint(
                            endpoint_name=SERVING_ENDPOINT,
                            messages=st.session_state.messages,
                            max_tokens=400,
                        )["content"]
                        st.markdown(assistant_response)
                    except Exception as e:
                        error_msg = str(e)
                        if "400" in error_msg or "Request size" in error_msg:
                            st.error("❌ **Image Analysis Failed**: The image was too large or the endpoint doesn't support multimodal input. Try uploading a smaller image or ask a text-only question.")
                            st.info("💡 **Tip**: The app will automatically try to process your image, but some endpoints may have limitations.")
                        else:
                            st.error(f"❌ **Error**: {error_msg}")
                        # Don't add to chat history if there was an error
                        assistant_response = None

                # Add assistant response to chat history only if successful
                if assistant_response:
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})


with tab4:
    st.header("Holiday Manager")
    st.markdown("Review, approve, or decline holiday requests from your team.")
    
    # Add new request section
    with st.expander("Add New Holiday Request", expanded=False):
        st.subheader("Add New Request")
        
        col1, col2 = st.columns(2)
        with col1:
            employee_name = st.text_input("Employee Name", key="new_employee_name")
            start_date = st.date_input("Start Date", key="new_start_date")
        with col2:
            end_date = st.date_input("End Date", key="new_end_date")
            status = st.selectbox("Status", ["Pending", "Approved", "Declined"], key="new_status")
        
        manager_note = st.text_area("Manager Note (Optional)", key="new_manager_note")
        
        if st.button("Add Request", type="primary", key="add_request_btn"):
            if employee_name and start_date and end_date:
                if add_holiday_request(employee_name, start_date, end_date, status, manager_note):
                    st.success("✅ Holiday request added successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to add holiday request.")
            else:
                st.error("Please fill in all required fields.")
    
    # Display existing requests
    st.subheader("📋 Holiday Requests")
    
    # Get holiday requests
    requests_df = get_holiday_requests()
    
    if not requests_df.empty:
        # Display requests in a table
        for _, request in requests_df.iterrows():
            with st.expander(f"Request #{request['request_id']} - {request['employee_name']} ({request['status']})", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Employee:** {request['employee_name']}")
                    st.write(f"**Start Date:** {request['start_date']}")
                with col2:
                    st.write(f"**End Date:** {request['end_date']}")
                    st.write(f"**Status:** {request['status']}")
                with col3:
                    st.write(f"**Manager Note:** {request['manager_note'] or 'None'}")
                
                # Action buttons
                if request['status'] == 'Pending':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"Approve", type="primary", key=f"approve_{request['request_id']}"):
                            if update_holiday_request(request['request_id'], 'Approved', 'Request approved by manager'):
                                st.success("✅ Request approved!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to approve request.")
                    with col2:
                        if st.button(f"Decline", type="primary", key=f"decline_{request['request_id']}"):
                            if update_holiday_request(request['request_id'], 'Declined', 'Request declined by manager'):
                                st.success("✅ Request declined!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to decline request.")
                    with col3:
                        if st.button(f"Add Note", type="primary", key=f"note_{request['request_id']}"):
                            note = st.text_input("Manager Note", key=f"note_input_{request['request_id']}")
                            if st.button("Save Note", type="primary", key=f"save_note_{request['request_id']}"):
                                if update_holiday_request(request['request_id'], request['status'], note):
                                    st.success("✅ Note added!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to add note.")
    else:
        st.info("No holiday requests found.")
