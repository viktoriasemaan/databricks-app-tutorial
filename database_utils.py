aseis import os
import time
import json
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from sqlalchemy import create_engine, event, text
import streamlit as st

# Initialize Databricks SDK components (following blog post exactly)
app_config = Config()
workspace_client = WorkspaceClient()

def get_lakebase_connection():
    """Get connection to Lakebase database using the exact pattern from the blog post."""
    try:
        # Get connection details from environment variables
        postgres_host = os.getenv("PGHOST")
        postgres_username = app_config.client_id  # Use client_id as username
        postgres_port = 5432
        postgres_database = os.getenv("PGDATABASE", "databricks_postgres")
        
        if not postgres_host:
            st.warning("⚠️ PGHOST not found. Make sure you've added the Lakebase resource to your Databricks App.")
            return None
        
        # Create engine with psycopg driver (as shown in blog)
        postgres_pool = create_engine(
            f"postgresql+psycopg://{postgres_username}:@{postgres_host}:{postgres_port}/{postgres_database}"
        )
        
        # Add event listener to provide OAuth token (exactly as in blog)
        @event.listens_for(postgres_pool, "do_connect")
        def provide_token(dialect, conn_rec, cargs, cparams):
            """Provide the App's OAuth token. Caching is managed by WorkspaceClient"""
            cparams["password"] = workspace_client.config.oauth_token().access_token
        
        return postgres_pool
        
    except Exception as e:
        st.error(f"Error connecting to Lakebase: {e}")
        return None

def get_holiday_requests():
    """Fetch all holiday requests from the database."""
    engine = get_lakebase_connection()
    if not engine:
        return pd.DataFrame()
    
    try:
        query = text("SELECT * FROM holidays.holiday_requests ORDER BY request_id")
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"Error fetching holiday requests: {e}")
        return pd.DataFrame()

def update_holiday_request(request_id, status, manager_note=""):
    """Update a holiday request status and manager note."""
    engine = get_lakebase_connection()
    if not engine:
        return False
    
    try:
        query = text("""
        UPDATE holidays.holiday_requests 
        SET status = :status, manager_note = :manager_note 
        WHERE request_id = :request_id
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {
                "status": status,
                "manager_note": manager_note,
                "request_id": request_id
            })
            conn.commit()
            return result.rowcount > 0
    except Exception as e:
        st.error(f"Error updating holiday request: {e}")
        return False

def add_holiday_request(employee_name, start_date, end_date, status="Pending", manager_note=""):
    """Add a new holiday request."""
    engine = get_lakebase_connection()
    if not engine:
        return False
    
    try:
        query = text("""
        INSERT INTO holidays.holiday_requests (employee_name, start_date, end_date, status, manager_note)
        VALUES (:employee_name, :start_date, :end_date, :status, :manager_note)
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {
                "employee_name": employee_name,
                "start_date": start_date,
                "end_date": end_date,
                "status": status,
                "manager_note": manager_note
            })
            conn.commit()
            return result.rowcount > 0
    except Exception as e:
        st.error(f"Error adding holiday request: {e}")
        return False

# Quiz App Database Functions

def save_pdf(pdf_filename, content, file_size, page_count):
    """Save uploaded PDF to database."""
    engine = get_lakebase_connection()
    if not engine:
        st.error("❌ No database connection available")
        return None
    
    try:
        query = text("""
        INSERT INTO quiz_app.pdfs (filename, content, file_size, page_count)
        VALUES (:filename, :content, :file_size, :page_count)
        RETURNING pdf_id
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {
                "filename": pdf_filename,
                "content": content,
                "file_size": file_size,
                "page_count": page_count
            })
            conn.commit()
            row = result.fetchone()
            if row:
                pdf_id = row[0]
                return pdf_id
            else:
                st.error("❌ No PDF ID returned from database")
                return None
    except Exception as e:
        st.error(f"❌ Database error saving PDF: {e}")
        return None

def create_quiz(quiz_name, description, pdf_id, total_questions):
    """Create a new quiz and return the quiz_id."""
    engine = get_lakebase_connection()
    if not engine:
        st.error("❌ No database connection")
        return None
    
    try:
        query = text("""
        INSERT INTO quiz_app.quizzes (quiz_name, description, pdf_id, total_questions)
        VALUES (:quiz_name, :description, :pdf_id, :total_questions)
        RETURNING quiz_id
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {
                "quiz_name": quiz_name,
                "description": description,
                "pdf_id": pdf_id,
                "total_questions": total_questions
            })
            quiz_id = result.fetchone()[0]
            conn.commit()
            return quiz_id
    except Exception as e:
        st.error(f"❌ Database error creating quiz: {e}")
        return None

def save_questions(questions, quiz_id):
    """Save generated questions to database with quiz_id."""
    engine = get_lakebase_connection()
    if not engine:
        st.error("❌ No database connection")
        return False
    
    try:
        query = text("""
        INSERT INTO quiz_app.questions (quiz_id, question_text, option_a, option_b, option_c, option_d, correct_answer, difficulty)
        VALUES (:quiz_id, :question_text, :option_a, :option_b, :option_c, :option_d, :correct_answer, :difficulty)
        """)
        
        with engine.connect() as conn:
            for i, question in enumerate(questions, 1):
                try:
                    conn.execute(query, {
                        "quiz_id": quiz_id,
                        "question_text": question["question"],
                        "option_a": question["option_a"],
                        "option_b": question["option_b"],
                        "option_c": question["option_c"],
                        "option_d": question["option_d"],
                        "correct_answer": question["correct_answer"],
                        "difficulty": question.get("difficulty", "medium")
                    })
                    conn.commit()  # Commit each question individually
                except Exception as q_error:
                    st.error(f"❌ Error saving question {i}: {q_error}")
                    return False
            
            return True
    except Exception as e:
        st.error(f"❌ Database error saving questions: {e}")
        return False

def get_all_quizzes():
    """Get all available quizzes."""
    engine = get_lakebase_connection()
    if not engine:
        return pd.DataFrame()
    
    try:
        query = text("""
        SELECT quiz_id, quiz_name, description, created_date, total_questions, is_active
        FROM quiz_app.quizzes 
        WHERE is_active = TRUE 
        ORDER BY created_date DESC
        """)
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"❌ Database error fetching quizzes: {e}")
        return pd.DataFrame()

def get_all_pdfs():
    """Get all uploaded PDFs."""
    engine = get_lakebase_connection()
    if not engine:
        return pd.DataFrame()
    
    try:
        query = text("SELECT * FROM quiz_app.pdfs ORDER BY upload_date DESC")
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"Error fetching PDFs: {e}")
        return pd.DataFrame()

def get_questions_by_pdf(pdf_id):
    """Get all questions for a specific PDF."""
    engine = get_lakebase_connection()
    if not engine:
        return pd.DataFrame()
    
    try:
        query = text("""
        SELECT * FROM quiz_app.questions 
        WHERE pdf_id = :pdf_id AND is_active = TRUE 
        ORDER BY created_date
        """)
        df = pd.read_sql_query(query, engine, params={"pdf_id": pdf_id})
        return df
    except Exception as e:
        st.error(f"Error fetching questions: {e}")
        return pd.DataFrame()

def get_random_questions(num_questions=5, quiz_id=None):
    """Get random questions for testing from a specific quiz or all quizzes."""
    engine = get_lakebase_connection()
    if not engine:
        return pd.DataFrame()
    
    try:
        if quiz_id:
            query = text("""
            SELECT * FROM quiz_app.questions 
            WHERE is_active = TRUE AND quiz_id = :quiz_id
            ORDER BY RANDOM() 
            LIMIT :num_questions
            """)
            df = pd.read_sql_query(query, engine, params={"num_questions": num_questions, "quiz_id": quiz_id})
        else:
            query = text("""
            SELECT * FROM quiz_app.questions 
            WHERE is_active = TRUE 
            ORDER BY RANDOM() 
            LIMIT :num_questions
            """)
            df = pd.read_sql_query(query, engine, params={"num_questions": num_questions})
        return df
    except Exception as e:
        st.error(f"Error fetching random questions: {e}")
        return pd.DataFrame()

def save_test_result(total_questions, correct_answers, score_percentage, time_taken, questions_used, quiz_id=None, quiz_name=None):
    """Save test results to database."""
    engine = get_lakebase_connection()
    if not engine:
        return False
    
    try:
        query = text("""
        INSERT INTO quiz_app.test_results (total_questions, correct_answers, score_percentage, time_taken_seconds, questions_used, quiz_id, quiz_name)
        VALUES (:total_questions, :correct_answers, :score_percentage, :time_taken, :questions_used, :quiz_id, :quiz_name)
        """)
        
        with engine.connect() as conn:
            conn.execute(query, {
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "score_percentage": score_percentage,
                "time_taken": time_taken,
                "questions_used": json.dumps(questions_used),
                "quiz_id": quiz_id,
                "quiz_name": quiz_name
            })
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error saving test result: {e}")
        return False

def get_test_history():
    """Get test history."""
    engine = get_lakebase_connection()
    if not engine:
        return pd.DataFrame()
    
    try:
        query = text("""
        SELECT * FROM quiz_app.test_results 
        ORDER BY test_date DESC 
        LIMIT 20
        """)
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"Error fetching test history: {e}")
        return pd.DataFrame()


