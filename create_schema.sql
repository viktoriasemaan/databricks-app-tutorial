-- Create schema for question generation app
CREATE SCHEMA IF NOT EXISTS quiz_app;

-- Table to store uploaded PDFs
CREATE TABLE IF NOT EXISTS quiz_app.pdfs (
    pdf_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size INTEGER,
    page_count INTEGER
);

-- Table to store generated questions
CREATE TABLE IF NOT EXISTS quiz_app.questions (
    question_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pdf_id INTEGER REFERENCES quiz_app.pdfs(pdf_id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    option_a TEXT NOT NULL,
    option_b TEXT NOT NULL,
    option_c TEXT NOT NULL,
    option_d TEXT NOT NULL,
    correct_answer CHAR(1) NOT NULL CHECK (correct_answer IN ('A', 'B', 'C', 'D')),
    difficulty VARCHAR(20) DEFAULT 'medium',
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Table to store test results
CREATE TABLE IF NOT EXISTS quiz_app.test_results (
    result_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_questions INTEGER NOT NULL,
    correct_answers INTEGER NOT NULL,
    score_percentage DECIMAL(5,2) NOT NULL,
    time_taken_seconds INTEGER,
    questions_used TEXT -- JSON array of question IDs used
);

-- Grant permissions to the app's client ID
-- Replace with your actual client ID from the environment
GRANT USAGE ON SCHEMA quiz_app TO "your client id";
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE quiz_app.pdfs TO "your client id";
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE quiz_app.questions TO "your client id";
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE quiz_app.test_results TO "your client id";
