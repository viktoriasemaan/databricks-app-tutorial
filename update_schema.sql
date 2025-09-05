-- Update schema to support reusable quizzes
-- Run this SQL to update your existing database

-- Create quizzes table to store quiz metadata
CREATE TABLE IF NOT EXISTS quiz_app.quizzes (
    quiz_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    quiz_name VARCHAR(255) NOT NULL,
    description TEXT,
    pdf_id INTEGER REFERENCES quiz_app.pdfs(pdf_id) ON DELETE CASCADE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    total_questions INTEGER DEFAULT 0
);

-- Add quiz_id column to questions table
ALTER TABLE quiz_app.questions 
ADD COLUMN IF NOT EXISTS quiz_id INTEGER REFERENCES quiz_app.quizzes(quiz_id) ON DELETE CASCADE;

-- Update test_results table to reference quizzes
ALTER TABLE quiz_app.test_results 
ADD COLUMN IF NOT EXISTS quiz_id INTEGER REFERENCES quiz_app.quizzes(quiz_id) ON DELETE SET NULL;

-- Add quiz_name column to test_results for easier display
ALTER TABLE quiz_app.test_results 
ADD COLUMN IF NOT EXISTS quiz_name VARCHAR(255);

-- Grant permissions on the new table
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE quiz_app.quizzes TO "b9c8c5c3-ba96-43f4-bcfe-09405a42f7a3";

-- Update existing questions to have a default quiz (optional - for existing data)
-- This creates a default quiz for any existing questions
INSERT INTO quiz_app.quizzes (quiz_name, description, pdf_id, total_questions)
SELECT 
    'Default Quiz - ' || p.filename as quiz_name,
    'Auto-created quiz from existing questions' as description,
    p.pdf_id,
    COUNT(q.question_id) as total_questions
FROM quiz_app.pdfs p
LEFT JOIN quiz_app.questions q ON p.pdf_id = q.pdf_id
WHERE q.question_id IS NOT NULL
GROUP BY p.pdf_id, p.filename
ON CONFLICT DO NOTHING;

-- Update existing questions to reference the default quiz
UPDATE quiz_app.questions 
SET quiz_id = (
    SELECT quiz_id 
    FROM quiz_app.quizzes 
    WHERE pdf_id = quiz_app.questions.pdf_id 
    LIMIT 1
)
WHERE quiz_id IS NULL;
