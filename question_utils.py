import json
import re
from typing import List, Dict, Any
import streamlit as st
from model_serving_utils import query_endpoint

def generate_questions_from_text(text: str, num_questions: int = 5, serving_endpoint: str = None, system_prompt: str = None) -> List[Dict[str, Any]]:
    """Generate multiple choice questions from text using LLM."""
    
    if not serving_endpoint:
        st.error("No serving endpoint provided")
        return []
    
    # Create messages for the LLM
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Smart text selection - look for question-related content first
    def find_question_content(text, max_length=50000):  # Much higher limit for Sonnet 4
        """Find the most relevant part of the text that likely contains questions."""
        text_lower = text.lower()
        
        # Look for question-related keywords
        question_keywords = [
            'sample questions', 'questions', 'question 1', 'question 2', 'question 3',
            'multiple choice', 'choose the best', 'select the', 'which of the following',
            'exam questions', 'practice questions', 'test questions'
        ]
        
        # Find the best section containing questions
        best_start = 0
        best_score = 0
        
        # Check different sections of the text
        for i in range(0, len(text) - 1000, 1000):
            section = text[i:i+max_length]
            section_lower = section.lower()
            
            # Score this section based on question keywords
            score = sum(1 for keyword in question_keywords if keyword in section_lower)
            
            if score > best_score:
                best_score = score
                best_start = i
        
        # If we found a good section, use it; otherwise use the beginning
        if best_score > 0:
            return text[best_start:best_start+max_length]
        else:
            return text[:max_length]
    
    # Get the most relevant text section
    # If document is small enough, use the entire document
    if len(text) <= 100000:  # If document is under 100k characters, use it all
        relevant_text = text
    else:
        relevant_text = find_question_content(text, max_length=50000)  # Much higher limit for Sonnet 4
    
    
    # Create user prompt for question generation
    user_prompt = f"""
    Follow the system instructions above to extract or generate {num_questions} multiple choice questions.
    
    Text to analyze:
    {relevant_text}
    
    IMPORTANT: Follow the system instructions exactly. If questions exist in the document, extract them first. Only generate new questions if no existing questions are found.
    
    Format your response as a JSON array where each question has this structure:
    {{
        "question": "Your question here?",
        "option_a": "First option",
        "option_b": "Second option", 
        "option_c": "Third option",
        "option_d": "Fourth option",
        "correct_answer": "A",
        "difficulty": "easy|medium|hard"
    }}
    
    Return only the JSON array, no other text.
    """
    
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        
        # Query the LLM
        response = query_endpoint(
            endpoint_name=serving_endpoint,
            messages=messages,
            max_tokens=4000  # Increased for Sonnet 4's capabilities
        )
        
        content = response["content"]
        
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            questions = json.loads(json_str)
            
            # Validate questions
            validated_questions = []
            for q in questions:
                if validate_question_format(q):
                    validated_questions.append(q)
                else:
                    st.warning(f"Invalid question format: {q}")
            
            return validated_questions
        else:
            st.error("Could not extract JSON from LLM response")
            return []
            
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

def validate_question_format(question: Dict[str, Any]) -> bool:
    """Validate that a question has the required format."""
    required_fields = ["question", "option_a", "option_b", "option_c", "option_d", "correct_answer"]
    
    for field in required_fields:
        if field not in question or not question[field]:
            return False
    
    # Check correct answer is valid
    if question["correct_answer"] not in ["A", "B", "C", "D"]:
        return False
    
    return True

def format_questions_for_display(questions: List[Dict[str, Any]]) -> str:
    """Format questions for display in Streamlit."""
    if not questions:
        return "No questions generated."
    
    formatted = ""
    for i, q in enumerate(questions, 1):
        formatted += f"**Question {i}:** {q['question']}\n\n"
        formatted += f"A) {q['option_a']}\n"
        formatted += f"B) {q['option_b']}\n"
        formatted += f"C) {q['option_c']}\n"
        formatted += f"D) {q['option_d']}\n"
        formatted += f"**Correct Answer:** {q['correct_answer']}\n"
        formatted += f"**Difficulty:** {q.get('difficulty', 'medium')}\n\n"
        formatted += "---\n\n"
    
    return formatted
