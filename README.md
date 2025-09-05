# Databricks App Tutorial - Complete AI-Powered Application

A comprehensive Streamlit application built for Databricks Apps with multiple AI-powered features including multimodal chatbot, holiday management, and intelligent quiz generation.

## 🚀 Features

### 1. 🤖 AI Chatbot
- **Multimodal Support**: Upload images and chat with Claude Sonnet 4
- **Smart Image Processing**: Automatic resizing and compression for large images
- **Context-Aware Responses**: Maintains conversation history
- **Error Handling**: Graceful fallback for image processing issues

### 2. 📅 Holiday Request Manager
- **Request Management**: View, approve, and decline holiday requests
- **Manager Comments**: Add notes to requests
- **Real-time Updates**: Live database integration with Lakebase
- **Clean Interface**: Easy-to-use table with action buttons

### 3. 📚 Question Generation
- **PDF Upload**: Upload PDFs and extract text content
- **AI-Powered Generation**: Use Claude Sonnet 4 to generate multiple-choice questions
- **Smart Text Selection**: Intelligently selects relevant content from large documents
- **Customizable Prompts**: System prompt customization for better question extraction
- **Database Storage**: Save PDFs and questions to Lakebase for future use

### 4. 🧠 Knowledge Testing
- **Random Questions**: Get random questions from your question bank
- **Interactive Testing**: Answer questions with immediate feedback
- **Results Tracking**: View test results and performance
- **Flexible Settings**: Choose number of questions and time limits

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: Databricks Model Serving (Claude Sonnet 4)
- **Database**: Lakebase (PostgreSQL)
- **PDF Processing**: PyPDF2
- **Authentication**: Databricks OAuth
- **Deployment**: Databricks Apps

## 📋 Prerequisites

1. **Databricks Workspace** with:
   - Model Serving endpoint for Claude Sonnet 4
   - Lakebase database instance
   - Databricks Apps enabled

2. **Environment Variables** (set in your Databricks App):
   ```
   SERVING_ENDPOINT=your-claude-sonnet-4-endpoint
   PGHOST=your-lakebase-host
   PGDATABASE=your-database-name
   PGUSER=your-client-id
   PGPORT=5432
   DATABRICKS_CLIENT_ID=your-client-id
   DATABRICKS_CLIENT_SECRET=your-client-secret
   DATABRICKS_HOST=your-workspace-host
   ```

## 🚀 Quick Start

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/viktoriasemaan/databricks-app-tutorial.git
   cd databricks-app-tutorial
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Database Schema**:
   Run the SQL commands in `create_schema.sql` in your Lakebase database:
   ```sql
   -- Create quiz app schema
   CREATE SCHEMA IF NOT EXISTS quiz_app;
   
   -- Create PDFs table
   CREATE TABLE IF NOT EXISTS quiz_app.pdfs (
       pdf_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
       filename VARCHAR(255) NOT NULL,
       content TEXT NOT NULL,
       file_size INTEGER NOT NULL,
       page_count INTEGER NOT NULL,
       uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   
   -- Create questions table
   CREATE TABLE IF NOT EXISTS quiz_app.questions (
       question_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
       pdf_id INTEGER REFERENCES quiz_app.pdfs(pdf_id),
       question_text TEXT NOT NULL,
       option_a TEXT NOT NULL,
       option_b TEXT NOT NULL,
       option_c TEXT NOT NULL,
       option_d TEXT NOT NULL,
       correct_answer CHAR(1) NOT NULL,
       difficulty VARCHAR(20) NOT NULL,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

4. **Deploy to Databricks Apps**:
   ```bash
   databricks sync --watch . /Workspace/Users/your-email@domain.com/databricks_apps/your-app-name
   ```

## 📁 Project Structure

```
databricks-app-tutorial/
├── app.py                 # Main Streamlit application
├── app.yaml              # Databricks App configuration
├── requirements.txt      # Python dependencies
├── create_schema.sql     # Database schema setup
├── model_serving_utils.py # Databricks Model Serving integration
├── database_utils.py     # Lakebase database utilities
├── pdf_utils.py          # PDF processing utilities
├── question_utils.py     # Question generation utilities
└── README.md            # This file
```

## 🔧 Configuration

### App Configuration (`app.yaml`)
```yaml
name: databricks-app-tutorial
resources:
  lakebase:
    - name: lakebase
      database: databricks_postgres
```

### Environment Variables
Set these in your Databricks App environment:
- `SERVING_ENDPOINT`: Your Claude Sonnet 4 model serving endpoint
- `PGHOST`: Lakebase database host
- `PGDATABASE`: Database name
- `PGUSER`: Your Databricks client ID
- `PGPORT`: Database port (usually 5432)
- `DATABRICKS_CLIENT_ID`: Your Databricks client ID
- `DATABRICKS_CLIENT_SECRET`: Your Databricks client secret
- `DATABRICKS_HOST`: Your Databricks workspace host

## 🎯 Usage

### AI Chatbot
1. Upload an image (optional)
2. Type your message
3. Get AI-powered responses from Claude Sonnet 4

### Holiday Request Manager
1. View pending holiday requests
2. Click "Approve" or "Decline" with optional comments
3. Submit your decision

### Question Generation
1. Upload a PDF document
2. Customize the system prompt (optional)
3. Set number of questions and difficulty
4. Generate questions using AI
5. Review and save to database

### Knowledge Testing
1. Choose number of questions and time limit
2. Answer questions from your question bank
3. View results and performance

## 🔒 Security Features

- **OAuth Authentication**: Secure Databricks authentication
- **Environment Variables**: Sensitive data stored securely
- **Input Validation**: PDF and image upload validation
- **Error Handling**: Graceful error handling throughout

## 🐛 Troubleshooting

### Common Issues

1. **"Unable to determine serving endpoint"**
   - Set the `SERVING_ENDPOINT` environment variable

2. **Lakebase connection errors**
   - Verify your Lakebase credentials and permissions
   - Check that the database schema is created

3. **Image upload errors**
   - Images are automatically resized and compressed
   - Maximum size is 10MB

4. **Question generation issues**
   - Check that your PDF contains readable text
   - Try adjusting the system prompt for better results

## 📚 Documentation

- [Databricks Apps Documentation](https://docs.databricks.com/apps)
- [Lakebase Documentation](https://docs.databricks.com/lakebase)
- [Streamlit Documentation](https://docs.streamlit.io)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Powered by [Databricks](https://databricks.com)
- AI capabilities provided by [Anthropic Claude](https://anthropic.com)
- Database powered by [Lakebase](https://docs.databricks.com/lakebase)

---

**Repository**: [https://github.com/viktoriasemaan/databricks-app-tutorial](https://github.com/viktoriasemaan/databricks-app-tutorial)
