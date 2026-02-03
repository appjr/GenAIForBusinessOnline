# Week 3: Python with ML and GenAI Toolkits - Slides Batch 3 (Slides 21-30)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Date:** February 4, 2026

---

## Slide 21: LangChain - Building AI Applications

### Framework for LLM-Powered Apps

**What is LangChain?**
- Framework for developing applications powered by language models
- Chains together multiple components (LLMs, prompts, data sources)
- Simplifies complex AI workflows
- Supports memory, agents, and tool integration

**Installation:**
```bash
pip install langchain langchain-openai
```

**Basic Example:**
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a business analyst. {question}"
)

# Create chain
chain = prompt | llm

# Run chain
response = chain.invoke({
    "question": "What are the key benefits of AI in retail?"
})
print(response.content)
```

---

## Slide 22: LangChain - Advanced Features

### Memory, Agents, and RAG

**Conversation Memory:**
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Create memory
memory = ConversationBufferMemory()

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Chat with memory
conversation.predict(input="Hi, I'm working on an AI project.")
conversation.predict(input="What did I just tell you?")
```

**RAG (Retrieval Augmented Generation):**
```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load and split documents
loader = TextLoader("business_data.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Ask questions about your documents
result = qa_chain.run("What are our Q4 sales numbers?")
```

---

## Slide 23: API Integration Best Practices

### Working with GenAI APIs Effectively

**Environment Variables for API Keys:**
```python
import os
from dotenv import load_dotenv

# Load from .env file
load_dotenv()

# Access keys securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
```

**.env file example:**
```bash
OPENAI_API_KEY=sk-your-key-here
HUGGINGFACE_TOKEN=hf_your-token-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Error Handling:**
```python
from openai import OpenAI, APIError, RateLimitError
import time

client = OpenAI()

def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        
        except RateLimitError:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
        
        except APIError as e:
            print(f"API error: {e}")
            if attempt == max_retries - 1:
                raise
    
    return None
```

---

## Slide 24: Cost Management & Optimization

### Controlling AI API Costs

**Token Counting:**
```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

prompt = "Explain artificial intelligence in business"
token_count = count_tokens(prompt)
print(f"Tokens: {token_count}")

# Estimate cost (GPT-4: ~$0.03 per 1K tokens input)
estimated_cost = (token_count / 1000) * 0.03
print(f"Estimated cost: ${estimated_cost:.4f}")
```

**Cost Optimization Strategies:**
```python
# 1. Use cheaper models when appropriate
def choose_model(task_complexity):
    if task_complexity == "simple":
        return "gpt-3.5-turbo"  # Cheaper
    elif task_complexity == "moderate":
        return "gpt-4-turbo"     # Balanced
    else:
        return "gpt-4"           # Most capable

# 2. Limit response length
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=150  # Limit output
)

# 3. Cache responses
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_generation(prompt):
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
```

**Budget Alerts:**
- Set up usage limits in API dashboards
- Monitor daily/monthly spending
- Use environment-specific keys (dev vs. prod)

---

## Slide 25: Building Your First GenAI App

### Hands-on Lab Part 1 - Setup

**Project: AI-Powered Business Analyzer**

**Step 1: Create Project Structure**
```bash
mkdir genai-business-app
cd genai-business-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create files
touch app.py
touch .env
touch requirements.txt
```

**Step 2: Install Dependencies**
```bash
# requirements.txt
openai==1.12.0
python-dotenv==1.0.0
pandas==2.0.0
streamlit==1.30.0
```

```bash
pip install -r requirements.txt
```

**Step 3: Configure API Keys**
```bash
# .env file
OPENAI_API_KEY=your-key-here
```

---

## Slide 26: Building Your First GenAI App

### Hands-on Lab Part 2 - Core Functionality

**Create the Application (app.py):**
```python
import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

# Load environment
load_dotenv()
client = OpenAI()

class BusinessAnalyzer:
    def __init__(self):
        self.client = OpenAI()
    
    def analyze_text(self, business_text, analysis_type):
        """
        Analyze business text using AI
        """
        prompts = {
            "summary": "Summarize this business text in 3 key points:",
            "sentiment": "Analyze the sentiment of this business text:",
            "insights": "Extract key business insights from this text:",
            "recommendations": "Provide actionable recommendations based on:"
        }
        
        prompt = f"{prompts[analysis_type]}\n\n{business_text}"
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a business analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def analyze_data(self, data_file):
        """
        Analyze business data from CSV
        """
        # Load data
        df = pd.read_csv(data_file)
        
        # Generate summary
        summary = df.describe().to_string()
        
        # Ask AI for insights
        prompt = f"Analyze this business data summary:\n\n{summary}"
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    analyzer = BusinessAnalyzer()
    
    text = """
    Our Q4 sales exceeded expectations with a 25% increase.
    Customer satisfaction scores improved significantly.
    However, supply chain challenges remain a concern.
    """
    
    result = analyzer.analyze_text(text, "insights")
    print(result)
```

---

## Slide 27: Building Your First GenAI App

### Hands-on Lab Part 3 - Web Interface

**Create Streamlit UI:**
```python
import streamlit as st
from app import BusinessAnalyzer

st.set_page_config(page_title="AI Business Analyzer", page_icon="📊")

st.title("📊 AI-Powered Business Analyzer")
st.markdown("Analyze business text and data using GPT-4")

# Initialize analyzer
analyzer = BusinessAnalyzer()

# Sidebar
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["summary", "sentiment", "insights", "recommendations"]
)

# Main content
tab1, tab2 = st.tabs(["Text Analysis", "Data Analysis"])

with tab1:
    st.header("Analyze Business Text")
    
    user_input = st.text_area(
        "Enter business text to analyze:",
        height=200,
        placeholder="Paste your business text here..."
    )
    
    if st.button("Analyze Text", type="primary"):
        if user_input:
            with st.spinner("Analyzing..."):
                result = analyzer.analyze_text(user_input, analysis_type)
                st.success("Analysis Complete!")
                st.markdown("### Results:")
                st.write(result)
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Analyze Business Data")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        if st.button("Analyze Data", type="primary"):
            with st.spinner("Analyzing data..."):
                result = analyzer.analyze_data(uploaded_file)
                st.success("Analysis Complete!")
                st.markdown("### Insights:")
                st.write(result)

# Footer
st.markdown("---")
st.markdown("Built with OpenAI GPT-4 and Streamlit")
```

**Run the App:**
```bash
streamlit run app.py
```

---

## Slide 28: Testing and Debugging

### Ensuring Quality AI Applications

**Unit Testing:**
```python
import unittest
from app import BusinessAnalyzer

class TestBusinessAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = BusinessAnalyzer()
    
    def test_analyze_text_summary(self):
        text = "Sales increased by 20% in Q4."
        result = self.analyzer.analyze_text(text, "summary")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
    
    def test_analyze_text_sentiment(self):
        text = "We are very excited about the new product launch!"
        result = self.analyzer.analyze_text(text, "sentiment")
        self.assertIn("positive", result.lower())
    
    def test_invalid_analysis_type(self):
        text = "Test text"
        with self.assertRaises(KeyError):
            self.analyzer.analyze_text(text, "invalid_type")

if __name__ == "__main__":
    unittest.main()
```

**Logging:**
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def analyze_with_logging(text, analysis_type):
    logger.info(f"Starting analysis: {analysis_type}")
    logger.debug(f"Input length: {len(text)} characters")
    
    try:
        result = analyzer.analyze_text(text, analysis_type)
        logger.info("Analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
```

---

## Slide 29: Deployment Considerations

### Taking Your App to Production

**Docker Containerization:**
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

```bash
# Build and run
docker build -t genai-business-app .
docker run -p 8501:8501 --env-file .env genai-business-app
```

**Deployment Options:**

1. **Streamlit Cloud** (Easiest)
   ```bash
   # Push to GitHub
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   
   # Deploy at streamlit.io/cloud
   ```

2. **AWS/Azure/GCP**
   - Use container services (ECS, Azure Container Apps, Cloud Run)
   - Set environment variables in cloud console
   - Configure auto-scaling

3. **Heroku**
   ```bash
   # Procfile
   web: streamlit run app.py --server.port=$PORT
   ```

**Security Checklist:**
- ✅ Never commit API keys to Git
- ✅ Use environment variables
- ✅ Implement rate limiting
- ✅ Add user authentication
- ✅ Validate all inputs
- ✅ Monitor usage and costs

---

## Slide 30: Real-World Applications

### GenAI in Business - Use Cases

**1. Customer Service Automation**
```python
# AI-powered chatbot
def customer_service_bot(customer_query):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": 
             "You are a helpful customer service representative."},
            {"role": "user", "content": customer_query}
        ]
    )
    return response.choices[0].message.content
```

**2. Content Generation**
- Marketing copy
- Product descriptions
- Email templates
- Social media posts

**3. Data Analysis & Insights**
- Sales trend analysis
- Customer feedback analysis
- Market research summarization

**4. Document Processing**
- Contract analysis
- Invoice extraction
- Report summarization

**5. Personalization**
- Product recommendations
- Email personalization
- Dynamic pricing suggestions

**Business Impact:**
- **Cost Reduction**: 30-50% in customer service
- **Speed**: 10x faster content creation
- **Accuracy**: 95%+ in data extraction
- **Scale**: Handle 1000s of requests simultaneously

---

**End of Batch 3 (Slides 21-30)**

*Continue to Batch 4 for Best Practices, Ethics, and Assignment*
