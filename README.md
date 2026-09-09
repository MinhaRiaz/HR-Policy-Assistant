# 📘 HR Policy Assistant

An AI-powered **HR Policy Assistant** built with **Retrieval-Augmented Generation (RAG)**.

Upload an HR Policy PDF, ask a question, and the application retrieves the most relevant parts of the document before asking **Groq's `openai/gpt-oss-120b`** model to answer.

The assistant is designed to **stay grounded in the uploaded PDF**. If the requested information is not found, it returns:

> Sorry, this information is not found in the uploaded HR policy.

## 🚀 Live Demo

After deployment, your Streamlit Cloud URL will look similar to:

`https://hr-policy-assistant-app.streamlit.app/`

---

## ✨ Features

- Upload an HR Policy PDF
- Extract PDF text using **PyMuPDF**
- Split the policy into overlapping chunks
- Generate semantic embeddings using **Sentence Transformers**
- Store embeddings in a **FAISS** vector index
- Retrieve the most relevant policy sections
- Generate answers using **Groq `openai/gpt-oss-120b`**
- Display RAG source pages and retrieved text
- Refuse to answer when the question is not supported by the uploaded policy
- No database required
- No Google Colab required
- Designed for deployment on **Streamlit Community Cloud**

---

## 🧠 How the RAG Pipeline Works

```text
HR Policy PDF
      ↓
PyMuPDF
      ↓
Extract text + page numbers
      ↓
Chunking
      ↓
Sentence Transformer embeddings
      ↓
FAISS vector index
      ↓
User question
      ↓
Question embedding
      ↓
Similarity search
      ↓
Top relevant policy chunks
      ↓
Groq GPT-OSS 120B
      ↓
Grounded answer + source pages
```

### Why this is RAG

The LLM is not asked to answer from general knowledge.

Instead:

1. The PDF is converted into text.
2. The text is divided into chunks.
3. Each chunk becomes a vector embedding.
4. FAISS searches for chunks semantically related to the user's question.
5. Only the retrieved chunks are supplied to the LLM.
6. The LLM generates an answer based on those chunks.
7. The application shows the source pages used by retrieval.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Streamlit | Web interface |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| Groq | LLM inference |
| `openai/gpt-oss-120b` | Answer generation |
| Python | Application logic |

---

## 📁 Project Structure

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔐 API Key

The application reads:

```text
GROQ_API_KEY
```

Never put the actual API key inside `app.py` or commit it to GitHub.

For Streamlit Community Cloud, add it through:

**App → Settings → Secrets**

Use:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

---

## 💻 Run Locally

If you ever want to run the application locally, install the dependencies:

```bash
pip install -r requirements.txt
```

Then configure your Groq API key and start Streamlit:

```bash
streamlit run app.py
```

This project does not require Google Colab, VS Code, or a terminal for GitHub → Streamlit Cloud deployment.

---

# ☁️ Deploy Without VS Code or Terminal

You can create and deploy the entire project through your browser using **GitHub + Streamlit Community Cloud**.

## Step 1 — Create a GitHub repository

1. Go to GitHub.
2. Sign in.
3. Click **+ → New repository**.
4. Repository name:

```text
hr-policy-assistant
```

5. Choose **Public** if you want the easiest setup.
6. You can add a README or leave the repository empty.
7. Click **Create repository**.

---

## Step 2 — Upload the four files

Inside your new GitHub repository:

1. Click **Add file**.
2. Select **Upload files**.
3. Upload:

```text
app.py
requirements.txt
README.md
.gitignore
```

4. Scroll down.
5. Click **Commit changes**.

Your repository should look like:

```text
hr-policy-assistant
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Step 3 — Create a Groq API key

Create a Groq API key from the Groq Console.

Keep the key private.

Do **not** paste the key into `app.py`.

---

## Step 4 — Open Streamlit Community Cloud

Go to:

https://share.streamlit.io/

Sign in with GitHub and authorize Streamlit to access your repository.

---

## Step 5 — Create the Streamlit app

In Streamlit Community Cloud:

1. Click **Create app**.
2. Choose **Yup, I have an app**.
3. Select your GitHub repository:

```text
hr-policy-assistant
```

4. Branch:

```text
main
```

5. Main file:

```text
app.py
```

6. Optionally choose an app URL.
7. Open **Advanced settings**.

---

## Step 6 — Add the Groq secret

In the **Secrets** field, paste:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

Do not add this key to GitHub.

Then save the settings and deploy.

---

## Step 7 — Wait for deployment

Streamlit Cloud will install the packages from:

```text
requirements.txt
```

and start:

```text
app.py
```

The first deployment may take longer because `sentence-transformers` and `faiss-cpu` need to be installed and the embedding model needs to be downloaded.

---

# 🧪 Test Questions

After deployment, upload an HR policy PDF and try questions such as:

### Questions that SHOULD be answered

```text
How many annual leave days are employees entitled to?
```

```text
What is the maternity leave policy?
```

```text
How many hours can an employee work per week?
```

```text
What is the company's policy on remote work?
```

```text
What happens if an employee arrives late?
```

### Questions that SHOULD return "not found"

If your PDF is only about leave policies, try:

```text
What is the company's stock price?
```

```text
Who is the president of Pakistan?
```

```text
What is the weather today?
```

The assistant should not use general knowledge to answer these.

---

# ⚠️ Important PDF Limitation

This version works best with **text-based PDFs**.

If the HR policy is a scanned PDF where every page is just an image, PyMuPDF may not extract useful text.

For scanned documents, OCR would need to be added as a separate processing step.

---

# 🔒 Security Notes

- Never commit `GROQ_API_KEY`.
- Use Streamlit Secrets for deployment.
- Do not upload confidential HR documents to a public repository.
- The uploaded PDF is processed in the running Streamlit session; it is not stored in the GitHub repository by this application.
- HR policies can contain sensitive information, so use appropriate organizational privacy controls before using this application with real employee data.

---

## 📌 RAG Source Design

Every retrieved chunk keeps its original PDF page number.

The interface therefore shows:

```text
Source 1 — Page 4
Source 2 — Page 7
Source 3 — Page 8
```

and the corresponding retrieved text.

This makes the answer easier to verify against the original HR policy.

---

## 📄 License

This project is provided for educational and portfolio use.
