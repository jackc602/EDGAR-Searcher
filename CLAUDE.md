# ♊ GEMINI Project Guidelines

This document outlines the core principles and guidelines for developing this project. Adhering to these standards ensures consistency, maintainability, and correctness.

This project will extract SEC filing data of a public company, then embed all the text found into a vector database, then connect this data to an LLM to be run locally so a user can ask questions about the filings of the company they have chosen.
---

## 🏛️ Core Principles

### 1. Tech Stack
The entire technology stack for this project must be **Python**. All new code, scripts, and tooling should be written in Python unless an exception is explicitly granted. The following packages should be used for their corresponding purposes.
* requests. Purpose: Fetching data from the SEC EDGAR API.
* Ollama. Purpose: Retreiving the LLMs and embedding models to use.
* Streamlit. Purpose: Building a barebones frontend of the application.
* Chroma. Purpose: Building a barebones vector database to store and retreive embeddings from.

### 2. Dependency Management
All external Python libraries and dependencies **must be added to the `requirements.txt` file**. This is crucial for ensuring a reproducible environment for all users.
* **Before adding:** Verify the library is necessary and well-maintained.
* **After adding:** Run `pip install -r requirements.txt` to confirm installation.

### 3. Code Style
All Python code must strictly conform to the **PEP 8 style guidelines**. Use tools like `flake8` or `black` to automatically check and format your code before committing. Consistent style makes the codebase more readable and easier to maintain.

### 4. Correctness Over All
The absolute highest priority is **correctness**.
* Code must produce the correct, expected results.
* It is better to have a simple, slow, but correct implementation than a complex, fast, but buggy one.
* Thoroughly test your changes to validate their correctness.

### 5. Cautious Contribution
When in doubt, **ask clarifying questions**.
* If a requirement is ambiguous, do not make assumptions. Ask for clarification first.
* When making changes, **always err on the side of doing too little rather than too much**. It is easier to add new functionality later than to remove a poorly-considered one.

### 6. Local-First Development
This application is designed to be run **entirely locally on a user's machine**.
* Be mindful of **limited hardware constraints**. Avoid solutions that require excessive RAM, CPU power, or disk space.
* Do not assume the user has a powerful GPU or a high-speed internet connection.
* Optimize for efficiency and a small footprint where possible, but never at the expense of correctness (see Principle #4).
