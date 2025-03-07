# Text to SQL Query Generator

A web application that converts natural language questions into SQL queries using large language models (LLMs).

## Description

This project provides a Flask-based web interface that allows users to:

1. Upload data files (CSV, Excel, SQL) containing database schemas and data
2. Ask questions in natural language about the data
3. Get SQL queries automatically generated by a large language model
4. Execute the generated queries against the uploaded data
5. View the query results

The application uses Ollama to run local LLMs (primarily CodeLlama) for processing natural language questions and converting them to SQL queries.

## Features

- File upload support for CSV, Excel, and SQL files
- Automatic schema extraction from uploaded files
- Natural language to SQL query conversion
- Query execution and result visualization
- Interactive web interface

## Prerequisites

- Python 3.x
- [Ollama](https://ollama.ai/) installed locally with the CodeLlama model
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository
```bash
git clone https://github.com/Shayan5422/Text_to_sql.git
cd LLM_to_sql
```

2. Install the required dependencies
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is installed and the CodeLlama model is available
```bash
ollama pull codellama:latest
```

## Usage

1. Start the Flask application
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload a data file (CSV, Excel, or SQL)

4. Ask questions about your data in natural language

5. View the generated SQL queries and their results

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates for the web interface
- `uploads/`: Directory for storing uploaded files
- `modified/`: Directory for processed data files
- `prompt.md`: Template for LLM prompts
- `metadata.sql`: Example SQL schema data

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Uses Ollama for running local LLM models
- Built with Flask, Pandas, and SQLite 
