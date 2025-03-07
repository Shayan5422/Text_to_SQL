import os
import subprocess
import logging
import pandas as pd
import sqlite3
import traceback
import re
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask App Setup
app = Flask(__name__)
app.secret_key = "ollama"
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv', 'sql'}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global LLM memory to store previous conversation/context
llm_memory = []

# ========= LLM AND DATA PROCESSING FUNCTIONS ==========

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_Qwen32_avec_texte(prompt, texte):
    """
    ارسال پیام به مدل Qwen2.5:32b و دریافت پاسخ تولیدی.
    حافظه‌ی قبلی (llm_memory) نیز به پیام نهایی اضافه می‌شود.
    """
    global llm_memory

    # اضافه کردن حافظه (در صورت موجود بودن پیام‌های قبلی)
    memory_text = "\n".join(llm_memory) if llm_memory else ""
    full_prompt = f"{memory_text}\n{prompt}\n\n{texte}".strip()
    
    command = ['ollama', 'run', 'codellama:latest']

    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate(input=full_prompt)
        if process.returncode != 0:
            logger.error(f"Error executing the model: {error}")
            return None
        result = output.strip()
        # ذخیره پیام ارسالی و پاسخ دریافتی در حافظه
        llm_memory.append(f"PROMPT:\n{full_prompt}")
        llm_memory.append(f"RESPONSE:\n{result}")
        return result
    except Exception as e:
        logger.error(f"Exception during calling the LLM: {e}")
        return None

def process_sql_file(filepath):
    logger.info("Processing SQL file...")
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            sql_commands = file.read()
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        cursor.executescript(sql_commands)
        conn.commit()
        logger.info("SQL script executed successfully on an in-memory database.")
        return conn
    except Exception as e:
        logger.error(f"Failed to execute SQL commands: {e}")
        return None

def process_csv_file(filepath):
    logger.info("Processing CSV file...")
    try:
        df = pd.read_csv(filepath)
        df.columns = [col.strip() for col in df.columns]
        logger.info("CSV file loaded successfully with cleaned column names.")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return None

def process_excel_file(filepath):
    logger.info("Processing Excel file...")
    try:
        df = pd.read_excel(filepath)
        df.columns = [col.strip() for col in df.columns]
        logger.info("Excel file loaded successfully with cleaned column names.")
        return df
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        return None

def extract_column_names(data):
    if isinstance(data, pd.DataFrame):
        return list(data.columns)
    elif isinstance(data, sqlite3.Connection):
        try:
            cursor = data.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if not tables:
                return []
            first_table = tables[0][0]
            cursor.execute(f"PRAGMA table_info({first_table});")
            columns_info = cursor.fetchall()
            columns = [col[1] for col in columns_info]
            return columns
        except Exception as e:
            logger.error(f"Error extracting columns from SQL database: {e}")
            return []
    else:
        return []

def clean_generated_code(code_str):
    if code_str is None:
        return None
    code_match = re.search(r"```python(.*?)```", code_str, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        lines = code_str.splitlines()
        code_lines = [line for line in lines if re.match(r"^\s*(import|def|class|result|if|else|for|while|#|\w)", line)]
        code = "\n".join(code_lines)
    return code.strip()

def execute_generated_code(code_str, globals_dict):
    try:
        cleaned_code = clean_generated_code(code_str)
        if not cleaned_code:
            logger.error("No valid code found after cleaning.")
            return None, "No valid code after cleaning"
        local_vars = {}
        exec(cleaned_code, globals_dict, local_vars)
        return local_vars.get("result", None), None
    except Exception as e:
        err_msg = traceback.format_exc()
        logger.error("Error during execution of generated code:")
        logger.error(err_msg)
        return None, err_msg

# ========= Flask Routes ==========

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    adapted_query = ""
    generated_code = ""
    output_log = ""
    if request.method == "POST":
        # Check file upload
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            # Process file based on extension
            ext = filename.rsplit(".", 1)[1].lower()
            if ext == "sql":
                data = process_sql_file(file_path)
            elif ext == "csv":
                data = process_csv_file(file_path)
            elif ext in ["xlsx", "xls"]:
                data = process_excel_file(file_path)
            else:
                data = None

            columns = extract_column_names(data)
            output_log += f"Loaded file successfully.<br>Detected columns: {columns}<br><br>"
            
            # Prepare a sample of 5 rows from data (if available)
            data_sample = ""
            if isinstance(data, pd.DataFrame):
                data_sample = data.head(5).to_string()
            elif isinstance(data, sqlite3.Connection):
                try:
                    cursor = data.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    if tables:
                        first_table = tables[0][0]
                        query = f"SELECT * FROM {first_table} LIMIT 5"
                        sample = pd.read_sql_query(query, data)
                        data_sample = sample.to_string()
                except Exception as e:
                    data_sample = "Could not extract sample rows."
            if data_sample:
                output_log += f"Data Sample (first 5 rows):<br><pre>{data_sample}</pre><br><br>"

            # Get query from form
            user_query = request.form.get("user_query", "").strip()
            if not user_query:
                flash("Please enter a natural language query.")
                return redirect(request.url)
            
            # --- First LLM Prompt: Adapt Query (Include Data Sample) ---
            prompt_adaptation = (
                "You are an expert data engineer. The available columns in the dataset are:\n"
                f"{columns}\n\n"
                "Here is a sample of the first 5 rows of the data:\n"
                f"{data_sample}\n\n"
                "Adapt the following query so that it references the appropriate columns and data structures. "
                "Return only the adapted query (no explanation or markdown formatting):"
            )
            output_log += "Sending adaptation prompt...<br>"
            adapted_query = run_Qwen32_avec_texte(prompt_adaptation, user_query)
            if not adapted_query:
                output_log += "Failed to get adapted query from LLM.<br>"
                return render_template("index.html", output_log=output_log)
            output_log += f"Adapted Query:<br>{adapted_query}<br><br>"
            
            # --- Second LLM Prompt: Generate Python Code ---
            prompt_code_generation = (
                "You are an expert Python developer. Convert the following adapted query into "
                "Python code that extracts the relevant data. Assume that a variable named 'data' exists that "
                "contains the dataset (a pandas DataFrame or a sqlite3.Connection). Assign the final extraction "
                "result to a variable called 'result'. Return only the Python code with no additional explanation or markdown formatting:"
            )
            output_log += "Sending code generation prompt...<br>"
            generated_code = run_Qwen32_avec_texte(prompt_code_generation, adapted_query)
            if not generated_code:
                output_log += "Failed to generate code from the LLM.<br>"
                return render_template("index.html", output_log=output_log)
            output_log += f"Generated Python Code:<br><pre>{generated_code}</pre><br>"
            
            # --- Execute Generated Code with Correction Loop ---
            exec_globals = {
                "data": data,
                "pd": pd,
                "sqlite3": sqlite3,
            }
            max_retries = 3
            retries = 0
            error_message = None

            output_log += "Executing generated code...<br>"
            while retries < max_retries:
                result, error_message = execute_generated_code(generated_code, exec_globals)
                # Check if result is valid (برای مثال اگر نتیجه DataFrame خالی نباشد)
                if result is not None and not (hasattr(result, "empty") and result.empty):
                    output_log += "Valid result obtained.<br>"
                    break

                if error_message:
                    output_log += f"Attempt {retries+1}/{max_retries} failed with error:<br><pre>{error_message}</pre><br>"
                else:
                    output_log += f"Attempt {retries+1}/{max_retries}: Result not valid, retrying...<br>"

                # Use LLM to fix the code – include error info in the prompt along with memory
                error_for_prompt = error_message if error_message is not None else "No error message captured."
                fix_prompt = (
                    "The following Python code generated for data extraction produced an error.\n\n"
                    "Original Adapted Query:\n" + adapted_query + "\n\n"
                    "Original Generated Python Code:\n" + generated_code + "\n\n"
                    "Error Message:\n" + error_for_prompt + "\n\n"
                    "Please provide a corrected version of the Python code that fixes the error. "
                    "Return only the corrected Python code with no additional explanation or markdown formatting:"
                )
                output_log += "Sending code correction prompt...<br>"
                fixed_code = run_Qwen32_avec_texte(fix_prompt, "")
                if not fixed_code:
                    output_log += "Failed to obtain a corrected code version from LLM.<br>"
                    break
                output_log += f"Corrected Python Code:<br><pre>{fixed_code}</pre><br>"
                generated_code = fixed_code  # update the code and retry execution
                retries += 1

            output_log += f"Final Result:<br><pre>{result}</pre><br>"
            # Optionally, remove the uploaded file after processing
            os.remove(file_path)
            
            return render_template("index.html", adapted_query=adapted_query,
                                   generated_code=generated_code, result=result,
                                   output_log=output_log)
    return render_template("index.html", output_log=output_log)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
