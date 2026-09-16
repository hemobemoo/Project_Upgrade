Getting a project from someone else can be a bit messy, as their local configurations and environment files won't necessarily work on your machine. Since you are on Windows, here is the cleanest way to reset and set up your environment.

## 1. Cleanup: What to Delete

Before you start, you should remove any folders that contain local machine-specific paths or compiled code.

- **`.venv`**: This is the previous owner's virtual environment. Virtual environments are not portable; they contain hardcoded paths to the original user's Python installation. Delete this so you can create a fresh one.
    
- **`__pycache__`**: These are compiled bytecode files generated when Python runs. They are harmless but unnecessary to keep when moving projects. Python will regenerate them automatically.
    
- **`.vscode` (Optional)**: This contains the other person's editor settings (like their specific Python path). If you want to use your own VS Code settings, you can delete this.
    

**Command to clean up via Terminal:**

PowerShell

```
rmdir /s /q .venv __pycache__
```

---

## 2. Create and Activate a New Virtual Environment

Open your terminal (PowerShell or CMD) in the project directory and run the following:

1. **Create the environment:**
    
    PowerShell
    
    ```
    python -m venv .venv
    ```
    
2. **Activate it:**
    
    - **In PowerShell:**
        
        PowerShell
        
        ```
        .\.venv\Scripts\Activate.ps1
        ```
        
    - **In CMD:**
        
        DOS
        
        ```
        .\.venv\Scripts\activate
        ```
        
    
    _(You should now see `(.venv)` appear in your terminal prompt.)_
    

---

## 3. Install Dependencies

Once the environment is active, use the `requirements.txt` file to install all necessary libraries:

PowerShell

```
pip install -r requirements.txt
```

---

## 4. Configuration Check (`.env`)

You have a `.env` file in your directory. This usually stores sensitive information like API keys or database URLs.

- Open the `.env` file in a text editor.
    
- Check if there are any keys or paths that belong to the previous owner (e.g., `OPENAI_API_KEY=...`). You will likely need to update these with your own credentials for the project to function.
    

---

## 5. Run the Project

Based on your file list, the entry point for the application is likely `app.py`. To run it:

PowerShell

```
python app.py
```


## 6. Initialize the Streamlit web application


Paste this in the cmd terminal:

```
streamlit run app.py
```

**Note:** If this is a web application (like Flask or Streamlit), the terminal will provide a URL (usually `http://127.0.0.1:5000` or similar) that you can open in your browser.

---

## Note after running the app
You may encounter a proxy 'problem' when you upload the data samples, which is version problem, here is how to solve:

Run this after activating the .venv, before running the app 

Command for PowerShell or CMD

```
python -m pip install httpx==0.27.2
```
