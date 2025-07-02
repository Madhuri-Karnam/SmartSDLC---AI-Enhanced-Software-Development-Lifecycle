from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import fitz # PyMuPDF
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline # Use langchain_community for HuggingFacePipeline
import json # Import json for debugging chat history
import re # Import re for improved response cleaning

MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"

app = FastAPI(title="SmartSDLC Backend")

# Configure CORS to allow requests from your local HTML file
origins = [
    "http://localhost",
    "http://localhost:8000", # If you serve index.html with a simple server
    "null", # Allows requests from file:/// protocol (when opening index.html directly)
    "*" # In a real production app, restrict this to your specific frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global AI Model Loading (Load once at startup) ---
llm_pipeline = None
chatbot_chain = None

@app.on_event("startup")
async def load_model():
    global llm_pipeline, chatbot_chain
    try:
        print(f"Loading model: {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGING_FACE_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            load_in_8bit=True, # Load in 8-bit for efficiency on Colab T4 GPUs
            token=HUGGING_FACE_TOKEN
        )
        model.eval() # Set model to evaluation mode

        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512, # Increased for potentially longer, more complete answers
            do_sample=True,
            temperature=0.7, # Adjusted for more natural chat output
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id, # Important for generation
        )
        print("Model loaded successfully!")

        # Initialize LangChain for Chatbot
        llm_langchain = HuggingFacePipeline(pipeline=llm_pipeline)
        # Refined prompt template for chatbot
        prompt_template = PromptTemplate(
            input_variables=["history", "question"],
            template="""
            If you do not know the answer, state that you do not know.

            --- Chat History ---
            {history}
            --- End Chat History ---

            Human: {question}
            AI Assistant:""" # Changed from "AI:" to be more distinct
        )
        chatbot_chain = LLMChain(llm=llm_langchain, prompt=prompt_template)
        print("Chatbot chain initialized!")

    except Exception as e:
        print(f"Error loading model or initializing LangChain: {e}")
        raise RuntimeError(f"Failed to load AI model: {e}")


# --- Utility Function for Text Extraction from PDF ---
def extract_text_from_pdf(pdf_file: UploadFile):
    text_content = ""
    print(f"Received file: {pdf_file.filename}, Content-Type: {pdf_file.content_type}, Size: {pdf_file.size} bytes")

    if pdf_file.size == 0:
        raise ValueError("Uploaded PDF file is empty.")

    try:
        # Read the file content once
        file_bytes = pdf_file.file.read()
        if not file_bytes:
            raise ValueError("File stream read returned no bytes.")

        # Try to open the PDF stream
        try:
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        except fitz.FileDataError as e:
            raise ValueError(f"File is not a valid PDF or is corrupted: {e}")
        except Exception as e:
            raise ValueError(f"Failed to open PDF stream: {e}")

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text_content += page.get_text()
        print(f"Successfully extracted {len(text_content)} characters from PDF.")
        return text_content
    except Exception as e:
        # Catch any other unexpected errors during processing
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")

# --- AI Scenarios as FastAPI Endpoints ---

@app.post("/classify-requirements")
async def classify_requirements(pdf_file: UploadFile = File(...)):
    if not llm_pipeline:
        raise HTTPException(status_code=503, detail="AI model not loaded yet. Please wait or restart the server.")

    try:
        raw_text = extract_text_from_pdf(pdf_file)
        print(f"\n--- PDF Raw Text Extracted ---\n{raw_text[:500]}...\n------------------------------\n") # Debug print
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions from extract_text_from_pdf
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"PDF Processing Error: {e}") # Bad request for invalid PDF
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during PDF extraction: {e}")


    classified_data = {
        "Requirements": [],
        "Design": [],
        "Development": [],
        "Testing": [],
        "Deployment": [],
        "Other": [] # Added an "Other" category for unclassified
    }

    # Split by common sentence delimiters, then filter empty strings
    sentences = [s.strip() for s in raw_text.replace('\n', ' ').split('.') if s.strip()]
    if not sentences:
        print("No sentences found in extracted text.")
        # If no sentences found but raw_text exists, add it to 'Other'
        if raw_text.strip():
            classified_data["Other"].append(f"Could not parse sentences, raw content: {raw_text.strip()[:200]}...")
        return JSONResponse(content={"classified_data": classified_data})

    for i, sentence in enumerate(sentences):
        # Even more strict prompt for classification
        prompt = (
            f"Classify the following sentence into ONE of these specific SDLC phases: "
            f"Requirements, Design, Development, Testing, Deployment. "
            f"If it doesn't fit any, classify as Other.\n"
            f"Sentence: \"{sentence}\"\n"
            f"Output ONLY the chosen phase name (e.g., Requirements, Design, Development, Testing, Deployment, Other). Do not include any other words or punctuation."
        )
        print(f"\n--- Sentence {i+1} ---")
        print(f"Sentence: \"{sentence}\"")
        print(f"Prompt sent to LLM:\n{prompt}") # Debug print

        try:
            # Generate a response, ensuring it's short
            llm_response_raw = llm_pipeline(prompt, max_new_tokens=15, do_sample=False, temperature=0.1)[0]['generated_text']
            print(f"LLM Raw Response: \"{llm_response_raw}\"") # Debug print

            # Attempt to clean the response by finding where the AI's actual answer starts
            # This is a common issue: LLMs repeating the prompt
            split_key = "Output ONLY the chosen phase name (e.g., Requirements, Design, Development, Testing, Deployment, Other). Do not include any other words or punctuation."
            if split_key in llm_response_raw:
                clean_response = llm_response_raw.split(split_key)[-1].strip()
            else:
                clean_response = llm_response_raw.strip() # Fallback if split key not found

            # Remove any trailing periods or non-alphanumeric characters,
            # but keep spaces if they are part of a phase name (though our phases are single words)
            clean_response = re.sub(r'[^a-zA-Z\s]', '', clean_response).strip()
            clean_response_lower = clean_response.lower()

            print(f"Cleaned LLM Response (parsed): \"{clean_response}\"") # Debug print

            # Map the cleaned response to the categories
            assigned = False
            if "requirements" in clean_response_lower:
                classified_data["Requirements"].append(f"As a user/stakeholder, the system must: {sentence.lower()}")
                assigned = True
            elif "design" in clean_response_lower:
                classified_data["Design"].append(f"Design aspect: {sentence.lower()}")
                assigned = True
            elif "development" in clean_response_lower:
                classified_data["Development"].append(f"Development task: {sentence.lower()}")
                assigned = True
            elif "testing" in clean_response_lower:
                classified_data["Testing"].append(f"Test case/strategy: {sentence.lower()}")
                assigned = True
            elif "deployment" in clean_response_lower:
                classified_data["Deployment"].append(f"Deployment step: {sentence.lower()}")
                assigned = True
            elif "other" in clean_response_lower: # Catch "Other" explicitly
                classified_data["Other"].append(f"General requirement: {sentence.lower()}")
                assigned = True

            if not assigned:
                # If still not assigned, force it into "Requirements" as a last resort
                classified_data["Requirements"].append(f"As a general requirement, the system: {sentence.lower()}")
                print(f"Failed to classify specifically. Assigned to Requirements (Fallback).")

        except Exception as e:
            print(f"Error processing sentence '{sentence}': {e}")
            classified_data["Other"].append(f"Could not classify (error): {sentence.lower()}") # Fallback for errors

    print(f"\n--- Final Classified Data ---\n{json.dumps(classified_data, indent=2)}\n------------------------------\n") # Debug print
    return JSONResponse(content={"classified_data": classified_data})

@app.post("/generate-code")
async def generate_code(prompt: str = Form(...)):
    if not llm_pipeline:
        raise HTTPException(status_code=503, detail="AI model not loaded yet. Please wait or restart the server.")
    try:
        # **SIMPLIFIED PROMPT FOR CODE GENERATION - Direct and Minimal**
        full_prompt = (
            f"Generate a complete Python program or function for the following request. "
            f"Provide ONLY the functional code. Do NOT include any test cases, explanations, or extra text.\n\n"
            f"Request: {prompt}\n\n"
            f"```python\n" # Start the code block
        )
        print(f"\n--- Code Gen Prompt ---\n{full_prompt}\n-----------------------\n") # Debug print

        generated_text = llm_pipeline(full_prompt, max_new_tokens=1024, do_sample=True, temperature=0.5)[0]['generated_text']
        print(f"\n--- Code Gen Raw Response ---\n{generated_text}\n---------------------------\n") # Debug print

        # **SIMPLIFIED POST-PROCESSING FOR CODE GENERATION*
        code = ""
        # 1. Try to find the Python code block explicitly
        code_match = re.search(r"```python\s*(.*?)\s*```", generated_text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            print(f"--- Code Extracted from ````python```` block ---\n{code}\n----------------------------------")
        else:
            # 2. Fallback: If no explicit markdown block, try to clean the raw text.
            # This attempts to remove the prompt itself and common conversational filler from the start.
            print(f"Warning: No ````python```` block found. Attempting direct text cleanup.")

            # Remove the exact prompt from the beginning of the generated text
            code_candidate = generated_text.replace(full_prompt, '', 1).strip()

            # Remove common LLM conversational intros/outros and the "block." artifact
            patterns_to_strip = [
                re.compile(r"^(?:Here\'s the Python code:|Below is the Python function:|I can help you with that\. Here\'s the code:|block\.)\s*", re.IGNORECASE | re.DOTALL),
                re.compile(r"\s*(?:Please let me know if you need any further assistance\.|Let me know if you want to test it\.|Feel free to modify it as needed\.|This function calculates the factorial of a number\.).*$", re.IGNORECASE | re.DOTALL),
                re.compile(r"^\s*Request:.*$", re.MULTILINE), # remove repeated request if any
                re.compile(r"^\s*block\.\s*$", re.MULTILINE), # specifically remove 'block.' if isolated
            ]

            for pattern in patterns_to_strip:
                code_candidate = pattern.sub('', code_candidate).strip()

            # Remove any remaining ``` or ` characters
            code_candidate = code_candidate.replace('```', '').replace('`', '').strip()

            code = code_candidate.strip()
            print(f"--- Code Extracted after cleanup ---\n{code}\n----------------------------------")

        if not code or code.lower().startswith("# could not generate functional code"):
            code = "# Could not generate functional code based on your prompt.\n" \
                   "# Please try a different request, be more specific, or simplify the request.\n" \
                   "# Review Colab output's 'Code Gen Raw Response' for AI's complete answer to diagnose."

        return JSONResponse(content={"code": code})
    except Exception as e:
        print(f"Error generating code: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating code: {e}")

@app.post("/fix-bug")
async def fix_bug(code_snippet: str = Form(...)):
    if not llm_pipeline:
        raise HTTPException(status_code=503, detail="AI model not loaded yet. Please wait or restart the server.")
    try:
        prompt = f"Analyze the following code snippet for syntactical and logical errors and provide an optimized and corrected version. Clearly show the corrected code.\n\nBuggy Code:\n```\n{code_snippet}\n```\n\nCorrected Code:\n```\n"
        generated_text = llm_pipeline(prompt, max_new_tokens=1024, do_sample=True)[0]['generated_text']
        fixed_code = generated_text.split("```")[-2].strip() # Assuming the LLM wraps fixed code in ```
        return JSONResponse(content={"original_code": code_snippet, "fixed_code": fixed_code})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fixing bug: {e}")

@app.post("/generate-test-cases")
async def generate_test_cases(code_or_req: str = Form(...)):
    if not llm_pipeline:
        raise HTTPException(status_code=503, detail="AI model not loaded yet. Please wait or restart the server.")
    try:
        # Explicitly ask for test cases using specific frameworks
        prompt = (
            f"Generate suitable test cases in Python using either the unittest or pytest framework "
            f"for the following code or requirement. Do NOT generate functional code.\n\n"
            f"Input:\n```\n{code_or_req}\n```\n\n"
            f"Test Cases (Python):\n```python\n"
        )
        generated_text = llm_pipeline(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)[0]['generated_text']
        test_cases = generated_text.split("```python")[-1].split("```")[0].strip()

        # Fallback to general text if code block extraction fails or is empty
        if not test_cases and "```python" not in generated_text:
            test_cases = generated_text.replace(prompt, '').strip() # Attempt to strip prompt if no code block

        return JSONResponse(content={"test_cases": test_cases})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating test cases: {e}")

@app.post("/summarize-code")
async def summarize_code(code_snippet: str = Form(...)):
    if not llm_pipeline:
        raise HTTPException(status_code=503, detail="AI model not loaded yet. Please wait or restart the server.")
    try:
        prompt = f"Provide a human-readable explanation and summary for the following code snippet, including its function and use cases:\n\n```\n{code_snippet}\n```\n\nSummary:"
        summary = llm_pipeline(prompt, max_new_tokens=512, do_sample=True)[0]['generated_text'].strip()
        # Try to get the part after "Summary:" if the LLM repeats the prompt
        summary_parts = summary.split("Summary:")
        if len(summary_parts) > 1:
            summary = summary_parts[-1].strip()
        return JSONResponse(content={"summary": summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing code: {e}")

@app.post("/chatbot")
async def chatbot_response(user_message: str = Form(...), chat_history: str = Form("[]")):
    if not chatbot_chain:
        raise HTTPException(status_code=503, detail="Chatbot not initialized yet. Please wait or restart the server.")
    try:
        # chat_history will be a JSON string, parse it
        history_list = json.loads(chat_history)
        # Prepare history for the prompt
        formatted_history_parts = []
        for msg in history_list:
            if msg['role'] == 'user':
                formatted_history_parts.append(f"Human: {msg['text']}")
            elif msg['role'] == 'AI':
                formatted_history_parts.append(f"AI Assistant: {msg['text']}")
        formatted_history = "\n".join(formatted_history_parts)

        # LangChain's LLMChain takes prompt input variables
        response = chatbot_chain.run(history=formatted_history, question=user_message)

        # --- MORE ROBUST POST-PROCESSING ---
        # Define patterns to remove. Order matters: more specific first.
        patterns_to_remove = [
            re.compile(r'AI Assistant:\s*'),  # Remove direct AI Assistant prefix
            re.compile(r'Human:\s*.*', re.IGNORECASE), # Remove any human input repeated in AI output
            re.compile(r'--- Chat History ---.*?--- End Chat History ---', re.DOTALL), # Remove chat history block
            re.compile(r'You are an AI assistant for Software Development Lifecycle \(SDLC\).*?concisely\.\s*', re.DOTALL), # Remove instruction block
            re.compile(r'If you do not know the answer, state that you do not know\.\s*'), # Remove the "if you don't know" part
            re.compile(r'\s*AI Assistant:$', re.IGNORECASE) # Remove trailing "AI Assistant:" if any
        ]

        cleaned_response = response.strip()
        for pattern in patterns_to_remove:
            cleaned_response = pattern.sub('', cleaned_response).strip()

        # Final cleanup for any leftover whitespace or special characters
        cleaned_response = cleaned_response.strip().replace('\\n', '\n')
        # Check if the cleaned response is empty, and if so, provide a default fallback
        if not cleaned_response:
             cleaned_response = "I am sorry, I could not generate a clear response for that query."


        return JSONResponse(content={"ai_response": cleaned_response})
    except Exception as e:
        print(f"Error getting chatbot response: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting chatbot response: {e}")
