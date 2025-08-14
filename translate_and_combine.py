import os
from openai import OpenAI

# Best Practice: The client automatically finds the API key if you set it as an
# environment variable named OPENAI_API_KEY. Avoid hardcoding keys in your script.
# If you must, you can pass the key directly: client = OpenAI(api_key="sk-proj-...")

# $env:OPENAI_API_KEY="sk-proj-..."
client = OpenAI()

def translate_text(text_to_translate):
    """
    Translates text to Chinese using the OpenAI API, ensuring LaTeX formatting is preserved.
    """
    print("   Sending content to OpenAI for translation...")
    try:
        # NEW: Use the client object to create a chat completion
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                # It's good practice to give instructions to the model in the "system" role.
                {"role": "system", "content": "You are an expert translator. Translate the user's text to Chinese. Crucially, you must preserve all LaTeX commands (like \\section, \\textbf, $, $$, etc.) and formatting exactly as they appear in the original text. Reply only with the translation without any additional text or comments or block codes. Don't complete latex blocks this is a chunck of the original text."},
                {"role": "user", "content": text_to_translate}
            ]
        )
        # NEW: Access the response content using attributes (. instead of ['...'])
        translated_content = response.choices[0].message.content
        print("   ...translation received successfully.")
        return translated_content.strip()
    except Exception as e:
        print(f"   An error occurred during translation: {e}")
        # Return an empty string on error so the main loop can continue
        return ""

def main():
    directory = 'ch'
    combined_content = ""
    output_filename = 'ch-businessplan.tex'

    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found. Please create it and place your .tex files inside.")
        return

    # Get a sorted list of .tex files to process them in a consistent order
    filenames = sorted([f for f in os.listdir(directory) if f.endswith('.tex')])

    if not filenames:
        print(f"No .tex files found in the '{directory}' directory.")
        return

    print(f"Found {len(filenames)} LaTeX files to translate.")

    # Process each LaTeX file in the directory
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        print(f"\n--- Processing '{filename}' ---")
        
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            if not content.strip():
                print("   File is empty, skipping.")
                continue
            
            translated_content = translate_text(content)
            
            # Add a LaTeX comment as a separator for better readability in the final file
            combined_content += f"% ----- Start of translated content from: {filename} -----\n\n"
            combined_content += translated_content
            combined_content += f"\n\n% ----- End of translated content from: {filename} -----\n\n"

    # Write the combined translated content to a new LaTeX file
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(combined_content)
    
    print(f"\n✅ All files have been translated and combined into '{output_filename}'")

if __name__ == "__main__":
    main()