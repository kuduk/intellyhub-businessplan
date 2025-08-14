import os
import google.generativeai as genai

# Best Practice: Configura la chiave API tramite una variabile d'ambiente.
# Eseguire nel terminale (PowerShell): $env:GOOGLE_API_KEY="IL_TUA_CHIAVE_API"
# Oppure (bash/zsh): export GOOGLE_API_KEY="IL_TUA_CHIAVE_API"
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("Errore: La variabile d'ambiente GOOGLE_API_KEY non è stata impostata.")
    print("Per favore, imposta la tua chiave API di Google AI Studio.")
    exit()

def translate_text(text_to_translate):
    """
    Translates text to Chinese using the Gemini API, ensuring LaTeX formatting is preserved.
    """
    print("   Sending content to Gemini for translation...")
    try:
        # Inizializza il modello Gemini. 'gemini-1.5-flash' è veloce ed efficiente.
        # Per una qualità potenzialmente superiore, puoi usare 'gemini-1.5-pro-latest'.
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Le istruzioni (system prompt) vengono unite alla richiesta dell'utente.
        # Gemini gestisce questo formato in modo molto efficace.
        prompt = f"""You are an expert translator. Translate the user's text to Chinese.
Crucially, you must preserve all LaTeX commands (like \\section, \\textbf, $, $$, etc.) and formatting exactly as they appear in the original text.
Reply only with the translation without any additional text or comments or block codes.
Don't complete latex blocks this is a chunck of the original text.

--- START OF TEXT TO TRANSLATE ---
{text_to_translate}
--- END OF TEXT TO TRANSLATE ---"""

        # NUOVO: Usa model.generate_content per chiamare l'API di Gemini
        response = model.generate_content(prompt)

        # NUOVO: Accedi al testo della risposta direttamente tramite l'attributo .text
        translated_content = response.text
        print("   ...translation received successfully.")
        return translated_content.strip()
    except Exception as e:
        print(f"   An error occurred during translation with Gemini: {e}")
        # Restituisce una stringa vuota in caso di errore per continuare il ciclo principale
        return ""

def main():
    directory = 'ch'
    combined_content = ""
    output_filename = 'ch-businessplan.tex'

    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found. Please create it and place your .tex files inside.")
        return

    # Ottiene una lista ordinata di file .tex per processarli in ordine consistente
    filenames = sorted([f for f in os.listdir(directory) if f.endswith('.tex')])

    if not filenames:
        print(f"No .tex files found in the '{directory}' directory.")
        return

    print(f"Found {len(filenames)} LaTeX files to translate.")

    # Processa ogni file LaTeX nella directory
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        print(f"\n--- Processing '{filename}' ---")
        
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            if not content.strip():
                print("   File is empty, skipping.")
                continue
            
            translated_content = translate_text(content)
            
            # Aggiunge un commento LaTeX come separatore per una migliore leggibilità
            combined_content += f"% ----- Start of translated content from: {filename} -----\n\n"
            combined_content += translated_content
            combined_content += f"\n\n% ----- End of translated content from: {filename} -----\n\n"

    # Scrive il contenuto tradotto e combinato in un nuovo file LaTeX
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(combined_content)
    
    print(f"\n✅ All files have been translated and combined into '{output_filename}'")

if __name__ == "__main__":
    main()