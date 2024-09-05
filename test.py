from langchain import chat_models
from langchain.prompts import PromptTemplate

# Initialize the Google Gemini model
def initialize_gemini_model():
    # Instantiate the Google Gemini model with the specified version
    model = chat_models.ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    return model

# Method to reframe a long string
def reframe_long_string(long_string: str) -> str:
    try:
        # Initialize the model
        model = initialize_gemini_model()
        
        # Define the prompt template for reframing
        prompt_template = PromptTemplate(
            template="Reframe the following text to make it more concise and clear:\n\n{input_text}",
            input_variables=["input_text"]
        )
        
        # Create the prompt
        prompt = prompt_template.format(input_text=long_string)
        
        # Generate the reframed text using the model
        response = model.generate(prompt)
        
        # Extract and return the reframed text from the response
        reframed_text = response['text']
        return reframed_text
    except Exception as e:
        # Log the error
        print(f"Error during text reframing: {e}")
        return "An error occurred while reframing the text."

# Example usage
if __name__ == "__main__":
    long_string = """
    Here is a long and complex string that needs to be reframed. It includes multiple sentences and might 
    be difficult to read or understand at a glance. The goal is to make this text more concise and to the 
    point while retaining its original meaning.
    """
    
    reframed_string = reframe_long_string(long_string)
    print("Reframed String:")
    print(reframed_string)
