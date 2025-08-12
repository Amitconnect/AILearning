import os
import panel as pn
import google.generativeai as genai
import asyncio

# --- Step 1: Initialize Panel Extension ---
# This is needed to render Panel components.
pn.extension()

# --- Step 2: Configure the Google Gemini Model ---
# This section is the same as the command-line version.

try:
    # Configure API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)

    # The system instruction to guide the chatbot
    system_instruction = """
    You are an insurance agent Bot, an automated insurance agent to provide details about insurance and convince customer to why the insurance is beneficial. 
    You first greet the customer, then understand the requirement of customer step by step. 
    and then provide option as per the customer requirement in details. 
    You wait to collect the entire requirement, then summarize it and share details.
    Ask them for the feedback at the end of coversation.
    """

    # Create the model and start a chat session with initial history
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_instruction,
        generation_config={"temperature": 0},
    )
    
    # The chat object holds the conversation history for the backend
    chat = model.start_chat(history=[
        {"role": "user", "parts": ["Hi there"]},
        {"role": "model", "parts": ["Hey there! I am your insurance advisor bot . How may I help you today?"]}
    ])

    # --- Step 3: Define the Chat Interface Callback ---
    # This async function is called whenever the user sends a message.
    async def get_bot_response(contents: str, user: str, instance: pn.chat.ChatInterface):
        """
        Callback function to get a response from the Gemini model.
        
        Args:
            contents: The user's message.
            user: The name of the user.
            instance: The ChatInterface instance.
        """
        # Send the user's message to the Gemini chat session
        response = chat.send_message(contents)
        
        # Yield each part of the response to stream it to the UI
        # In this case, response.text is not a stream, so we yield the whole text.
        # If the API supported streaming chunks, you could yield each chunk.
        yield response.text

    # --- Step 4: Create the Panel Chat Interface Widget ---
    chat_interface = pn.chat.ChatInterface(
        callback=get_bot_response,
        callback_user="InsuranceAdvisorBot",
        height=600,
        help_text="Enter your Insurance requirement details below."
    )

    # Send the initial greeting message to the UI
    chat_interface.send(
        "Hey there! I am your Insurance advisor. How may i help you?",
        user="InsuranceAdvisorBot",
        respond=False # Set to False to prevent triggering the callback
    )

    # --- Step 5: Create a Template and Serve the App ---
    # We use a template to give our app a nice look and feel.
    template = pn.template.FastListTemplate(
        site="Insurance advisor",
        title="Insurance Advisor Assistant",
        main=[chat_interface],
        header_background="#D22B2B", # A pizza-sauce red color
        sidebar_footer="""
        <div style='text-align: center; color: #A0A0A0;'>
            Powered by <b>Google Gemini</b> & <b>Panel</b>
        </div>
        """,
    )

    template.servable()

except Exception as e:
    # If the API key is missing or something goes wrong, display an error.
    error_card = pn.Card(
        f"🔴 **Application Error:**<br>{e}",
        title="Error",
        styles={'background': '#FFF0F0'}
    )
    error_card.servable()