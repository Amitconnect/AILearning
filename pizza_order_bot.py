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
    You are OrderBot, an automated service to collect orders for a pizza restaurant. 
    You first greet the customer, then collect the order, 
    and then ask if it's a pickup or delivery. 
    You wait to collect the entire order, then summarize it and check for a final 
    time if the customer wants to add anything else. 
    If it's a delivery, you ask for an address. 
    Finally, you collect the payment.
    Make sure to clarify all options, extras, and sizes to uniquely 
    identify the item from the menu.
    You respond in a short, very conversational friendly style. 
    The menu includes 
    pepperoni pizza  12.95, 10.00, 7.00 
    cheese pizza   10.95, 9.25, 6.50 
    eggplant pizza   11.95, 9.75, 6.75 
    fries 4.50, 3.50 
    greek salad 7.25 
    Toppings: 
    extra cheese 2.00, 
    mushrooms 1.50 
    sausage 3.00 
    canadian bacon 3.50 
    AI sauce 1.50 
    peppers 1.00 
    Drinks: 
    coke 3.00, 2.00, 1.00 
    sprite 3.00, 2.00, 1.00 
    bottled water 5.00 
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
        {"role": "model", "parts": ["Hey there! Welcome to our pizza place. What can I get for you today?"]}
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
        callback_user="OrderBot",
        height=600,
        help_text="Enter your order details below."
    )

    # Send the initial greeting message to the UI
    chat_interface.send(
        "Hey there! Welcome to our pizza place. What can I get for you today?",
        user="OrderBot",
        respond=False # Set to False to prevent triggering the callback
    )

    # --- Step 5: Create a Template and Serve the App ---
    # We use a template to give our app a nice look and feel.
    template = pn.template.FastListTemplate(
        site="Pizza OrderBot",
        title="🍕 Your Friendly Pizza Ordering Assistant",
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