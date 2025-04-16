import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re
import sys
import torch.nn.functional as F # For softmax
import numpy as np # For basic display

# --- Constants, Explanations, Model Loading, Generation ---
# (Keep all your previous code: MODEL_PATH, TRANSFORMER_EXPLANATIONS,
#  path checks, load_model_and_tokenizer, generate_chat_response)
# --- (Previous code omitted for brevity) ---

MODEL_PATH = "./model/final-new" # Adjust if your model is elsewhere

TRANSFORMER_EXPLANATIONS = {
    "--- Select a component ---": {
        "text": "Please choose a component from the dropdown above to see its explanation.",
        "formulas": []
    },
    "1. Text Preprocessing & Tokenization": {
        "text": """
        **Goal:** Convert raw text into numerical input the model can understand.
        *   **Preprocessing:** Cleaning text (removing extra spaces, etc.).
        *   **Tokenization:** Breaking text into 'tokens' (words/subwords) based on a vocabulary. Each token gets a unique ID. Special tokens (`[PAD]`, `[EOS]`, etc.) might be added.
        """,
        "formulas": [] # No core mathematical formula here, it's more procedural.
    },
    "2. Input Embeddings": {
        "text": """
        **Goal:** Convert token IDs into dense vectors capturing semantic meaning.
        *   Each token ID maps to a learned high-dimensional vector.
        *   **Positional Embeddings:** Vectors representing positions (0, 1, 2, ...) are **added** to token embeddings to provide word order information.
        """,
        "formulas": [
            (r"InputEmbedding = TokenEmbedding + PositionalEmbedding", "Combined input representation.")
        ]
    },
    "3. The Attention Mechanism": {
        "text": """
        **Goal:** Allow the model to weigh the importance of different words when processing a specific word.
        *   **Query (Q), Key (K), Value (V):** Input embeddings are projected into Q, K, V spaces using learned weight matrices (W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup>).
        *   **Scaled Dot-Product Attention:** Calculates attention weights by comparing Queries and Keys, scales them, applies softmax, and multiplies by Values.
        *   **Multi-Head Attention:** Performs attention multiple times in parallel ("heads") with different projections, allowing focus on different relationships. Results are concatenated and projected.
        """,
        "formulas": [
            (r"Q = X W^Q, \quad K = X W^K, \quad V = X W^V", "Projection into Q, K, V spaces (X is input embeddings)."),
            (r"Scores = \frac{QK^T}{\sqrt{d_k}}", "Calculating raw attention scores."),
            (r"Weights = \text{softmax}(Scores)", "Normalizing scores into weights."),
            (r"AttentionOutput = Weights \cdot V", "Weighted sum of Value vectors.")
        ]
    },
    "4. Feed-Forward Neural Networks": {
        "text": """
        **Goal:** Process the context-rich output from the attention layer further for each word independently.
        *   A standard feed-forward network (typically 2 linear layers with a non-linear activation like ReLU or GeLU) is applied to each token's representation individually.
        """,
        "formulas": [
            (r"FFN(x) = \text{Activation}(xW_1 + b_1)W_2 + b_2", "Typical Feed-Forward Network structure (e.g., Activation = ReLU/GeLU). Applied position-wise.")
        ]
    },
    "5. Residual Connections & Layer Normalization": {
        "text": """
        **Goal:** Improve training stability and information flow.
        *   **Residual Connections:** The input `x` to a sub-layer (Attention or FFN) is added to the sub-layer's output. `output = x + Sublayer(x)`.
        *   **Layer Normalization:** Normalizes activations within a layer across features for each token independently. Stabilizes training. Usually applied *before* the residual connection (Pre-LN) or *after* (Post-LN).
        """,
        "formulas": [
            (r"Output = \text{LayerNorm}(x + \text{Sublayer}(x)) \quad \text{(Post-LN variant)}", "Residual connection followed by Layer Normalization."),
            (r"Output = x + \text{Sublayer}(\text{LayerNorm}(x)) \quad \text{(Pre-LN variant)}", "Layer Normalization followed by Residual connection."),
            (r"\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta", "Layer Normalization formula. \(\mu\) = mean, \(\sigma^2\) = variance, \(\gamma, \beta\) are learnable parameters, \(\epsilon\) prevents division by zero.")
        ]
    },
    "6. Encoder & Decoder Structure": {
        "text": """
        **Goal:** Define the overall architecture.
        *   **Encoder:** Reads input, builds representation (Stack of Self-Attention + FFN).
        *   **Decoder:** Generates output token by token (Stack of Masked Self-Attention, Encoder-Decoder Attention, FFN).
        *   **Note:** GPT models (like this one) are **Decoder-only**. They use Masked Self-Attention to predict the next token based on previous ones.
        """,
        "formulas": [] # More architectural than mathematical formula based at this level.
    },
    "7. Output Decoding": {
        "text": """
        **Goal:** Convert final vectors back into text probabilities.
        *   Final vectors passed through a Linear layer (size = vocabulary size) to get raw scores (logits).
        *   **Softmax:** Converts logits into probabilities over the vocabulary for the next token prediction.
        *   **Generation Strategy:** Sampling (used here), Greedy, or Beam Search selects the actual next token based on these probabilities.
        """,
        "formulas": [
            (r"\text{Logits} = \text{Linear}(\text{FinalHiddenState})", "Final projection layer."),
            (r"P(\text{next\_token}) = \text{softmax}(\text{Logits})", "Calculating probabilities over the vocabulary."),
            (r"\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{|\text{Vocab}|} e^{z_j}}", "Softmax function applied to logits \(z\).")
        ]
    }
}


# --- Model Loading (Cached) ---
@st.cache_resource(show_spinner="Loading Amharic Chatbot Model...")
def load_model_and_tokenizer(model_path):
    # (Existing implementation)
    if not os.path.isdir(model_path):
        st.error(f"Error: Model directory not found at {model_path}. Cannot load the model.")
        return None, None, None # Return None for model, tokenizer, device
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        model.to(device)
        model.eval()
        st.success(f"Model loaded successfully on {device_name.upper()}!")
        return model, tokenizer, device
    except OSError as e:
        st.error(f"Error loading model/tokenizer files from {model_path}: {e}. Ensure all necessary files (config, weights, tokenizer) are present.")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        return None, None, None

# --- Generation Function ---
def generate_chat_response(prompt, model, tokenizer, device, max_new, temp, top_k, top_p, rep_pen, no_repeat_ngram):
    # (Existing implementation)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new, temperature=temp, top_k=top_k, top_p=top_p,
                repetition_penalty=rep_pen, no_repeat_ngram_size=no_repeat_ngram,
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = full_text
        if prompt and full_text.startswith(prompt): response = full_text[len(prompt):].strip()
        response = response.split('\n')[0]
        response = response.replace("...", "").strip()
        response = re.sub(r'^[፡።፣፤]+|[፡።፣፤]+$', '', response).strip()
        return response if response else "ይቅርታ፣ ምላሽ ማመንጨት አልቻልኩም።"
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return "ይቅርታ፣ በምላሹ ወቅት ስህተት አጋጥሟል።"

# --- Function for Illustrative Calculation ---
def show_illustrative_calculations(text, model, tokenizer, device):
    st.subheader(f"Illustrative Calculations for: '{text}'")
    if not model or not tokenizer:
        st.warning("Model not loaded, cannot perform calculations.")
        return

    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        st.markdown("**1. Tokenization:**")
        st.write(f"Input Text: `{text}`")
        st.write(f"Tokens: `{tokens}`")
        st.write(f"Input IDs: `{input_ids[0].tolist()}`")

        with torch.no_grad():
            # Run forward pass requesting hidden states and attentions
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

            # --- Input Embeddings ---
            st.markdown("**2. Input Embeddings (Initial):**")
            initial_embeddings = outputs.hidden_states[0] # First hidden state is embeddings
            st.write(f"Shape: `{list(initial_embeddings.shape)}` (Batch, Sequence Length, Hidden Size)")
            # Show first few dimensions of the first token's embedding
            st.write("Embedding vector for first token (first 5 dims):")
            st.code(f"{initial_embeddings[0, 0, :5].cpu().numpy()}")

            # --- Attention (Example from First Layer, First Head) ---
            st.markdown("**3. Attention Weights (Example):**")
            # Let's look at the attention weights from the first layer, first head
            attention_weights = outputs.attentions[0][0, 0] # Batch 0, Head 0
            st.write("Shape (1st layer, 1st head): "
                     f"`{list(attention_weights.shape)}` (Seq Len 'Query', Seq Len 'Key')")
            st.write("Attention weights (softmax output):")
            # Display as a dataframe for better readability
            st.dataframe(np.round(attention_weights.cpu().numpy(), 3),
                         index=[f"To: {t}" for t in tokens],
                         columns=[f"From: {t}" for t in tokens])
            st.caption("Rows attend TO columns. Higher values mean the row token paid more attention to the column token.")

            # --- Output Logits (Prediction for Next Token) ---
            st.markdown("**4. Output Logits (Prediction after input):**")
            logits = outputs.logits
            # Logits for the *last* token in the input sequence predict the *next* token
            last_token_logits = logits[0, -1, :]
            st.write("Shape of all logits: "
                     f"`{list(logits.shape)}` (Batch, Seq Length, Vocab Size)")
            st.write("Logits for predicting token *after* the last input token:"
                     f" Shape `{list(last_token_logits.shape)}`")

            # Show top 5 predicted next tokens based on these logits
            probs = F.softmax(last_token_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, 5)
            top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

            st.write("Top 5 predicted next tokens (and their probabilities):")
            for token, prob in zip(top_k_tokens, top_k_probs.cpu().numpy()):
                st.write(f"- `{token}`: {prob:.4f}")

    except Exception as e:
        st.error(f"Error during illustrative calculation: {e}")


# --- Streamlit App UI ---
# (Keep st.set_page_config and Sidebar code as before)
# --- (Sidebar code omitted for brevity) ---
st.set_page_config(
    page_title="የአማርኛ Chatbot | Amharic Chatbot",
    page_icon="🇪🇹",
    layout="wide"
)

# --- Sidebar for Settings AND Explanations ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Flag_of_Ethiopia.svg/1200px-Flag_of_Ethiopia.svg.png", width=80)
    st.title("⚙️ መቆጣጠሪያ | Settings")
    st.markdown("Adjust generation parameters:")

    # Generation Parameter Sliders
    temperature = st.slider("🌡️ Temperature", 0.1, 1.5, 0.8, 0.05, help="Controls randomness.")
    max_new_tokens = st.slider("📝 Max New Tokens", 10, 250, 70, 10, help="Max generated response length.")
    top_k = st.slider("🔝 Top-K", 0, 100, 40, 5, help="Considers only top K words.")
    top_p = st.slider("🅿️ Top-P", 0.0, 1.0, 0.9, 0.05, help="Considers words cumulative prob > P.")
    repetition_penalty = st.slider("🔄 Repetition Penalty", 1.0, 2.0, 1.2, 0.1, help="Penalizes repeated words.")
    no_repeat_ngram = st.slider("🚫 No Repeat N-grams", 0, 5, 3, 1, help="Prevents repeating N-word sequences.")

    if st.button("🔄 Reset Settings"):
        st.rerun()

    st.markdown("---") # Separator

    # --- Explanation Section ---
    st.title("🧠 Learn About the Model")
    st.markdown("How does the underlying Transformer model work?")

    explanation_topics = list(TRANSFORMER_EXPLANATIONS.keys())
    selected_topic_key = st.selectbox(
        "Choose a component:",
        options=explanation_topics,
        index=0 # Default to "--- Select a component ---"
    )

    # Display the explanation and formulas for the selected topic
    if selected_topic_key != "--- Select a component ---":
        explanation_data = TRANSFORMER_EXPLANATIONS[selected_topic_key]
        # Handle cases where the key might not have a number prefix
        topic_name_parts = selected_topic_key.split('. ')
        topic_name = topic_name_parts[1] if len(topic_name_parts) > 1 else selected_topic_key

        with st.expander(f"Explanation for: {topic_name}", expanded=True):
            # Display text description
            st.markdown(explanation_data["text"])

            # Display formulas if they exist
            if explanation_data["formulas"]:
                st.markdown("**Key Formula(s):**")
                for formula, description in explanation_data["formulas"]:
                    st.latex(formula)
                    if description:
                        st.caption(description) # Add a small caption explaining the formula


# --- Main Chat Area ---
# (Keep title, markdown, model loading as before)
# --- (Code omitted for brevity) ---
st.title("🇪🇹 የአማርኛ Chatbot (Amharic Chatbot)")
st.markdown("በ **rasyosef/gpt2-small-amharic** ሞዴል የተሻሻለ | _Fine-tuned from rasyosef/gpt2-small-amharic_")
st.markdown("---")

# Load model and tokenizer
model, tokenizer, device = load_model_and_tokenizer(MODEL_PATH)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "ሰላም! እንዴት ልረዳዎት እችላለሁ? | Hello! How can I help you?"}]

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("የሚፈልጉትን ይጻፉ... | Type your message here..."):
    if model is None or tokenizer is None:
        st.error("Model is not loaded. Cannot generate response.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 ቦቱ እያሰበ ነው... | Bot is thinking..."):
                response = generate_chat_response(
                    prompt, model, tokenizer, device,
                    max_new=max_new_tokens, temp=temperature, top_k=top_k, top_p=top_p,
                    rep_pen=repetition_penalty, no_repeat_ngram=no_repeat_ngram
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Illustrative Calculation Section ---
st.markdown("---") # Separator
example_text = "ሰላም" # Fixed example text
if st.button(f"📊 Show Example Calculations for '{example_text}'"):
    show_illustrative_calculations(example_text, model, tokenizer, device)


# Footer
st.markdown("---")
st.caption("Disclaimer: Designed by Abdullah Farid, this chatbot is based on a fine-tuned model and may produce unexpected or inaccurate responses.")