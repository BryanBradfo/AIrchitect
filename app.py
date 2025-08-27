import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)
import gradio as gr

# --- 1. Initialisation du Modèle ---
if not torch.cuda.is_available():
    raise SystemError("CUDA is not available. This script requires a NVIDIA GPU.")

print("CUDA is available! Initializing the model...")

model_id = "mistral-community/pixtral-12b"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

print("Model and processor loaded successfully. The web app is ready.")

# --- 2. Simulation de la base de données produit ---
PRODUCT_DATABASE = {
    "chaise_design_scandinave": {
        "name": "Chaise Design Scandinave 'Nordik'",
        "material": "Bois de chêne massif, assise en tissu gris clair",
        "dimensions_cm": "H: 85, L: 48, P: 50",
        "price_eur": 129.99,
        "style_tags": ["scandinave", "minimaliste", "bois clair", "moderne"],
    },
    "lampe_industrielle_acier": {
        "name": "Lampe sur pied 'Atelier'",
        "material": "Acier noir mat, ampoule Edison visible",
        "dimensions_cm": "H: 150, Diamètre base: 25",
        "price_eur": 89.90,
        "style_tags": ["industriel", "vintage", "métal", "loft"],
    },
}


# --- 3. Logique du modèle avec un prompt amélioré ---
def get_pixtral_response(product_image, user_image, product_data, user_question):
    product_info_text = (
        f"Nom du produit : {product_data['name']}\n"
        f"Matériaux : {product_data['material']}\n"
        f"Style : {', '.join(product_data['style_tags'])}"
    )

    # On donne des instructions plus claires et structurées.
    chat_template = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "SYSTEM: Tu es un assistant expert en décoration d'intérieur. "
                        "Ta mission est d'analyser les informations et les images fournies pour répondre à la question du client. "
                        "Sois amical et donne des conseils concrets.\n\n"
                        "CONTEXTE:\n"
                        f"Voici les informations sur le produit concerné :\n{product_info_text}\n\n"
                        "La première image est le PRODUIT. La deuxième image est la PIÈCE du client.\n\n"
                        f'QUESTION DU CLIENT: "{user_question}"\n\n'
                        "RÉPONSE:"
                    ),
                },
                {"type": "image"},
                {"type": "image"},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        chat_template, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=prompt, images=[product_image, user_image], return_tensors="pt"
    )

    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    inputs = inputs.to(model.device)

    print("Generating response...")
    generate_ids = model.generate(
        **inputs, max_new_tokens=300, pad_token_id=processor.tokenizer.eos_token_id
    )
    raw_output = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Pour le débogage, on peut afficher la sortie brute du modèle
    print(f"--- RAW MODEL OUTPUT ---\n{raw_output}\n-------------------------")

    try:
        # On cherche la partie qui commence par "RÉPONSE:" et on prend ce qui suit.
        # C'est plus fiable que de se baser sur "assistant\n".
        assistant_response = raw_output.split("RÉPONSE:")[1].strip()
        if assistant_response == "":  # Si la réponse est vide après le split
            # On tente une autre méthode de secours
            assistant_response = raw_output.split("assistant\n")[-1].strip()

    except IndexError:
        assistant_response = "Désolé, une erreur est survenue lors de la génération de la réponse. Veuillez reformuler votre question."

    return assistant_response


# --- 4. Interface Chatbot avec un bouton Envoyer ---

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🛋️ Déco-Vision AI (Projet de démo Mistral)")
    gr.Markdown(
        "Un assistant IA qui vous aide à visualiser nos produits dans votre intérieur."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")
            product_id = gr.Dropdown(
                choices=list(PRODUCT_DATABASE.keys()),
                value="chaise_design_scandinave",
                label="1. Choisissez un produit",
            )
            user_image = gr.Image(
                type="pil", label="2. Téléchargez une photo de votre pièce"
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation avec l'Expert IA",
                bubble_full_width=False,
                avatar_images=(None, "https://i.imgur.com/Phd3S8y.png"),
            )
            with gr.Row():
                msg = gr.Textbox(
                    label="3. Posez votre question",
                    placeholder="Ex: Est-ce que cette chaise irait bien avec mon salon ?",
                    scale=4,  # La zone de texte prend plus de place
                )
                send_button = gr.Button(
                    "Envoyer", variant="primary", scale=1
                )  # Le bouton

    def respond(product_id_selected, room_image, user_question, chat_history):
        if not user_question.strip():
            gr.Warning("Veuillez entrer une question !")
            return "", chat_history
        if room_image is None:
            gr.Warning("N'oubliez pas de télécharger une photo de votre pièce !")
            return (
                user_question,
                chat_history,
            )

        chat_history.append((user_question, None))
        yield "", chat_history

        product_info = PRODUCT_DATABASE[product_id_selected]
        image_paths = {
            "chaise_design_scandinave": "data/chaise_design_scandinave.jpg",
            "lampe_industrielle_acier": "data/lampadaire.jpg",
        }
        product_image = Image.open(image_paths[product_id_selected])

        bot_response = get_pixtral_response(
            product_image, room_image, product_info, user_question
        )

        chat_history[-1] = (user_question, bot_response)

        yield "", chat_history

    send_button.click(respond, [product_id, user_image, msg, chatbot], [msg, chatbot])
    msg.submit(respond, [product_id, user_image, msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True)
