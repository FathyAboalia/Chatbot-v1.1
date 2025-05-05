import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_products():
    """Load product names from JSON data."""
    products = [
        "Farmer", "Accountant", "Teacher", "Engineer", "Credit Hour Engineering 2023",
        "Keratin Shampoo 1 L", "Keratin Shampoo 500 ML", "Zakhir Perfume oil 100 GMS",
        "new Parent Item", "فاصوليا خام", "Mobile Samsung 8 Gega", "فاصوليا مجمده(نص مصنع)",
        "New Child Item", "شكاره فاصوليا 20ك", "Item1", "شكاره", "شكاره فاصوليا 25ك",
        "Office Chair With a stand", "Item2"
    ]
    return products

def create_training_data(products):
    """Create training dataset from product names."""
    data = []
    for product in products:
        # Place order examples (Arabic and English, varied phrasing)
        data.append({
            "input": f"اطلب 50 وحدة من {product}",
            "output": f"product: {product}, quantity: 50"
        })
        data.append({
            "input": f"Order 20 units of {product}",
            "output": f"product: {product}, quantity: 20"
        })
        data.append({
            "input": f"اشتري 30 {product}",
            "output": f"product: {product}, quantity: 30"
        })
        data.append({
            "input": f"Place an order for 15 {product}",
            "output": f"product: {product}, quantity: 15"
        })
        data.append({
            "input": f"أريد 10 من {product}",
            "output": f"product: {product}, quantity: 10"
        })
        data.append({
            "input": f"Buy 5 units of {product}",
            "output": f"product: {product}, quantity: 5"
        })
        # Check price examples
        data.append({
            "input": f"كم سعر {product}؟",
            "output": f"product: {product}, quantity: 1"
        })
        data.append({
            "input": f"What’s the price of {product}?",
            "output": f"product: {product}, quantity: 1"
        })
        # Additional variations matching user input
        data.append({
            "input": f"Order 20 bottles of {product}",
            "output": f"product: {product}, quantity: 20"
        })
        data.append({
            "input": f"اطلب 50 كيس من {product}",
            "output": f"product: {product}, quantity: 50"
        })
        data.append({
            "input": f"أطلب 100 {product} بسرعة",
            "output": f"product: {product}, quantity: 100"
        })
        data.append({
            "input": f"I need 10 {product} urgently",
            "output": f"product: {product}, quantity: 10"
        })
    return data

def main():
    try:
        # Load model and tokenizer
        model_name = "google/flan-t5-base"
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        logger.info(f"Loaded model {model_name}")

        # Load and create training data
        products = load_products()
        data = create_training_data(products)
        dataset = Dataset.from_list(data)

        # Preprocess data
        def preprocess_function(examples):
            inputs = [
                f"Extract the product name and quantity from the following input. "
                f"Return in the format: 'product: <name>, quantity: <number>'. "
                f"Available products: {', '.join(products)}. "
                f"Match the product name exactly as listed. "
                f"If no product or quantity is found, return 'product: None, quantity: 1'. "
                f"Input: '{x}'" for x in examples["input"]
            ]
            outputs = examples["output"]
            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
            labels = tokenizer(outputs, max_length=64, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./flan-t5-finetuned",
            num_train_epochs=10,  # Increased epochs for better training
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=50,
            learning_rate=2e-5,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        # Train model
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed")

        # Save model
        model.save_pretrained("./flan-t5-finetuned")
        tokenizer.save_pretrained("./flan-t5-finetuned")
        logger.info("Model saved to ./flan-t5-finetuned")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()