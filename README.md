# Voice Command Classification using DistilBERT

## Project Overview
This ALM (Applied Learning Module) demonstrates a transformer-based Natural Language Processing (NLP) model for classifying voice-assistant/autonomous-vehicle-style text commands into four movement intents:

- **FORWARD** – move forward
- **BACKWARD** – move backward/reverse
- **LEFT** – turn left
- **RIGHT** – turn right

The notebook fine-tunes **DistilBERT (`distilbert-base-uncased`)** for sequence classification using a custom CSV dataset. It includes dataset loading, train/test splitting, tokenization, model training, testing with sample commands, saving the trained model and tokenizer, and downloading the model as a ZIP file.

> Note: The notebook classifies text commands. It does not itself perform speech-to-text or control a physical vehicle. A speech-recognition or hardware-control module would be required for those functions.

## Objectives
1. Prepare a labeled command dataset.
2. Load and split the dataset into training and testing subsets.
3. Use a pretrained DistilBERT tokenizer and sequence-classification model.
4. Fine-tune the model to recognize four movement-command classes.
5. Test predictions on English and Telugu transliterated command examples.
6. Save and export the trained model and tokenizer.

## Requirements

### Software
- Google Colab or Jupyter Notebook
- Python 3.x (Google Colab provides a compatible Python environment)
- Required Python packages:
  - `transformers`
  - `datasets`
  - `torch`

The notebook installs the packages with:

```bash
pip install transformers datasets torch
```

### Hardware
- CPU: sufficient for a small demonstration, but training may take longer.
- GPU: recommended for faster model fine-tuning. In Colab, select **Runtime → Change runtime type → T4 GPU** (or another available GPU).

### Dataset
Upload a CSV file named `commands.csv` when prompted.

The CSV must contain these columns:

| Column | Description |
|---|---|
| `text` | Command phrase or text input |
| `label` | Integer class label |

The notebook configures four classes (`num_labels=4`). Use integer labels consistently, with the following mapping:

| Label | Class |
|---:|---|
| 0 | FORWARD |
| 1 | BACKWARD |
| 2 | LEFT |
| 3 | RIGHT |

Example dataset rows:

```csv
text,label
move forward,0
go ahead,0
venakki vellandi,1
reverse,1
turn left,2
edama tirugu,2
turn right,3
kudi tirugu,3
```

**Dataset notes:**
- Ensure the CSV has both `text` and `label` columns.
- Labels should be integer values from `0` to `3`.
- Include multiple examples per class.
- Use varied phrases, spelling variations, and language/transliteration patterns to improve generalization.
- The sample phrases above are illustrative; prepare a larger balanced dataset for meaningful evaluation.

## Notebook Workflow
1. Install the required libraries.
2. Upload `commands.csv`.
3. Load the CSV dataset using Hugging Face `datasets`.
4. Split the dataset into training and testing sets (80% training / 20% testing).
5. Load the pretrained DistilBERT tokenizer and sequence-classification model.
6. Tokenize command text.
7. Format the tokenized dataset for PyTorch.
8. Train using Hugging Face `Trainer` and `TrainingArguments`.
9. Predict movement intents for sample English and Telugu-transliterated phrases.
10. Save the model and tokenizer in `command_model/`.
11. ZIP the saved model and download `command_model.zip`.
12. Run the eight example test phrases and inspect predicted classes.

## Model Details
- **Model:** DistilBERT
- **Pretrained checkpoint:** `distilbert-base-uncased`
- **Task:** Text classification / intent classification
- **Number of classes:** 4
- **Training epochs in notebook:** 15
- **Training batch size:** 8 per device
- **Evaluation batch size:** 8 per device
- **Train/test split:** 80% / 20%

## How to Run in Google Colab
1. Open the notebook in Google Colab.
2. Select a GPU runtime if available: **Runtime → Change runtime type → GPU**.
3. Run the package-installation cell.
4. Upload your `commands.csv` dataset when prompted.
5. Execute the notebook cells in order.
6. Review the prediction outputs.
7. Download the generated `command_model.zip` file.

## Example Commands
The notebook tests commands such as:

| Command | Expected intent |
|---|---|
| `move forward` | FORWARD |
| `go ahead` | FORWARD |
| `venakki vellandi` | BACKWARD |
| `reverse` | BACKWARD |
| `turn left` | LEFT |
| `edama tirugu` | LEFT |
| `turn right` | RIGHT |
| `kudi tirugu` | RIGHT |

Expected outputs are examples of the intended class mapping; actual predictions depend on the quality, size, balance, and training of the dataset.

## Output Files
After execution, the notebook produces:

- `command_model/` – saved model weights and tokenizer files
- `command_model.zip` – compressed model package for download

## Limitations
- This notebook performs text-based command classification only.
- It does not include speech recognition, Bluetooth communication, ESP32 integration, or motor-control code.
- The dataset quality and quantity directly affect model performance.
- The notebook's current test examples are illustrative; evaluate the trained model with additional unseen commands before using it in a real application.
- Do not connect the classifier directly to a real vehicle or robotic platform without an independent safety layer, validation, and fail-safe controls.

## Repository Structure (Suggested)

```text
voice-command-classification/
├── README.md
├── project.ipynb
├── commands.csv              # Add your dataset (if permitted)
└── command_model.zip         # Optional: exported trained model
```

## License
Add an appropriate license if you plan to distribute this project publicly. Ensure you have permission to share the dataset and any model artifacts.
