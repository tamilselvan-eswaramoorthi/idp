from transformers import AutoProcessor, LayoutLMv2ForQuestionAnswering
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2ForQuestionAnswering.from_pretrained("microsoft/layoutlmv2-base-uncased")

image_path = '/home/tamilselvan/Downloads/R.jpeg'
image = Image.open(image_path).convert("RGB")
question = "What is the ticket type?"
encoding = processor(image, question, return_tensors="pt")

outputs = model(**encoding)
predicted_start_idx = outputs.start_logits.argmax(-1).item()
predicted_end_idx = outputs.end_logits.argmax(-1).item()
predicted_start_idx, predicted_end_idx

predicted_answer_tokens = encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1]
predicted_answer = processor.tokenizer.decode(predicted_answer_tokens)
print(predicted_answer)