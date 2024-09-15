import json


# Function to read and parse the JSON file containing the predictions
def read_json_file(file_path):
    # Open the file and read each line, parse each line as a JSON object
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    return data


# Function to read and parse the entity2text mapping file
# This file contains entity IDs mapped to their natural language descriptions
def read_entity2text(file_path):
    entity2text = {}
    # Open the mapping file and split each line into entity ID and description
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entity_id, entity_text = line.strip().split('\t')
            entity2text[entity_id] = entity_text  # Store the mapping in a dictionary
    return entity2text


# Function to replace Answer IDs with their corresponding natural language descriptions
def map_answer_to_text(data, entity2text):
    # For each entry in the JSON data, map the 'Answer' ID to its natural language text
    for entry in data:
        answer_id = entry['Answer']
        # If the answer ID exists in the entity2text mapping, replace it with the text
        entry['AnswerText'] = entity2text.get(answer_id, answer_id)  # If not found, keep the ID
    return data


# Function to calculate MRR (Mean Reciprocal Rank) and Hit@1, Hit@3, Hit@10
def calculate_metrics(data):
    mrr = 0.0  # Mean Reciprocal Rank accumulator
    hits_at_1, hits_at_3, hits_at_10 = 0, 0, 0  # Hit counters for different thresholds
    total = len(data)  # Total number of samples

    # Loop over each entry to calculate ranks and hits
    for entry in data:
        answer = entry['AnswerText']  # The correct answer (in natural language)
        # Split the predicted answers (comma-separated) and strip extra spaces
        predictions = [pred.strip() for pred in entry['Prediction'].split(',')]

        # Find the rank of the correct answer in the list of predictions
        if answer in predictions:
            rank = predictions.index(answer) + 1  # Rank is 1-based
        else:
            rank = float('inf')  # If the answer is not in the predictions

        # MRR calculation: Reciprocal of the rank of the correct answer
        if rank != float('inf'):
            mrr += 1.0 / rank

        # Hit@1, Hit@3, Hit@10 calculations: Check if the correct answer is within the top-k predictions
        if rank == 1:
            hits_at_1 += 1
        if rank <= 3:
            hits_at_3 += 1
        if rank <= 10:
            hits_at_10 += 1

    # Normalize the results by the total number of samples to get average scores
    mrr /= total
    hit_at_1 = hits_at_1 / total
    hit_at_3 = hits_at_3 / total
    hit_at_10 = hits_at_10 / total

    return mrr, hit_at_1, hit_at_3, hit_at_10


# Main function to execute all steps
def main(json_file, entity2text_file):
    # Step 1: Read and parse the JSON data
    data = read_json_file(json_file)

    # Step 2: Read and parse the entity2text mapping
    entity2text = read_entity2text(entity2text_file)

    # Step 3: Replace the Answer IDs in the JSON data with natural language descriptions
    data = map_answer_to_text(data, entity2text)

    # Step 4: Calculate MRR and Hit@1, Hit@3, Hit@10 based on the predictions
    mrr, hit_at_1, hit_at_3, hit_at_10 = calculate_metrics(data)

    # Step 5: Output the calculated metrics
    print(f"MRR: {mrr:.4f}")
    print(f"Hit@1: {hit_at_1:.4f}")
    print(f"Hit@3: {hit_at_3:.4f}")
    print(f"Hit@10: {hit_at_10:.4f}")


# Call the main function with the paths to the JSON file and entity2text mapping file
json_file_path = 'outputs/wn18rr/100_test/Mistral7B/tsa/output_tail_mistral7b_test_50can_tsa.txt'
entity2text_file_path = 'dataset/wn18rr/entity2text.txt'

main(json_file_path, entity2text_file_path)
