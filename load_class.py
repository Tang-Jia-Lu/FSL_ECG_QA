import os
import glob
import json
import argparse
from collections import Counter
import numpy as np

def get_temp_qa(question_type, paraphrased_path):
    """
    Load all question data and return a list of filtered template IDs based on question_type.

    Args:
        question_type (str): Type of question to filter by ('single-query', 'single-verify', etc., or 'all').
        paraphrased_path (str): Base path for the dataset.

    Returns:
        List[int]: A list of unique template IDs matching the filter criteria.
    """
    data = []
    
    for split in ["train", "valid", "test"]:
        file_pattern = os.path.join(paraphrased_path, split, "*.json")
        for fname in sorted(glob.glob(file_pattern)):
            with open(fname, "r") as f:
                records = json.load(f)
                data.extend(records)

    print(f"Loaded {len(data)} samples!")
    
    temp_list = []
    if question_type != "all":
        for item in data:
            if item['question_type'] == question_type:
                temp_list.append(item['template_id'])
    else:
        for item in data:
            if "single-" in item['question_type']:
                temp_list.append(item['template_id'])
    
    temp_ids = list(set(temp_list))
    
    # Filter template IDs to avoid conflict class
    temp_ids = [i for i in temp_ids if i not in [17, 14, 26, 28, 4, 40, 31]]
    return temp_ids

def prepare_ecg_qa_data(args):
    if args.question_type == "all":
        class_qa = get_temp_qa("single-verify")
        ecg_qa_list_1 = change_ecg_to_qa(
            class_qa, "single-verify",
            paraphrased_path=args.paraphrased_path,
            dif_exp=args.dif_exp
        )

        class_qa += get_temp_qa("single-choose")
        ecg_qa_list_2 = change_ecg_to_qa(
            class_qa, "single-choose",
            paraphrased_path=args.paraphrased_path,
            dif_exp=args.dif_exp
        )

        class_qa += get_temp_qa("single-query")
        ecg_qa_list_3 = change_ecg_to_qa(
            class_qa, "single-query",
            paraphrased_path=args.paraphrased_path,
            dif_exp=args.dif_exp
        )

        ecg_qa_list = {**ecg_qa_list_1, **ecg_qa_list_2, **ecg_qa_list_3}
    else:
        class_qa = get_temp_qa(args.question_type, args.paraphrased_path)
        ecg_qa_list = change_ecg_to_qa(
            class_qa, args.question_type,
            paraphrased_path=args.paraphrased_path,
            dif_exp=args.dif_exp
        )

    all_ecg_qa_temp = list(ecg_qa_list.keys())
    sample_size = int(len(all_ecg_qa_temp) * 0.8)
    train_temp = np.random.choice(all_ecg_qa_temp, sample_size, replace=False).tolist()
    test_temp = [i for i in all_ecg_qa_temp if i not in train_temp]

    return class_qa, train_temp, test_temp

def change_ecg_to_qa(
    sample_ids, 
    question_type, 
    paraphrased_path,
    in_template=None, 
    dif_exp=1, 
    attr="",
    test_dataset="ptb-xl"
):
    """
    Process ECG data to create a dictionary of question-answer pairs categorized by template.
    
    Args:
        sample_ids (List[int]): List of template IDs to include.
        question_type (str): Type of question to filter.
        paraphrased_path (str): Base path for all data.
        data_root_path (str, optional): Path to the data root directory.
        in_template (List, optional): List of templates to include.
        dif_exp (int, optional): Controls return format. Default is 1.
        attr (str, optional): Attribute type filter.
        test_dataset (str, optional): Dataset to use ('ptb-xl' or 'mimic'). Default is 'ptb-xl'.
        
    Returns:
        Dict: Dictionary of ECG QA data categorized by template.
    """
    data = []
    
        
    # Define subdirectories to process
    subdirectories = ["train", "valid", "test"]
    
    # Load data from all subdirectories
    for subdir in subdirectories:
        directory_path = os.path.join(paraphrased_path, subdir)
        if not os.path.exists(directory_path):
            print(f"Warning: Directory {directory_path} does not exist!")
            continue
            
        file_pattern = os.path.join(directory_path, "*.json")
        files = sorted(glob.glob(file_pattern))
        
        if not files:
            print(f"Warning: No JSON files found in {directory_path}")
            continue
            
        for fname in files:
            with open(fname, "r") as f:
                json_data = json.load(f)
                data.extend(json_data)
    
    # Filter by attribute type if specified
    if attr != "":
        sample_data = [item for item in data if item['attribute_type'] == attr]
    else:
        sample_data = data
    
    # Filter samples by file existence if using MIMIC dataset
    if test_dataset == "mimic":
        ecg_data_path = os.path.join(paraphrased_path, "mimic_iv_ecg/processed_test_30k/")
        if not os.path.exists(ecg_data_path):
            print(f"Warning: MIMIC ECG data path {ecg_data_path} does not exist!")
        sample_data = [sample for sample in sample_data if os.path.isfile(os.path.join(ecg_data_path, f"{sample['ecg_id'][0]}.mat"))]

    ecg_qa_dict = {}
    
    if len(sample_data) == 0:
        print(f"Cannot find template_id == {sample_ids} or no data available")
    else:
        for sample in sample_data:
            template_id = sample['template_id']
            if template_id in sample_ids:
                process_sample_by_type(sample, template_id, ecg_qa_dict, in_template)
    
    filter_ecg_qa_dict_by_question_type(ecg_qa_dict, question_type)
    
    if dif_exp == 1:
        return {key: [value] for key, value in ecg_qa_dict.items()}
    
    return filter_by_question_frequency(ecg_qa_dict)


def process_sample_by_type(sample, template_id, ecg_qa_dict, in_template):
    """
    Process a sample based on its question type.
    
    Args:
        sample (Dict): The sample data.
        template_id (int): The template ID.
        ecg_qa_dict (Dict): Dictionary to populate with processed data.
        in_template (List, optional): List of templates to include.
    """
    question_type = sample['question_type']
    
    if question_type == 'single-verify':
        process_single_verify_sample(sample, template_id, ecg_qa_dict, in_template)
    elif question_type == 'single-choose':
        process_single_choose_sample(sample, template_id, ecg_qa_dict, in_template)
    elif question_type == 'single-query':
        process_single_query_sample(sample, template_id, ecg_qa_dict, in_template)


def process_single_verify_sample(sample, template_id, ecg_qa_dict, in_template):
    """
    Process a single-verify type sample.
    
    Args:
        sample (Dict): The sample data.
        template_id (int): The template ID.
        ecg_qa_dict (Dict): Dictionary to populate with processed data.
        in_template (List, optional): List of templates to include.
    """
    if sample['answer'][0] in ["yes", "no"]:
        answer = sample['answer'][0]
        all_attributes = "_".join(sorted(sample['attribute']))
        dict_key = f"{template_id}_{all_attributes}_{answer}"
        
        if in_template is None or dict_key in in_template:
            if dict_key not in ecg_qa_dict:
                ecg_qa_dict[dict_key] = [sample]
            else:
                ecg_qa_dict[dict_key].append(sample)


def process_single_choose_sample(sample, template_id, ecg_qa_dict, in_template):
    """
    Process a single-choose type sample.
    
    Args:
        sample (Dict): The sample data.
        template_id (int): The template ID.
        ecg_qa_dict (Dict): Dictionary to populate with processed data.
        in_template (List, optional): List of templates to include.
    """
    if len(sample['answer']) == 1:
        answer = sample['answer'][0]
    elif len(sample['answer']) == 2:
        answer = "both"
        sample['answer'] = ["both"]
    else:
        print("single-choose data have more than 2 answers!")
        return
    
    all_attributes = "_".join(sorted(sample['attribute']))
    
    if in_template is None:
        handle_single_choose_without_template(sample, template_id, answer, all_attributes, ecg_qa_dict)
    else:
        handle_single_choose_with_template(sample, template_id, answer, all_attributes, in_template, ecg_qa_dict)


def handle_single_choose_without_template(sample, template_id, answer, all_attributes, ecg_qa_dict):
    """
    Handle a single-choose type sample when no template is provided.
    
    Args:
        sample (Dict): The sample data.
        template_id (int): The template ID.
        answer (str): The answer string.
        all_attributes (str): The joined attributes string.
        ecg_qa_dict (Dict): Dictionary to populate with processed data.
    """
    if answer == "both":
        dict_key = f"{template_id}_{all_attributes}_{answer}"
        add_to_ecg_qa_dict(dict_key, sample, ecg_qa_dict)
    elif answer == "none":
        for attr in sample['attribute']:
            dict_key = f"{template_id}_{attr}_{answer}"
            add_to_ecg_qa_dict(dict_key, sample, ecg_qa_dict)
    else:
        dict_key = f"{template_id}_{answer}_{answer}"
        add_to_ecg_qa_dict(dict_key, sample, ecg_qa_dict)


def handle_single_choose_with_template(sample, template_id, answer, all_attributes, in_template, ecg_qa_dict):
    """
    Handle a single-choose type sample when a template is provided.
    
    Args:
        sample (Dict): The sample data.
        template_id (int): The template ID.
        answer (str): The answer string.
        all_attributes (str): The joined attributes string.
        in_template (List): List of templates to include.
        ecg_qa_dict (Dict): Dictionary to populate with processed data.
    """
    full_key = f"{template_id}_{all_attributes}_{answer}"
    short_key = f"{template_id}_{answer}"
    
    if full_key in in_template or short_key in in_template:
        if answer == "both":
            dict_key = f"{template_id}_{all_attributes}_{answer}"
            add_to_ecg_qa_dict(dict_key, sample, ecg_qa_dict)
        elif answer == "none":
            for attr in sample['attribute']:
                dict_key = f"{template_id}_{attr}_{answer}"
                add_to_ecg_qa_dict(dict_key, sample, ecg_qa_dict)
        else:
            dict_key = f"{template_id}_{answer}_{answer}"
            add_to_ecg_qa_dict(dict_key, sample, ecg_qa_dict)


def process_single_query_sample(sample, template_id, ecg_qa_dict, in_template):
    """
    Process a single-query type sample.
    
    Args:
        sample (Dict): The sample data.
        template_id (int): The template ID.
        ecg_qa_dict (Dict): Dictionary to populate with processed data.
        in_template (List, optional): List of templates to include.
    """
    if len(sample['answer']) == 1:
        answer = sample['answer'][0]
    elif len(sample['answer']) >= 2:
        answer = ", ".join(sample['answer'])
        sample['answer'] = [answer]
    
    all_attributes = "_".join(sorted(sample['attribute']))
    dict_key = f"{template_id}_{all_attributes}_{answer}"
    
    if in_template is None or dict_key in in_template:
        if dict_key not in ecg_qa_dict:
            ecg_qa_dict[dict_key] = [sample]
        else:
            ecg_qa_dict[dict_key].append(sample)


def add_to_ecg_qa_dict(key, sample, ecg_qa_dict):
    """
    Add a sample to the ECG QA dictionary.
    
    Args:
        key (str): The dictionary key.
        sample (Dict): The sample data.
        ecg_qa_dict (Dict): Dictionary to populate with processed data.
    """
    if key not in ecg_qa_dict:
        ecg_qa_dict[key] = [sample]
    else:
        ecg_qa_dict[key].append(sample)


def filter_ecg_qa_dict_by_question_type(ecg_qa_dict, question_type):
    """
    Filter the ECG QA dictionary by question type and minimum sample counts.
    
    Args:
        ecg_qa_dict (Dict): Dictionary to filter.
        question_type (str): Type of question to filter by.
    """
    for key in list(ecg_qa_dict.keys()):
        if question_type in ["single-verify", "single-choose", "single-query"]:
            qt = question_type
        elif question_type == "all":
            if len(ecg_qa_dict[key]) == 0:
                del ecg_qa_dict[key]
                continue
            qt = ecg_qa_dict[key][0]['question_type']
        else:
            continue
        
        if qt == "single-verify" and len(ecg_qa_dict[key]) < 140:
            del ecg_qa_dict[key]
        elif qt == "single-choose" and len(ecg_qa_dict[key]) < 14:
            del ecg_qa_dict[key]
        elif qt == "single-query" and len(ecg_qa_dict[key]) < 50:
            del ecg_qa_dict[key]


def filter_by_question_frequency(ecg_qa_dict, min_question_count=10):
    """
    Filter the ECG QA dictionary by question frequency.
    
    Args:
        ecg_qa_dict (Dict): Dictionary to filter.
        min_question_count (int, optional): Minimum count for a question to be included. Default is 10.
        
    Returns:
        Dict: Filtered dictionary.
    """
    filtered_dict = {}
    
    for key, samples in ecg_qa_dict.items():
        # Count occurrences of each question_id
        question_id_counter = Counter(sample['question_id'] for sample in samples)
        
        # Get frequent question IDs
        frequent_question_ids = [q_id for q_id, count in question_id_counter.items() 
                                if count >= min_question_count]
        
        # Group samples by question_id
        samples_by_question = []
        for question_id in frequent_question_ids:
            question_samples = [sample for sample in samples if sample['question_id'] == question_id]
            samples_by_question.append(question_samples)
        
        # Only include keys with at least one question group
        if samples_by_question:
            filtered_dict[key] = samples_by_question
    
    return filtered_dict


def validate_path(path):
    """
    Validate if a path exists.
    
    Args:
        path (str): Path to validate.
        
    Returns:
        bool: True if path exists, False otherwise.
    """
    if not os.path.exists(path):
        return False
    return True
