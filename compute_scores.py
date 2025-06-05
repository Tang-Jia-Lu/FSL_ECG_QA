import evaluate
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

def compute_bmr_metrics(true_captions, generated_captions):
  f1_scores = bertscore.compute(predictions=generated_captions, references=true_captions,
      lang="en", model_type='bert-base-uncased')["f1"]
  meteor_scores = meteor.compute(predictions=generated_captions, references=true_captions)['meteor']
  rouge_scores = rouge.compute(predictions=generated_captions, references=true_captions, rouge_types=["rougeL"])["rougeL"]
  return {"f1_bertscore": np.mean(f1_scores),
    "meteor": np.mean(meteor_scores),
    "rouge": np.mean(rouge_scores)
  }

def compute_bleu_metrics(true_captions, generated_captions):
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []

    smoother = SmoothingFunction().method1

    for true, generated in zip(true_captions, generated_captions):
        # Tokenize captions
        true_tokens = true.split()
        generated_tokens = generated.split()

        # Compute BLEU scores
        bleu1 = sentence_bleu([true_tokens], generated_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother)
        bleu2 = sentence_bleu([true_tokens], generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
        bleu3 = sentence_bleu([true_tokens], generated_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother)
        bleu4 = sentence_bleu([true_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)
        bleu4_scores.append(bleu4)

    # Calculate average scores
    avg_bleu1 = np.mean(bleu1_scores)
    avg_bleu2 = np.mean(bleu2_scores)
    avg_bleu3 = np.mean(bleu3_scores)
    avg_bleu4 = np.mean(bleu4_scores)

    return {
        "BLEU-1": avg_bleu1,
        "BLEU-2": avg_bleu2,
        "BLEU-3": avg_bleu3,
        "BLEU-4": avg_bleu4,
    }
    
if __name__ == '__main__':
    # Example true and generated captions
    true_captions = ["The cat is on the mat", "A dog is playing with a ball"]
    generated_captions = ["The cat is on the carpet", "The dog plays with a ball"]
    
    # Computing BMR metrics
    bmr_results = compute_bmr_metrics(true_captions, generated_captions)
    print("BMR Results:", bmr_results)
    
    # Computing BLEU metrics
    bleu_results = compute_bleu_metrics(true_captions, generated_captions)
    print("BLEU Results:", bleu_results)
    
    # pip install absl-py rouge-score
    # pip install bert_score
    # pip install tqdm
    # pip install nltk
    # pip install evaluate