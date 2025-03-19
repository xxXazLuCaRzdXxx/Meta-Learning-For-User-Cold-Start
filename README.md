# Meta-Learning for Cold-Start Recommendations

This repository demonstrates a meta-learning approach for improving recommendations in cold-start scenarios. We use a matrix factorization model enhanced with a MAML-style (Model-Agnostic Meta-Learning) adaptation process so that the system can quickly personalize recommendations for new users with only a few ratings.

## Overview

Cold-start problems occur when a recommender system has very limited data for new users. Our solution combines:
- **Matrix Factorization:** A classic approach that represents users and items as vectors.
- **Meta-Learning:** We use a meta-training process to learn an initialization that can be quickly adapted to a new user’s limited data.

In practice, the meta-learning strategy helps the model adjust its predictions after seeing just a few ratings (support set) before evaluating on new ones (query set).

## Project Structure

- **Preprocessing:**  
  The code ensures that your dataset (e.g., MovieLens) is ready for training. If the DataFrame lacks `user_idx` or `movie_idx`, these are created from the original `UserID` and `MovieID` columns.

- **Model Definition:**  
  A simple matrix factorization model is implemented using PyTorch. The model learns separate embeddings for users and movies and computes the predicted rating as the dot product of these embeddings.

- **Meta-Learning Task Setup:**  
  A custom dataset class (`UserTaskDataset`) groups ratings by user and splits them into a support set (for adaptation) and a query set (for evaluation).

- **Meta-Training Loop:**  
  Using the `higher` library, we perform inner-loop updates (i.e., task-specific adaptation) on the support sets. The meta-training loop updates the model’s global parameters based on performance on the query sets across multiple tasks.

- **Evaluation:**  
  The evaluation is done in three parts:
  - **Baseline Evaluation:** The meta-trained model is applied directly (without additional adaptation) to predict ratings.
  - **Meta Evaluation:** For each user, the model adapts using their support set and then predicts on the query set.
  - **Direct Comparison:** Both predictions (baseline and meta-adapted) are computed on the same query set, allowing a fair comparison of the impact of meta-adaptation.

## Requirements

- Python 3.x
- PyTorch
- higher
- Pandas
- NumPy
- scikit-learn

Install the required dependencies using:

```bash
pip install torch higher pandas numpy scikit-learn
