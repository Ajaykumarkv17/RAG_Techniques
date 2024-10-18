"""
RAG Evaluation Script

This script evaluates the performance of a Retrieval-Augmented Generation (RAG) system
using various metrics from the deepeval library.

Dependencies:
- deepeval
- langchain_openai
- json

Custom modules:
- helper_functions (for RAG-specific operations)
"""
import os
import json
from typing import List, Tuple

from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("GITHUB_TOKEN")


from deepeval.models.base_model import DeepEvalBaseLLM

# 09/15/24 kimmeyh Added path where helper functions is located to the path
# Add the parent directory to the path since we work with notebooks
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from helper_functions import (
    create_question_answer_from_context_chain,
    answer_question_from_context,
    retrieve_context_per_question
)


class OpenAI_proxy(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Github Model"

# Replace these with real values
custom_model = ChatOpenAI(base_url="https://models.inference.ai.azure.com",api_key=api_key,model="gpt-4o-mini")
openai_p = OpenAI_proxy(model=custom_model)

def create_deep_eval_test_cases(
    questions: List[str],
    gt_answers: List[str],
    generated_answers: List[str],
    retrieved_documents: List[str]
) -> List[LLMTestCase]:
    """
    Create a list of LLMTestCase objects for evaluation.

    Args:
        questions (List[str]): List of input questions.
        gt_answers (List[str]): List of ground truth answers.
        generated_answers (List[str]): List of generated answers.
        retrieved_documents (List[str]): List of retrieved documents.

    Returns:
        List[LLMTestCase]: List of LLMTestCase objects.
    """
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]

# Define evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    model=openai_p,
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=openai_p,
    include_reason=False
)

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model=openai_p,
    include_reason=True
)

def evaluate_rag(chunks_query_retriever, num_questions: int = 5) -> None:
    """
    Evaluate the RAG system using predefined metrics.

    Args:
        chunks_query_retriever: Function to retrieve context chunks for a given query.
        num_questions (int): Number of questions to evaluate (default: 5).
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini",base_url="https://models.inference.ai.azure.com",api_key=api_key)
    question_answer_from_context_chain = create_question_answer_from_context_chain(llm)

    # Load questions and answers from JSON file
    q_a_file_name = "../data/q_a.json"
    with open(q_a_file_name, "r", encoding="utf-8") as json_file:
        q_a = json.load(json_file)

    questions = [qa["question"] for qa in q_a][:num_questions]
    ground_truth_answers = [qa["answer"] for qa in q_a][:num_questions]
    generated_answers = []
    retrieved_documents = []

    # Generate answers and retrieve documents for each question
    for question in questions:
        context = retrieve_context_per_question(question, chunks_query_retriever)
        retrieved_documents.append(context)
        context_string = " ".join(context)
        result = answer_question_from_context(question, context_string, question_answer_from_context_chain)
        generated_answers.append(result["answer"])

    # Create test cases and evaluate
    test_cases = create_deep_eval_test_cases(questions, ground_truth_answers, generated_answers, retrieved_documents)
    evaluate(
        test_cases=test_cases,
        metrics=[correctness_metric, faithfulness_metric, relevance_metric]
    )

if __name__ == "__main__":
    # Add any necessary setup or configuration here
    # Example: evaluate_rag(your_chunks_query_retriever_function)
    pass
