import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.distance import cosine
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm
from google.api_core import retry
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Configure Google Generative AI with API key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=API_KEY)

@dataclass
class ModelConfig:
    rpm: int
    tpm: int
    purpose: str
    complexity_threshold: float

MODELS: Dict[str, ModelConfig] = {
    "gemini-1.5-flash": ModelConfig(rpm=15, tpm=1_000_000, purpose="balanced", complexity_threshold=0.3),
    "gemini-1.5-flash-8b": ModelConfig(rpm=4000, tpm=4_000_000, purpose="high-frequency", complexity_threshold=0.1),
    "gemini-1.5-pro": ModelConfig(rpm=2, tpm=32_000, purpose="complex", complexity_threshold=0.7),
}

MAX_TOKENS = 8192
PERFORMANCE_LOG = "model_performance.json"
KNOWLEDGE_BASE = "knowledge_base.json"
FEW_SHOT_EXAMPLES = "few_shot_examples.json"
SYNTHETIC_DATA = "synthetic_data.json"

class Geminio1:
    def __init__(self):
        self.load_performance_data()
        self.load_knowledge_base()
        self.load_few_shot_examples()
        self.load_synthetic_data()
        self.session_context = []
        self.initialize_faiss_index()
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.prompt_optimizations = {}

    def load_performance_data(self):
        try:
            with open(PERFORMANCE_LOG, 'r') as f:
                self.performance_data = json.load(f)
        except FileNotFoundError:
            self.performance_data = {model: {"success_rate": 0.5, "avg_tokens": MAX_TOKENS // 2} for model in MODELS}

    def save_performance_data(self):
        with open(PERFORMANCE_LOG, 'w') as f:
            json.dump(self.performance_data, f)

    def load_knowledge_base(self):
        try:
            with open(KNOWLEDGE_BASE, 'r') as f:
                self.knowledge_base = json.load(f)
        except FileNotFoundError:
            self.knowledge_base = {}

    def save_knowledge_base(self):
        with open(KNOWLEDGE_BASE, 'w') as f:
            json.dump(self.knowledge_base, f)

    def load_few_shot_examples(self):
        try:
            with open(FEW_SHOT_EXAMPLES, 'r') as f:
                self.few_shot_examples = json.load(f)
        except FileNotFoundError:
            self.few_shot_examples = {}

    def load_synthetic_data(self):
        try:
            with open(SYNTHETIC_DATA, 'r') as f:
                self.synthetic_data = json.load(f)
        except FileNotFoundError:
            self.synthetic_data = []

    def save_synthetic_data(self):
        with open(SYNTHETIC_DATA, 'w') as f:
            json.dump(self.synthetic_data, f)

    def initialize_faiss_index(self):
        self.embedding_dim = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.id_to_text = {}

    async def get_embedding(self, text: str) -> np.ndarray:
        return self.sentence_transformer.encode([text])[0]

    async def add_to_index(self, text: str, embedding: np.ndarray):
        idx = len(self.id_to_text)
        self.id_to_text[idx] = text
        self.index.add(embedding.reshape(1, -1))

    async def search_similar(self, query: str, k: int = 5) -> List[str]:
        query_embedding = await self.get_embedding(query)
        _, I = self.index.search(query_embedding.reshape(1, -1), k)
        return [self.id_to_text[i] for i in I[0] if i in self.id_to_text]

    @retry.AsyncRetry(predicate=retry.if_exception_type(Exception), initial=1, maximum=60, multiplier=2, timeout=120)
    async def call_gemini_api(self, prompt: str, model_name: str) -> Tuple[str, int]:
        try:
            generation_config = {
                "temperature": 0.9,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 2048,
            }

            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            model = genai.GenerativeModel(model_name, generation_config=generation_config, safety_settings=safety_settings)
            chat = model.start_chat(history=[])
            
            response = await chat.send_message_async(prompt)
            visible_tokens = len(response.text) if response and response.text else 0
            logging.debug(f"API Prompt: {prompt}")
            logging.debug(f"API Response: {response.text}")
            return response.text if response and response.text else "No output from API.", visible_tokens
        except Exception as e:
            logging.error(f"Error during API call to {model_name}: {str(e)}")
            raise

    async def select_model(self, query: str, complexity: float) -> str:
        model_scores = {}
        for model, config in MODELS.items():
            performance = self.performance_data[model]
            score = (
                (1 - abs(complexity - config.complexity_threshold)) * 0.5 +
                performance["success_rate"] * 0.3 +
                (1 - performance["avg_tokens"] / MAX_TOKENS) * 0.2
            )
            model_scores[model] = score
        
        return max(model_scores, key=model_scores.get)

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        analysis_prompt = f"""
        Analyze this query: '{query}'.
        Provide a JSON output with the following keys:
        - complexity (float 0-1)
        - category (string)
        - estimated_depth (int 1-5)
        - key_concepts (list of strings)
        - domain (string)
        - reasoning_strategy (string: 'analytical', 'creative', 'evaluative')
        """
        model_name = "gemini-1.5-flash"
        response, _ = await self.call_gemini_api(analysis_prompt, model_name)
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in the response")
        except (json.JSONDecodeError, ValueError):
            logging.error(f"Failed to parse JSON from analysis: {response}")
            return {
                "complexity": 0.5,
                "category": "general",
                "estimated_depth": 3,
                "key_concepts": [],
                "domain": "general",
                "reasoning_strategy": "analytical"
            }

    async def generate_meta_prompt(self, query: str, analysis: Dict[str, Any]) -> str:
        meta_prompt = f"""
        Given the following query and its analysis, generate a meta-prompt that will guide the reasoning process:
        
        Query: {query}
        Analysis: {json.dumps(analysis)}
        
        The meta-prompt should:
        1. Outline the overall strategy for answering the query.
        2. Suggest key points to consider or steps to follow.
        3. Indicate any specific knowledge domains to draw upon.
        4. Propose potential angles or perspectives to explore.
        5. Include guidance on how to structure the response.
        6. Suggest any relevant examples or analogies to use.
        
        Format the meta-prompt as a clear, step-by-step guide for the AI to follow when generating the final answer.
        """
        model_name = await self.select_model(query, analysis['complexity'])
        response, _ = await self.call_gemini_api(meta_prompt, model_name)
        return response

    async def retrieve_context(self, query: str, analysis: Dict[str, Any]) -> str:
        relevant_context = await self.search_similar(query)
        domain_examples = self.few_shot_examples.get(analysis['domain'], [])
        
        context_prompt = f"""
        Relevant context from previous interactions:
        {' '.join(relevant_context)}
        
        Domain-specific examples:
        {' '.join(domain_examples)}
        
        Session context:
        {' '.join(self.session_context[-5:])}  # Last 5 interactions
        """
        return context_prompt

    async def generate_response(self, query: str, meta_prompt: str, context: str) -> str:
        full_prompt = f"""
        Query: {query}
        
        Meta-Prompt (Strategy):
        {meta_prompt}
        
        Context:
        {context}
        
        Based on the above information, provide a comprehensive answer to the query.
        Follow the strategy outlined in the meta-prompt and incorporate relevant context where appropriate.
        Structure your response clearly, using headings and bullet points where relevant.
        """
        model_name = await self.select_model(query, 0.8)  # Use a high complexity for the main response
        response, _ = await self.call_gemini_api(full_prompt, model_name)
        return response

    async def self_critique_and_improve(self, query: str, initial_response: str) -> str:
        critique_prompt = f"""
        Original Query: {query}
        
        Initial Response:
        {initial_response}
        
        Please critique the above response based on the following criteria:
        1. Accuracy and factual correctness
        2. Completeness in addressing all aspects of the query
        3. Clarity and coherence of explanation
        4. Logical flow and reasoning
        5. Appropriate use of context and background knowledge
        6. Relevance to the original query
        7. Depth of analysis
        
        Provide a detailed critique highlighting strengths and areas for improvement.
        Format your critique as a structured list of points.
        """
        
        model_name = await self.select_model(query, 0.7)
        critique, _ = await self.call_gemini_api(critique_prompt, model_name)
        
        improvement_prompt = f"""
        Original Query: {query}
        
        Initial Response:
        {initial_response}
        
        Critique:
        {critique}
        
        Based on the above critique, please provide an improved and refined answer to the original query.
        Address the weaknesses identified in the critique while maintaining the strengths of the initial response.
        Ensure that your improved response is:
        1. More accurate and factually correct
        2. More comprehensive in addressing all aspects of the query
        3. Clearer and more coherent in its explanation
        4. More logical in its flow and reasoning
        5. Making better use of context and background knowledge
        6. More relevant to the original query
        7. Providing a deeper analysis
        
        Structure your improved response clearly, using headings and bullet points where appropriate.
        """
        
        improved_response, _ = await self.call_gemini_api(improvement_prompt, model_name)
        return improved_response

    async def evaluate_response(self, query: str, response: str) -> float:
        evaluation_prompt = f"""
        Evaluate the following response to the query: '{query}'
        
        Response: {response}
        
        Provide a score from 0 to 1, where 1 is the best possible answer and 0 is completely irrelevant or incorrect.
        Consider the following factors in your evaluation:
        1. Accuracy and factual correctness
        2. Completeness in addressing all aspects of the query
        3. Clarity and coherence of explanation
        4. Logical flow and reasoning
        5. Appropriate use of context and background knowledge
        6. Relevance to the original query
        7. Depth of analysis
        8. Overall quality and usefulness of the response

        Return only the numeric score, rounded to two decimal places.
        """
        model_name = "gemini-1.5-flash"  # Use a fast model for evaluation
        eval_response, _ = await self.call_gemini_api(evaluation_prompt, model_name)
        try:
            return round(float(eval_response), 2)
        except ValueError:
            logging.error(f"Failed to parse evaluation score: {eval_response}")
            return 0.5

    async def update_performance(self, model: str, success_rate: float, tokens_used: int):
        current = self.performance_data[model]
        current["success_rate"] = round((current["success_rate"] * 0.9) + (success_rate * 0.1), 3)
        current["avg_tokens"] = int((current["avg_tokens"] * 0.9) + (tokens_used * 0.1))
        self.save_performance_data()

    async def update_knowledge_base(self, query: str, response: str, score: float):
        if score > 0.8:  # Only store high-quality responses
            embedding = await self.get_embedding(query)
            await self.add_to_index(response, embedding)
            self.knowledge_base[query] = {"response": response, "score": score}
            self.save_knowledge_base()

    async def generate_synthetic_data(self, query: str, response: str, score: float):
        if score > 0.9:  # Only use very high-quality responses for synthetic data
            synthetic_prompt = f"""
            Based on the following high-quality query-response pair, generate 3 new, similar but distinct query-response pairs.
            Ensure that the new pairs maintain the high quality and depth of the original.

            Original Query: {query}
            Original Response: {response}

            For each new pair, provide the following in JSON format:
            {{
                "query": "New query here",
                "response": "New response here"
            }}

            Generate 3 such JSON objects.
            """
            model_name = await self.select_model(query, 0.8)
            synthetic_data_str, _ = await self.call_gemini_api(synthetic_prompt, model_name)
            
            try:
                # Extract JSON objects from the response
                json_objects = re.findall(r'\{[^}]+\}', synthetic_data_str)
                synthetic_data = [json.loads(obj) for obj in json_objects]
                self.synthetic_data.extend(synthetic_data)
                self.save_synthetic_data()
            except json.JSONDecodeError:
                logging.error(f"Failed to parse synthetic data: {synthetic_data_str}")

    async def optimize_prompts(self):
        # Cluster similar queries
        embeddings = np.array([await self.get_embedding(query) for query in self.knowledge_base.keys()])
        kmeans = KMeans(n_clusters=min(5, len(self.knowledge_base)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        for cluster_id in range(kmeans.n_clusters):
            cluster_queries = [query for query, cluster in zip(self.knowledge_base.keys(), clusters) if cluster == cluster_id]
            if cluster_queries:
                representative_query = cluster_queries[0]
                cluster_responses = [self.knowledge_base[query]['response'] for query in cluster_queries]
                
                optimization_prompt = f"""
                Analyze the following set of similar queries and their responses:

                {json.dumps(dict(zip(cluster_queries, cluster_responses)), indent=2)}

                Based on this analysis:
                1. Identify common patterns in successful responses.
                2. Suggest an optimized prompt structure that could work well for this type of query.
                3. Propose key elements or phrases that should be included in prompts for such queries.
                4. Recommend any specific techniques (e.g., chain-of-thought, analogies) that seem effective.

                Provide your recommendations in a structured JSON format with the following keys:
                - optimized_prompt_template
                - key_elements
                - recommended_techniques

                Ensure the JSON is valid and can be parsed.
                """
                
                model_name = await self.select_model(representative_query, 0.7)
                optimization_result, _ = await self.call_gemini_api(optimization_prompt, model_name)
                
                try:
                    # Extract JSON from the response
                    json_start = optimization_result.find('{')
                    json_end = optimization_result.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = optimization_result[json_start:json_end]
                        optimization_data = json.loads(json_str)
                        self.prompt_optimizations[cluster_id] = optimization_data
                    else:
                        raise ValueError("No valid JSON found in the response")
                except (json.JSONDecodeError, ValueError):
                    logging.error(f"Failed to parse prompt optimization data: {optimization_result}")

        # Save optimizations
        with open('prompt_optimizations.json', 'w') as f:
            json.dump(self.prompt_optimizations, f, indent=2)

    async def get_optimized_prompt(self, query: str, analysis: Dict[str, Any]) -> str:
        query_embedding = await self.get_embedding(query)
        _, I = self.index.search(query_embedding.reshape(1, -1), 1)
        nearest_cluster = I[0][0]
        
        if nearest_cluster in self.prompt_optimizations:
            optimization = self.prompt_optimizations[nearest_cluster]
            optimized_prompt = optimization['optimized_prompt_template'].format(
                query=query,
                **analysis
            )
            return optimized_prompt
        else:
            return None  # Fall back to default prompt if no optimization is available

    async def autonomous_reasoning(self, query: str) -> str:
        try:
            analysis = await self.analyze_query(query)
            
            # Try to get an optimized prompt, fall back to generating a new one if not available
            optimized_prompt = await self.get_optimized_prompt(query, analysis)
            if optimized_prompt:
                meta_prompt = optimized_prompt
            else:
                meta_prompt = await self.generate_meta_prompt(query, analysis)
            
            context = await self.retrieve_context(query, analysis)
            
            initial_response = await self.generate_response(query, meta_prompt, context)
            improved_response = await self.self_critique_and_improve(query, initial_response)
            
            final_answer = improved_response
            
            # Generate a concise response
            concise_response_prompt = f"""
            Based on the following analysis, provide a single, concise response to the original query: "{query}"
            The response should be no longer than one or two sentences.

            Analysis:
            {final_answer}

            Concise response:
            """
            concise_response, _ = await self.call_gemini_api(concise_response_prompt, "gemini-1.5-flash")
            
            # Combine concise response with detailed analysis
            combined_response = f"Concise response: {concise_response.strip()}\n\nDetailed analysis:\n{final_answer}"
            
            evaluation_score = await self.evaluate_response(query, combined_response)
            
            avg_model_name = await self.select_model(query, analysis['complexity'])
            await self.update_performance(avg_model_name, evaluation_score, len(combined_response))
            
            await self.update_knowledge_base(query, combined_response, evaluation_score)
            await self.generate_synthetic_data(query, combined_response, evaluation_score)
            
            self.session_context.append(f"Q: {query}\nA: {combined_response}")
            
            # Periodically optimize prompts (e.g., every 10 queries)
            if len(self.knowledge_base) % 10 == 0:
                await self.optimize_prompts()
            
            return combined_response
        except Exception as e:
            logging.error(f"Error in autonomous_reasoning: {str(e)}")
            return "I apologize, but I'm having trouble processing your request at the moment. Could you please try asking something else?"

    async def run_interactive_mode(self):
        print("Welcome to the Gemini o1!")
        print("Type 'exit' to quit the system.")
        
        while True:
            query = input("\nEnter your question: ")
            if query.lower() == 'exit':
                print("Thank you for using the Gemini o1. Goodbye!")
                break
            
            print("\nProcessing your query...")
            response = await self.autonomous_reasoning(query)
            print("\nResponse:")
            print(response)
            print("\n" + "="*50)

async def main():
    system = Geminio1()
    await system.run_interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())