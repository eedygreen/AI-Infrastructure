# TODO: 1 - import the OpenAI class from the openai library
import numpy as np
import pandas as pd
import re, os
import csv
import uuid
from datetime import datetime
from openai import OpenAI
from enum import Enum
from dotenv import load_dotenv
from utils import logger

load_dotenv()

BASE_URL = "https://openai.vocareum.com/v1"


class Model(str, Enum):
    GPT4="gpt-4"
    GPT3="gpt-3.5-turbo"
    Embeddings="text-embedding-3-large"


# === Utility Functions ===
def _client(api_key: str, base_url: str=BASE_URL):
    try:
        logger.info(f"[Accessing LLM] connecting to {BASE_URL}")
        client = OpenAI(
            base_url = base_url,
            api_key = api_key
        )
        logger.info("[Accessing LLM] connection Successfull!")
        return client
    except Exception as e:
        logger.error(f"[Accessing LLM] failed to connect with error: {e}")

def llm_call(
        system_prompt: str,
        user_prompt: str,
        client,
        model: str = Model.GPT3,
        temperature: float= 0.3,
    ) -> str:
    """Basic LLM call wrapper.
    
    Parameters:
        systtem_prompt (str): Instructions for the Agent and Agents role.
        user_prompt (str): User instructions as request to Agent.
        client (OpenAI): OpenAI API initialization request.
        model (str): The specific LLM Model.
        temperature (float): The number for controling Agent's behaviour.
    
    Returns:
        LLM reponse
    """
    try:
        logger.info("[LLM Chat] sending prompt...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        logger.info("[LLM Chat] chat completed!")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[LLM Chat Failure] error: {e}")

# DirectPromptAgent class definition
class DirectPromptAgent:
    """directly relays a user's input (prompt) to the LLM and \
        returns the LLM's response without incorporating additional context, memory, \
        or specialized tools.
    """
    def __init__(self, openai_api_key):
        """
        Parameters:
            openai_api_key (str): API key for accessing OpenAI. 
        """
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        """
        Parameters:
            prompt: user query

        Returns:
            LLM reponse  
        """
        client = _client(self.openai_api_key)
        try:
            response = client.chat.completions.create(
                model=Model.GPT3,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            logger.info("[Knowledge Source] DirectPromptAgent has no injected knowledge — "
            "this response is generated purely from the LLM's own training data.\n\n")

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[LLM Chat Failure] error: {e}")
        

# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    """Augmented Prompt Agent is a specialized agent designed to respond according \
    to a predefined persona.
    """
    def __init__(self, openai_api_key, persona):
        """
        Parameters:
            openai_api_key (str): API key for accessing OpenAI.
            persona (str): Persona description for the agent.
        """
        self.openai_api_key = openai_api_key
        self.persona = persona
        
    def respond(self, input_text):
        """Generate a response using OpenAI API.

        Parameters:
            input_text: user query
            
        Returns:
            LLM reponse
        """
        client = _client(api_key=self.openai_api_key)
        system_prompt = f"""
        Assume this persona: {self.persona}, and forget every previous context before now.
        You are {self.persona}.
        """
        response = llm_call(
            system_prompt=system_prompt,
            user_prompt=input_text,
            client=client,
            temperature=0
        )
        logger.info(f"[Knowledge Source] Using the LLM's own internal knowledge, filtered through persona: '{self.persona}'. "
          f"No external knowledge base provided — persona shapes tone/framing, not factual content.\n\n")

        return response


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    """
    An Agent that follows only the knowledge users want llm to utilize without using external reference.
    """
    def __init__(self, openai_api_key, persona, knowledge):
        """
        Initialize the agent with provided attributes.
        
        Parameters:
            openai_api_key (str): API key for accessing OpenAI.
            persona (str): Persona description for the agent.
            knowledge (str): For storing the Agent's specific knowledge.
        """
        self.persona = persona
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def respond(self, input_text):
        """Generate a response using the OpenAI API.

        Parameters:
            input_text: user query
            
        Returns:
            LLM reponse
        """
        system_message = (
            f"""You are {self.persona} knowledge-based assistant. Forget all previous context.
            Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}.
            Answer the prompt based on this knowledge, not your own."""
        )
        client = _client(api_key=self.openai_api_key)

        response = llm_call(
            system_prompt=system_message,
            user_prompt=input_text,
            client=client,
            temperature=0
        )
        return response


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
            openai_api_key (str): API key for accessing OpenAI.
            persona (str): Persona description for the agent.
            chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
            chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
            text (str): Text to embed.

        Returns:
            list: The embedding vector.
        """
        client = _client(api_key=self.openai_api_key)
        response = client.embeddings.create(
            model=Model.Embeddings,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two) -> float:
        """
        Calculates cosine similarity between two vectors.

        Parameters:
            vector_one (list): First embedding vector.
            vector_two (list): Second embedding vector.

        Returns:
            float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text) -> list[dict[str | None]]:
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
            text (str): Text to split into chunks.

        Returns:
            list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            # break the loop if we have reached the end of the text
            if end == len(text):
                break

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
            DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt) -> str:
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
            prompt (str): User input prompt.

        Returns:
            str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = _client(api_key=self.openai_api_key)
        system_prompt = f"You are {self.persona}, a knowledge-based assistant. Forget previous context."
        user_prompt = f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"
        response = llm_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            client=client,
            temperature=0
        )

        return response


class EvaluationAgent:
    """
    An agent that evaluate LLM responses and benchmark the response against the
    given criteria.
    """
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        """
        Initializes the EvaluationAgent.

        Parameters:
            openai_api_key (str): API key for accessing OpenAI.
            persona (str): Persona description for the agent.
            evaluation_criteria: Benchmark to score the LLM response.
            worker_agent: Other Agent
            max_interactions (int): Number of interactions before terminating loop
        """
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt, initial_response=None) -> dict:
        """ 
        This method manages interactions between agents to achieve a solution.
            
        Parameters:
            initial_prompt: the original prompt that produced `initial_response`
                (also used to rebuild context on correction rounds).
            initial_response: optional. If provided, this is judged directly on
                round 1 instead of calling worker_agent.respond(initial_prompt)
                again — avoids generating (and discarding) a second, different
                response than the one the caller already has.
        Returns:
            dict (str): Agent reponse, response evaluations, and iterations
        """
        client = _client(api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):
            logger.info(f"\n--- Interaction {i+1} ---")

            logger.info(" Step 1: Worker agent generates a response to the prompt\n")
            if i == 0 and initial_response is not None:
                response_from_worker = initial_response
                logger.info(f"(using pre-computed response)\n Worker Agent Resposnse:\n{response_from_worker}")
            else:
                logger.info(f"Prompt:\n{prompt_to_evaluate}")
                response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
                logger.info(f"Worker Agent Response:\n{response_from_worker}")

            logger.info(" Step 2: Evaluator agent judges the response\n")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            evaluation = llm_call(
                system_prompt=self.persona,
                user_prompt=eval_prompt,
                client=client,
                temperature=0
            )

            logger.info(f"Evaluator Agent Evaluation:\n{evaluation}")

            logger.info(" Step 3: Check if evaluation is positive\n")
            if evaluation.lower().startswith("yes"):
                logger.info("✅ Final solution accepted.")
                break
            else:
                logger.info(" Step 4: Generate instructions to correct the response\n")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                instructions = llm_call(
                    system_prompt=self.persona,
                    user_prompt=instruction_prompt,
                    client=client,
                    temperature=0
                )

                logger.info(f"Instructions to fix:\n{instructions}\n")

                logger.info(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iteration": i + 1,
        }   


class RoutingAgent():
    """
    An Agent that dynamically route request to corresponding agents based on cosine similarities.
    """
    def __init__(self, openai_api_key: str | None, agents: list[dict[str, any]]):
        """Initialize the agent with given attributes
        Parameters:
            openai_api_key (str): API key for accessing OpenAI.
            persona (str): Persona description for the agent.
        """
        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text):
        """
        Parameters:
            text: user's query

        Returns:
            embeddings: the embeddings for the user's query
        """
        client = _client(api_key=self.openai_api_key)
        try:
            logger.info("[Embeddings] creating embeddings...")
            response = client.embeddings.create(
                model=Model.Embeddings,
                input=text,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            logger.info("[Embeddings] Successfully created embeddings")
            return embedding
        except Exception as e:
            logger.error(f"[Embeddings] failed with error: {e}")
        
    def route(self, user_input):
        """
        Parameters:
            user_input: task (users' query)

        Returns:
            agent: The agent responsible for the task (user_input)
        """
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            logger.info(similarity)

            # logic to select the best agent based on the similarity score between the user prompt and the agent descriptions
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        logger.info(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:
    """
    An Agent that plan the actions to be taken on any prompt.
    """
    def __init__(self, openai_api_key: str, knowledge: str | None):
        """Initialize the agent with given attributes
        Parameters:
            openai_api_key (str): API key for accessing OpenAI.
            persona (str): Persona description for the agent.
        """
        self.openai_api_key=openai_api_key
        self.knowledge=knowledge

    def extract_steps_from_prompt(self, prompt):
        """
        Parameters:
            prompt (str): The task (user's query)

        Returns:
            steps: Sequention (execution) steps to solve the task 
        """
        client = _client(api_key=self.openai_api_key)
        logger.info("[Steps Extractions] extracting steps...")
        system_message = f"""
            You are an action planning agent. Using your knowledge, 
            you extract from the user prompt the steps requested to complete the action the user is asking for.
            You return the steps as a list. Only return the steps in your knowledge. Forget any previous context.
            This is your knowledge: {self.knowledge}
        """
        response_text = llm_call(
            system_prompt=system_message,
            user_prompt=prompt,
            client=client,
            temperature=0
        )
        steps = [s.strip() for s in response_text.split("\n") if s.strip()]
        logger.info("[Steps Extraction] extraction completed!")
        return steps
