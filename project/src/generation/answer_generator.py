"""
Answer generation using retrieved subgraph context.

Takes a RetrievalResult and generates a scaffolded answer with citations.

Usage:
    from src.generation.answer_generator import AnswerGenerator
    
    generator = AnswerGenerator()
    answer = generator.generate(retrieval_result)
"""

import os
from typing import Optional
from openai import OpenAI

from src.retrieval.hybrid_retriever import RetrievalResult


class AnswerGenerator:
    """
    Generates answers using Qwen via OpenRouter.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen/qwen-2.5-72b-instruct",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable."
            )
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
    
    def generate(
        self,
        retrieval_result: RetrievalResult,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate an answer based on retrieved context.
        
        Args:
            retrieval_result: Result from HybridRetriever
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated answer with citations
        """
        # Build the context prompt
        context = self._build_context(retrieval_result)
        
        # System prompt for tutoring
        system_prompt = """You are Erica, an AI tutor for an Introduction to AI course.

Your role is to explain concepts clearly, building from foundational ideas to more complex ones.

Guidelines:
1. Start with prerequisites before explaining the main concept
2. Use the provided examples to illustrate ideas
3. Cite resources when referencing specific information (use [Resource: URL] format)
4. Keep explanations clear and accessible for students
5. If examples include code or math, explain them step by step
6. Connect related concepts to build a complete picture

Always be encouraging and supportive of student learning."""

        # User prompt with query and context
        user_prompt = f"""Student Question: {retrieval_result.query}

{context}

Please answer the student's question using the knowledge graph context above. 
Start with foundational concepts and build up to the main topic.
Include relevant examples and cite resources where appropriate."""

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content
    
    def _build_context(self, result: RetrievalResult) -> str:
        """Build context string from retrieval result."""
        sections = []
        
        # Section 1: Concepts in topological order
        sections.append("## Relevant Concepts (ordered from foundational to advanced)")
        
        # Create a lookup for concept details
        concept_lookup = {c.title: c for c in result.subgraph.concepts}
        
        for i, title in enumerate(result.ordered_concepts, 1):
            concept = concept_lookup.get(title)
            if concept:
                difficulty = concept.difficulty or "unknown"
                definition = concept.definition or "No definition available."
                sections.append(f"\n### {i}. {title} [{difficulty}]")
                sections.append(definition)
                
                # Add relationship info
                if concept.relation_to_seed != "seed":
                    sections.append(f"*(Relationship: {concept.relation_to_seed} of {concept.seed_concept})*")
        
        # Section 2: Examples
        if result.subgraph.examples:
            sections.append("\n## Examples")
            
            # Group examples by concept
            examples_by_concept = {}
            for ex in result.subgraph.examples:
                if ex.concept not in examples_by_concept:
                    examples_by_concept[ex.concept] = []
                examples_by_concept[ex.concept].append(ex)
            
            for concept_title in result.ordered_concepts:
                if concept_title in examples_by_concept:
                    sections.append(f"\n### Examples for {concept_title}")
                    for ex in examples_by_concept[concept_title]:
                        sections.append(f"- [{ex.example_type}] {ex.text}")
        
        # Section 3: Resources
        if result.subgraph.resources:
            sections.append("\n## Resources")
            
            # Group by type
            by_type = {}
            for r in result.subgraph.resources:
                rtype = r.resource_type or "other"
                if rtype not in by_type:
                    by_type[rtype] = []
                by_type[rtype].append(r)
            
            for rtype, resources in by_type.items():
                sections.append(f"\n### {rtype.upper()} Resources")
                for r in resources[:5]:  # Limit per type
                    concepts_str = ", ".join(r.concepts_explained[:3])
                    sections.append(f"- {r.url}")
                    sections.append(f"  Explains: {concepts_str}")
                    if r.page_numbers:
                        sections.append(f"  Pages: {r.page_numbers}")
                    if r.timecodes:
                        sections.append(f"  Time: {r.timecodes['start']}s - {r.timecodes['end']}s")
        
        # Section 4: Prerequisite chains
        if result.subgraph.prereq_chain:
            sections.append("\n## Learning Path")
            for chain in result.subgraph.prereq_chain:
                if len(chain) > 1:
                    sections.append(f"- {' → '.join(chain)}")
        
        return "\n".join(sections)
    
    def generate_simple(
        self,
        query: str,
        concepts: list[dict],
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a simple answer without full retrieval result.
        
        Useful for quick responses with just concept matches.
        """
        context = "## Relevant Concepts\n"
        for c in concepts:
            context += f"\n### {c['title']} [{c.get('difficulty', 'unknown')}]\n"
            context += c.get('definition', 'No definition.') + "\n"
        
        system_prompt = """You are Erica, an AI tutor. Explain concepts clearly and helpfully."""
        
        user_prompt = f"""Question: {query}

{context}

Provide a clear, educational answer based on these concepts."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        
        return response.choices[0].message.content