"""
Modal-based entity extraction using vLLM on A100-80GB GPU.

Uses Qwen3-32B for extraction with thinking mode DISABLED for clean JSON output.

Usage:
    # First, export chunks from MongoDB to JSON
    python export_chunks.py
    
    # Run extraction on Modal
    modal run extract.py --input chunks.json --output extractions.json
    
    # For testing with a small batch
    modal run extract.py --input chunks.json --output extractions.json --max-chunks 100
"""

import modal
import json
from typing import Optional

# =============================================================================
# Modal App Definition
# =============================================================================

app = modal.App("erica-extraction")

# Docker image with vLLM and dependencies
# Need vLLM >= 0.8.5 for Qwen3 support
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.8.5",
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "huggingface_hub",
    )
)


# =============================================================================
# Extraction Prompt (System + User)
# =============================================================================

SYSTEM_PROMPT = """You are an expert at extracting AI/ML concepts from educational content.
You always respond with valid JSON only, no markdown, no explanations, no thinking."""

USER_PROMPT_TEMPLATE = """Extract AI/ML concepts and their relationships from this educational text.

For each concept, provide:
- title: The canonical name (e.g., "Gradient Descent")
- definition: A brief 1-2 sentence definition
- difficulty: "beginner", "intermediate", or "advanced"
- aliases: Alternative names as a list

For relations between concepts found:
- prereq_of: Concept A must be understood before Concept B
- is_a: Concept A is a type of Concept B (e.g., "CNN is_a Neural Network")
- part_of: Concept A is a component of Concept B
- contrasts_with: Concepts that are alternatives or opposites
- sibling: Concepts at the same level

IMPORTANT:
- Only extract concepts actually discussed in the text
- Use exact concept titles from your extracted concepts in relations
- Return ONLY valid JSON, no markdown code blocks, no explanations

Return format:
{{"concepts": [{{"title": "...", "definition": "...", "difficulty": "...", "aliases": []}}], "relations": [{{"source": "...", "target": "...", "relation_type": "..."}}]}}

If no AI/ML concepts found: {{"concepts": [], "relations": []}}

TEXT:
{text}"""


# =============================================================================
# vLLM Model Class
# =============================================================================

@app.cls(
    gpu=modal.gpu.A100(size="80GB"),
    image=vllm_image,
    timeout=3600,  # 1 hour max
    container_idle_timeout=300,  # Keep warm for 5 min
)
class Extractor:
    """vLLM-based extraction using Qwen3-32B with thinking disabled."""
    
    model_name: str = "Qwen/Qwen3-32B"
    
    @modal.enter()
    def load_model(self):
        """Load vLLM model on container start."""
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        
        print(f"Loading {self.model_name}...")
        
        # Load tokenizer for chat template
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        
        # Load vLLM engine
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        
        # Sampling params for non-thinking mode (lower temperature for structured output)
        self.sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.95,
            top_k=20,
            max_tokens=2048,
            stop=["```", "\n\n\n"],
        )
        print("Model loaded!")
    
    def _build_prompt(self, text: str) -> str:
        """Build chat prompt with thinking DISABLED."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}
        ]
        
        # Apply chat template with enable_thinking=False
        # This disables the <think>...</think> reasoning mode
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # CRITICAL: disable thinking for clean JSON
        )
        return prompt
    
    @modal.method()
    def extract_batch(self, chunks: list[dict]) -> list[dict]:
        """
        Extract concepts and relations from a batch of chunks.
        
        Args:
            chunks: List of chunk dicts with 'chunk_id', 'text', 'source_url'
        
        Returns:
            List of extraction results with same chunk_id
        """
        # Build prompts with chat template
        prompts = [
            self._build_prompt(chunk["text"][:3500])  # Leave room for prompt template
            for chunk in chunks
        ]
        
        # Batch inference
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        results = []
        for chunk, output in zip(chunks, outputs):
            result = {
                "chunk_id": chunk["chunk_id"],
                "source_url": chunk.get("source_url", ""),
                "concepts": [],
                "relations": [],
                "error": None,
            }
            
            try:
                raw_text = output.outputs[0].text.strip()
                
                # Remove any thinking tags if they somehow appeared
                if "<think>" in raw_text:
                    # Extract content after </think>
                    think_end = raw_text.find("</think>")
                    if think_end != -1:
                        raw_text = raw_text[think_end + 8:].strip()
                
                # Clean up common formatting issues
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()
                
                # Try to find JSON object in response
                if not raw_text.startswith("{"):
                    # Look for first { in response
                    json_start = raw_text.find("{")
                    if json_start != -1:
                        raw_text = raw_text[json_start:]
                
                # Find matching closing brace
                brace_count = 0
                json_end = 0
                for i, c in enumerate(raw_text):
                    if c == "{":
                        brace_count += 1
                    elif c == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > 0:
                    raw_text = raw_text[:json_end]
                
                parsed = json.loads(raw_text)
                result["concepts"] = parsed.get("concepts", [])
                result["relations"] = parsed.get("relations", [])
                result["raw_response"] = raw_text
                
            except json.JSONDecodeError as e:
                result["error"] = f"JSON parse error: {str(e)}"
                result["raw_response"] = output.outputs[0].text[:500]
            except Exception as e:
                result["error"] = str(e)
            
            results.append(result)
        
        return results


# =============================================================================
# Local Entry Point
# =============================================================================

@app.local_entrypoint()
def main(
    input: str = "chunks.json",
    output: str = "extractions.json",
    batch_size: int = 32,
    max_chunks: Optional[int] = None,
):
    """
    Run extraction on all chunks.
    
    Args:
        input: Path to JSON file with chunks
        output: Path to save extraction results
        batch_size: Number of chunks per batch (affects GPU memory)
        max_chunks: Limit number of chunks for testing
    """
    import time
    
    # Load chunks
    print(f"Loading chunks from {input}...")
    with open(input) as f:
        chunks = json.load(f)
    
    total_chunks = len(chunks)
    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f"Limited to {max_chunks} chunks (test mode)")
    
    print(f"Processing {len(chunks)} chunks in batches of {batch_size}...")
    print(f"Using Qwen3-32B with thinking DISABLED")
    
    # Initialize extractor
    extractor = Extractor()
    
    all_results = []
    start_time = time.time()
    
    # Process in batches
    num_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"  Batch {batch_num}/{num_batches} ({len(batch)} chunks)...")
        batch_start = time.time()
        
        results = extractor.extract_batch.remote(batch)
        all_results.extend(results)
        
        batch_time = time.time() - batch_start
        chunks_per_sec = len(batch) / batch_time if batch_time > 0 else 0
        print(f"    Done in {batch_time:.1f}s ({chunks_per_sec:.1f} chunks/sec)")
    
    # Save results
    print(f"\nSaving results to {output}...")
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    elapsed = time.time() - start_time
    n_concepts = sum(len(r.get("concepts", [])) for r in all_results)
    n_relations = sum(len(r.get("relations", [])) for r in all_results)
    n_errors = sum(1 for r in all_results if r.get("error"))
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Time:      {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Chunks:    {len(all_results)}")
    print(f"Concepts:  {n_concepts}")
    print(f"Relations: {n_relations}")
    print(f"Errors:    {n_errors} ({100*n_errors/len(all_results):.1f}%)")
    print(f"Output:    {output}")
    
    if n_errors > 0:
        print(f"\nSample errors:")
        error_count = 0
        for r in all_results:
            if r.get("error") and error_count < 5:
                print(f"  - {r['chunk_id']}: {r['error'][:80]}")
                error_count += 1
