#!/usr/bin/env python3
"""
KLI Knowledge Component Extraction Pipeline
Extracts Knowledge Components from Elasticsearch using LLM analysis
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

from elasticsearch import Elasticsearch
from llama_index.core import StorageContext
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeComponent:
    id: str
    name: str
    description: str
    knowledge_type: str  # Factual, Procedural, Conceptual, Strategic
    primary_learning_process: str  # Memory & Fluency, Induction & Refinement, Understanding & Sense-Making
    difficulty_level: int  # 1-5
    content_chunks: List[str]
    parent_chunk_id: str
    key_concepts: List[str]
    assessment_criteria: List[str]
    estimated_time_minutes: int

@dataclass
class LearningObjective:
    id: str
    title: str
    description: str
    knowledge_components: List[str]
    dominant_knowledge_type: str
    learning_processes: List[str]
    recommended_strategies: List[str]
    chapter: str
    section: str
    difficulty_level: int
    estimated_total_time: int

@dataclass
class PrerequisiteRelationship:
    prerequisite_kc: str
    target_kc: str
    relationship_type: str  # hard, soft
    strength: float  # 0.1-1.0
    reasoning: str

class KLIExtractionPipeline:
    def __init__(self, es_config: str, neo4j_config: Dict, storage_dir: str):
        """Initialize the extraction pipeline"""
        self.es_client = Elasticsearch([es_config])  # es_config is now a URL string
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_config['uri'], 
            auth=(neo4j_config['user'], neo4j_config['password'])
        )
        self.storage_dir = storage_dir
        
        # Load storage context to access hierarchical structure
        self.storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        self.docstore = self.storage_context.docstore
        
    def get_parent_nodes_with_children(self) -> List[Dict]:
        """Get parent nodes with their child content for analysis"""
        logger.info("Extracting parent nodes with hierarchical structure...")
        
        parent_data = []
        all_nodes = list(self.docstore.docs.values())
        
        for node in all_nodes:
            if hasattr(node, 'child_nodes') and node.child_nodes:
                # This is a parent node
                parent_info = {
                    'node_id': node.node_id,
                    'content': node.get_content(),
                    'metadata': getattr(node, 'metadata', {}),
                    'child_contents': []
                }
                
                # Get child node contents
                for child_node_info in node.child_nodes:
                    # Extract the actual node ID from RelatedNodeInfo
                    child_id = child_node_info.node_id
                    if child_id in self.docstore.docs:
                        child_node = self.docstore.docs[child_id]
                        parent_info['child_contents'].append({
                            'child_id': child_id,
                            'content': child_node.get_content(),
                            'metadata': getattr(child_node, 'metadata', {})
                        })
                
                parent_data.append(parent_info)
        
        logger.info(f"Found {len(parent_data)} parent nodes with children")
        return parent_data
    
    def extract_knowledge_components_from_content(self, parent_data: Dict) -> List[KnowledgeComponent]:
        """Extract KCs from a parent node and its children using LLM"""
        logger.info(f"Analyzing content for node {parent_data['node_id']} with LLM")
        
        # Use LLM for proper KLI mapping
        return self.extract_kcs_with_llm(parent_data)
    
    def _create_kli_framework_context(self) -> str:
        """Create detailed KLI framework context for the LLM"""
        return """
KNOWLEDGE-LEARNING-INSTRUCTION (KLI) FRAMEWORK CONTEXT:

The KLI framework uses three interconnected taxonomies:

1. KNOWLEDGE TAXONOMY (What students need to know):
   - Factual Knowledge: Discrete facts, definitions, names, dates, symbols
     * Examples: "CPU stands for Central Processing Unit", "Process ID is a unique identifier"
   - Procedural Knowledge: Step-by-step processes, algorithms, how-to sequences  
     * Examples: "Steps to implement round-robin scheduling", "How to calculate response time"
   - Conceptual Knowledge: Principles, theories, models, relationships between ideas
     * Examples: "Understanding process lifecycle", "How scheduling affects system performance"
   - Strategic Knowledge: When/why to apply knowledge, problem-solving strategies
     * Examples: "Choosing appropriate scheduling algorithm", "Debugging deadlock situations"

2. LEARNING PROCESSES (How students acquire knowledge):
   - Memory & Fluency: Strengthening recall, building automatic responses
     * Best for: Factual knowledge, basic procedures
   - Induction & Refinement: Pattern discovery, generalization, skill improvement
     * Best for: Procedural knowledge, concept formation
   - Understanding & Sense-Making: Building mental models, explaining "why"
     * Best for: Conceptual and Strategic knowledge

3. COGNITIVE COMPLEXITY PROGRESSION:
   - Factual → Procedural → Conceptual → Strategic (increasing complexity)
   - Lower difficulty levels before higher difficulty levels
   - Foundation concepts before advanced applications

ASSESSMENT CRITERIA GUIDELINES:
- Factual: Can recall, recognize, define
- Procedural: Can execute steps, apply procedure
- Conceptual: Can explain, compare, analyze relationships
- Strategic: Can choose appropriate approach, justify decisions
"""

    def _create_kc_extraction_prompt(self, parent_content: str, child_contents: List[str]) -> str:
        """Create detailed prompt for KC extraction based on KLI framework"""
        
        child_text = "\n---\n".join(child_contents)
        kli_context = self._create_kli_framework_context()
        
        return f"""
{kli_context}

TASK: Analyze this Operating Systems educational content and extract Knowledge Components (KCs) following KLI principles.

PARENT SECTION CONTENT:
{parent_content[:1500]}

DETAILED CHILD SECTIONS:
{child_text[:4000]}

ANALYSIS INSTRUCTIONS:
1. Identify discrete, teachable knowledge units that can be independently mastered
2. For EACH KC, determine the Knowledge Type using these criteria:
   - Factual: If it's a definition, fact, or piece of information to memorize
   - Procedural: If it's a sequence of steps or algorithm to follow
   - Conceptual: If it's understanding principles, relationships, or "why" something works
   - Strategic: If it's about when/how to choose between alternatives or solve problems

3. For EACH KC, determine the Primary Learning Process:
   - Memory & Fluency: If students mainly need to memorize and recall quickly
   - Induction & Refinement: If students need to discover patterns or improve through practice
   - Understanding & Sense-Making: If students need to build deep comprehension and mental models

4. Set difficulty (1-5) based on cognitive complexity and prerequisites needed

OUTPUT FORMAT - Return ONLY valid JSON array, no other text:
[
  {{
    "name": "Specific, clear KC name (e.g., 'Process State Transitions')",
    "description": "What students will know/understand after mastering this KC",
    "knowledge_type": "Factual|Procedural|Conceptual|Strategic",
    "primary_learning_process": "Memory & Fluency|Induction & Refinement|Understanding & Sense-Making",
    "difficulty_level": 1-5,
    "key_concepts": ["concept1", "concept2", "concept3"],
    "assessment_criteria": ["How to measure mastery criterion 1", "criterion 2"],
    "estimated_time_minutes": 15-90
  }}
]

REQUIREMENTS:
- Extract 2-8 KCs per content section
- Ensure each KC is independently teachable and assessable
- Focus on Operating Systems domain knowledge
- Be specific and actionable in descriptions and criteria
"""

    def call_llm_for_kc_extraction(self, prompt: str) -> List[Dict]:
        """Call local Ollama LLM to extract Knowledge Components"""
        from llama_index.llms.ollama import Ollama
        
        try:
            llm = Ollama(model="qwen3:4b", request_timeout=300.0)  # Fixed model name
            response = llm.complete(prompt)
            
            # Parse JSON response from LLM
            response_text = response.text.strip()
            
            # Handle case where LLM might wrap JSON in markdown
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Parse the JSON
            kc_data = json.loads(response_text)
            
            # Ensure it's a list
            if isinstance(kc_data, dict):
                kc_data = [kc_data]
            
            logger.info(f"LLM extracted {len(kc_data)} Knowledge Components")
            return kc_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.error(f"Raw response: {response.text}")
            return []
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return []
    
    def extract_kcs_with_llm(self, parent_data: Dict) -> List[KnowledgeComponent]:
        """Extract KCs using LLM analysis with proper KLI mapping"""
        
        # Create the extraction prompt
        extraction_prompt = self._create_kc_extraction_prompt(
            parent_data['content'], 
            [child['content'] for child in parent_data['child_contents']]
        )
        
        # Call LLM for analysis
        try:
            llm_response = self.call_llm_for_kc_extraction(extraction_prompt)
            
            # Convert LLM response to KnowledgeComponent objects
            kcs = []
            for i, kc_data in enumerate(llm_response):
                kc = KnowledgeComponent(
                    id=f"KC_{parent_data['node_id']}_{i+1:03d}",
                    name=kc_data['name'],
                    description=kc_data['description'],
                    knowledge_type=kc_data['knowledge_type'],
                    primary_learning_process=kc_data['primary_learning_process'],
                    difficulty_level=kc_data['difficulty_level'],
                    content_chunks=[child['child_id'] for child in parent_data['child_contents']],
                    parent_chunk_id=parent_data['node_id'],
                    key_concepts=kc_data['key_concepts'],
                    assessment_criteria=kc_data['assessment_criteria'],
                    estimated_time_minutes=kc_data['estimated_time_minutes']
                )
                kcs.append(kc)
            
            logger.info(f"LLM extracted {len(kcs)} KCs from node {parent_data['node_id']}")
            return kcs
            
        except Exception as e:
            logger.error(f"LLM extraction failed for node {parent_data['node_id']}: {e}")
            return []
    
    def call_llm_for_prerequisites(self, prompt: str) -> List[Dict]:
        """Call local Ollama LLM to identify prerequisite relationships"""
        from llama_index.llms.ollama import Ollama
        
        try:
            llm = Ollama(model="qwen3:4b", request_timeout=300.0)  # Fixed model name
            response = llm.complete(prompt)
            
            # Parse JSON response from LLM
            response_text = response.text.strip()
            
            # Handle case where LLM might wrap JSON in markdown
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Parse the JSON
            rel_data = json.loads(response_text)
            
            # Ensure it's a list
            if isinstance(rel_data, dict):
                rel_data = [rel_data]
            
            logger.info(f"LLM identified {len(rel_data)} prerequisite relationships")
            return rel_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.error(f"Raw response: {response.text}")
            return []
        except Exception as e:
            logger.error(f"LLM prerequisite call failed: {e}")
            return []
    
    def identify_prerequisite_relationships(self, all_kcs: List[KnowledgeComponent]) -> List[PrerequisiteRelationship]:
        """Identify prerequisite relationships between KCs using LLM analysis"""
        
        # Create prompt for relationship analysis
        kc_summary = "\n".join([
            f"- {kc.id}: {kc.name} ({kc.knowledge_type}, {kc.primary_learning_process}, Level {kc.difficulty_level})"
            for kc in all_kcs
        ])
        
        relationship_prompt = f"""
{self._create_kli_framework_context()}

TASK: Analyze these Operating Systems Knowledge Components and identify prerequisite relationships using KLI framework principles.

KNOWLEDGE COMPONENTS:
{kc_summary}

PREREQUISITE ANALYSIS RULES (KLI-based):
1. KNOWLEDGE TYPE PROGRESSION:
   - Factual knowledge typically comes before Procedural
   - Procedural knowledge typically comes before Conceptual  
   - Conceptual knowledge typically comes before Strategic

2. LEARNING PROCESS COMPLEXITY:
   - Memory & Fluency processes before Understanding & Sense-Making
   - Simple recall before complex reasoning

3. COGNITIVE COMPLEXITY:
   - Lower difficulty levels before higher difficulty levels
   - Foundation concepts before advanced applications

4. DOMAIN-SPECIFIC OS KNOWLEDGE FLOW:
   - Basic computer architecture → Process concepts → Scheduling algorithms
   - Basic memory concepts → Virtual memory → Memory management strategies
   - Simple synchronization → Complex coordination mechanisms

RELATIONSHIP TYPES:
- hard: Must master prerequisite before target (strength 0.7-1.0)
  * Example: Must understand "Process States" before "Process Scheduling"
- soft: Helpful background knowledge (strength 0.3-0.6)  
  * Example: "Computer Architecture" helps with "Memory Management" but isn't strictly required

ANALYSIS STEPS:
1. Compare knowledge_type progression (Factual → Procedural → Conceptual → Strategic)
2. Compare learning_process complexity
3. Compare difficulty_level values
4. Consider OS domain conceptual dependencies
5. Assign appropriate relationship_type and strength

OUTPUT FORMAT - Return ONLY valid JSON array:
[
  {{
    "prerequisite_kc": "KC_ID", 
    "target_kc": "KC_ID",
    "relationship_type": "hard|soft",
    "strength": 0.1-1.0,
    "reasoning": "KLI-based explanation: why this prerequisite is necessary based on knowledge type, learning process, and domain concepts"
  }}
]

REQUIREMENTS:
- Only include relationships where prerequisite genuinely aids target learning
- Provide clear KLI-based reasoning for each relationship
- Ensure KC IDs match exactly from the provided list
"""
        
        try:
            llm_response = self.call_llm_for_prerequisites(relationship_prompt)
            
            relationships = []
            for rel_data in llm_response:
                # Validate that KCs exist
                prereq_exists = any(kc.id == rel_data['prerequisite_kc'] for kc in all_kcs)
                target_exists = any(kc.id == rel_data['target_kc'] for kc in all_kcs)
                
                if prereq_exists and target_exists:
                    rel = PrerequisiteRelationship(
                        prerequisite_kc=rel_data['prerequisite_kc'],
                        target_kc=rel_data['target_kc'],
                        relationship_type=rel_data['relationship_type'],
                        strength=rel_data['strength'],
                        reasoning=rel_data['reasoning']
                    )
                    relationships.append(rel)
                else:
                    logger.warning(f"Skipping invalid relationship: {rel_data}")
            
            logger.info(f"LLM identified {len(relationships)} valid prerequisite relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"LLM prerequisite analysis failed: {e}")
            return []
    
    def create_learning_objectives(self, kcs_by_section: Dict[str, List[KnowledgeComponent]]) -> List[LearningObjective]:
        """Create learning objectives that group related KCs"""
        
        learning_objectives = []
        
        for section, kcs in kcs_by_section.items():
            if not kcs:
                continue
                
            # Group KCs into learning objectives
            lo = LearningObjective(
                id=f"LO_{section}",
                title=f"Master {section} Concepts",
                description=f"Students will understand key concepts in {section}",
                knowledge_components=[kc.id for kc in kcs],
                dominant_knowledge_type=self._find_dominant_type(kcs),
                learning_processes=list(set(kc.primary_learning_process for kc in kcs)),
                recommended_strategies=self._recommend_strategies(kcs),
                chapter=section,
                section=section,
                difficulty_level=max(kc.difficulty_level for kc in kcs),
                estimated_total_time=sum(kc.estimated_time_minutes for kc in kcs)
            )
            
            learning_objectives.append(lo)
        
        return learning_objectives
    
    def _find_dominant_type(self, kcs: List[KnowledgeComponent]) -> str:
        """Find the most common knowledge type in a group of KCs"""
        if not kcs:
            return "Conceptual"  # Default fallback
            
        type_counts = {}
        for kc in kcs:
            type_counts[kc.knowledge_type] = type_counts.get(kc.knowledge_type, 0) + 1
        return max(type_counts, key=type_counts.get)
    
    def _recommend_strategies(self, kcs: List[KnowledgeComponent]) -> List[str]:
        """Recommend instructional strategies based on KC types and learning processes"""
        if not kcs:
            return []
            
        strategies = set()
        
        for kc in kcs:
            if kc.knowledge_type == "Factual":
                strategies.update(["spaced practice", "retrieval practice", "flashcards"])
            elif kc.knowledge_type == "Procedural":
                strategies.update(["worked examples", "deliberate practice", "scaffolding"])
            elif kc.knowledge_type == "Conceptual":
                strategies.update(["self-explanation prompts", "concept mapping", "contrasting cases"])
            elif kc.knowledge_type == "Strategic":
                strategies.update(["metacognitive prompts", "authentic problems", "reflection"])
        
        return list(strategies)
    
    def save_to_neo4j(self, kcs: List[KnowledgeComponent], 
                      relationships: List[PrerequisiteRelationship],
                      learning_objectives: List[LearningObjective]):
        """Save extracted knowledge graph to Neo4j"""
        
        with self.neo4j_driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create Knowledge Components
            for kc in kcs:
                query = """
                CREATE (kc:KnowledgeComponent {
                    id: $id,
                    name: $name,
                    description: $description,
                    knowledge_type: $knowledge_type,
                    primary_learning_process: $primary_learning_process,
                    difficulty_level: $difficulty_level,
                    content_chunks: $content_chunks,
                    parent_chunk_id: $parent_chunk_id,
                    key_concepts: $key_concepts,
                    assessment_criteria: $assessment_criteria,
                    estimated_time_minutes: $estimated_time_minutes,
                    created_at: datetime()
                })
                """
                session.run(query, **asdict(kc))
            
            # Create prerequisite relationships
            for rel in relationships:
                query = """
                MATCH (prereq:KnowledgeComponent {id: $prerequisite_kc})
                MATCH (target:KnowledgeComponent {id: $target_kc})
                CREATE (prereq)-[:PREREQUISITE_OF {
                    type: $relationship_type,
                    strength: $strength,
                    reasoning: $reasoning
                }]->(target)
                """
                session.run(query, **asdict(rel))
            
            # Create Learning Objectives
            for lo in learning_objectives:
                query = """
                CREATE (lo:LearningObjective {
                    id: $id,
                    title: $title,
                    description: $description,
                    knowledge_components: $knowledge_components,
                    dominant_knowledge_type: $dominant_knowledge_type,
                    learning_processes: $learning_processes,
                    recommended_strategies: $recommended_strategies,
                    chapter: $chapter,
                    section: $section,
                    difficulty_level: $difficulty_level,
                    estimated_total_time: $estimated_total_time,
                    created_at: datetime()
                })
                """
                session.run(query, **asdict(lo))
                
                # Link LOs to KCs
                for kc_id in lo.knowledge_components:
                    link_query = """
                    MATCH (lo:LearningObjective {id: $lo_id})
                    MATCH (kc:KnowledgeComponent {id: $kc_id})
                    CREATE (lo)-[:COMPOSED_OF {required: true}]->(kc)
                    """
                    session.run(link_query, lo_id=lo.id, kc_id=kc_id)
        
        logger.info(f"Saved {len(kcs)} KCs, {len(relationships)} relationships, {len(learning_objectives)} LOs to Neo4j")
    
    def run_full_extraction(self):
        """Run the complete extraction pipeline"""
        logger.info("Starting KLI knowledge extraction pipeline...")
        
        # Step 1: Get hierarchical content
        parent_nodes = self.get_parent_nodes_with_children()
        
        # Step 2: Extract KCs from each parent section
        all_kcs = []
        kcs_by_section = {}
        
        for parent_data in parent_nodes[:3]:  # Start with small batch for testing
            section_kcs = self.extract_knowledge_components_from_content(parent_data)
            all_kcs.extend(section_kcs)
            
            section_name = f"Section_{parent_data['node_id']}"
            kcs_by_section[section_name] = section_kcs
        
        logger.info(f"Extracted {len(all_kcs)} Knowledge Components")
        
        # Step 3: Identify relationships
        relationships = self.identify_prerequisite_relationships(all_kcs)
        logger.info(f"Identified {len(relationships)} prerequisite relationships")
        
        # Step 4: Create learning objectives
        learning_objectives = self.create_learning_objectives(kcs_by_section)
        logger.info(f"Created {len(learning_objectives)} Learning Objectives")
        
        # Step 5: Save to Neo4j
        self.save_to_neo4j(all_kcs, relationships, learning_objectives)
        
        logger.info("KLI extraction pipeline completed successfully!")
        
        return {
            'knowledge_components': len(all_kcs),
            'relationships': len(relationships), 
            'learning_objectives': len(learning_objectives)
        }

# Usage example
if __name__ == "__main__":
    es_config = "http://localhost:9200"  # Fixed: Use URL string instead of dict
    neo4j_config = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j", 
        "password": "knowledge123"  # Updated to match docker-compose password
    }
    storage_dir = "./elasticsearch_storage"
    
    pipeline = KLIExtractionPipeline(es_config, neo4j_config, storage_dir)
    results = pipeline.run_full_extraction()
    
    print(f"✅ Extraction completed: {results}")