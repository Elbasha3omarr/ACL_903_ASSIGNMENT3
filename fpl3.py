#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install streamlit pandas networkx plotly neo4j watchdog


# In[ ]:


# Cell 1: Install required packages
# get_ipython().system('pip install sentence-transformers neo4j')

# Cell 2: Import libraries
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import json

import pandas as pd
import numpy as np
import json
import re
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import atexit
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[ ]:


# Cell 3: Read Neo4j configuration
def read_config(config_file="Neo4j-5c3078ea-Created-2025-11-17.txt"):
    """Read Neo4j configuration from file"""
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()

    config['URI'] = config.get('NEO4J_URI') or config.get('URI')
    config['USERNAME'] = config.get('NEO4J_USERNAME') or config.get('USERNAME')
    config['PASSWORD'] = config.get('NEO4J_PASSWORD') or config.get('PASSWORD')
    return config


# In[ ]:


@dataclass
class ProcessedInput:
    """Container for preprocessed input data"""
    raw_query: str
    intent: str
    confidence: float
    entities: Dict[str, List[str]]
    embedding: Optional[np.ndarray]
    cypher_params: Dict[str, any]

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'raw_query': self.raw_query,
            'intent': self.intent,
            'confidence': self.confidence,
            'entities': self.entities,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'cypher_params': self.cypher_params
        }


# In[ ]:


# Cell 5: FPL Input Preprocessor Class (Part 1 - Initialization)
class FPLInputPreprocessor:
    """Preprocesses user queries for FPL Graph-RAG system"""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", team_names_from_kg: Optional[List[str]] = None, player_names_from_kg: Optional[List[str]] = None):
        """Initialize the preprocessor with intent patterns and entity patterns"""

        if team_names_from_kg:
            team_names_regex = '|'.join(re.escape(name) for name in team_names_from_kg)
            print(f"Team names: {team_names_regex}")
        else:
            team_names_regex = (
                "arsenal|liverpool|man city|manchester city|chelsea|tottenham|man utd|manchester united|"
                "newcastle|brighton|aston villa|west ham|crystal palace|wolves|fulham|brentford|nottm forest|"
                "everton|leicester|leeds|southampton|bournemouth|burnley|norwich|watford|luton"
            )
            print("Warning: There are no team names in the KG.")

        if player_names_from_kg:
            player_names_regex = '|'.join(re.escape(name) for name in player_names_from_kg)
            print(f"Player names: {player_names_regex}")
        else:
            player_names_regex = (
                "mohamed salah|harry kane|kevin de bruyne|bruno fernandes|"
                "son heung-min|erling haaland|gabriel jesus|martin ødegaard|"
                "marcus rashford|jack grealish"
            )
            print("Warning: There are no player names in the KG.")

        # Intent classification keywords (pattern, weight) tuples
        # Weights reflect how strongly a pattern indicates a specific intent
        self.intent_patterns = {
            'player_performance': [
                (r'\b(goals?|assists?|points?|stats?|performance|total|sum|how many|how much)\b', 1.8),
                (r'\b(scored|assisted|earned)\b', 2.2),
            ],
            'player_comparison': [
                (r'\b(compare|comparison|versus|vs\\.?|better|who.*better)\b', 2.5),
                (r'\b(vs?\\.?|versus)\b.*\b(with)\b', 2.0),
                (r'\b(vs\\.?|versus)\b.*\b(performance|comparison)\b', 2.5),
                (rf'({player_names_regex}).*({player_names_regex})', 3.6)
            ],
            'team_analysis': [
                (rf'\b({team_names_regex})\b.*\b(performance|stats?|defence|attack|clean sheets?)\b', 2.4),
                (r'\b(how.*(did|does)|performance.*(team|club))\b', 1.8),
                (rf'\b({team_names_regex})\b.*\b(defensive|attacking)\s+stats?\b', 2.5),
            ],
            'fixture_query': [
                (r'\b(fixture|match|game|gameweek|gw|when.*play|schedule|play against|plays against)\b', 2.5),
                (r'\b(gw|gameweek)\s*\d+\b', 2.8),
            ],
            'season_comparison': [
                (r'\b(compare|comparison|across seasons?|different seasons?|versus|vs\\.?|compared? to)\b', 2.2),
                (r'\b(202[1-9]-[2-9][0-9])\b.*\b(202[1-9]-[2-9][0-9])\b', 3.6),
            ],
            'position_analysis': [
                (r'\b(top|best|highest|leading)\b.*\b(defenders?|midfielders?|forwards?|goalkeepers?|def|mid|fwd|gk)\b', 2.2),
                (r'\b(defenders?|midfielders?|forwards?|goalkeepers?|def|mid|fwd|gk)\b.*\b(top|best|points?|goals?)\b', 2.0),
            ],
            'recommendation': [
                (r'\b(should i |recommend|suggest|who to (pick|buy|transfer)|good pick|captain)\b', 3.0),
            ],
            'search_player': [
                (r'\b(tell me about|who is|info|information about|find|search)\b.*\b([A-Z][a-z]+)\b', 2.5),
            ],
        }

        # FPL-specific entity patterns
        self.entity_patterns = {
            'player_name': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
            'team_name': rf'\b({team_names_regex})\b',
            'position': r'\b(DEF|MID|FWD|GK|defender|midfielder|forward|goalkeeper)\b',
            'season': r'\b(2021-22|2022-23|2023-24)\b',
            'gameweek': r'\b(GW|gameweek|week)\s*(\d+)\b',
            'stat_metric': r'\b(goals?|assists?|points?|minutes|bonus|clean sheets?|'
                          r'saves?|yellow cards?|red cards?|BPS|ICT|influence|'
                          r'creativity|threat|form)\b',
            'comparison_operator': r'\b(more than|less than|at least|exactly|minimum|maximum|'
                                   r'above|below|over|under|>=|<=|>|<|=)\b',
            'numeric_value': r'\b(\d+(?:\\.\d+)?)\b'
        }

        # Position mappings
        self.position_mappings = {
            'defender': 'DEF', 'defenders': 'DEF', 'defence': 'DEF',
            'midfielder': 'MID', 'midfielders': 'MID', 'midfield': 'MID',
            'forward': 'FWD', 'forwards': 'FWD', 'striker': 'FWD', 'strikers': 'FWD',
            'goalkeeper': 'GK', 'goalkeepers': 'GK', 'keeper': 'GK', 'keepers': 'GK'
        }

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"✓ Embedding model loaded successfully")

    def classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify user intent based on query patterns"""
        query_lower = query.lower()
        intent_scores = {intent: 0.0 for intent in self.intent_patterns.keys()}

        for intent, patterns_with_weights in self.intent_patterns.items():
            score = 0.0
            # Calculate total possible weighted score for this intent, needed for confidence
            total_possible_weighted_score = sum(weight for _, weight in patterns_with_weights)

            for pattern, weight in patterns_with_weights:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    score += weight * len(matches) # Add weighted score for each match

            intent_scores[intent] = score

        # Handle case where no patterns matched at all
        if not intent_scores or max(intent_scores.values()) == 0:
            return 'general_query', 0.5

        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]

        # Recalculate total possible weighted score for the best intent to avoid issues with different len(matches)
        best_intent_total_possible_weight = sum(weight for _, weight in self.intent_patterns[best_intent])

        if best_intent_total_possible_weight == 0:
            confidence = 0.5 # Fallback if no patterns for the best intent or all have zero weight
        else:
            confidence = min(1.0, max_score / best_intent_total_possible_weight)

        return best_intent, confidence

# === PATCH 1: Fix player name extraction (case-insensitive + better pattern) ===
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        entities = {k: [] for k in self.entity_patterns.keys()}
        entities['player_name'] = []

        # Extract all other entities first
        for entity_type, pattern in self.entity_patterns.items():
            if entity_type == 'player_name':
                continue
            matches = re.findall(pattern, query, re.IGNORECASE)
            if entity_type == 'gameweek':
                entities[entity_type].extend([m[1] if isinstance(m, tuple) else str(m) for m in matches])
            elif entity_type == 'position':
                # FIX: Improved position mapping with case-insensitive matching
                pos_map = {
                    'defender': 'DEF', 'defenders': 'DEF', 'defence': 'DEF', 'def': 'DEF',
                    'midfielder': 'MID', 'midfielders': 'MID', 'midfield': 'MID', 'mid': 'MID',
                    'forward': 'FWD', 'forwards': 'FWD', 'striker': 'FWD', 'strikers': 'FWD', 'fwd': 'FWD',
                    'goalkeeper': 'GK', 'goalkeepers': 'GK', 'keeper': 'GK', 'keepers': 'GK', 'gk': 'GK'
                }

                normalized = []
                for m in matches:
                    term = m.lower() if isinstance(m, str) else m[0].lower()
                    if term in pos_map:
                        normalized.append(pos_map[term])
                    elif term.upper() in ['DEF', 'MID', 'FWD', 'GK']:
                        normalized.append(term.upper())

                # ADDITIONAL FIX: Check query directly for position keywords if pattern didn't match
                if not normalized:
                    query_lower = query.lower()
                    for key, value in pos_map.items():
                        if key in query_lower and value not in normalized:
                            normalized.append(value)
                            break

                entities[entity_type].extend(normalized)
            else:
                flattened = [m[0] if isinstance(m, tuple) else m for m in matches]
                entities[entity_type].extend(flattened)

        # Player name extraction (existing code)
        banned = {
            'arsenal','liverpool','city','united','man','chelsea','tottenham','how','what','who','when',
            'compare','top','best','show','tell','vs','versus','and','against','the','me','are'
        }

        query_clean = re.sub(r'\b(vs\\.?|versus|and|against)\b', '|||', query, flags=re.IGNORECASE)
        query_clean = re.sub(r'\b(compare|show|tell|find)\s+', '', query_clean, flags=re.IGNORECASE)

        parts = [p.strip() for p in query_clean.split('|||') if p.strip()]

        candidates = []
        for part in parts:
            found = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', part)
            for cand in found:
                words = cand.split()
                filtered_words = [w for w in words if w.lower() not in banned]
                if filtered_words and 1 <= len(filtered_words) <= 3:
                    candidates.append(' '.join(filtered_words))

        entities['player_name'] = list(dict.fromkeys(candidates))

        if not entities['player_name']:
            single_names = re.findall(r'\b([A-Z][a-z]{2,})\b', query)
            for name in single_names:
                if name.lower() not in banned and name not in entities['player_name']:
                    entities['player_name'].append(name)

        return {k: v for k, v in entities.items() if v}



    def generate_embedding(self, query: str) -> np.ndarray:
        """Generate embedding vector for semantic search"""
        embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        return embedding

    # === PATCH 3: Fix cypher_params to always set player1/player2 when comparison ===
    def build_cypher_params(self, entities: Dict[str, List[str]], intent: str) -> Dict[str, any]:
        params = {}

        if 'player_name' in entities and entities['player_name']:
            players = entities['player_name']
            params['player_names'] = players
            if len(players) >= 1:
                params['player_name'] = players[0]
            if intent == 'player_comparison':
                params['player1'] = players[0]
                if len(players) >= 2:
                    params['player2'] = players[1]
                else:
                    params['player2'] = players[0]  # fallback if only one player found
        if 'team_name' in entities and entities['team_name']:
            team_list = entities['team_name']
            params['team_names'] = team_list
            if len(team_list) == 1:
                params['team_name'] = team_list[0]

        if 'position' in entities and entities['position']:
            pos_list = list(set(entities['position']))
            params['positions'] = pos_list
            if len(pos_list) == 1:
                params['position'] = pos_list[0]

        if 'season' in entities and entities['season']:
            season_list = entities['season']
            params['seasons'] = season_list
            if len(season_list) == 1:
                params['season'] = season_list[0]

        if 'gameweek' in entities:
            try:
                gws = [int(gw) for gw in entities['gameweek']]
                params['gameweeks'] = gws
                if len(gws) == 1:
                    params['gameweek'] = gws[0]
            except:
                pass

        return params

    def process(self, query: str) -> ProcessedInput:
        intent, confidence = self.classify_intent(query)
        entities = self.extract_entities(query)
        embedding = self.generate_embedding(query)
        cypher_params = self.build_cypher_params(entities, intent)
        return ProcessedInput(query, intent, confidence, entities, embedding, cypher_params)


# In[ ]:


class CypherQueryGenerator:
    """Generate Cypher queries based on intent and entities"""

    @staticmethod
    def get_query_template(processed: ProcessedInput) -> str:
        """Get appropriate Cypher query template"""
        intent = processed.intent
        params = processed.cypher_params

        templates = {
            'player_performance': """
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                WHERE p.player_name = $player_name
                RETURN p.player_name AS player,
                       SUM(r.total_points) AS total_points,
                       SUM(r.goals_scored) AS total_goals,
                       SUM(r.assists) AS total_assists,
                       SUM(r.minutes) AS total_minutes,
                       COUNT(f) AS matches_played
            """,

            'player_comparison': """
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                WHERE p.player_name IN [$player1, $player2]
                WITH p.player_name AS player,
                     SUM(r.total_points) AS points,
                     SUM(r.goals_scored) AS goals,
                     SUM(r.assists) AS assists,
                     SUM(r.minutes) AS minutes
                RETURN player, points, goals, assists, minutes
                ORDER BY points DESC
            """,

            'team_analysis': """
                MATCH (t:Team)<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture)
                WHERE t.name = $team_name
                WITH f
                MATCH (p:Player)-[r:PLAYED_IN]->(f)
                RETURN COUNT(DISTINCT f) AS matches,
                       SUM(r.goals_scored) AS goals_scored,
                       SUM(r.clean_sheets) AS clean_sheets
            """,

            'position_analysis': """
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: $position})
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                WITH p, SUM(r.total_points) AS total_points,
                     SUM(r.goals_scored) AS goals,
                     SUM(r.assists) AS assists
                ORDER BY total_points DESC
                LIMIT 10
                RETURN p.player_name, total_points, goals, assists
            """,

            'season_comparison': """
                MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture)
                WITH f.season AS season,
                     SUM(r.total_points) AS points,
                     SUM(r.goals_scored) AS goals,
                     SUM(r.assists) AS assists
                RETURN season, points, goals, assists
                ORDER BY season
            """
        }

        return templates.get(intent, """
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:HAS_HOME_TEAM]->(t:Team)
            WITH p, SUM(r.total_points) AS total_points, COLLECT(DISTINCT f.season) AS seasons, COLLECT(DISTINCT t.name) AS team
            ORDER BY total_points DESC
            LIMIT 10
            RETURN p.player_name AS player, total_points, team, seasons
        """)


# In[ ]:


class Neo4jQueryExecutor:
    """Execute queries against Neo4j database"""

    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]


# In[ ]:


class FPLGraphRAGPreprocessor:
    """Complete preprocessing pipeline with Neo4j integration"""

    def __init__(self, config_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        # Load configuration
        config = read_config(config_path)

        # Initialize executor first to fetch dynamic data
        self.executor = Neo4jQueryExecutor(
            config['URI'],
            config['USERNAME'],
            config['PASSWORD']
        )

        team_names = []
        player_names = []
        try:
            print("Loading team names from Neo4j...")
            # Fetch all unique team names from the KG
            team_records = self.executor.execute_query("MATCH (t:Team) RETURN t.name AS name")
            team_names = [record['name'] for record in team_records]
            print(f"\u2713 {len(team_names)} team names loaded from KG.")
        except Exception as e:
            print(f"\u2717 Failed to load team names from KG: {e}. Using hardcoded defaults.")
            # Fallback to a hardcoded list if KG query fails
            team_names = ['Arsenal', 'Liverpool', 'Man City', 'Manchester City', 'Chelsea', 'Tottenham',
                            'Man Utd', 'Manchester United', 'Newcastle', 'Brighton', 'Aston Villa',
                            'West Ham', 'Crystal Palace', 'Wolves', 'Fulham', 'Brentford', 'Nottm Forest',
                            'Everton', 'Leicester', 'Leeds', 'Southampton', 'Bournemouth', 'Burnley',
                            'Norwich', 'Watford', 'Luton']

        try:
            print("Loading player names from Neo4j...")
            # Fetch all unique player names from the KG
            player_records = self.executor.execute_query("MATCH (p:Player) RETURN p.player_name AS name")
            player_names = [record['name'] for record in player_records]
            print(f"\u2713 {len(player_names)} player names loaded from KG.")
        except Exception as e:
            print(f"\u2717 Failed to load player names from KG: {e}. Using hardcoded defaults.")
            # Fallback to a hardcoded list if KG query fails
            player_names = [
                'Mohamed Salah', 'Harry Kane', 'Kevin De Bruyne', 'Bruno Fernandes',
                'Son Heung-min', 'Erling Haaland', 'Gabriel Jesus', 'Martin Ødegaard',
                'Marcus Rashford', 'Jack Grealish'
            ]

        self.player_names = [name.lower() for name in player_names]
        self.team_names = [name.lower() for name in team_names]

        self.preprocessor = FPLInputPreprocessor(embedding_model, team_names_from_kg=team_names, player_names_from_kg=player_names)
        self.query_generator = CypherQueryGenerator()

        print("\u2713 FPL Graph-RAG Preprocessor initialized successfully")

    def process_and_query(self, user_query: str) -> Dict:
        """Process query and execute against knowledge graph"""
        # Step 1: Preprocess input
        processed = self.preprocessor.process(user_query)

        # Step 2: Generate Cypher query
        cypher_query = self.query_generator.get_query_template(processed)

        # Step 3: Execute query
        try:
            results = self.executor.execute_query(
                cypher_query,
                processed.cypher_params
            )
        except Exception as e:
            results = []
            print(f"Query execution error: {e}")

        # Step 4: Return complete results
        return {
            'input': {
                'query': processed.raw_query,
                'intent': processed.intent,
                'confidence': processed.confidence,
                'entities': processed.entities,
                'embedding_shape': processed.embedding.shape if processed.embedding is not None else None
            },
            'cypher': {
                'query': cypher_query.strip(),
                'params': processed.cypher_params
            },
            'results': results,
            'result_count': len(results)
        }

    def close(self):
        """Close database connection"""
        self.executor.close()


# In[ ]:


# Initialize pipeline
config_path = "Neo4j-5c3078ea-Created-2025-11-17.txt"
pipeline = FPLGraphRAGPreprocessor(config_path)

team_names = pipeline.team_names
player_names = pipeline.player_names

# Test queries
test_queries = [
    "How many goals did Mohamed Salah score in 2022-23?",
    "Compare Erling Haaland and Harry Kane",
    "Show me the top scoring defenders",
    "What was Mohamed Salah's performance across different seasons?"
]

print("\n" + "="*80)
print("FPL GRAPH-RAG INPUT PREPROCESSING DEMONSTRATION")
print("="*80)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"Query {i}: {query}")
    print("="*80)

    result = pipeline.process_and_query(query)

    print(f"\n📊 PREPROCESSING RESULTS:")
    print(f"  Intent: {result['input']['intent']} "
          f"(confidence: {result['input']['confidence']:.2f})")
    print(f"  Entities: {result['input']['entities']}")
    print(f"  Embedding Shape: {result['input']['embedding_shape']}")

    print(f"\n🔍 CYPHER QUERY:")
    print(result['cypher']['query'])
    print(f"\n  Parameters: {result['cypher']['params']}")

    print(f"\n✅ RESULTS ({result['result_count']} records):")
    for record in result['results'][:3]:  # Show first 3 results
        print(f"  {record}")

    if result['result_count'] > 3:
        print(f"  ... and {result['result_count'] - 3} more")

pipeline.close()
print("\n" + "="*80)
print("✓ Demonstration complete")
print("="*80)


# In[ ]:


# def save_preprocessing_module(output_path='fpl_preprocessor.py'):
#     """Save the preprocessor as a module for team integration"""
#     # This exports the preprocessor for other team members to use
#     with open(output_path, 'w') as f:
#         f.write('''
# # Import this module in other notebooks:
# # from fpl_preprocessor import FPLInputPreprocessor, ProcessedInput

# # Usage:
# # preprocessor = FPLInputPreprocessor()
# # result = preprocessor.process("Your query here")
# # print(result.intent, result.entities)
# ''')
#     print(f"✓ Module saved to {output_path}")


# In[ ]:


# get_ipython().system('pip install pandas matplotlib seaborn -q')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# In[ ]:


# FPL Graph-RAG Preprocessor - Comprehensive Testing Notebook
# This notebook validates all preprocessing functionality

# Cell 2: Load the preprocessor (copy the FPLInputPreprocessor class here)
# ... (Include the full FPLInputPreprocessor class from artifact 1)

# Cell 3: Test Dataset
# Comprehensive test queries covering all intents
test_dataset = [
    # Player Performance
    {
        'query': 'How many goals did Mohamed Salah score in 2022-23?',
        'expected_intent': 'player_performance',
        'expected_entities': ['player_name', 'stat_metric', 'season']
    },
    {
        'query': 'What was Erling Haaland total points last season?',
        'expected_intent': 'player_performance',
        'expected_entities': ['player_name', 'stat_metric']
    },
    {
        'query': 'Show me Harry Kane assists',
        'expected_intent': 'player_performance',
        'expected_entities': ['player_name', 'stat_metric']
    },

    # Player Comparison
    {
        'query': 'Compare Erling Haaland and Harry Kane',
        'expected_intent': 'player_comparison',
        'expected_entities': ['player_name']
    },
    {
        'query': 'Who is better Mohamed Salah versus Kevin De Bruyne',
        'expected_intent': 'player_comparison',
        'expected_entities': ['player_name']
    },
    {
        'query': 'Haaland vs Kane performance comparison',
        'expected_intent': 'player_comparison',
        'expected_entities': ['player_name']
    },

    # Team Analysis
    {
        'query': 'How did Arsenal perform this season?',
        'expected_intent': 'team_analysis',
        'expected_entities': ['team_name']
    },
    {
        'query': 'Liverpool defensive stats',
        'expected_intent': 'team_analysis',
        'expected_entities': ['team_name']
    },
    {
        'query': 'Man City clean sheets analysis',
        'expected_intent': 'team_analysis',
        'expected_entities': ['team_name', 'stat_metric']
    },

    # Fixture Query
    {
        'query': 'Arsenal fixtures in gameweek 10',
        'expected_intent': 'fixture_query',
        'expected_entities': ['team_name', 'gameweek']
    },
    {
        'query': 'When does Liverpool play against Man City?',
        'expected_intent': 'fixture_query',
        'expected_entities': ['team_name']
    },
    {
        'query': 'Show me all fixtures in GW 15',
        'expected_intent': 'fixture_query',
        'expected_entities': ['gameweek']
    },

    # Season Comparison
    {
        'query': 'Compare Mohamed Salah performance in 2021-22 versus 2022-23',
        'expected_intent': 'season_comparison',
        'expected_entities': ['player_name', 'season']
    },
    {
        'query': 'How did Kane do across different seasons?',
        'expected_intent': 'season_comparison',
        'expected_entities': ['player_name']
    },

    # Position Analysis
    {
        'query': 'Who are the top scoring defenders?',
        'expected_intent': 'position_analysis',
        'expected_entities': ['position']
    },
    {
        'query': 'Best midfielders in the league',
        'expected_intent': 'position_analysis',
        'expected_entities': ['position']
    },
    {
        'query': 'Show me all forwards with more than 10 goals',
        'expected_intent': 'position_analysis',
        'expected_entities': ['position', 'numeric_value', 'stat_metric']
    },

    # Recommendation
    {
        'query': 'Which midfielder should I pick for my team?',
        'expected_intent': 'recommendation',
        'expected_entities': ['position']
    },
    {
        'query': 'Recommend a good defender',
        'expected_intent': 'recommendation',
        'expected_entities': ['position']
    },

    # Search Player
    {
        'query': 'Tell me about Bukayo Saka',
        'expected_intent': 'search_player',
        'expected_entities': ['player_name']
    },
    {
        'query': 'Find information about Erling Haaland',
        'expected_intent': 'search_player',
        'expected_entities': ['player_name']
    }
]

# Cell 4: Initialize Preprocessor
print("Initializing FPL Input Preprocessor...")
preprocessor = FPLInputPreprocessor(embedding_model_name="all-MiniLM-L6-v2", team_names_from_kg=team_names, player_names_from_kg=player_names)
print("✓ Preprocessor initialized\n")

# Cell 5: Run Tests
print("="*80)
print("RUNNING COMPREHENSIVE TESTS")
print("="*80)

results = []
for i, test in enumerate(test_dataset, 1):
    query = test['query']
    processed = preprocessor.process(query)

    # Check intent match
    intent_correct = processed.intent == test['expected_intent']

    # Check if expected entities are present
    extracted_entity_types = set(processed.entities.keys())
    expected_entity_types = set(test['expected_entities'])
    entities_match = len(expected_entity_types & extracted_entity_types) / len(expected_entity_types) if expected_entity_types else 1.0

    results.append({
        'test_id': i,
        'query': query,
        'expected_intent': test['expected_intent'],
        'actual_intent': processed.intent,
        'intent_correct': intent_correct,
        'confidence': processed.confidence,
        'expected_entities': test['expected_entities'],
        'extracted_entities': list(processed.entities.keys()),
        'entity_match_score': entities_match,
        'has_embedding': processed.embedding is not None
    })

    # Print result
    status = "✓" if intent_correct and entities_match > 0.5 else "✗"
    print(f"\n{status} Test {i}: {query}")
    print(f"  Intent: {processed.intent} (expected: {test['expected_intent']}) - {'PASS' if intent_correct else 'FAIL'}")
    print(f"  Confidence: {processed.confidence:.2f}")
    print(f"  Entities: {processed.entities}")
    print(f"  Entity Match: {entities_match:.0%}")

# Cell 6: Calculate Test Statistics
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

total_tests = len(results_df)
intent_accuracy = (results_df['intent_correct'].sum() / total_tests) * 100
avg_confidence = results_df['confidence'].mean()
avg_entity_match = results_df['entity_match_score'].mean() * 100
embedding_success = (results_df['has_embedding'].sum() / total_tests) * 100

print(f"\nTotal Tests: {total_tests}")
print(f"Intent Classification Accuracy: {intent_accuracy:.1f}%")
print(f"Average Confidence Score: {avg_confidence:.2f}")
print(f"Average Entity Match Score: {avg_entity_match:.1f}%")
print(f"Embedding Generation Success: {embedding_success:.1f}%")

# Per-intent accuracy
print("\n" + "-"*80)
print("PER-INTENT ACCURACY")
print("-"*80)
intent_accuracy_df = results_df.groupby('expected_intent').agg({
    'intent_correct': lambda x: f"{(x.sum()/len(x)*100):.1f}%",
    'confidence': 'mean',
    'entity_match_score': 'mean'
}).round(2)
print(intent_accuracy_df)

# Cell 7: Visualize Results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Intent Distribution
intent_counts = results_df['expected_intent'].value_counts()
axes[0, 0].bar(range(len(intent_counts)), intent_counts.values)
axes[0, 0].set_xticks(range(len(intent_counts)))
axes[0, 0].set_xticklabels(intent_counts.index, rotation=45, ha='right')
axes[0, 0].set_title('Intent Distribution in Test Set')
axes[0, 0].set_ylabel('Count')

# 2. Confidence Scores by Intent
results_df.boxplot(column='confidence', by='expected_intent', ax=axes[0, 1])
axes[0, 1].set_title('Confidence Scores by Intent')
axes[0, 1].set_xlabel('Intent')
axes[0, 1].set_ylabel('Confidence')
plt.sca(axes[0, 1])
plt.xticks(rotation=45, ha='right')

# 3. Accuracy Comparison
metrics = ['Intent Accuracy', 'Entity Match', 'Embedding Success']
scores = [intent_accuracy, avg_entity_match, embedding_success]
axes[1, 0].bar(metrics, scores, color=['green', 'blue', 'orange'])
axes[1, 0].set_ylim(0, 100)
axes[1, 0].set_title('Overall Performance Metrics')
axes[1, 0].set_ylabel('Score (%)')
for i, v in enumerate(scores):
    axes[1, 0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')

# 4. Entity Extraction Success
all_entities = []
for entities in results_df['extracted_entities']:
    all_entities.extend(entities)
entity_counts = Counter(all_entities)
axes[1, 1].barh(list(entity_counts.keys()), list(entity_counts.values()))
axes[1, 1].set_title('Entity Type Frequency')
axes[1, 1].set_xlabel('Count')

plt.tight_layout()
plt.savefig('preprocessing_test_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Cell 8: Test Embedding Quality
print("\n" + "="*80)
print("EMBEDDING QUALITY TEST")
print("="*80)

# Test semantic similarity
test_pairs = [
    ("Mohamed Salah goals", "How many goals did Salah score?"),
    ("Compare Haaland and Kane", "Haaland versus Kane comparison"),
    ("Arsenal fixtures", "When does Arsenal play?"),
    ("Top defenders", "Best defensive players"),
]

print("\nSemantic Similarity Tests:")
for query1, query2 in test_pairs:
    emb1 = preprocessor.generate_embedding(query1)
    emb2 = preprocessor.generate_embedding(query2)

    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    print(f"\nQuery 1: {query1}")
    print(f"Query 2: {query2}")
    print(f"Similarity: {similarity:.3f} {'✓ High' if similarity > 0.7 else '✗ Low'}")

# Cell 9: Entity Extraction Edge Cases
print("\n" + "="*80)
print("EDGE CASE TESTING")
print("="*80)

edge_cases = [
    "mohamed salah goals",  # lowercase player name
    "ARSENAL FIXTURES",  # all caps
    "Salah vs Kane vs Haaland",  # 3 players
    "defenders with more than 5 goals and less than 2 yellow cards",  # complex
    "who scored the most?",  # vague
    "gw10 fixtures",  # no space in GW
    "2021-22 vs 2022-23 season comparison",  # multiple seasons
]

print("\nEdge Case Tests:")
for query in edge_cases:
    processed = preprocessor.process(query)
    print(f"\nQuery: {query}")
    print(f"  Intent: {processed.intent} ({processed.confidence:.2f})")
    print(f"  Entities: {processed.entities}")

# Cell 10: Generate Test Report
report = {
    'test_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_tests': int(total_tests),
    'intent_accuracy': float(intent_accuracy),
    'avg_confidence': float(avg_confidence),
    'avg_entity_match': float(avg_entity_match),
    'embedding_success': float(embedding_success),
    'per_intent_accuracy': intent_accuracy_df.to_dict(),
    'failed_tests': results_df[~results_df['intent_correct']]['query'].tolist()
}

with open('preprocessing_test_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*80)
print("TEST REPORT SAVED")
print("="*80)
print(f"✓ Report saved to: preprocessing_test_report.json")
print(f"✓ Visualization saved to: preprocessing_test_results.png")

# Cell 11: Performance Benchmarking
print("\n" + "="*80)
print("PERFORMANCE BENCHMARKING")
print("="*80)

import time

# Time each component
query = "How many goals did Mohamed Salah score in 2022-23?"

# Intent classification
start = time.time()
for _ in range(100):
    intent, conf = preprocessor.classify_intent(query)
intent_time = (time.time() - start) / 100

# Entity extraction
start = time.time()
for _ in range(100):
    entities = preprocessor.extract_entities(query)
entity_time = (time.time() - start) / 100

# Embedding generation
start = time.time()
for _ in range(100):
    embedding = preprocessor.generate_embedding(query)
embedding_time = (time.time() - start) / 100

# Full pipeline
start = time.time()
for _ in range(100):
    result = preprocessor.process(query)
total_time = (time.time() - start) / 100

print(f"\nAverage Processing Times (100 iterations):")
print(f"  Intent Classification: {intent_time*1000:.2f} ms")
print(f"  Entity Extraction: {entity_time*1000:.2f} ms")
print(f"  Embedding Generation: {embedding_time*1000:.2f} ms")
print(f"  Total Pipeline: {total_time*1000:.2f} ms")
print(f"\nThroughput: {1/total_time:.1f} queries/second")

# Cell 12: Export Results for Team
print("\n" + "="*80)
print("EXPORTING FOR TEAM INTEGRATION")
print("="*80)

# Export sample preprocessed queries
sample_exports = []
for test in test_dataset[:5]:
    processed = preprocessor.process(test['query'])
    sample_exports.append({
        'query': processed.raw_query,
        'intent': processed.intent,
        'confidence': float(processed.confidence),
        'entities': processed.entities,
        'cypher_params': processed.cypher_params,
        'embedding_shape': processed.embedding.shape if processed.embedding is not None else None
    })

with open('sample_preprocessed_queries.json', 'w') as f:
    json.dump(sample_exports, f, indent=2)

print("✓ Sample preprocessed queries exported to: sample_preprocessed_queries.json")
print("\n" + "="*80)
print("ALL TESTS COMPLETED SUCCESSFULLY")
print("="*80)


# In[ ]:


def prepare_cypher_params(processed: ProcessedInput) -> Dict:
    """Map extracted entities to Cypher parameters"""
    intent = processed.intent
    entities = processed.entities  # dict from your FPLInputPreprocessor

    if intent == 'player_performance':
        return {
            'player_name': entities.get('player_name', [None])[0]
        }
    elif intent == 'player_comparison':
        players = entities.get('player_name', [])
        # Fill with None if less than 2 players for query safety
        if len(players) == 1:
            players.append(None)
        return {
            'player1': players[0],
            'player2': players[1]
        }
    elif intent == 'team_analysis':
        return {
            'team_name': entities.get('team_name', [None])[0]
        }
    elif intent == 'position_analysis':
        return {
            'position': entities.get('position', [None])[0]
        }
    elif intent == 'season_comparison':
        return {
            'player_name': entities.get('player_name', [None])[0]
        }
    else:
        return {}


# In[ ]:


def run_processed_query(executor: Neo4jQueryExecutor, processed: ProcessedInput) -> List[Dict]:
    """Select query template and execute it with entity parameters"""
    # Prepare Cypher params from entities
    processed.cypher_params = prepare_cypher_params(processed)

    # Get the query template
    query = CypherQueryGenerator.get_query_template(processed)

    # Execute query with parameters
    results = executor.execute_query(query, processed.cypher_params)
    return results

# get_ipython().system('pip install huggingface_hub --quiet')
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
from huggingface_hub import InferenceClient

config = read_config(config_path)

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(api_key=HF_TOKEN)

def llm_generate(prompt: str,
                        model="deepseek-ai/DeepSeek-V3",
                        max_tokens=128):

    MAX_ATTEMPTS = 1
    retries = 0
    wait_period = 2

    response, tokens_used = None, None

    while retries < MAX_ATTEMPTS:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )

            response = completion.choices[0].message["content"]
            tokens_used = completion.usage.total_tokens
        
        except Exception as e:
            retries += 1
            print(f"LLM call failed, no response.")
            if retries < MAX_ATTEMPTS:
                print(f"Waiting {wait_period} seconds before attempting to reconnect...")
                time.sleep(wait_period)
            else:
                print("Maximum attempts reached, ignoring response.\n")
                return None, None
            
    return response, tokens_used


def format_player_performance(results: List[Dict]) -> str:
    lines = []
    for r in results:
        line = (
            f"{r.get('player', 'Unknown')} played {r.get('matches_played', 'N/A')} matches, "
            f"scored {r.get('total_goals', 0)} goals, "
            f"provided {r.get('total_assists', 0)} assists, "
            f"and accumulated {r.get('total_points', 0)} points "
            f"in {r.get('total_minutes', 0)} minutes."
        )
        lines.append(line)
    return "\n".join(lines)

def format_player_comparison(results: List[Dict]) -> str:
    results_sorted = sorted(results, key=lambda x: x.get('points', 0), reverse=True)
    lines = []
    for r in results_sorted:
        line = (
            f"{r.get('player', 'Unknown')} scored {r.get('goals', 0)} goals, "
            f"made {r.get('assists', 0)} assists, "
            f"played {r.get('minutes', 0)} minutes, "
            f"earning {r.get('points', 0)} points."
        )
        lines.append(line)
    return "\n".join(lines)


def format_generic(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        parts = []
        for key, value in row.items():
            clean_key = key.split(".")[-1].replace("_", " ")
            parts.append(f"{value} {clean_key}")
        lines.append(", ".join(parts))
    return "\n".join(lines)


def userQueryKG(userQuery: str, model="deepseek-ai/DeepSeek-V3", traceQuery=False):
    processed = preprocessor.process(userQuery)

    executor = Neo4jQueryExecutor(config['URI'], config['USERNAME'], config['PASSWORD'])
    results = run_processed_query(executor, processed)
    executor.close()

    if results is None or len(results) == 0:
        print(f"No results found for query: {userQuery}")
        return

    if processed.intent == 'player_performance':
        context_text = format_player_performance(results)
        output_template = "{player} scored {total_goals} goals."
    elif processed.intent == 'player_comparison':
        context_text = format_player_comparison(results)
        output_template = "{player} scored {goals} goals and earned {points} points."
    else:
        context_text = format_generic(results)

    persona = "You are an expert FPL AI assistant."
    task = """Answer the user's question in one sentence using ONLY the data provided. Omit any unecessary context.
    The contex is provided from the knowledge graph."""

    llm_prompt = f"""
        Persona:
        {persona}

        Task:
        {task}

        Context (Retreived from the knowledge graph):
        {context_text}

        User Query:
        {userQuery}
    """
        
    response, tokens_used = llm_generate(llm_prompt, model=model)

    if traceQuery:
        print(f"Query: {userQuery}")
        print(f"Entities: {processed.entities}")
        print(f"Context: {context_text}")
        print(f"Response: {response}")
        print("-" * 80)

    return response, tokens_used


hf_model1 = "deepseek-ai/DeepSeek-V3"
hf_model2 = "google/gemma-2-9b-it"
hf_model3 = "meta-llama/Llama-3.3-70B-Instruct"
queries = [
           "What was Erling Haaland total points last season?",
           "How many goals did Mohamed Salah score in 2022-23?",
           "How did Arsenal perform this season?",
           "Which midfielder should I pick for my team?",
           "Who are the top scoring defenders?",
           "When does Liverpool play against Man City?"
          ]

# Test run
# userQueryKG(queries[0], hf_model1, True)
import json
import pandas as pd
from huggingface_hub import InferenceClient
from neo4j import GraphDatabase

# =================================================
# MODELS UNDER TEST (UNCHANGED)
# =================================================
MODELS = {
    "deepseekV3": hf_model1,
    "gemma-2-9b-it": hf_model2,
    "Llama-3.3": hf_model3
}

# =================================================
# QUALITATIVE EVALUATOR
# =================================================
EVAL_MODEL = "meta-llama/Llama-3.1-8B-Instruct",

# =================================================
# NEO4J CONFIG
# =================================================
neo4j_uri = "neo4j+s://5c3078ea.databases.neo4j.io"
neo4j_user = "neo4j"
neo4j_pass = "B1StqtGDu90Z9BFqB8SWlcKYgtwLfdWH0xVkmyYzzm4"
neo4j_db   = "neo4j"

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

# =================================================
# CYPHER QUERIES
# =================================================
QUERIES = {
    "midfielders": """
        MATCH (p:Player)-[:PLAYS_AS]->(:Position {name:'MID'})
        MATCH (p)-[r:PLAYED_IN]->(:Fixture)
        RETURN p.player_name AS name, SUM(r.total_points) AS points
        ORDER BY points DESC LIMIT 5
    """,
    "clean_sheets": """
        MATCH (p:Player)-[:PLAYS_AS]->(:Position {name:'DEF'})
        MATCH (p)-[r:PLAYED_IN]->(:Fixture)
        RETURN p.player_name AS name, SUM(r.clean_sheets) AS cs
        ORDER BY cs DESC LIMIT 5
    """,
    "bonus_forwards": """
        MATCH (p:Player)-[:PLAYS_AS]->(:Position {name:'FWD'})
        MATCH (p)-[r:PLAYED_IN]->(:Fixture)
        RETURN p.player_name AS name, SUM(r.bonus) AS bonus
        ORDER BY bonus DESC LIMIT 5
    """
}

def run_query(q):
    with driver.session(database=neo4j_db) as s:
        return [r.data() for r in s.run(q)]

# =================================================
# BUILD KG CONTEXT
# =================================================
def build_kg_text():
    mids = run_query(QUERIES["midfielders"])
    defs = run_query(QUERIES["clean_sheets"])
    fwds = run_query(QUERIES["bonus_forwards"])

    txt = "FPL Knowledge Graph (Live From Neo4j):\n\n"

    txt += "Highest scoring midfielders:\n"
    for m in mids:
        txt += f"- {m['name']} ({m['points']} pts)\n"

    txt += "\nMost clean sheets (defenders):\n"
    for d in defs:
        txt += f"- {d['name']} ({d['cs']} CS)\n"

    txt += "\nMost bonus points (forwards):\n"
    for f in fwds:
        txt += f"- {f['name']} ({f['bonus']} bonus)\n"

    return txt

# =================================================
# TEST CASES
# =================================================
test_cases = [
    {"query": "Who is the highest-scoring midfielder in FPL this season?", "expected_keywords": ["midfielder"]},
    {"query": "List three defenders with the most clean sheets.", "expected_keywords": ["defender", "clean"]},
    {"query": "Which forward has the most bonus points?", "expected_keywords": ["forward", "bonus"]}
    # {"query": "Give me the best captain pick for this week.", "expected_keywords": []}
]

# =================================================
# MODEL CALL
# =================================================
def call_model(modelUsed, userQuery):
    start = time.time()

    response, tokens_used = userQueryKG(userQuery, modelUsed)

    return (
        response,
        time.time() - start,
        tokens_used
    )

# =================================================
# METRICS
# =================================================
def compute_accuracy(out, keys):
    if not keys:
        return 0
    out = out.lower()
    return sum(k in out for k in keys) / len(keys)

def evaluate_quality(query, output):
    prompt = f"""
    Rate the answer from 0 to 5 and return JSON only.

    Query: {query}
    Answer: {output}

    {{"relevance":0,"correctness":0,"naturalness":0}}
    """

    try:
        response, _ = llm_generate(prompt, EVAL_MODEL)
        return json.loads(response)
    except:
        return {"relevance": None, "correctness": None, "naturalness": None}

# =================================================
# RUN EVALUATION
# =================================================

rows = []
for model_name, model_id in MODELS.items():
    print(f"Evaluating {model_name}")
    for case in test_cases:
        print(f"Current query execution: {case['query']}")
        out, lat, tok = call_model(model_id, case["query"])
        q = evaluate_quality(case["query"], out)

        if not out:
            continue

        rows.append({
            "model": model_name,
            "query": case["query"],
            "latency": round(lat, 3),
            "tokens": tok,
            "accuracy": compute_accuracy(out, case["expected_keywords"]),
            "relevance": q["relevance"],
            "correctness": q["correctness"],
            "naturalness": q["naturalness"]
        })

df = pd.DataFrame(rows)

# =================================================
# OUTPUT
# =================================================
print(df)
df.to_csv("llm_eval_results_full.csv", index=False)


# In[ ]:


# get_ipython().system('pip install streamlit pyvis')


# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import os
import streamlit.components.v1 as components
from pyvis.network import Network


# In[ ]:


# ============================================================================
# REPLACE YOUR STREAMLIT SECTION WITH THIS CODE
# ============================================================================

# Cell: Streamlit App Integration
import streamlit as st
import pandas as pd

# ============================================================================
# FORMAT RESPONSE FUNCTION (ADD THIS)
# ============================================================================

def format_response(intent: str, data: List[Dict]) -> str:
    """Format KG data into natural language"""
    if not data:
        return "I couldn't find any relevant data in the Knowledge Graph for your query."
    
    if intent == 'player_performance':
        row = data[0]
        return f"**{row.get('player', 'Unknown')}** has played {row.get('matches_played', 'N/A')} matches, scoring **{row.get('total_goals', 0)} goals** and providing **{row.get('total_assists', 0)} assists**, totaling **{row.get('total_points', 0)} points**."
    
    elif intent == 'player_comparison':
        response = "**Player Comparison:**\n\n"
        for row in data:
            response += f"- **{row.get('player', 'Unknown')}**: {row.get('points', 0)} pts | {row.get('goals', 0)} goals | {row.get('assists', 0)} assists\n"
        return response
    
    elif intent == 'team_analysis':
        row = data[0]
        return f"The team has scored **{row.get('goals_scored', 0)} goals** and kept **{row.get('clean_sheets', 0)} clean sheets** in {row.get('matches', 0)} matches."
    
    elif intent == 'position_analysis':
        response = "**Top Players:**\n\n"
        for i, row in enumerate(data, 1):
            response += f"{i}. **{row.get('player', 'Unknown')}** - {row.get('total_points', 0)} pts ({row.get('goals', 0)} goals)\n"
        return response
    
    elif intent == 'fixture_query':
        response = "**Fixtures:**\n\n"
        for row in data:
            response += f"- {row.get('home', 'TBD')} vs {row.get('away', 'TBD')}\n"
        return response
    
    elif intent == 'recommendation':
        response = "**Recommended Players:**\n\n"
        for i, row in enumerate(data[:5], 1):
            response += f"{i}. **{row.get('player', 'Unknown')}** - {row.get('total_points', 0)} pts\n"
        return response
    
    else:
        # General/fallback format
        response = f"**Found {len(data)} results:**\n\n"
        for i, row in enumerate(data[:5], 1):
            player = row.get('player') or row.get('player_name') or row.get('name', 'Unknown')
            points = row.get('total_points') or row.get('points', 'N/A')
            response += f"{i}. **{player}** - {points} pts\n"
        if len(data) > 5:
            response += f"\n...and {len(data) - 5} more"
        return response

# ============================================================================
# STREAMLIT APP (REPLACE YOUR ENTIRE STREAMLIT SECTION)
# ============================================================================

def run_streamlit_app():
    """Main Streamlit application"""
    st.set_page_config(page_title="FPL Graph-RAG Assistant", layout="wide", page_icon="⚽")
    
    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "preprocessor" not in st.session_state:
        with st.spinner("🔄 Loading models and connecting to Knowledge Graph..."):
            try:
                # Initialize preprocessor with KG data
                config = read_config(config_path)
                executor = Neo4jQueryExecutor(config['URI'], config['USERNAME'], config['PASSWORD'])
                
                # Fetch team and player names
                team_records = executor.execute_query("MATCH (t:Team) RETURN t.name AS name")
                team_names = [record['name'] for record in team_records] if team_records else []
                
                player_records = executor.execute_query("MATCH (p:Player) RETURN p.player_name AS name")
                player_names = [record['name'] for record in player_records] if player_records else []
                
                st.session_state.preprocessor = FPLInputPreprocessor(
                    embedding_model_name="all-MiniLM-L6-v2",
                    team_names_from_kg=team_names,
                    player_names_from_kg=player_names
                )
                st.session_state.executor = executor
                st.session_state.query_generator = CypherQueryGenerator()
                
            except Exception as e:
                st.error(f"❌ Initialization Error: {str(e)}")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ FPL Assistant Settings")
        
        show_debug = st.checkbox("🔧 Show Debug Info", value=False)
        show_kg_data = st.checkbox("📊 Show Raw KG Data", value=True)
        
        st.divider()
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        
        st.divider()
        st.success(f"✅ Connected to Neo4j")
        st.info("💡 **Try asking:**\n\n"
                "• How many goals did Salah score?\n"
                "• Compare Haaland and Kane\n"
                "• Top scoring defenders\n"
                "• Arsenal's performance\n"
                "• Recommend a midfielder")
    
    # Main UI
    st.title("⚽ FPL Graph-RAG Assistant")
    st.markdown("Ask questions about Fantasy Premier League players, teams, and statistics powered by Knowledge Graph!")
    
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Your question:",
            placeholder="e.g., How many goals did Mohamed Salah score?",
            label_visibility="collapsed",
            key="query_input"
        )
    
    with col2:
        run_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    # Process query
    if run_button and query:
        with st.spinner("🤔 Processing your query..."):
            try:
                # Step 1: Preprocess
                processed = st.session_state.preprocessor.process(query)
                
                # Step 2: Generate Cypher
                cypher_query = st.session_state.query_generator.get_query_template(processed)
                
                # Step 3: Execute query
                results = st.session_state.executor.execute_query(
                    cypher_query,
                    processed.cypher_params
                )
                
                # Step 4: Format response
                answer = format_response(processed.intent, results)
                
                # Step 5: Save to history
                st.session_state.history.append({
                    "query": query,
                    "intent": processed.intent,
                    "confidence": processed.confidence,
                    "entities": processed.entities,
                    "cypher": cypher_query,
                    "params": processed.cypher_params,
                    "results": results,
                    "answer": answer
                })
                
                st.success("✅ Query processed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display conversation history
    if st.session_state.history:
        st.divider()
        st.subheader("💬 Conversation History")
        
        for i, turn in enumerate(reversed(st.session_state.history)):
            with st.container():
                # Query header
                st.markdown(f"### 🧑 Query {len(st.session_state.history) - i}: {turn['query']}")
                
                # Create columns for KG data and answer
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.markdown("**📊 Knowledge Graph Results**")
                    
                    if show_kg_data:
                        if turn['results']:
                            # Display as dataframe if possible
                            try:
                                df = pd.DataFrame(turn['results'])
                                st.dataframe(df, use_container_width=True)
                            except:
                                st.json(turn['results'])
                        else:
                            st.info("No data found in Knowledge Graph")
                    else:
                        st.info(f"Found {len(turn['results'])} records")
                
                with col2:
                    st.markdown("**💬 Answer**")
                    st.info(turn['answer'])
                
                # Debug information
                if show_debug:
                    with st.expander("🔧 Technical Details"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**Intent Classification:**")
                            st.write(f"- Intent: `{turn['intent']}`")
                            st.write(f"- Confidence: `{turn['confidence']:.2f}`")
                            st.write(f"- Entities: `{turn['entities']}`")
                        
                        with col_b:
                            st.write("**Query Parameters:**")
                            st.json(turn['params'])
                        
                        st.write("**Generated Cypher Query:**")
                        st.code(turn['cypher'].strip(), language="cypher")
                
                st.divider()
    
    else:
        # Empty state
        st.info("👋 Welcome! Ask your first FPL question above to get started.")

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    run_streamlit_app()






