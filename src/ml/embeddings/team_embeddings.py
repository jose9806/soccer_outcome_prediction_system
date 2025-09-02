"""
Team embeddings system for intelligent team name lookup and representation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import pickle
try:
    from rapidfuzz import fuzz, process
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
    except ImportError:
        # Fallback simple matching
        class SimpleFuzz:
            @staticmethod
            def ratio(a, b):
                a, b = a.lower(), b.lower()
                if a == b: return 100
                if a in b or b in a: return 80
                return 50
        class SimpleProcess:
            @staticmethod 
            def extract(query, choices, limit=3, scorer=None):
                scores = [(choice, SimpleFuzz.ratio(query, choice)) for choice in choices]
                scores.sort(key=lambda x: x[1], reverse=True)
                return scores[:limit]
        fuzz = SimpleFuzz()
        process = SimpleProcess()
from datetime import datetime
import logging

from src.config.logging_config import get_logger


@dataclass
class TeamEmbeddingConfig:
    """Configuration for team embeddings."""
    embedding_dim: int = 128
    similarity_threshold: float = 0.8
    fuzzy_threshold: int = 80  # For fuzzy string matching
    cache_dir: str = "data/embeddings"
    update_frequency: int = 100  # Update embeddings every N new matches
    include_aliases: bool = True
    league_weighting: bool = True


class TeamEmbeddings:
    """
    Intelligent team embeddings system for soccer prediction.
    
    Handles:
    - Team name normalization and fuzzy matching
    - Historical performance embeddings  
    - Team aliases and variations
    - League-specific context
    """
    
    def __init__(self, config: Optional[TeamEmbeddingConfig] = None):
        self.config = config or TeamEmbeddingConfig()
        self.logger = get_logger("TeamEmbeddings", color="magenta")
        
        # Core data structures
        self.team_registry: Dict[str, str] = {}  # variations -> canonical name
        self.canonical_teams: Set[str] = set()
        self.team_aliases: Dict[str, List[str]] = {}  # canonical -> aliases
        self.team_embeddings: Dict[str, np.ndarray] = {}
        self.team_stats: Dict[str, Dict] = {}
        
        # Cache and persistence
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Team embeddings system initialized")
    
    def learn_from_data(self, match_data: pd.DataFrame) -> None:
        """
        Learn team names and create embeddings from match data.
        
        Args:
            match_data: DataFrame with match data including team names
        """
        self.logger.info(f"Learning from {len(match_data)} matches")
        
        # Extract all team names
        home_teams = set(match_data['home_team'].dropna().astype(str))
        away_teams = set(match_data['away_team'].dropna().astype(str))
        all_teams = home_teams.union(away_teams)
        
        # Remove invalid entries
        all_teams = {team for team in all_teams if team and team != 'nan' and team != 'None'}
        
        self.logger.info(f"Found {len(all_teams)} unique team names")
        
        # Build team registry with fuzzy matching for aliases
        self._build_team_registry(all_teams)
        
        # Create performance-based embeddings
        self._create_performance_embeddings(match_data)
        
        # Save to cache
        self._save_to_cache()
        
        self.logger.info(f"Team embeddings created for {len(self.canonical_teams)} teams")
    
    def find_team(self, team_name: str) -> Optional[str]:
        """
        Find canonical team name using fuzzy matching.
        
        Args:
            team_name: Team name to search for
            
        Returns:
            Canonical team name or None if not found
        """
        if not team_name or team_name in ['nan', 'None', '']:
            return None
        
        team_name = str(team_name).strip()
        
        # Direct lookup
        if team_name in self.team_registry:
            return self.team_registry[team_name]
        
        # Fuzzy matching
        matches = process.extract(
            team_name, 
            list(self.team_registry.keys()), 
            limit=3,
            scorer=fuzz.ratio
        )
        
        if matches and matches[0][1] >= self.config.fuzzy_threshold:
            canonical = self.team_registry[matches[0][0]]
            
            # Add this variation to registry for future lookups
            self.team_registry[team_name] = canonical
            
            self.logger.debug(f"Fuzzy matched '{team_name}' -> '{canonical}' (score: {matches[0][1]})")
            return canonical
        
        self.logger.warning(f"No match found for team: '{team_name}'")
        return None
    
    def get_embedding(self, team_name: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a team.
        
        Args:
            team_name: Team name
            
        Returns:
            Embedding vector or None if not found
        """
        canonical = self.find_team(team_name)
        if canonical and canonical in self.team_embeddings:
            return self.team_embeddings[canonical]
        return None
    
    def get_team_stats(self, team_name: str) -> Optional[Dict]:
        """
        Get performance statistics for a team.
        
        Args:
            team_name: Team name
            
        Returns:
            Stats dictionary or None if not found
        """
        canonical = self.find_team(team_name)
        if canonical and canonical in self.team_stats:
            return self.team_stats[canonical]
        return None
    
    def get_similarity(self, team1: str, team2: str) -> float:
        """
        Calculate similarity between two teams based on embeddings.
        
        Args:
            team1, team2: Team names
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.get_embedding(team1)
        emb2 = self.get_embedding(team2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_teams(self, team_name: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar teams to given team.
        
        Args:
            team_name: Target team name
            n: Number of similar teams to return
            
        Returns:
            List of (team_name, similarity_score) tuples
        """
        canonical = self.find_team(team_name)
        if not canonical:
            return []
        
        similarities = []
        for other_team in self.canonical_teams:
            if other_team != canonical:
                sim = self.get_similarity(canonical, other_team)
                if sim > 0:
                    similarities.append((other_team, sim))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    def get_all_teams(self) -> List[str]:
        """Get list of all canonical team names."""
        return sorted(list(self.canonical_teams))
    
    def get_team_aliases(self, team_name: str) -> List[str]:
        """Get all known aliases for a team."""
        canonical = self.find_team(team_name)
        if canonical and canonical in self.team_aliases:
            return self.team_aliases[canonical]
        return []
    
    def load_from_cache(self) -> bool:
        """
        Load embeddings from cache.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            registry_file = self.cache_dir / "team_registry.json"
            embeddings_file = self.cache_dir / "embeddings.pkl"
            stats_file = self.cache_dir / "team_stats.json"
            
            if not all(f.exists() for f in [registry_file, embeddings_file, stats_file]):
                return False
            
            # Load registry
            with open(registry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.team_registry = data['registry']
                self.canonical_teams = set(data['canonical_teams'])
                self.team_aliases = data['aliases']
            
            # Load embeddings
            with open(embeddings_file, 'rb') as f:
                self.team_embeddings = pickle.load(f)
            
            # Load stats
            with open(stats_file, 'r', encoding='utf-8') as f:
                self.team_stats = json.load(f)
            
            self.logger.info(f"Loaded embeddings for {len(self.canonical_teams)} teams from cache")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {e}")
            return False
    
    def _build_team_registry(self, team_names: Set[str]) -> None:
        """Build registry mapping team name variations to canonical names."""
        sorted_teams = sorted(team_names, key=len, reverse=True)  # Longer names first
        
        for team in sorted_teams:
            if team in self.team_registry:
                continue
            
            # Find potential matches using fuzzy matching
            matches = process.extract(
                team, 
                list(self.canonical_teams), 
                limit=1,
                scorer=fuzz.ratio
            )
            
            if matches and matches[0][1] >= self.config.fuzzy_threshold:
                # This is a variation of an existing team
                canonical = matches[0][0]
                self.team_registry[team] = canonical
                
                # Add to aliases
                if canonical not in self.team_aliases:
                    self.team_aliases[canonical] = []
                if team not in self.team_aliases[canonical]:
                    self.team_aliases[canonical].append(team)
            else:
                # This is a new canonical team
                canonical = self._normalize_team_name(team)
                self.canonical_teams.add(canonical)
                self.team_registry[team] = canonical
                
                # Initialize aliases
                self.team_aliases[canonical] = [team] if team != canonical else []
                
                # Also register normalized name if different
                if canonical != team:
                    self.team_registry[canonical] = canonical
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team name to canonical form."""
        # Basic normalization
        name = team_name.strip()
        
        # Common replacements
        replacements = {
            'Atl ': 'Atletico ',
            'Dep ': 'Deportes ',
            'CF ': 'Club de Futbol ',
            'FC ': 'Football Club ',
            'SC ': 'Sporting Club '
        }
        
        for old, new in replacements.items():
            if name.startswith(old):
                name = name.replace(old, new, 1)
        
        return name
    
    def _create_performance_embeddings(self, match_data: pd.DataFrame) -> None:
        """Create performance-based embeddings for teams."""
        self.logger.info("Creating performance-based embeddings")
        
        for team in self.canonical_teams:
            # Get all matches for this team
            team_variations = [var for var, canonical in self.team_registry.items() 
                              if canonical == team]
            
            home_matches = match_data[match_data['home_team'].isin(team_variations)]
            away_matches = match_data[match_data['away_team'].isin(team_variations)]
            
            # Calculate performance statistics
            stats = self._calculate_team_stats(team, home_matches, away_matches)
            self.team_stats[team] = stats
            
            # Create embedding vector from stats
            embedding = self._stats_to_embedding(stats)
            self.team_embeddings[team] = embedding
    
    def _calculate_team_stats(self, team: str, home_matches: pd.DataFrame, 
                            away_matches: pd.DataFrame) -> Dict:
        """Calculate comprehensive team statistics."""
        total_matches = len(home_matches) + len(away_matches)
        
        if total_matches == 0:
            return self._default_stats()
        
        # Goals
        home_goals = home_matches['home_score'].fillna(0).sum()
        away_goals = away_matches['away_score'].fillna(0).sum()
        home_conceded = home_matches['away_score'].fillna(0).sum()
        away_conceded = away_matches['home_score'].fillna(0).sum()
        
        total_goals = home_goals + away_goals
        total_conceded = home_conceded + away_conceded
        
        # Win/Draw/Loss
        home_wins = (home_matches['home_score'] > home_matches['away_score']).sum()
        away_wins = (away_matches['away_score'] > away_matches['home_score']).sum()
        home_draws = (home_matches['home_score'] == home_matches['away_score']).sum()
        away_draws = (away_matches['away_score'] == away_matches['home_score']).sum()
        
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        total_losses = total_matches - total_wins - total_draws
        
        # Calculate rates
        win_rate = total_wins / total_matches if total_matches > 0 else 0
        draw_rate = total_draws / total_matches if total_matches > 0 else 0
        loss_rate = total_losses / total_matches if total_matches > 0 else 0
        
        goals_per_game = total_goals / total_matches if total_matches > 0 else 0
        conceded_per_game = total_conceded / total_matches if total_matches > 0 else 0
        
        # Home/Away splits
        home_matches_count = len(home_matches)
        away_matches_count = len(away_matches)
        
        home_win_rate = home_wins / home_matches_count if home_matches_count > 0 else 0
        away_win_rate = away_wins / away_matches_count if away_matches_count > 0 else 0
        
        return {
            'total_matches': total_matches,
            'wins': int(total_wins),
            'draws': int(total_draws),
            'losses': int(total_losses),
            'win_rate': float(win_rate),
            'draw_rate': float(draw_rate),
            'loss_rate': float(loss_rate),
            'goals_scored': int(total_goals),
            'goals_conceded': int(total_conceded),
            'goals_per_game': float(goals_per_game),
            'conceded_per_game': float(conceded_per_game),
            'goal_difference': int(total_goals - total_conceded),
            'home_matches': int(home_matches_count),
            'away_matches': int(away_matches_count),
            'home_win_rate': float(home_win_rate),
            'away_win_rate': float(away_win_rate),
            'last_updated': datetime.now().isoformat()
        }
    
    def _default_stats(self) -> Dict:
        """Default stats for teams with no data."""
        return {
            'total_matches': 0,
            'wins': 0, 'draws': 0, 'losses': 0,
            'win_rate': 0.33, 'draw_rate': 0.33, 'loss_rate': 0.34,
            'goals_scored': 0, 'goals_conceded': 0,
            'goals_per_game': 1.0, 'conceded_per_game': 1.0,
            'goal_difference': 0,
            'home_matches': 0, 'away_matches': 0,
            'home_win_rate': 0.45, 'away_win_rate': 0.25,
            'last_updated': datetime.now().isoformat()
        }
    
    def _stats_to_embedding(self, stats: Dict) -> np.ndarray:
        """Convert team statistics to embedding vector."""
        # Key performance indicators for embedding
        features = [
            stats['win_rate'],
            stats['draw_rate'],
            stats['loss_rate'],
            stats['goals_per_game'],
            stats['conceded_per_game'],
            min(stats['goal_difference'] / 10.0, 3.0),  # Normalized goal difference
            stats['home_win_rate'],
            stats['away_win_rate'],
            min(stats['total_matches'] / 50.0, 1.0),  # Experience factor
        ]
        
        # Pad or truncate to desired embedding dimension
        embedding = np.array(features, dtype=np.float32)
        
        if len(embedding) < self.config.embedding_dim:
            # Pad with random values based on existing features
            padding_size = self.config.embedding_dim - len(embedding)
            noise = np.random.normal(0, 0.1, padding_size).astype(np.float32)
            embedding = np.concatenate([embedding, noise])
        elif len(embedding) > self.config.embedding_dim:
            # Truncate
            embedding = embedding[:self.config.embedding_dim]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _save_to_cache(self) -> None:
        """Save embeddings to cache."""
        try:
            # Save registry
            registry_data = {
                'registry': self.team_registry,
                'canonical_teams': list(self.canonical_teams),
                'aliases': self.team_aliases,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.cache_dir / "team_registry.json", 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            with open(self.cache_dir / "embeddings.pkl", 'wb') as f:
                pickle.dump(self.team_embeddings, f)
            
            # Save stats
            with open(self.cache_dir / "team_stats.json", 'w', encoding='utf-8') as f:
                json.dump(self.team_stats, f, indent=2, ensure_ascii=False)
            
            self.logger.info("Embeddings saved to cache")
            
        except Exception as e:
            self.logger.error(f"Failed to save to cache: {e}")


# Factory functions
def create_team_embeddings(config: Optional[TeamEmbeddingConfig] = None) -> TeamEmbeddings:
    """Create team embeddings system."""
    return TeamEmbeddings(config)


def load_or_create_embeddings(match_data: Optional[pd.DataFrame] = None,
                             config: Optional[TeamEmbeddingConfig] = None) -> TeamEmbeddings:
    """Load embeddings from cache or create new ones."""
    embeddings = TeamEmbeddings(config)
    
    if not embeddings.load_from_cache():
        if match_data is not None:
            embeddings.learn_from_data(match_data)
        else:
            logger = get_logger("TeamEmbeddings")
            logger.warning("No cached embeddings found and no data provided")
    
    return embeddings