"""Memory manager for orchestrating storage and retrieval."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.storage.base_storage import BaseStorage
from src.storage.sqlite_storage import SQLiteStorage
from src.storage.chromadb_storage import ChromaDBStorage
from src.storage.pinecone_storage import PineconeStorage
from src.models.user_profile import UserProfile, UserInteraction, UserPreference, PreferenceType


logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages user memory across multiple storage backends."""

    def __init__(self, backend: str = "sqlite", **backend_config):
        """Initialize memory manager with specified backend."""
        self.backend_type = backend
        self.storage = self._initialize_backend(backend, **backend_config)
        logger.info(f"Memory manager initialized with {backend} backend")

    def _initialize_backend(self, backend: str, **config) -> BaseStorage:
        """Initialize the appropriate storage backend."""
        if backend.lower() == "sqlite":
            return SQLiteStorage(
                database_path=config.get("database_path", "./data/memory.db")
            )
        elif backend.lower() == "chromadb":
            return ChromaDBStorage(
                persist_directory=config.get("persist_directory", "./data/chromadb"),
                collection_name=config.get("collection_name", "interactions")
            )
        elif backend.lower() == "pinecone":
            return PineconeStorage(
                api_key=config.get("api_key"),
                environment=config.get("environment", "us-west1-gcp"),
                index_name=config.get("index_name", "memory-agent"),
                vector_dimension=config.get("vector_dimension", 1536)
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get existing user profile or create a new one."""
        profile = self.storage.get_user_profile(user_id)
        return profile

    def save_profile(self, profile: UserProfile) -> None:
        """Save user profile."""
        self.storage.save_user_profile(profile)
        logger.debug(f"Profile saved for user {profile.user_id}")

    def record_interaction(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        agent_response: str,
        feedback_score: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> UserInteraction:
        """Record a user-agent interaction."""
        interaction = UserInteraction(
            session_id=session_id,
            user_message=user_message,
            agent_response=agent_response,
            feedback_score=feedback_score,
            tags=tags or []
        )

        # Save to storage
        self.storage.record_interaction(user_id, interaction)

        # Update user profile
        profile = self.get_or_create_profile(user_id)
        profile.record_interaction(interaction)
        self.save_profile(profile)

        logger.debug(f"Interaction recorded for user {user_id}")
        return interaction

    def extract_preferences(
        self, user_id: str, interaction: UserInteraction
    ) -> List[UserPreference]:
        """Extract user preferences from interaction."""
        preferences = []

        # Analyze user message for preference hints
        user_msg = interaction.user_message.lower()

        # Detect detail preference
        if any(word in user_msg for word in ["detailed", "comprehensive", "in depth", "explain"]):
            pref = UserPreference(
                type=PreferenceType.DETAIL_LEVEL,
                value="detailed"
            )
            preferences.append(pref)
        elif any(word in user_msg for word in ["brief", "short", "quick", "summary", "tldr"]):
            pref = UserPreference(
                type=PreferenceType.DETAIL_LEVEL,
                value="brief"
            )
            preferences.append(pref)

        # Detect communication style
        if any(word in user_msg for word in ["formal", "professional", "business"]):
            pref = UserPreference(
                type=PreferenceType.COMMUNICATION_STYLE,
                value="formal"
            )
            preferences.append(pref)
        elif any(word in user_msg for word in ["casual", "friendly", "conversational"]):
            pref = UserPreference(
                type=PreferenceType.COMMUNICATION_STYLE,
                value="casual"
            )
            preferences.append(pref)

        return preferences

    def get_interaction_history(
        self, user_id: str, limit: int = 10, offset: int = 0
    ) -> List[UserInteraction]:
        """Get user's interaction history."""
        return self.storage.get_interactions(user_id, limit=limit, offset=offset)

    def search_interactions(
        self, user_id: str, query: str, limit: int = 5
    ) -> List[UserInteraction]:
        """Search user's past interactions."""
        return self.storage.search_interactions(user_id, query, limit=limit)

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context for agent."""
        profile = self.get_or_create_profile(user_id)
        recent_interactions = self.get_interaction_history(user_id, limit=5)
        stats = self.storage.get_user_stats(user_id)

        context = {
            "user_id": user_id,
            "preferences": profile.get_preference_summary(),
            "recent_interactions": [
                {
                    "timestamp": i.timestamp.isoformat(),
                    "user_message": i.user_message,
                    "feedback": i.feedback_score
                }
                for i in recent_interactions
            ],
            "top_topics": profile.get_top_topics(3),
            "stats": stats
        }

        return context

    def provide_personalized_context(self, user_id: str) -> str:
        """Generate personalized context string for LLM."""
        profile = self.get_or_create_profile(user_id)
        
        context_parts = ["User Profile Information:"]
        
        if profile.preferences:
            context_parts.append("Preferences:")
            for pref_type, pref in profile.preferences.items():
                context_parts.append(f"  - {pref_type.value}: {pref.value}")
        
        if profile.common_topics:
            top_topics = profile.get_top_topics(3)
            context_parts.append("Interests:")
            for topic, freq in top_topics:
                context_parts.append(f"  - {topic} (mentioned {int(freq)} times)")
        
        recent = self.get_interaction_history(user_id, limit=3)
        if recent:
            context_parts.append("Recent Context:")
            for interaction in recent:
                context_parts.append(f"  - {interaction.user_message[:100]}...")
        
        return "\n".join(context_parts)

    def cleanup_old_data(self, user_id: str, days: int = 90) -> int:
        """Clean up old interactions beyond retention period."""
        deleted = self.storage.clear_old_interactions(user_id, days=days)
        logger.info(f"Cleaned up {deleted} old interactions for user {user_id}")
        return deleted

    def get_all_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a user."""
        profile = self.get_or_create_profile(user_id)
        storage_stats = self.storage.get_user_stats(user_id)
        
        return {
            **storage_stats,
            "profile_preferences": profile.get_preference_summary(),
            "total_preferences": len(profile.preferences),
            "average_feedback_score": profile.get_average_feedback_score(),
            "top_topics": profile.get_top_topics(5)
        }

    def close(self) -> None:
        """Close memory manager and storage backend."""
        self.storage.close()
        logger.info("Memory manager closed")
