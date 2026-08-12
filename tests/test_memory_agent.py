"""Integration test suite for memory agent."""

import pytest
import json
from datetime import datetime, timedelta
from src.agent import MemoryAgent
from src.memory_manager import MemoryManager
from src.models.user_profile import UserProfile, UserInteraction, UserPreference, PreferenceType


class TestMemoryAgent:
    """Test suite for MemoryAgent functionality."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        agent = MemoryAgent(user_id="test_user", backend="sqlite")
        yield agent
        agent.close()

    @pytest.fixture
    def memory_manager(self):
        """Create a test memory manager."""
        manager = MemoryManager(backend="sqlite", database_path=":memory:")
        yield manager
        manager.close()

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.user_id == "test_user"
        assert agent.session_id is not None
        assert agent.memory is not None

    def test_user_profile_creation(self, memory_manager):
        """Test creating a new user profile."""
        user_id = "test_user_1"
        profile = memory_manager.get_or_create_profile(user_id)
        
        assert profile.user_id == user_id
        assert profile.interaction_count == 0
        assert len(profile.preferences) == 0

    def test_user_preference_addition(self, memory_manager):
        """Test adding preferences to user profile."""
        user_id = "test_user_2"
        profile = memory_manager.get_or_create_profile(user_id)
        
        # Add preference
        pref = UserPreference(
            type=PreferenceType.DETAIL_LEVEL,
            value="detailed"
        )
        profile.add_preference(pref)
        memory_manager.save_profile(profile)
        
        # Retrieve and verify
        retrieved = memory_manager.get_or_create_profile(user_id)
        assert PreferenceType.DETAIL_LEVEL in retrieved.preferences
        assert retrieved.preferences[PreferenceType.DETAIL_LEVEL].value == "detailed"

    def test_interaction_recording(self, memory_manager):
        """Test recording interactions."""
        user_id = "test_user_3"
        
        interaction = memory_manager.record_interaction(
            user_id=user_id,
            session_id="session_1",
            user_message="Hello, agent!",
            agent_response="Hello, user!"
        )
        
        assert interaction.user_message == "Hello, agent!"
        assert interaction.agent_response == "Hello, user!"
        
        # Verify stored
        history = memory_manager.get_interaction_history(user_id)
        assert len(history) > 0
        assert history[0].user_message == "Hello, agent!"

    def test_preference_extraction(self, memory_manager):
        """Test automatic preference extraction from interactions."""
        user_id = "test_user_4"
        
        interaction = UserInteraction(
            session_id="session_1",
            user_message="I prefer detailed and comprehensive explanations",
            agent_response="Understood"
        )
        
        preferences = memory_manager.extract_preferences(user_id, interaction)
        
        assert len(preferences) > 0
        assert any(p.type == PreferenceType.DETAIL_LEVEL for p in preferences)
        assert any(p.value == "detailed" for p in preferences)

    def test_interaction_search(self, memory_manager):
        """Test searching past interactions."""
        user_id = "test_user_5"
        
        # Record multiple interactions
        for msg in ["machine learning", "neural networks", "deep learning"]:
            memory_manager.record_interaction(
                user_id=user_id,
                session_id="session_1",
                user_message=msg,
                agent_response=f"Response to {msg}"
            )
        
        # Search
        results = memory_manager.search_interactions(user_id, "learning", limit=5)
        
        assert len(results) > 0
        assert any("learning" in r.user_message for r in results)

    def test_user_statistics(self, memory_manager):
        """Test retrieving user statistics."""
        user_id = "test_user_6"
        
        # Record interactions with feedback
        for i in range(3):
            memory_manager.record_interaction(
                user_id=user_id,
                session_id="session_1",
                user_message=f"Message {i}",
                agent_response=f"Response {i}",
                feedback_score=0.8 + (i * 0.05)
            )
        
        stats = memory_manager.get_all_user_stats(user_id)
        
        assert stats["total_interactions"] == 3
        assert stats["average_feedback_score"] > 0

    def test_old_interaction_cleanup(self, memory_manager):
        """Test clearing old interactions."""
        user_id = "test_user_7"
        
        # Record interaction
        memory_manager.record_interaction(
            user_id=user_id,
            session_id="session_1",
            user_message="Test message",
            agent_response="Test response"
        )
        
        # Clear interactions older than 0 days (should delete all)
        deleted = memory_manager.cleanup_old_data(user_id, days=0)
        
        assert deleted > 0

    def test_multi_session_context(self, memory_manager):
        """Test maintaining context across sessions."""
        user_id = "test_user_8"
        
        # Session 1
        interaction1 = memory_manager.record_interaction(
            user_id=user_id,
            session_id="session_1",
            user_message="I like Python",
            agent_response="Python is great!"
        )
        
        # Session 2
        interaction2 = memory_manager.record_interaction(
            user_id=user_id,
            session_id="session_2",
            user_message="Tell me about Django",
            agent_response="Django is a Python framework"
        )
        
        # Get all interactions
        history = memory_manager.get_interaction_history(user_id, limit=10)
        
        assert len(history) >= 2
        sessions = {i.session_id for i in history}
        assert "session_1" in sessions
        assert "session_2" in sessions

    def test_personalized_context_generation(self, memory_manager):
        """Test generating personalized context for LLM."""
        user_id = "test_user_9"
        
        profile = memory_manager.get_or_create_profile(user_id)
        pref = UserPreference(
            type=PreferenceType.COMMUNICATION_STYLE,
            value="formal"
        )
        profile.add_preference(pref)
        memory_manager.save_profile(profile)
        
        # Record interaction
        memory_manager.record_interaction(
            user_id=user_id,
            session_id="session_1",
            user_message="Explain quantum computing",
            agent_response="Quantum computing uses quantum mechanics..."
        )
        
        # Generate context
        context = memory_manager.provide_personalized_context(user_id)
        
        assert "User Profile Information" in context
        assert "formal" in context or "Preferences" in context

    def test_backend_switching(self):
        """Test switching between backends."""
        user_id = "test_user_10"
        
        # Use SQLite
        manager_sqlite = MemoryManager(backend="sqlite", database_path=":memory:")
        manager_sqlite.record_interaction(
            user_id=user_id,
            session_id="session_1",
            user_message="SQLite test",
            agent_response="SQLite response"
        )
        manager_sqlite.close()
        
        # Use ChromaDB
        manager_chromadb = MemoryManager(backend="chromadb")
        manager_chromadb.record_interaction(
            user_id=user_id,
            session_id="session_1",
            user_message="ChromaDB test",
            agent_response="ChromaDB response"
        )
        manager_chromadb.close()
        
        # Both should work
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
