"""Example: Basic SQLite-based agent with preference learning."""

import logging
from src.agent import MemoryAgent
from src.models.user_profile import PreferenceType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate basic SQLite-based agent with multi-session memory."""
    
    # Initialize agent
    user_id = "user_123"
    agent = MemoryAgent(user_id=user_id, backend="sqlite")
    
    print("=" * 60)
    print("Memory Agent - SQLite Example")
    print("=" * 60)
    
    # Session 1: User expresses preferences
    print("\n[Session 1] Initial interaction...")
    response1 = agent.chat("I prefer detailed, comprehensive explanations")
    print(f"Agent: {response1['response'][:200]}...")
    
    response2 = agent.chat("Tell me about machine learning")
    print(f"Agent: {response2['response'][:200]}...")
    
    # Get session summary
    session_summary = agent.get_session_summary()
    print(f"\nSession Summary: {session_summary['interaction_count']} interactions")
    
    # New session - preferences should be remembered
    print("\n[Session 2] New session - agent remembers preferences...")
    agent2 = MemoryAgent(user_id=user_id, backend="sqlite")
    
    # No need to re-state preferences - agent learned them
    response3 = agent2.chat("Explain quantum computing")
    print(f"Agent: {response3['response'][:200]}...")
    print("(Notice: Agent remembered the preference for detailed explanations!)")
    
    # View user profile
    profile = agent2.get_user_profile()
    print(f"\nUser Profile Summary:")
    print(f"  Total Interactions: {profile['statistics']['total_interactions']}")
    print(f"  Preferences: {profile['preferences']}")
    print(f"  Top Topics: {profile['statistics'].get('top_topics', [])}")
    
    # Explicit preference setting
    print("\n[Preferences] Explicitly setting preferences...")
    agent2.set_preference("communication_style", "formal")
    
    response4 = agent2.chat("What is natural language processing?")
    print(f"Agent: {response4['response'][:200]}...")
    
    # Search past interactions
    print("\n[Search] Searching past interactions...")
    results = agent2.search_past_interactions("machine learning")
    print(f"Found {len(results)} interactions related to 'machine learning'")
    for result in results:
        print(f"  - {result['user_message'][:60]}...")
    
    # Cleanup
    agent.close()
    agent2.close()
    print("\n✓ SQLite example completed!")


if __name__ == "__main__":
    main()
