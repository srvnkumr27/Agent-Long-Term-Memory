"""Example: Multi-session interaction tracking and context building."""

import logging
from src.agent import MemoryAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simulate_session(user_id: str, session_num: int, messages: list):
    """Simulate a conversation session."""
    agent = MemoryAgent(user_id=user_id, backend="sqlite")
    
    print(f"\n{'='*60}")
    print(f"Session {session_num} - {len(messages)} interactions")
    print('='*60)
    
    for i, msg in enumerate(messages, 1):
        print(f"\n[{i}] User: {msg}")
        response = agent.chat(msg)
        print(f"Agent: {response['response'][:150]}...")
    
    # Get session info
    profile = agent.get_user_profile()
    print(f"\nAfter Session {session_num}:")
    print(f"  Total interactions: {profile['statistics']['total_interactions']}")
    print(f"  Learned preferences: {profile['preferences']}")
    
    agent.close()


def main():
    """Demonstrate multi-session memory and context."""
    
    user_id = "data_analyst_001"
    
    # Session 1: User establishes initial preferences
    session1_messages = [
        "I'm a data analyst working with Python",
        "I prefer detailed explanations with code examples",
        "Tell me about pandas for data manipulation"
    ]
    simulate_session(user_id, 1, session1_messages)
    
    # Session 2: Different day, agent remembers preferences
    session2_messages = [
        "How do I handle missing data in pandas?",
        "Show me some visualization techniques",
    ]
    simulate_session(user_id, 2, session2_messages)
    
    # Session 3: Another context - preferences still applied
    session3_messages = [
        "Explain SQL for data analysis",
        "What's the difference between JOIN operations?"
    ]
    simulate_session(user_id, 3, session3_messages)
    
    # Final session - comprehensive profile review
    print(f"\n{'='*60}")
    print("Final User Profile Across All Sessions")
    print('='*60)
    
    final_agent = MemoryAgent(user_id=user_id, backend="sqlite")
    profile = final_agent.get_user_profile()
    
    print(f"\nUser: {profile['user_id']}")
    print(f"Profile Created: {profile['created_at']}")
    print(f"Last Interaction: {profile['last_interaction']}")
    print(f"\nStatistics:")
    stats = profile['statistics']
    print(f"  Total Interactions: {stats['total_interactions']}")
    print(f"  Average Feedback Score: {stats.get('average_feedback_score', 0):.2f}")
    if stats.get('first_interaction'):
        print(f"  First Interaction: {stats['first_interaction']}")
    if stats.get('last_interaction'):
        print(f"  Last Interaction: {stats['last_interaction']}")
    
    print(f"\nLearned Preferences:")
    for pref_name, pref_value in profile['preferences'].items():
        print(f"  - {pref_name}: {pref_value}")
    
    print(f"\nTop Topics of Interest:")
    for topic, freq in stats.get('top_topics', []):
        print(f"  - {topic}: mentioned {int(freq)} times")
    
    final_agent.close()
    print("\n✓ Multi-session example completed!")


if __name__ == "__main__":
    main()
