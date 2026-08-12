"""Example: ChromaDB with vector-based semantic search."""

import logging
from src.agent import MemoryAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate ChromaDB-based agent with semantic search."""
    
    print("=" * 60)
    print("Memory Agent - ChromaDB Semantic Search Example")
    print("=" * 60)
    
    # Initialize agent with ChromaDB backend
    user_id = "user_456"
    agent = MemoryAgent(
        user_id=user_id,
        backend="chromadb",
        persist_directory="./data/chromadb",
        collection_name="conversations"
    )
    
    print("\n[Interactions] Recording various interactions...")
    
    # Record multiple interactions on different topics
    interactions_data = [
        "I want to learn about Python programming",
        "How do decorators work in Python?",
        "Explain async/await in Python",
        "Tell me about web development frameworks",
        "What is Django?",
        "How does machine learning work?",
        "Explain neural networks"
    ]
    
    for user_msg in interactions_data:
        response = agent.chat(user_msg)
        print(f"User: {user_msg}")
        print(f"Agent: {response['response'][:100]}...\n")
    
    # Semantic search - find related interactions
    print("\n[Semantic Search] Finding related interactions...")
    
    search_queries = [
        "Python programming concepts",
        "web frameworks",
        "deep learning"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = agent.search_past_interactions(query, limit=3)
        print(f"Found {len(results)} related interactions:")
        for result in results:
            print(f"  - {result['user_message'][:60]}...")
    
    # Get comprehensive profile
    profile = agent.get_user_profile()
    print(f"\nUser Profile:")
    print(f"  Total Interactions: {profile['statistics']['total_interactions']}")
    print(f"  Preferences: {profile['preferences']}")
    
    agent.close()
    print("\n✓ ChromaDB semantic search example completed!")


if __name__ == "__main__":
    main()
