import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import shutil # Added for deleting user directories

# Base directory for storing all user data
USER_DATA_BASE_DIR = "user_data"

def _get_user_dir(user_id: str) -> str:
    """Returns the absolute path to a user's data directory."""
    user_dir = os.path.join(USER_DATA_BASE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def _get_discussion_filepath(user_id: str, discussion_id: str) -> str:
    """Returns the absolute path to a specific discussion file."""
    return os.path.join(_get_user_dir(user_id), f"{discussion_id}.json")

def create_new_discussion(user_id: str, title: str = "New Discussion") -> Dict[str, Any]:
    """
    Creates a new discussion entry.
    Returns the new discussion metadata.
    """
    discussion_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    new_discussion = {
        "id": discussion_id,
        "title": title,
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": []
    }
    # Save immediately to create the file
    save_discussion(user_id, discussion_id, title, [])
    print(f"Created new discussion for user {user_id}: {title} ({discussion_id})")
    return new_discussion

def save_discussion(user_id: str, discussion_id: str, title: str, messages: List[Dict[str, Any]]):
    """
    Saves a discussion's chat history and metadata.
    """
    filepath = _get_discussion_filepath(user_id, discussion_id)
    timestamp = datetime.now().isoformat()
    discussion_data = {
        "id": discussion_id,
        "title": title,
        "updated_at": timestamp,
        "messages": messages
    }
    # If the file exists, try to preserve 'created_at'
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                discussion_data["created_at"] = existing_data.get("created_at", timestamp)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode existing JSON for {filepath}. Overwriting.")
        except FileNotFoundError: # Should not happen if os.path.exists is true, but for robustness
            pass
    else:
        discussion_data["created_at"] = timestamp # New file, set created_at

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(discussion_data, f, indent=4)
        print(f"Saved discussion {discussion_id} for user {user_id} to {filepath}")
    except IOError as e:
        print(f"Error saving discussion {discussion_id} for user {user_id}: {e}")
        raise

def load_discussion(user_id: str, discussion_id: str) -> Optional[Dict[str, Any]]:
    """
    Loads a specific discussion's chat history and metadata.
    Returns None if not found.
    """
    filepath = _get_discussion_filepath(user_id, discussion_id)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                discussion_data = json.load(f)
                print(f"Loaded discussion {discussion_id} for user {user_id} from {filepath}")
                return discussion_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for discussion {discussion_id} ({filepath}): {e}")
            return None
        except IOError as e:
            print(f"Error loading discussion {discussion_id} ({filepath}): {e}")
            return None
    print(f"Discussion {discussion_id} not found for user {user_id}.")
    return None

def list_discussions(user_id: str) -> List[Dict[str, Any]]:
    """
    Lists all discussions for a given user, returning their IDs and titles.
    """
    user_dir = _get_user_dir(user_id)
    discussions = []
    if os.path.exists(user_dir):
        for filename in os.listdir(user_dir):
            if filename.endswith(".json"):
                discussion_id = filename[:-5] # Remove .json extension
                filepath = os.path.join(user_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        discussions.append({
                            "id": discussion_id,
                            "title": data.get("title", f"Untitled Discussion {discussion_id[:8]}"),
                            "created_at": data.get("created_at"),
                            "updated_at": data.get("updated_at")
                        })
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not decode JSON for {filepath} when listing: {e}")
                except IOError as e:
                    print(f"Warning: Could not read file {filepath} when listing: {e}")
    # Sort by updated_at, most recent first
    discussions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    print(f"Listed {len(discussions)} discussions for user {user_id}.")
    return discussions

def delete_discussion(user_id: str, discussion_id: str) -> bool:
    """
    Deletes a specific discussion file.
    Returns True if successful, False otherwise.
    """
    filepath = _get_discussion_filepath(user_id, discussion_id)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"Deleted discussion {discussion_id} for user {user_id}.")
            return True
        except OSError as e:
            print(f"Error deleting discussion {discussion_id} ({filepath}): {e}")
            return False
    print(f"Attempted to delete discussion {discussion_id} but file not found.")
    return False

def delete_all_user_data(user_id: str) -> bool:
    """
    Deletes the entire directory for a given user.
    Use with extreme caution!
    """
    user_dir = _get_user_dir(user_id)
    if os.path.exists(user_dir):
        try:
            shutil.rmtree(user_dir)
            print(f"Deleted all data for user {user_id}.")
            return True
        except OSError as e:
            print(f"Error deleting user directory {user_dir}: {e}")
            return False
    print(f"Attempted to delete user data for {user_id} but directory not found.")
    return False

if __name__ == '__main__':
    # Example Usage for testing
    test_user_id = "test_user_123"
    
    print(f"\n--- Testing for user: {test_user_id} ---")

    # Clean up previous test data
    delete_all_user_data(test_user_id)

    # Create a new discussion
    new_disc_meta = create_new_discussion(test_user_id, "My First Dissertation Chat")
    current_disc_id = new_disc_meta["id"]
    print(f"Created: {new_disc_meta}")

    # Add some messages
    messages = [
        {"role": "user", "content": "Hello ESI!"},
        {"role": "assistant", "content": "Hello there! How can I help?"}
    ]
    save_discussion(test_user_id, current_disc_id, "My First Dissertation Chat", messages)

    # Load and verify
    loaded_disc = load_discussion(test_user_id, current_disc_id)
    print(f"\nLoaded: {loaded_disc}")
    assert loaded_disc["messages"] == messages

    # Create another discussion
    new_disc_meta_2 = create_new_discussion(test_user_id, "Research Methods Ideas")
    current_disc_id_2 = new_disc_meta_2["id"]
    messages_2 = [
        {"role": "user", "content": "What are qualitative methods?"},
        {"role": "assistant", "content": "Qualitative methods explore non-numerical data."}
    ]
    save_discussion(test_user_id, current_disc_id_2, "Research Methods Ideas", messages_2)

    # List discussions
    all_discs = list_discussions(test_user_id)
    print(f"\nAll discussions for {test_user_id}:")
    for d in all_discs:
        print(f"  - {d['title']} (ID: {d['id']})")
    assert len(all_discs) == 2

    # Update first discussion
    messages.append({"role": "user", "content": "Tell me more about ethics."})
    save_discussion(test_user_id, current_disc_id, "My First Dissertation Chat", messages)
    loaded_disc_updated = load_discussion(test_user_id, current_disc_id)
    print(f"\nUpdated first discussion: {loaded_disc_updated['messages'][-1]['content']}")
    assert len(loaded_disc_updated["messages"]) == 3

    # Delete one discussion
    print(f"\nDeleting discussion {current_disc_id_2}...")
    delete_discussion(test_user_id, current_disc_id_2)
    all_discs_after_delete = list_discussions(test_user_id)
    print(f"Discussions after delete: {len(all_discs_after_delete)}")
    assert len(all_discs_after_delete) == 1

    # Clean up
    print("\nCleaning up all test data...")
    delete_all_user_data(test_user_id)
    all_discs_after_cleanup = list_discussions(test_user_id)
    assert len(all_discs_after_cleanup) == 0
    print("Cleanup complete. All tests passed.")
