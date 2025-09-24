"""
Feedback storage module for the Joke CLI application.

Handles persistence and retrieval of user feedback data using JSON-based local storage.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .models import FeedbackEntry
from .config import get_feedback_storage_dir, FEEDBACK_STORAGE_FILENAME


class FeedbackStorage:
    """Handles feedback data persistence and retrieval."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize feedback storage with optional custom directory."""
        self.storage_dir = storage_dir or get_feedback_storage_dir()
        self.storage_file = self.storage_dir / FEEDBACK_STORAGE_FILENAME
        self._ensure_storage_directory()
    
    def _ensure_storage_directory(self) -> None:
        """Ensure the storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_feedback_data(self) -> Dict[str, Any]:
        """Load feedback data from storage file."""
        if not self.storage_file.exists():
            return {
                "feedback_entries": [],
                "stats": {
                    "total_jokes": 0,
                    "average_rating": 0.0,
                    "category_stats": {}
                }
            }
        
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # If file is corrupted or unreadable, start fresh
            return {
                "feedback_entries": [],
                "stats": {
                    "total_jokes": 0,
                    "average_rating": 0.0,
                    "category_stats": {}
                }
            }
    
    def _save_feedback_data(self, data: Dict[str, Any]) -> None:
        """Save feedback data to storage file."""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            raise RuntimeError(f"Failed to save feedback data: {e}")
    
    def _feedback_entry_to_dict(self, entry: FeedbackEntry) -> Dict[str, Any]:
        """Convert FeedbackEntry to dictionary for JSON serialization."""
        entry_dict = asdict(entry)
        # Convert datetime to ISO format string
        entry_dict['timestamp'] = entry.timestamp.isoformat()
        return entry_dict
    
    def _dict_to_feedback_entry(self, entry_dict: Dict[str, Any]) -> FeedbackEntry:
        """Convert dictionary to FeedbackEntry object."""
        # Convert ISO format string back to datetime
        if isinstance(entry_dict['timestamp'], str):
            entry_dict['timestamp'] = datetime.fromisoformat(entry_dict['timestamp'])
        
        return FeedbackEntry(**entry_dict)
    
    def save_feedback(self, feedback: FeedbackEntry) -> None:
        """
        Save a feedback entry to storage.
        
        Args:
            feedback: The FeedbackEntry object to save
            
        Raises:
            RuntimeError: If saving fails
        """
        data = self._load_feedback_data()
        
        # Add new feedback entry
        entry_dict = self._feedback_entry_to_dict(feedback)
        data["feedback_entries"].append(entry_dict)
        
        # Update statistics
        self._update_statistics(data)
        
        # Save updated data
        self._save_feedback_data(data)
    
    def _update_statistics(self, data: Dict[str, Any]) -> None:
        """Update statistics based on current feedback entries."""
        entries = data["feedback_entries"]
        
        if not entries:
            data["stats"] = {
                "total_jokes": 0,
                "average_rating": 0.0,
                "category_stats": {}
            }
            return
        
        # Calculate overall statistics
        total_jokes = len(entries)
        total_rating = sum(entry["rating"] for entry in entries)
        average_rating = total_rating / total_jokes if total_jokes > 0 else 0.0
        
        # Calculate category statistics
        category_stats = {}
        category_counts = {}
        category_ratings = {}
        
        for entry in entries:
            category = entry["category"]
            rating = entry["rating"]
            
            if category not in category_counts:
                category_counts[category] = 0
                category_ratings[category] = 0
            
            category_counts[category] += 1
            category_ratings[category] += rating
        
        for category in category_counts:
            count = category_counts[category]
            avg_rating = category_ratings[category] / count if count > 0 else 0.0
            category_stats[category] = {
                "count": count,
                "avg_rating": round(avg_rating, 2)
            }
        
        # Update stats in data
        data["stats"] = {
            "total_jokes": total_jokes,
            "average_rating": round(average_rating, 2),
            "category_stats": category_stats
        }
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Retrieve aggregated feedback statistics.
        
        Returns:
            Dictionary containing feedback statistics
        """
        data = self._load_feedback_data()
        return data["stats"]
    
    def get_all_feedback(self) -> List[FeedbackEntry]:
        """
        Retrieve all feedback entries.
        
        Returns:
            List of FeedbackEntry objects
        """
        data = self._load_feedback_data()
        entries = []
        
        for entry_dict in data["feedback_entries"]:
            try:
                entry = self._dict_to_feedback_entry(entry_dict)
                entries.append(entry)
            except (KeyError, ValueError, TypeError):
                # Skip corrupted entries
                continue
        
        return entries
    
    def get_feedback_by_category(self, category: str) -> List[FeedbackEntry]:
        """
        Retrieve feedback entries for a specific category.
        
        Args:
            category: The joke category to filter by
            
        Returns:
            List of FeedbackEntry objects for the specified category
        """
        all_feedback = self.get_all_feedback()
        return [entry for entry in all_feedback if entry.category == category]
    
    def export_feedback(self, export_path: Optional[Path] = None) -> Path:
        """
        Export feedback data to a JSON file.
        
        Args:
            export_path: Optional path for export file. If None, uses default location.
            
        Returns:
            Path to the exported file
            
        Raises:
            RuntimeError: If export fails
        """
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = self.storage_dir / f"feedback_export_{timestamp}.json"
        
        data = self._load_feedback_data()
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            raise RuntimeError(f"Failed to export feedback data: {e}")
        
        return export_path
    
    def clear_all_feedback(self) -> None:
        """
        Clear all feedback data (for testing purposes).
        
        Warning: This permanently deletes all feedback data.
        """
        data = {
            "feedback_entries": [],
            "stats": {
                "total_jokes": 0,
                "average_rating": 0.0,
                "category_stats": {}
            }
        }
        self._save_feedback_data(data)


# Convenience functions for module-level access
_default_storage = None

def get_default_storage() -> FeedbackStorage:
    """Get the default feedback storage instance."""
    global _default_storage
    if _default_storage is None:
        _default_storage = FeedbackStorage()
    return _default_storage

def save_feedback(feedback: FeedbackEntry) -> None:
    """Save feedback using the default storage instance."""
    storage = get_default_storage()
    storage.save_feedback(feedback)

def get_feedback_stats() -> Dict[str, Any]:
    """Get feedback statistics using the default storage instance."""
    storage = get_default_storage()
    return storage.get_feedback_stats()

def get_all_feedback() -> List[FeedbackEntry]:
    """Get all feedback using the default storage instance."""
    storage = get_default_storage()
    return storage.get_all_feedback()

def export_feedback(export_path: Optional[Path] = None) -> Path:
    """Export feedback using the default storage instance."""
    storage = get_default_storage()
    return storage.export_feedback(export_path)