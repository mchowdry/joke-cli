"""
Unit tests for the feedback storage module.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, mock_open

from joke_cli.feedback_storage import FeedbackStorage, save_feedback, get_feedback_stats
from joke_cli.models import FeedbackEntry


class TestFeedbackStorage:
    """Test cases for FeedbackStorage class."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage instance for testing."""
        with TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            yield FeedbackStorage(storage_dir)
    
    @pytest.fixture
    def sample_feedback(self):
        """Create sample feedback entries for testing."""
        return [
            FeedbackEntry.create(
                joke_id="123e4567-e89b-12d3-a456-426614174000",
                joke_text="Why did the programmer quit his job? He didn't get arrays!",
                category="programming",
                rating=4,
                user_comment="Pretty good!"
            ),
            FeedbackEntry.create(
                joke_id="123e4567-e89b-12d3-a456-426614174001", 
                joke_text="I told my wife she was drawing her eyebrows too high. She looked surprised.",
                category="general",
                rating=5
            ),
            FeedbackEntry.create(
                joke_id="123e4567-e89b-12d3-a456-426614174002",
                joke_text="Why don't scientists trust atoms? Because they make up everything!",
                category="puns",
                rating=3,
                user_comment="Classic but good"
            )
        ]
    
    def test_init_creates_storage_directory(self):
        """Test that initialization creates the storage directory."""
        with TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir) / "test_storage"
            assert not storage_dir.exists()
            
            storage = FeedbackStorage(storage_dir)
            assert storage_dir.exists()
            assert storage.storage_dir == storage_dir
    
    def test_save_feedback_creates_new_file(self, temp_storage, sample_feedback):
        """Test saving feedback creates a new file when none exists."""
        feedback = sample_feedback[0]
        
        # Ensure file doesn't exist initially
        assert not temp_storage.storage_file.exists()
        
        # Save feedback
        temp_storage.save_feedback(feedback)
        
        # Verify file was created
        assert temp_storage.storage_file.exists()
        
        # Verify content
        with open(temp_storage.storage_file, 'r') as f:
            data = json.load(f)
        
        assert len(data["feedback_entries"]) == 1
        assert data["feedback_entries"][0]["joke_id"] == feedback.joke_id
        assert data["feedback_entries"][0]["rating"] == feedback.rating
    
    def test_save_multiple_feedback_entries(self, temp_storage, sample_feedback):
        """Test saving multiple feedback entries."""
        # Save all sample feedback
        for feedback in sample_feedback:
            temp_storage.save_feedback(feedback)
        
        # Verify all entries were saved
        with open(temp_storage.storage_file, 'r') as f:
            data = json.load(f)
        
        assert len(data["feedback_entries"]) == 3
        
        # Verify each entry
        saved_ids = [entry["joke_id"] for entry in data["feedback_entries"]]
        expected_ids = [feedback.joke_id for feedback in sample_feedback]
        assert saved_ids == expected_ids
    
    def test_statistics_calculation(self, temp_storage, sample_feedback):
        """Test that statistics are calculated correctly."""
        # Save sample feedback (ratings: 4, 5, 3)
        for feedback in sample_feedback:
            temp_storage.save_feedback(feedback)
        
        stats = temp_storage.get_feedback_stats()
        
        # Verify overall statistics
        assert stats["total_jokes"] == 3
        assert stats["average_rating"] == 4.0  # (4 + 5 + 3) / 3
        
        # Verify category statistics
        category_stats = stats["category_stats"]
        assert len(category_stats) == 3
        
        assert category_stats["programming"]["count"] == 1
        assert category_stats["programming"]["avg_rating"] == 4.0
        
        assert category_stats["general"]["count"] == 1
        assert category_stats["general"]["avg_rating"] == 5.0
        
        assert category_stats["puns"]["count"] == 1
        assert category_stats["puns"]["avg_rating"] == 3.0
    
    def test_get_all_feedback(self, temp_storage, sample_feedback):
        """Test retrieving all feedback entries."""
        # Save sample feedback
        for feedback in sample_feedback:
            temp_storage.save_feedback(feedback)
        
        # Retrieve all feedback
        retrieved_feedback = temp_storage.get_all_feedback()
        
        assert len(retrieved_feedback) == 3
        
        # Verify data integrity
        for i, feedback in enumerate(retrieved_feedback):
            assert isinstance(feedback, FeedbackEntry)
            assert feedback.joke_id == sample_feedback[i].joke_id
            assert feedback.rating == sample_feedback[i].rating
            assert feedback.category == sample_feedback[i].category
    
    def test_get_feedback_by_category(self, temp_storage, sample_feedback):
        """Test retrieving feedback by category."""
        # Save sample feedback
        for feedback in sample_feedback:
            temp_storage.save_feedback(feedback)
        
        # Test filtering by programming category
        programming_feedback = temp_storage.get_feedback_by_category("programming")
        assert len(programming_feedback) == 1
        assert programming_feedback[0].category == "programming"
        
        # Test filtering by non-existent category
        nonexistent_feedback = temp_storage.get_feedback_by_category("nonexistent")
        assert len(nonexistent_feedback) == 0
    
    def test_export_feedback(self, temp_storage, sample_feedback):
        """Test exporting feedback data."""
        # Save sample feedback
        for feedback in sample_feedback:
            temp_storage.save_feedback(feedback)
        
        # Export to default location
        export_path = temp_storage.export_feedback()
        
        assert export_path.exists()
        assert export_path.name.startswith("feedback_export_")
        assert export_path.suffix == ".json"
        
        # Verify exported content
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data["feedback_entries"]) == 3
        assert exported_data["stats"]["total_jokes"] == 3
    
    def test_export_feedback_custom_path(self, temp_storage, sample_feedback):
        """Test exporting feedback to a custom path."""
        # Save sample feedback
        for feedback in sample_feedback:
            temp_storage.save_feedback(feedback)
        
        # Export to custom path
        custom_path = temp_storage.storage_dir / "custom_export.json"
        export_path = temp_storage.export_feedback(custom_path)
        
        assert export_path == custom_path
        assert custom_path.exists()
    
    def test_clear_all_feedback(self, temp_storage, sample_feedback):
        """Test clearing all feedback data."""
        # Save sample feedback
        for feedback in sample_feedback:
            temp_storage.save_feedback(feedback)
        
        # Verify data exists
        assert len(temp_storage.get_all_feedback()) == 3
        
        # Clear all feedback
        temp_storage.clear_all_feedback()
        
        # Verify data is cleared
        assert len(temp_storage.get_all_feedback()) == 0
        stats = temp_storage.get_feedback_stats()
        assert stats["total_jokes"] == 0
        assert stats["average_rating"] == 0.0
        assert len(stats["category_stats"]) == 0
    
    def test_load_nonexistent_file(self, temp_storage):
        """Test loading data when storage file doesn't exist."""
        # Ensure file doesn't exist
        assert not temp_storage.storage_file.exists()
        
        # Should return empty data structure
        stats = temp_storage.get_feedback_stats()
        assert stats["total_jokes"] == 0
        assert stats["average_rating"] == 0.0
        assert len(stats["category_stats"]) == 0
        
        feedback = temp_storage.get_all_feedback()
        assert len(feedback) == 0
    
    def test_load_corrupted_file(self, temp_storage):
        """Test handling of corrupted JSON file."""
        # Create corrupted JSON file
        with open(temp_storage.storage_file, 'w') as f:
            f.write("invalid json content {")
        
        # Should handle gracefully and return empty data
        stats = temp_storage.get_feedback_stats()
        assert stats["total_jokes"] == 0
        
        feedback = temp_storage.get_all_feedback()
        assert len(feedback) == 0
    
    def test_save_feedback_io_error(self, temp_storage, sample_feedback):
        """Test handling of IO errors during save."""
        feedback = sample_feedback[0]
        
        # Mock open to raise IOError
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = IOError("Permission denied")
            
            with pytest.raises(RuntimeError, match="Failed to save feedback data"):
                temp_storage.save_feedback(feedback)
    
    def test_export_feedback_io_error(self, temp_storage, sample_feedback):
        """Test handling of IO errors during export."""
        # Save some feedback first
        temp_storage.save_feedback(sample_feedback[0])
        
        # Mock open to raise IOError for export
        with patch("builtins.open", mock_open()) as mock_file:
            def side_effect(*args, **kwargs):
                if 'export' in str(args[0]):
                    raise IOError("Permission denied")
                return mock_open()(*args, **kwargs)
            
            mock_file.side_effect = side_effect
            
            with pytest.raises(RuntimeError, match="Failed to export feedback data"):
                temp_storage.export_feedback()
    
    def test_datetime_serialization(self, temp_storage):
        """Test that datetime objects are properly serialized and deserialized."""
        # Create feedback with specific timestamp
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        feedback = FeedbackEntry(
            joke_id="123e4567-e89b-12d3-a456-426614174000",
            joke_text="Test joke",
            category="general",
            rating=4,
            timestamp=timestamp
        )
        
        # Save and retrieve
        temp_storage.save_feedback(feedback)
        retrieved_feedback = temp_storage.get_all_feedback()
        
        assert len(retrieved_feedback) == 1
        assert retrieved_feedback[0].timestamp == timestamp
    
    def test_statistics_with_empty_data(self, temp_storage):
        """Test statistics calculation with no feedback data."""
        stats = temp_storage.get_feedback_stats()
        
        assert stats["total_jokes"] == 0
        assert stats["average_rating"] == 0.0
        assert len(stats["category_stats"]) == 0
    
    def test_statistics_single_entry(self, temp_storage, sample_feedback):
        """Test statistics calculation with single feedback entry."""
        temp_storage.save_feedback(sample_feedback[0])
        
        stats = temp_storage.get_feedback_stats()
        
        assert stats["total_jokes"] == 1
        assert stats["average_rating"] == 4.0
        assert len(stats["category_stats"]) == 1
        assert stats["category_stats"]["programming"]["count"] == 1
        assert stats["category_stats"]["programming"]["avg_rating"] == 4.0


class TestModuleLevelFunctions:
    """Test cases for module-level convenience functions."""
    
    @pytest.fixture(autouse=True)
    def reset_default_storage(self):
        """Reset the default storage instance before each test."""
        import joke_cli.feedback_storage
        joke_cli.feedback_storage._default_storage = None
        yield
        joke_cli.feedback_storage._default_storage = None
    
    def test_save_feedback_function(self):
        """Test the module-level save_feedback function."""
        feedback = FeedbackEntry.create(
            joke_id="123e4567-e89b-12d3-a456-426614174000",
            joke_text="Test joke",
            category="general",
            rating=4
        )
        
        with patch('joke_cli.feedback_storage.FeedbackStorage') as mock_storage_class:
            mock_storage = mock_storage_class.return_value
            
            save_feedback(feedback)
            
            mock_storage_class.assert_called_once()
            mock_storage.save_feedback.assert_called_once_with(feedback)
    
    def test_get_feedback_stats_function(self):
        """Test the module-level get_feedback_stats function."""
        expected_stats = {"total_jokes": 5, "average_rating": 4.2}
        
        with patch('joke_cli.feedback_storage.FeedbackStorage') as mock_storage_class:
            mock_storage = mock_storage_class.return_value
            mock_storage.get_feedback_stats.return_value = expected_stats
            
            result = get_feedback_stats()
            
            assert result == expected_stats
            mock_storage.get_feedback_stats.assert_called_once()


@pytest.mark.unit
class TestFeedbackStorageIntegration:
    """Integration tests for feedback storage with real file operations."""
    
    def test_full_workflow(self):
        """Test complete feedback storage workflow."""
        with TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            storage = FeedbackStorage(storage_dir)
            
            # Create and save feedback
            feedback1 = FeedbackEntry.create(
                joke_id="123e4567-e89b-12d3-a456-426614174000",
                joke_text="Programming joke",
                category="programming",
                rating=4
            )
            
            feedback2 = FeedbackEntry.create(
                joke_id="123e4567-e89b-12d3-a456-426614174001",
                joke_text="General joke",
                category="general", 
                rating=5
            )
            
            # Save feedback
            storage.save_feedback(feedback1)
            storage.save_feedback(feedback2)
            
            # Verify statistics
            stats = storage.get_feedback_stats()
            assert stats["total_jokes"] == 2
            assert stats["average_rating"] == 4.5
            
            # Verify retrieval
            all_feedback = storage.get_all_feedback()
            assert len(all_feedback) == 2
            
            # Verify category filtering
            programming_feedback = storage.get_feedback_by_category("programming")
            assert len(programming_feedback) == 1
            assert programming_feedback[0].category == "programming"
            
            # Verify export
            export_path = storage.export_feedback()
            assert export_path.exists()
            
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            assert len(exported_data["feedback_entries"]) == 2