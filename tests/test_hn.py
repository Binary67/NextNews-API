from app.hn import is_valid_story


def test_is_valid_story_rejects_deleted_dead_and_non_story_items() -> None:
    assert not is_valid_story(None)
    assert not is_valid_story({"id": 1, "type": "comment", "title": "No"})
    assert not is_valid_story({"id": 1, "type": "story", "title": "No", "deleted": True})
    assert not is_valid_story({"id": 1, "type": "story", "title": "No", "dead": True})
    assert not is_valid_story({"id": 1, "type": "story"})


def test_is_valid_story_accepts_live_story_with_title() -> None:
    assert is_valid_story({"id": 1, "type": "story", "title": "A story"})

