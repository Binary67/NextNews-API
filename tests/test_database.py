from sqlalchemy import inspect

from app.config import Settings
from app.database import create_app_engine, init_database


def test_init_database_creates_query_matched_indexes() -> None:
    settings = Settings(database_url="sqlite://")
    engine = create_app_engine(settings.database_url)

    init_database(engine, settings.database_url)

    inspector = inspect(engine)
    expected_indexes = {
        "source_items": {
            "ix_source_items_fetched_at_id": ["fetched_at", "id"],
        },
        "generated_posts": {
            "ix_generated_posts_status_ready_at_id": ["status", "ready_at", "id"],
        },
        "conversation_threads": {
            "ix_conversation_threads_post_id_updated_at_id": [
                "post_id",
                "updated_at",
                "id",
            ],
        },
        "conversation_messages": {
            "ix_conversation_messages_thread_id_created_at_id": [
                "thread_id",
                "created_at",
                "id",
            ],
        },
    }

    for table_name, expected_table_indexes in expected_indexes.items():
        actual_indexes = {
            index["name"]: index["column_names"]
            for index in inspector.get_indexes(table_name)
        }

        for index_name, column_names in expected_table_indexes.items():
            assert actual_indexes[index_name] == column_names
