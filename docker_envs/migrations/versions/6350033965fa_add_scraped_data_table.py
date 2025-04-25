"""add_scraped_data_table

Revision ID: 6350033965fa
Revises: d602b547996a
Create Date: 2025-04-25 12:26:19.510089

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = '6350033965fa'
down_revision: Union[str, None] = 'd602b547996a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'scraped_data',
        sa.Column('scraped_data_id', sa.Integer(), nullable=False),
        sa.Column('trigger_id', sa.Integer(), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=False),
        sa.Column('source', sa.String(64), nullable=False),
        sa.Column('data_type', sa.String(64), nullable=False),
        sa.Column('raw_data', JSONB, nullable=False),
        sa.Column('processed_data', JSONB, nullable=True),
        sa.Column('processing_metadata', JSONB, nullable=True),
        sa.Column('is_processed', sa.Boolean(), server_default='false'),
        sa.Column('ingested_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('processed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('scraped_data_id'),
        sa.ForeignKeyConstraint(['trigger_id'], ['triggers.trigger_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['asset_id'], ['assets.asset_id'], ondelete='CASCADE')
    )

    op.create_index('idx_scraped_data_trigger_id', 'scraped_data', ['trigger_id'])
    op.create_index('idx_scraped_data_asset_id', 'scraped_data', ['asset_id'])
    op.create_index('idx_scraped_data_source_type', 'scraped_data', ['source', 'data_type'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_scraped_data_source_type')
    op.drop_index('idx_scraped_data_asset_id')
    op.drop_index('idx_scraped_data_trigger_id')
    op.drop_table('scraped_data')
