"""add_report_scores

Revision ID: 334e4181ae22
Revises: ae204aef849a
Create Date: 2025-07-11 11:39:00.271366

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '334e4181ae22'
down_revision: Union[str, None] = 'ae204aef849a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add score columns to reports table
    op.add_column('reports', sa.Column('relevance_score', sa.Integer(), nullable=True))
    op.add_column('reports', sa.Column('completeness_score', sa.Integer(), nullable=True))
    op.add_column('reports', sa.Column('usefulness_score', sa.Integer(), nullable=True))
    op.add_column('reports', sa.Column('data_quality_score', sa.Integer(), nullable=True))
    op.add_column('reports', sa.Column('actionability_score', sa.Integer(), nullable=True))
    op.add_column('reports', sa.Column('composite_score', sa.Integer(), nullable=True))
    op.add_column('reports', sa.Column('score_breakdown', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('reports', sa.Column('improvement_suggestions', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    
    # Add index for score-based queries
    op.create_index('idx_reports_composite_score', 'reports', ['composite_score'], unique=False, postgresql_using='btree')
    
    # Add index for low scoring reports queries
    op.create_index('idx_reports_low_scores', 'reports', ['composite_score', 'created_at'], unique=False, postgresql_using='btree')
    
    # Add index for model performance analysis
    op.create_index('idx_reports_model_scores', 'reports', ['llm_model_used', 'composite_score'], unique=False, postgresql_using='btree')


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index('idx_reports_model_scores', table_name='reports', postgresql_using='btree')
    op.drop_index('idx_reports_low_scores', table_name='reports', postgresql_using='btree')
    op.drop_index('idx_reports_composite_score', table_name='reports', postgresql_using='btree')
    
    # Drop score columns
    op.drop_column('reports', 'improvement_suggestions')
    op.drop_column('reports', 'score_breakdown')
    op.drop_column('reports', 'composite_score')
    op.drop_column('reports', 'actionability_score')
    op.drop_column('reports', 'data_quality_score')
    op.drop_column('reports', 'usefulness_score')
    op.drop_column('reports', 'completeness_score')
    op.drop_column('reports', 'relevance_score')
