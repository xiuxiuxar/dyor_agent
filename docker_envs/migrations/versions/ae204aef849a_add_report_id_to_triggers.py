"""add_report_id_to_triggers

Revision ID: ae204aef849a
Revises: 8e322c6e38bb
Create Date: 2025-05-28 12:27:36.831489

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ae204aef849a'
down_revision: Union[str, None] = '8e322c6e38bb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('triggers', sa.Column('report_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_triggers_report_id',
        'triggers', 'reports',
        ['report_id'], ['report_id'],
        ondelete='SET NULL'
    )

def downgrade():
    op.drop_constraint('fk_triggers_report_id', 'triggers', type_='foreignkey')
    op.drop_column('triggers', 'report_id')