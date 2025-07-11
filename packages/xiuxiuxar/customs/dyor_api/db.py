# ------------------------------------------------------------------------------
#
#   Copyright 2025 xiuxiuxar
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""This module contains the SQLAlchemy models for the DYOR API."""

from sqlalchemy import Text, Index, Column, String, Integer, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP


Base = declarative_base()


class Asset(Base):
    """SQLAlchemy model for assets table."""

    __tablename__ = "assets"

    asset_id = Column(Integer, primary_key=True)
    symbol = Column(String(32), unique=True, nullable=False)
    name = Column(String(128), nullable=False)
    coingecko_id = Column(String(128), unique=True)
    category = Column(String(64))
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_assets_symbol", "symbol"),
        Index("idx_assets_category", "category"),
    )


class Trigger(Base):
    """SQLAlchemy model for triggers table."""

    __tablename__ = "triggers"

    trigger_id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False)
    trigger_type = Column(String(64), nullable=False)
    trigger_details = Column(JSONB)
    status = Column(String(32), nullable=False, server_default="pending")
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    processing_started_at = Column(TIMESTAMP(timezone=True))
    completed_at = Column(TIMESTAMP(timezone=True))
    error_message = Column(Text)
    report_id = Column(Integer, ForeignKey("reports.report_id", ondelete="SET NULL"), nullable=True)

    __table_args__ = (
        Index("idx_triggers_asset_id_created_at", "asset_id", "created_at", postgresql_using="btree"),
        Index("idx_triggers_status", "status"),
    )


class Report(Base):
    """SQLAlchemy model for reports table."""

    __tablename__ = "reports"

    report_id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False)
    trigger_id = Column(Integer, ForeignKey("triggers.trigger_id", ondelete="RESTRICT"), unique=True, nullable=False)
    report_content_markdown = Column(Text, nullable=False)
    report_data_json = Column(JSONB)
    llm_model_used = Column(String(128))
    generation_time_ms = Column(Integer)
    token_usage = Column(JSONB)
    relevance_score = Column(Integer)
    completeness_score = Column(Integer)
    usefulness_score = Column(Integer)
    data_quality_score = Column(Integer)
    actionability_score = Column(Integer)
    composite_score = Column(Integer)
    score_breakdown = Column(JSONB)
    improvement_suggestions = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_reports_asset_id_created_at", "asset_id", "created_at", postgresql_using="btree"),
        Index("idx_reports_composite_score", "composite_score", postgresql_using="btree"),
        Index("idx_reports_low_scores", "composite_score", "created_at", postgresql_using="btree"),
        Index("idx_reports_model_scores", "llm_model_used", "composite_score", postgresql_using="btree"),
    )
