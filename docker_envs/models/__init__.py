from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column, Integer, String, ForeignKey, 
    Text, Index
)
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
import sqlalchemy as sa

Base = declarative_base()

class Asset(Base):
    __tablename__ = 'assets'
    
    asset_id = Column(Integer, primary_key=True)
    symbol = Column(String(32), unique=True, nullable=False)
    name = Column(String(128), nullable=False)
    coingecko_id = Column(String(128), unique=True)
    category = Column(String(64))
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index('idx_assets_symbol', 'symbol'),
        Index('idx_assets_category', 'category'),
    )

class Trigger(Base):
    __tablename__ = 'triggers'
    
    trigger_id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('assets.asset_id', ondelete='CASCADE'), nullable=False)
    trigger_type = Column(String(64), nullable=False)
    trigger_details = Column(JSONB)
    status = Column(String(32), nullable=False, server_default='pending')
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    processing_started_at = Column(TIMESTAMP(timezone=True))
    completed_at = Column(TIMESTAMP(timezone=True))
    error_message = Column(Text)

    __table_args__ = (
        Index('idx_triggers_asset_id_created_at', 'asset_id', 'created_at', postgresql_using='btree'),
        Index('idx_triggers_status', 'status'),
    )

class Report(Base):
    __tablename__ = 'reports'

    report_id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('assets.asset_id', ondelete='CASCADE'), nullable=False)
    trigger_id = Column(Integer, ForeignKey('triggers.trigger_id', ondelete='RESTRICT'), unique=True, nullable=False)
    report_content_markdown = Column(Text, nullable=False)
    report_data_json = Column(JSONB)
    llm_model_used = Column(String(128))
    generation_time_ms = Column(Integer)
    token_usage = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index('idx_reports_asset_id_created_at', 'asset_id', 'created_at', postgresql_using='btree'),
    )

class ScrapedData(Base):
    __tablename__ = 'scraped_data'

    scraped_data_id = Column(Integer, primary_key=True)
    trigger_id = Column(Integer, ForeignKey('triggers.trigger_id', ondelete='CASCADE'), nullable=False)
    asset_id = Column(Integer, ForeignKey('assets.asset_id', ondelete='CASCADE'), nullable=False)
    source = Column(String(64), nullable=False)
    data_type = Column(String(64), nullable=False)
    raw_data = Column(JSONB, nullable=False)
    processed_data = Column(JSONB)
    processing_metadata = Column(JSONB)
    is_processed = Column(sa.Boolean, server_default='false')
    ingested_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    processed_at = Column(TIMESTAMP(timezone=True))

    __table_args__ = (
        Index('idx_scraped_data_trigger_id', 'trigger_id'),
        Index('idx_scraped_data_asset_id', 'asset_id'),
        Index('idx_scraped_data_source_type', 'source', 'data_type'),
    )

