from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column, Integer, String, ForeignKey, 
    Text, Index
)
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

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