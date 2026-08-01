from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class WorkType(str, Enum):
    CONSTITUTION = "Constitution"
    ACT = "Act"
    RULE = "Rule"
    DIRECTIVE = "Directive"
    NOTIFICATION = "Notification"


class ComponentType(str, Enum):
    BHAG = "bhag"
    KHANDA = "khanda"
    DAFA = "dafa"
    UPDAFA = "updafa"
    PARICHHEDA = "parichheda"
    PROVISO = "proviso"
    SPASTIKARAN = "spastikaran"
    ANUSUCHI = "anusuchi"


class SourceKind(str, Enum):
    OFFICIAL_ORIGINAL = "official_original"
    AMENDING_INSTRUMENT = "amending_instrument"
    VERIFIED_INTERNAL_CONSOLIDATION = "verified_internal_consolidation"
    OFFICIAL_COPY_UNVERIFIED = "official_copy_unverified"
    DERIVED_VERIFIED = "derived_verified"


class EffectType(str, Enum):
    AMEND = "amend"
    REPEAL = "repeal"
    COMMENCE = "commence"
    EXPIRY = "expiry"
    SUSPEND = "suspend"
    CORRECT = "correct"
    DECLARED_INVALID = "declared_invalid"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Work(BaseModel):
    id: Optional[UUID] = None
    uri: str
    work_type: WorkType
    title_ne: str
    title_en: Optional[str] = None
    jurisdiction: str = "NP"
    created_at: Optional[datetime] = None


class Component(BaseModel):
    id: Optional[UUID] = None
    work_id: UUID
    uri: str
    component_type: ComponentType
    number: Optional[str] = None
    parent_uri: Optional[str] = None
    created_at: Optional[datetime] = None


class SourcePublication(BaseModel):
    id: Optional[UUID] = None
    work_id: UUID
    kind: SourceKind
    source_url: Optional[str] = None
    sha256: str = Field(min_length=64, max_length=64)
    ocr_confidence: Optional[float] = None
    ingested_at: Optional[datetime] = None


class LifecycleEffect(BaseModel):
    id: Optional[UUID] = None
    component_uri: str
    effect_type: EffectType
    legal_valid_time: str
    transaction_time: str
    effective_date: Optional[date] = None
    commencement_dependency: Optional[str] = None
    replacement_text: Optional[str] = None
    source_pub_id: Optional[UUID] = None
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by_1: Optional[UUID] = None
    approved_by_2: Optional[UUID] = None
    created_at: Optional[datetime] = None


class Expression(BaseModel):
    id: Optional[UUID] = None
    component_uri: str
    as_of: date
    text_ne: str
    text_hash: str = Field(min_length=64, max_length=64)
    is_derived: bool = True
    created_at: Optional[datetime] = None
