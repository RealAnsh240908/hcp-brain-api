"""
Brain API - HCP Engagement Scoring System
Python 3.11 | FastAPI | Pydantic

Production-ready deterministic scoring engine for Healthcare Professional engagement.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from collections import defaultdict
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# IN-MEMORY EVENT STORE
# ============================================================================

class EventStore:
    """
    In-memory event store for development/testing.
    In production, replace with persistent storage (PostgreSQL, MongoDB, etc.)
    """
    
    def __init__(self):
        self._events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._all_events: List[Dict[str, Any]] = []
    
    def add_event(self, event: Dict[str, Any], hcp_id: Optional[str] = None) -> str:
        """Store an event and return its ID"""
        event_id = f"evt_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        stored_event = {
            "event_id": event_id,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            **event
        }
        
        self._all_events.append(stored_event)
        
        if hcp_id:
            self._events[hcp_id].append(stored_event)
        
        return event_id
    
    def get_events_by_hcp(self, hcp_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve events for a specific HCP"""
        return self._events.get(hcp_id, [])[-limit:]
    
    def get_all_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all events"""
        return self._all_events[-limit:]
    
    def get_event_count(self) -> int:
        """Get total event count"""
        return len(self._all_events)
    
    def clear(self):
        """Clear all events (for testing)"""
        self._events.clear()
        self._all_events.clear()


# Global event store instance
event_store = EventStore()


# ============================================================================
# MODELS
# ============================================================================

class HCPProfile(BaseModel):
    """Healthcare Professional profile data"""
    hcp_id: str = Field(..., min_length=1, description="Unique identifier for the HCP")
    name: str = Field(..., min_length=1, description="HCP's full name")
    profile: Dict[str, Any] = Field(default_factory=dict, description="Additional profile metadata")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "hcp_id": "hcp_12345",
                "name": "Dr. Jane Smith",
                "profile": {
                    "specialty": "Cardiology",
                    "institution": "Metro Hospital",
                    "region": "Northeast"
                }
            }
        }
    }


class Event(BaseModel):
    """Individual engagement event"""
    event_type: str = Field(..., min_length=1, description="Type of engagement event")
    timestamp: str = Field(..., description="Event timestamp in ISO-8601 format")
    channel: str = Field(..., min_length=1, description="Channel through which engagement occurred")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate that timestamp is a valid ISO-8601 format"""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid ISO-8601 timestamp: {v}") from e
        return v
    
    @field_validator('event_type')
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Normalize event type to lowercase"""
        return v.strip().lower()
    
    @field_validator('channel')
    @classmethod
    def validate_channel(cls, v: str) -> str:
        """Normalize channel to lowercase"""
        return v.strip().lower()
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "event_type": "email_open",
                "timestamp": "2025-01-03T10:30:00Z",
                "channel": "email",
                "metadata": {"campaign_id": "winter_2025", "subject": "New Treatment Options"}
            }
        }
    }


class ScoreRequest(BaseModel):
    """Request payload for scoring endpoint"""
    hcp: HCPProfile
    recent_events: List[Event] = Field(default_factory=list, description="List of recent engagement events")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "hcp": {
                    "hcp_id": "hcp_12345",
                    "name": "Dr. Jane Smith",
                    "profile": {"specialty": "Cardiology"}
                },
                "recent_events": [
                    {
                        "event_type": "email_open",
                        "timestamp": "2025-01-03T10:30:00Z",
                        "channel": "email",
                        "metadata": {}
                    },
                    {
                        "event_type": "content_download",
                        "timestamp": "2025-01-02T14:00:00Z",
                        "channel": "website",
                        "metadata": {"content_id": "whitepaper_001"}
                    }
                ]
            }
        }
    }


class ScoreBreakdown(BaseModel):
    """Detailed score breakdown by dimension"""
    recency: int = Field(ge=0, le=100, description="Score based on how recent the last engagement was")
    frequency: int = Field(ge=0, le=100, description="Score based on number of engagements")
    depth: int = Field(ge=0, le=100, description="Score based on quality/depth of engagements")


class NextBestAction(BaseModel):
    """Recommended action with priority and reasoning"""
    action: str = Field(..., description="Recommended action to take")
    priority: Literal["low", "medium", "high"] = Field(..., description="Action priority level")
    why: str = Field(..., description="Rationale for the recommendation")
    asset_id: Optional[str] = Field(None, description="Optional reference to a specific asset or content")


class ScoreResponse(BaseModel):
    """Response payload from scoring endpoint"""
    hcp_id: str
    score_total: int = Field(ge=0, le=100, description="Overall engagement score")
    score_breakdown: ScoreBreakdown
    stage: Literal["Awareness", "Education", "Consideration", "Dormant"]
    stage_reason: List[str] = Field(..., description="Reasons for stage assignment")
    next_best_actions: List[NextBestAction] = Field(..., description="Recommended next actions")


class IngestEventRequest(BaseModel):
    """Single event ingestion payload"""
    event_type: str = Field(..., min_length=1, description="Type of engagement event")
    timestamp: str = Field(..., description="Event timestamp in ISO-8601 format")
    channel: str = Field(..., min_length=1, description="Channel through which engagement occurred")
    hcp_id: Optional[str] = Field(None, description="Optional HCP identifier to associate event with")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate that timestamp is a valid ISO-8601 format"""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid ISO-8601 timestamp: {v}") from e
        return v
    
    @field_validator('event_type')
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Normalize and validate event type"""
        normalized = v.strip().lower()
        # Replace spaces/hyphens with underscores for consistency
        normalized = re.sub(r'[\s-]+', '_', normalized)
        return normalized
    
    @field_validator('channel')
    @classmethod
    def validate_channel(cls, v: str) -> str:
        """Normalize channel to lowercase"""
        return v.strip().lower()
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "event_type": "email_open",
                "timestamp": "2025-01-03T10:30:00Z",
                "channel": "email",
                "hcp_id": "hcp_12345",
                "metadata": {"campaign_id": "winter_2025"}
            }
        }
    }


class IngestEventResponse(BaseModel):
    """Response from event ingestion"""
    status: str
    event_id: str = Field(..., description="Unique identifier for the ingested event")
    message: str = Field(..., description="Additional information about the ingestion")


class StatusResponse(BaseModel):
    """Generic status response"""
    status: str


class EventListResponse(BaseModel):
    """Response containing list of events"""
    events: List[Dict[str, Any]]
    total_count: int
    returned_count: int


class StatsResponse(BaseModel):
    """API statistics response"""
    status: str
    total_events_ingested: int
    supported_event_types: List[str]
    supported_channels: List[str]


# ============================================================================
# SCORING LOGIC
# ============================================================================

class EngagementScorer:
    """
    Deterministic scoring engine for HCP engagement.
    Uses recency, frequency, and depth dimensions.
    """
    
    # Event type weights for depth scoring
    EVENT_WEIGHTS = {
        "email_open": 5,
        "email_click": 10,
        "content_view": 15,
        "content_download": 25,
        "webinar_registration": 20,
        "webinar_attendance": 35,
        "form_submission": 30,
        "sample_request": 40,
        "meeting_scheduled": 45,
        "meeting_completed": 50,
        "prescription_written": 100,
    }
    
    # Channel multipliers
    CHANNEL_MULTIPLIERS = {
        "email": 1.0,
        "website": 1.2,
        "webinar": 1.5,
        "sales_rep": 2.0,
        "conference": 1.8,
        "phone": 1.3,
        "mobile_app": 1.1,
    }
    
    # Weights for total score calculation
    RECENCY_WEIGHT = 0.30
    FREQUENCY_WEIGHT = 0.30
    DEPTH_WEIGHT = 0.40
    
    @classmethod
    def get_supported_event_types(cls) -> List[str]:
        """Return list of supported event types"""
        return list(cls.EVENT_WEIGHTS.keys())
    
    @classmethod
    def get_supported_channels(cls) -> List[str]:
        """Return list of supported channels"""
        return list(cls.CHANNEL_MULTIPLIERS.keys())
    
    @staticmethod
    def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
        """Safely parse an ISO-8601 timestamp string"""
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
    
    @classmethod
    def calculate_recency_score(cls, events: List[Event]) -> int:
        """
        Calculate recency score based on most recent event.
        Score decays exponentially with time.
        """
        if not events:
            return 0
        
        now = datetime.now(timezone.utc)
        most_recent = None
        
        for event in events:
            event_time = cls.parse_timestamp(event.timestamp)
            if event_time and (most_recent is None or event_time > most_recent):
                most_recent = event_time
        
        if most_recent is None:
            return 0
        
        days_ago = (now - most_recent).days
        
        # Scoring logic: recent = higher score
        if days_ago <= 7:
            return 100
        elif days_ago <= 14:
            return 85
        elif days_ago <= 30:
            return 70
        elif days_ago <= 60:
            return 50
        elif days_ago <= 90:
            return 30
        elif days_ago <= 180:
            return 15
        else:
            return 5
    
    @classmethod
    def calculate_frequency_score(cls, events: List[Event], lookback_days: int = 90) -> int:
        """
        Calculate frequency score based on number of events in recent period.
        More events = higher engagement frequency.
        """
        if not events:
            return 0
        
        now = datetime.now(timezone.utc)
        recent_count = 0
        
        for event in events:
            event_time = cls.parse_timestamp(event.timestamp)
            if event_time:
                days_ago = (now - event_time).days
                if days_ago <= lookback_days:
                    recent_count += 1
        
        # Scoring based on event count
        if recent_count >= 20:
            return 100
        elif recent_count >= 15:
            return 85
        elif recent_count >= 10:
            return 70
        elif recent_count >= 7:
            return 55
        elif recent_count >= 5:
            return 40
        elif recent_count >= 3:
            return 25
        elif recent_count >= 1:
            return 10
        else:
            return 0
    
    @classmethod
    def calculate_depth_score(cls, events: List[Event]) -> int:
        """
        Calculate depth score based on event types and channels.
        High-value actions contribute more to depth.
        """
        if not events:
            return 0
        
        total_weighted_value = 0.0
        max_possible = 0.0
        
        for event in events:
            event_type = event.event_type.lower()
            channel = event.channel.lower()
            
            # Get base weight for event type (default to 10 for unknown types)
            base_weight = cls.EVENT_WEIGHTS.get(event_type, 10)
            
            # Apply channel multiplier (default to 1.0 for unknown channels)
            channel_multiplier = cls.CHANNEL_MULTIPLIERS.get(channel, 1.0)
            
            weighted_value = base_weight * channel_multiplier
            total_weighted_value += weighted_value
            max_possible += 100  # Max theoretical value per event
        
        if max_possible == 0:
            return 0
        
        # Normalize to 0-100 scale
        raw_score = (total_weighted_value / max_possible) * 100
        return min(int(raw_score), 100)
    
    @classmethod
    def calculate_total_score(cls, breakdown: ScoreBreakdown) -> int:
        """
        Calculate weighted total score from breakdown components.
        """
        total = (
            breakdown.recency * cls.RECENCY_WEIGHT +
            breakdown.frequency * cls.FREQUENCY_WEIGHT +
            breakdown.depth * cls.DEPTH_WEIGHT
        )
        return int(round(total))
    
    @classmethod
    def determine_stage(
        cls, 
        score_total: int, 
        breakdown: ScoreBreakdown, 
        events: List[Event]
    ) -> tuple[str, List[str]]:
        """
        Assign lifecycle stage based on scoring and event patterns.
        Returns (stage, reasons).
        """
        reasons = []
        
        # Dormant: Low engagement across all dimensions
        if score_total < 20:
            reasons.append(f"Total engagement score is low ({score_total}/100)")
            if breakdown.recency < 20:
                reasons.append("No recent activity detected")
            return "Dormant", reasons
        
        # Awareness: Low to moderate engagement, exploring
        if score_total < 45:
            reasons.append(f"Early stage engagement detected (score: {score_total})")
            if breakdown.frequency < 30:
                reasons.append("Limited interaction frequency")
            return "Awareness", reasons
        
        # Education: Moderate engagement, learning phase
        if score_total < 70:
            reasons.append(f"Active learning behavior observed (score: {score_total})")
            if breakdown.depth >= 40:
                reasons.append("Engaging with educational content")
            if breakdown.frequency >= 40:
                reasons.append("Regular interaction pattern established")
            return "Education", reasons
        
        # Consideration: High engagement, decision-making phase
        reasons.append(f"High engagement level indicates strong interest (score: {score_total})")
        if breakdown.depth >= 60:
            reasons.append("Deep engagement with high-value activities")
        if breakdown.recency >= 70:
            reasons.append("Very recent activity shows active consideration")
        
        # Check for high-intent events
        high_intent_types = {"sample_request", "meeting_scheduled", "meeting_completed", "prescription_written"}
        high_intent_events = [e for e in events if e.event_type.lower() in high_intent_types]
        if high_intent_events:
            reasons.append(f"High-intent actions detected ({len(high_intent_events)} events)")
        
        return "Consideration", reasons
    
    @classmethod
    def generate_next_best_actions(
        cls, 
        stage: str, 
        breakdown: ScoreBreakdown, 
        events: List[Event]
    ) -> List[NextBestAction]:
        """
        Generate 1-3 recommended next best actions based on stage and engagement pattern.
        """
        actions = []
        
        if stage == "Dormant":
            actions.append(NextBestAction(
                action="Re-engagement email campaign",
                priority="high",
                why="No recent activity - need to recapture attention with compelling content",
                asset_id=None
            ))
            actions.append(NextBestAction(
                action="Personalized outreach via sales rep",
                priority="medium",
                why="Direct human touch may be needed to understand disengagement",
                asset_id=None
            ))
        
        elif stage == "Awareness":
            actions.append(NextBestAction(
                action="Send educational content series",
                priority="high",
                why="Build foundational knowledge to move HCP toward education stage",
                asset_id=None
            ))
            if breakdown.frequency < 20:
                actions.append(NextBestAction(
                    action="Increase touchpoint frequency",
                    priority="medium",
                    why="Low interaction frequency - establish regular communication rhythm",
                    asset_id=None
                ))
        
        elif stage == "Education":
            actions.append(NextBestAction(
                action="Invite to upcoming webinar or virtual event",
                priority="high",
                why="Deeper engagement format will accelerate learning and consideration",
                asset_id=None
            ))
            actions.append(NextBestAction(
                action="Share clinical study data and efficacy reports",
                priority="high",
                why="Provide evidence-based content to support decision-making",
                asset_id=None
            ))
            if breakdown.depth < 50:
                actions.append(NextBestAction(
                    action="Offer product sample or trial",
                    priority="medium",
                    why="Hands-on experience will deepen engagement",
                    asset_id=None
                ))
        
        elif stage == "Consideration":
            actions.append(NextBestAction(
                action="Schedule one-on-one consultation with medical science liaison",
                priority="high",
                why="HCP shows high intent - personal interaction will facilitate decision",
                asset_id=None
            ))
            actions.append(NextBestAction(
                action="Provide prescribing information and formulary details",
                priority="high",
                why="Remove practical barriers to prescription with specific details",
                asset_id=None
            ))
            
            # Check if they've had recent meetings
            recent_meetings = [e for e in events if "meeting" in e.event_type.lower()]
            if not recent_meetings:
                actions.append(NextBestAction(
                    action="Offer in-person meeting or lunch-and-learn",
                    priority="medium",
                    why="No meeting detected yet - face-to-face interaction will strengthen relationship",
                    asset_id=None
                ))
        
        # Return maximum 3 actions
        return actions[:3]


# ============================================================================
# BACKGROUND TASKS (Async Processing Pipeline)
# ============================================================================

async def process_event_pipeline(event_data: Dict[str, Any], event_id: str):
    """
    Background task to process ingested events.
    In production, this could trigger:
    - Real-time analytics updates
    - Score recalculation
    - Alert generation
    - CRM synchronization
    """
    logger.info(f"Processing event {event_id} in background pipeline")
    
    # Simulate processing steps
    hcp_id = event_data.get("hcp_id")
    event_type = event_data.get("event_type")
    
    # Log processing completion
    logger.info(f"Event {event_id} processed: type={event_type}, hcp_id={hcp_id}")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown"""
    # Startup
    logger.info("Brain API starting up...")
    logger.info("Engagement scoring engine initialized")
    logger.info(f"Supported event types: {EngagementScorer.get_supported_event_types()}")
    logger.info(f"Supported channels: {EngagementScorer.get_supported_channels()}")
    
    yield
    
    # Shutdown
    logger.info("Brain API shutting down...")
    logger.info(f"Total events processed: {event_store.get_event_count()}")


app = FastAPI(
    title="Brain API - HCP Engagement Scoring",
    description="""
## HCP Engagement Scoring System

Production-ready API for calculating Healthcare Professional engagement scores and lifecycle stages.

### Features
- **Engagement Scoring**: Calculate scores based on recency, frequency, and depth dimensions
- **Lifecycle Stages**: Automatically assign HCPs to stages (Awareness, Education, Consideration, Dormant)
- **Next Best Actions**: Get AI-driven recommendations for optimal engagement strategies
- **Event Ingestion**: Ingest and track engagement events in real-time

### Scoring Dimensions
- **Recency (30%)**: How recently the HCP engaged
- **Frequency (30%)**: How often the HCP engages
- **Depth (40%)**: Quality and value of engagements
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=StatusResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    """
    return StatusResponse(status="ok")


@app.get("/stats", response_model=StatsResponse, tags=["Health"])
async def get_stats():
    """
    Get API statistics and supported configurations.
    """
    return StatsResponse(
        status="ok",
        total_events_ingested=event_store.get_event_count(),
        supported_event_types=EngagementScorer.get_supported_event_types(),
        supported_channels=EngagementScorer.get_supported_channels()
    )


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_hcp(request: ScoreRequest):
    """
    Calculate engagement score, assign lifecycle stage, and recommend next best actions.
    
    This endpoint processes HCP profile and recent events to generate:
    - Total engagement score (0-100)
    - Score breakdown (recency, frequency, depth)
    - Lifecycle stage assignment
    - 1-3 prioritized next best actions
    """
    try:
        scorer = EngagementScorer()
        
        # Calculate score components
        recency_score = scorer.calculate_recency_score(request.recent_events)
        frequency_score = scorer.calculate_frequency_score(request.recent_events)
        depth_score = scorer.calculate_depth_score(request.recent_events)
        
        # Build breakdown
        breakdown = ScoreBreakdown(
            recency=recency_score,
            frequency=frequency_score,
            depth=depth_score
        )
        
        # Calculate total score
        total_score = scorer.calculate_total_score(breakdown)
        
        # Determine stage
        stage, stage_reason = scorer.determine_stage(total_score, breakdown, request.recent_events)
        
        # Generate next best actions
        actions = scorer.generate_next_best_actions(stage, breakdown, request.recent_events)
        
        logger.info(f"Scored HCP {request.hcp.hcp_id}: total={total_score}, stage={stage}")
        
        return ScoreResponse(
            hcp_id=request.hcp.hcp_id,
            score_total=total_score,
            score_breakdown=breakdown,
            stage=stage,
            stage_reason=stage_reason,
            next_best_actions=actions
        )
    
    except Exception as e:
        logger.error(f"Error scoring HCP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/ingest-event", response_model=IngestEventResponse, tags=["Events"])
async def ingest_event(event: IngestEventRequest, background_tasks: BackgroundTasks):
    """
    Ingest a single engagement event for processing.
    
    This endpoint accepts individual events and:
    - Validates event data (via Pydantic models)
    - Persists to in-memory event store
    - Triggers async background processing pipeline
    
    In production, the event store would be replaced with persistent storage
    (PostgreSQL, MongoDB, etc.) and the background tasks would integrate
    with message queues or streaming platforms.
    """
    try:
        # Prepare event data for storage
        event_data = {
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "channel": event.channel,
            "hcp_id": event.hcp_id,
            "metadata": event.metadata
        }
        
        # Persist event to store
        event_id = event_store.add_event(event_data, event.hcp_id)
        
        # Trigger async processing pipeline
        background_tasks.add_task(process_event_pipeline, event_data, event_id)
        
        logger.info(f"Ingested event {event_id}: type={event.event_type}, channel={event.channel}, hcp_id={event.hcp_id}")
        
        return IngestEventResponse(
            status="ok",
            event_id=event_id,
            message=f"Event ingested successfully and queued for processing"
        )
    
    except Exception as e:
        logger.error(f"Error ingesting event: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Event ingestion failed: {str(e)}")


@app.get("/events", response_model=EventListResponse, tags=["Events"])
async def list_events(hcp_id: Optional[str] = None, limit: int = 100):
    """
    Retrieve ingested events.
    
    Optionally filter by HCP ID. Returns most recent events up to the limit.
    """
    try:
        if hcp_id:
            events = event_store.get_events_by_hcp(hcp_id, limit)
        else:
            events = event_store.get_all_events(limit)
        
        return EventListResponse(
            events=events,
            total_count=event_store.get_event_count(),
            returned_count=len(events)
        )
    
    except Exception as e:
        logger.error(f"Error retrieving events: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve events: {str(e)}")


@app.delete("/events", response_model=StatusResponse, tags=["Events"])
async def clear_events():
    """
    Clear all stored events (for testing/development purposes).
    
    **Warning**: This will permanently delete all stored events.
    """
    try:
        event_store.clear()
        logger.info("All events cleared from store")
        return StatusResponse(status="ok")
    
    except Exception as e:
        logger.error(f"Error clearing events: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear events: {str(e)}")
