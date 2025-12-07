import os
import time
import uuid
import json
import contextvars
from datetime import datetime
from typing import Literal
from fastapi import FastAPI, HTTPException, Depends, APIRouter, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import structlog
from prometheus_fastapi_instrumentator import Instrumentator
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres@postgres:5432/postgres")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Upload(Base):
    __tablename__ = "uploads"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    status = Column(String, nullable=False)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


app = FastAPI()

instrumentator = Instrumentator().instrument(app).expose(app)

# Sentry setup
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[FastApiIntegration()],
        traces_sample_rate=1.0,
        environment="development"
    )

# Configure CORS with explicit origins for local development
_origins_env = os.getenv("ALLOW_ORIGINS", "http://localhost:3000,http://localhost:8080")
_allow_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# correlation id context
correlation_id_var = contextvars.ContextVar("correlation_id", default=None)

# Structured JSON logging with structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger("metadata-api")

def log_json(event: str, **fields):
    cid = correlation_id_var.get()
    if cid:
        fields["correlation_id"] = cid
    logger.info(event, **fields)

# simple request metrics
_requests_total: dict[tuple[str, str, int], int] = {}
_latency_sum: dict[tuple[str, str], float] = {}
_latency_count: dict[tuple[str, str], int] = {}

@app.middleware("http")
async def add_observability(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    correlation_id_var.set(req_id)
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
    except Exception as exc:
        log_json(
            "request_error",
            method=request.method,
            path=request.url.path,
            error=str(exc),
        )
        raise
    finally:
        duration = time.perf_counter() - start
        status_code = getattr(response, "status_code", 500) if response else 500
        key = (request.method, request.url.path, status_code)
        _requests_total[key] = _requests_total.get(key, 0) + 1
        lkey = (request.method, request.url.path)
        _latency_sum[lkey] = _latency_sum.get(lkey, 0.0) + duration
        _latency_count[lkey] = _latency_count.get(lkey, 0) + 1
        log_json(
            "request_end",
            method=request.method,
            path=request.url.path,
            status=status_code,
            duration_ms=int(duration * 1000),
        )
    if response:
        response.headers["X-Correlation-ID"] = req_id
    return response

# error handlers for consistent JSON errors
def _code_for_status(status_code: int) -> str:
    if status_code == 400:
        return "BAD_REQUEST"
    if status_code == 404:
        return "NOT_FOUND"
    if status_code == 429:
        return "TOO_MANY_REQUESTS"
    if status_code == 422:
        return "VALIDATION_ERROR"
    if 500 <= status_code < 600:
        return "INTERNAL_ERROR"
    return f"HTTP_{status_code}"

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    payload = {"error": {"code": _code_for_status(exc.status_code), "message": exc.detail}}
    return Response(content=json.dumps(payload), status_code=exc.status_code, media_type="application/json")

@app.exception_handler(RequestValidationError)\nasync def validation_exception_handler(request: Request, exc: RequestValidationError):\n    payload = {"error": {"code": "VALIDATION_ERROR", "message": "Invalid request", "details": [str(e) for e in exc.errors()]}}\n    return Response(content=json.dumps(payload), status_code=422, media_type="application/json")\n\n@app.exception_handler(Exception)\nasync def general_exception_handler(request: Request, exc: Exception):\n    log_json("unhandled_exception", method=request.method, path=request.url.path, error=str(exc))\n    payload = {"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"}}\n    return Response(content=json.dumps(payload), status_code=500, media_type="application/json")


from collections import defaultdict
rate_store: dict[tuple[str, str], tuple[float, int]] = {}

def rate_limit(action: str, limit: int, per: int = 60):
    async def _limiter(request: Request):
        ip = request.client.host if request.client else "unknown"
        key = (ip, action)
        now = time.time()
        window_start, count = rate_store.get(key, (now, 0))
        if now - window_start >= per:
            window_start, count = now, 0
        count += 1
        rate_store[key] = (window_start, count)
        if count > limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return None
    return _limiter

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




from pydantic import constr

class VideoCreate(BaseModel):
    title: constr(strip_whitespace=True, min_length=1, max_length=200)
    status: Literal["new", "processing", "failed", "done"] = "new"


class VideoRead(BaseModel):
    id: int
    title: str
    status: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class UploadCreate(BaseModel):
    filename: constr(strip_whitespace=True, min_length=1, max_length=255)
    status: Literal["queued", "processing", "failed", "done"] = "queued"


class UploadRead(BaseModel):
    id: int
    filename: str
    status: Literal["queued", "processing", "failed", "done"]
    error_message: str | None = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ReviewCreate(BaseModel):
    video_id: int
    status: Literal["pending", "approved", "rejected"] = "pending"


class ReviewRead(BaseModel):
    id: int
    video_id: int
    status: Literal["pending", "approved", "rejected"]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class VideoDossier(BaseModel):
    video: VideoRead
    reviews: list[ReviewRead]

class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class VideoPerson(Base):
    __tablename__ = "video_person"
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, nullable=False)
    person_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class PersonCreate(BaseModel):\n    name: constr(strip_whitespace=True, min_length=1, max_length=100)


class PersonRead(BaseModel):
    id: int
    name: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class PersonDossier(BaseModel):
    person: PersonRead
    videos: list[VideoRead]
    reviews: list[ReviewRead]

Base.metadata.create_all(bind=engine)


@app.post("/videos", response_model=VideoRead)
def create_video(payload: VideoCreate, db: Session = Depends(get_db)):
    video = Video(title=payload.title, status=payload.status)
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


@app.get("/videos", response_model=list[VideoRead])
def list_videos(db: Session = Depends(get_db)):
    return db.query(Video).order_by(Video.created_at.desc()).all()


@app.get("/videos/{video_id}", response_model=VideoRead)
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Not found")
    return video


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def readiness():
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as exc:
        log_json("readiness_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Service unavailable")

v1 = APIRouter(prefix="/v1")

@v1.post("/videos", response_model=VideoRead)
def create_video_v1(payload: VideoCreate, db: Session = Depends(get_db)):
    video = Video(title=payload.title, status=payload.status)
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


@v1.get("/videos", response_model=list[VideoRead])
def list_videos_v1(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    return (
        db.query(Video)
        .order_by(Video.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


@v1.get("/videos/{video_id}", response_model=VideoRead)
def get_video_v1(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Not found")
    return video


@v1.get("/health")
def health_v1():
    return {"status": "ok"}

@v1.get("/ready")
def readiness_v1():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as exc:
        log_json("readiness_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Service unavailable")

@v1.post("/uploads", response_model=UploadRead)
def create_upload(payload: UploadCreate, db: Session = Depends(get_db)):
    upload = Upload(filename=payload.filename, status=payload.status)
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


@v1.get("/uploads", response_model=list[UploadRead])
def list_uploads(status: str | None = Query(None), db: Session = Depends(get_db)):
    q = db.query(Upload).order_by(Upload.created_at.desc())
    if status:
        statuses = {s.strip() for s in status.split(",") if s.strip()}
        if statuses:
            q = q.filter(Upload.status.in_(list(statuses)))
    return q.all()


@v1.get("/uploads/{upload_id}", response_model=UploadRead)
def get_upload(upload_id: int, db: Session = Depends(get_db)):
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Not found")
    return upload


def _set_status(upload: Upload, new_status: str, error_message: str | None, db: Session) -> Upload:
    upload.status = new_status
    upload.error_message = error_message
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


@v1.post("/uploads/{upload_id}/start", response_model=UploadRead, dependencies=[Depends(rate_limit("uploads_action", 30, 60))])
def start_upload(upload_id: int, db: Session = Depends(get_db)):
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Not found")
    if upload.status not in {"queued", "failed"}:
        raise HTTPException(status_code=400, detail="Invalid state")
    return _set_status(upload, "processing", None, db)


@v1.post("/uploads/{upload_id}/retry", response_model=UploadRead, dependencies=[Depends(rate_limit("uploads_action", 30, 60))])
def retry_upload(upload_id: int, db: Session = Depends(get_db)):
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Not found")
    if upload.status != "failed":
        raise HTTPException(status_code=400, detail="Invalid state")
    return _set_status(upload, "queued", None, db)


@v1.post("/uploads/{upload_id}/cancel", response_model=UploadRead, dependencies=[Depends(rate_limit("uploads_action", 30, 60))])
def cancel_upload(upload_id: int, db: Session = Depends(get_db)):
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Not found")
    if upload.status in {"done"}:
        raise HTTPException(status_code=400, detail="Invalid state")
    return _set_status(upload, "failed", "canceled", db)

# reviews
@v1.post("/reviews", response_model=ReviewRead)
def create_review(payload: ReviewCreate, db: Session = Depends(get_db)):
    review = Review(video_id=payload.video_id, status=payload.status)
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


@v1.get("/reviews", response_model=list[ReviewRead])
def list_reviews(status: str | None = Query(None), db: Session = Depends(get_db)):
    q = db.query(Review).order_by(Review.created_at.desc())
    if status:
        statuses = {s.strip() for s in status.split(",") if s.strip()}
        if statuses:
            q = q.filter(Review.status.in_(list(statuses)))
    return q.all()


def _set_review_status(review: Review, new_status: str, db: Session) -> Review:
    review.status = new_status
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


@v1.post("/reviews/{review_id}/approve", response_model=ReviewRead, dependencies=[Depends(rate_limit("reviews_action", 30, 60))])
def approve_review(review_id: int, db: Session = Depends(get_db)):
    review = db.query(Review).filter(Review.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Not found")
    if review.status != "pending":
        raise HTTPException(status_code=400, detail="Invalid state")
    return _set_review_status(review, "approved", db)


@v1.post("/reviews/{review_id}/reject", response_model=ReviewRead, dependencies=[Depends(rate_limit("reviews_action", 30, 60))])
def reject_review(review_id: int, db: Session = Depends(get_db)):
    review = db.query(Review).filter(Review.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Not found")
    if review.status != "pending":
        raise HTTPException(status_code=400, detail="Invalid state")
    return _set_review_status(review, "rejected", db)

@v1.get("/dossiers/videos/{video_id}", response_model=VideoDossier)
def get_video_dossier(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Not found")
    reviews = db.query(Review).filter(Review.video_id == video_id).order_by(Review.created_at.desc()).all()
    return VideoDossier(video=video, reviews=reviews)

# persons
@v1.post("/persons", response_model=PersonRead)
def create_person(payload: PersonCreate, db: Session = Depends(get_db)):
    person = Person(name=payload.name)
    db.add(person)
    db.commit()
    db.refresh(person)
    return person


@v1.get("/persons/{person_id}", response_model=PersonRead)
def get_person(person_id: int, db: Session = Depends(get_db)):
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Not found")
    return person


@v1.get("/persons", response_model=list[PersonRead])
def list_persons(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    return (
        db.query(Person)
        .order_by(Person.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


@v1.post("/persons/{person_id}/link/{video_id}", response_model=VideoRead)
def link_video_to_person(person_id: int, video_id: int, db: Session = Depends(get_db)):
    person = db.query(Person).filter(Person.id == person_id).first()
    video = db.query(Video).filter(Video.id == video_id).first()
    if not person or not video:
        raise HTTPException(status_code=404, detail="Not found")
    link = VideoPerson(video_id=video_id, person_id=person_id)
    db.add(link)
    db.commit()
    db.refresh(video)
    return video


@v1.get("/dossiers/persons/{person_id}", response_model=PersonDossier)
def get_person_dossier(person_id: int, db: Session = Depends(get_db)):
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Not found")
    link_rows = db.query(VideoPerson).filter(VideoPerson.person_id == person_id).all()
    video_ids = [l.video_id for l in link_rows]
    videos = []
    reviews = []
    if video_ids:
        videos = db.query(Video).filter(Video.id.in_(video_ids)).order_by(Video.created_at.desc()).all()
        reviews = db.query(Review).filter(Review.video_id.in_(video_ids)).order_by(Review.created_at.desc()).all()
    return PersonDossier(person=person, videos=videos, reviews=reviews)
app.include_router(v1)

 

# Rate limiting is applied via dependencies directly on route decorators above

# Prometheus metrics endpoint
@app.get("/metrics")
def metrics() -> Response:
    lines = [
        "# HELP http_requests_total Total HTTP requests",
        "# TYPE http_requests_total counter",
    ]
    for (method, path, status), count in _requests_total.items():
        lines.append(f'http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}')
    lines.append("# HELP http_request_duration_seconds_sum Total request duration seconds")
    lines.append("# TYPE http_request_duration_seconds_sum gauge")
    for (method, path), s in _latency_sum.items():
        lines.append(f'http_request_duration_seconds_sum{{method="{method}",path="{path}"}} {s}')
    lines.append("# HELP http_request_duration_seconds_count Request duration samples")
    lines.append("# TYPE http_request_duration_seconds_count counter")
    for (method, path), c in _latency_count.items():
        lines.append(f'http_request_duration_seconds_count{{method="{method}",path="{path}"}} {c}')
    return Response("\n".join(lines) + "\n", media_type="text/plain")
