from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import router, load_artifacts
from src.utils.logger import get_logger
from contextlib import asynccontextmanager
import time

from dotenv import load_dotenv
load_dotenv(override=True)

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing API application...")
    load_artifacts()
    yield
    logger.info("Shutting down API application...")

app = FastAPI(
    title="Fraud Detection Intelligence Platform",
    description="Enterprise-grade credit card fraud detection engine with explainable AI and GenAI Analyst Copilot.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Router
app.include_router(router)

# Request response logger middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"Method: {request.method} Path: {request.url.path} Status: {response.status_code} Duration: {duration:.4f}s")
    return response

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error. Transaction analysis failed.", "detail": str(exc)}
    )

@app.get("/")
def root():
    return {
        "service": "Fraud Intelligence Platform API",
        "status": "online",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fraud-intelligence-platform"}
