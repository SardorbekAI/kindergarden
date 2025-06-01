#!/usr/bin/env python3
"""
Daycare Food Tracking and Inventory Management System
A comprehensive backend system for managing kitchen inventory, food tracking, and daycare reporting.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from functools import wraps
import asyncio
from contextlib import asynccontextmanager

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

# Pydantic models
from pydantic import BaseModel, Field, ConfigDict
from fastapi import Query
from pydantic.json_schema import JsonSchemaValue

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func

# Authentication imports
from passlib.context import CryptContext
from jose import JWTError, jwt

# Celery imports
from celery import Celery
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = "sqlite:///./daycare_food_system.db"
SECRET_KEY = "fvyuygh87yughgt67f6fgyu87iu09uyhji98hyughjnji"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REDIS_URL = "redis://localhost:6379/0"

# Database setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Celery setup
celery_app = Celery(
    "daycare_food_system",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["main"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "daily-inventory-check": {
            "task": "main.check_low_inventory",
            "schedule": 86400.0,  # 24 hours
        },
        "monthly-report-generation": {
            "task": "main.generate_monthly_reports",
            "schedule": 2592000.0,  # 30 days
        },
    },
)

# Redis client
redis_client = redis.Redis.from_url(REDIS_URL)

# Database Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    firstname = Column(String(100), nullable=False)
    lastname = Column(String(100), nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)  # admin, manager, cook
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    meal_servings = relationship("MealServing", back_populates="user")
    inventory_transactions = relationship("InventoryTransaction", back_populates="user")
    notifications = relationship("Notification", back_populates="user")

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    current_weight_grams = Column(Float, nullable=False, default=0.0)
    delivery_date = Column(DateTime(timezone=True))
    threshold_warning_grams = Column(Float, nullable=False, default=1000.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    inventory_transactions = relationship("InventoryTransaction", back_populates="product")

class Meal(Base):
    __tablename__ = "meals"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text)
    ingredients_json = Column(Text, nullable=False)  # JSON string: [{"product_id": 1, "required_grams": 100}]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    meal_servings = relationship("MealServing", back_populates="meal")

class MealServing(Base):
    __tablename__ = "meal_servings"

    id = Column(Integer, primary_key=True, index=True)
    meal_id = Column(Integer, ForeignKey("meals.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    served_at = Column(DateTime(timezone=True), server_default=func.now())
    ingredients_used_json = Column(Text, nullable=False)  # JSON string of ingredients used
    success_status = Column(Boolean, nullable=False, default=True)

    meal = relationship("Meal", back_populates="meal_servings")
    user = relationship("User", back_populates="meal_servings")

class InventoryTransaction(Base):
    __tablename__ = "inventory_transactions"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    transaction_type = Column(String(50), nullable=False)  # delivery, usage, adjustment
    amount_grams = Column(Float, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text)

    product = relationship("Product", back_populates="inventory_transactions")
    user = relationship("User", back_populates="inventory_transactions")

class MonthlyReport(Base):
    __tablename__ = "monthly_reports"

    id = Column(Integer, primary_key=True, index=True)
    month = Column(Integer, nullable=False)
    year = Column(Integer, nullable=False)
    total_served = Column(Integer, nullable=False, default=0)
    total_possible = Column(Integer, nullable=False, default=0)
    discrepancy_rate = Column(Float, nullable=False, default=0.0)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("idx_month_year", "month", "year"),)

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String(50), nullable=False)  # low_inventory, misuse_alert, system_notification
    is_read = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="notifications")

# Pydantic Models
class UserBase(BaseModel):
    firstname: str = Field(..., min_length=1, max_length=100)
    lastname: str = Field(..., min_length=1, max_length=100)
    username: str = Field(..., min_length=3, max_length=100)
    role: str = Field(..., pattern="^(admin|manager|cook)$")

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserResponse(UserBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ProductBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    current_weight_grams: float = Field(..., ge=0)
    threshold_warning_grams: float = Field(default=1000.0, ge=0)

class ProductCreate(ProductBase):
    pass

class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    current_weight_grams: Optional[float] = Field(None, ge=0)
    threshold_warning_grams: Optional[float] = Field(None, ge=0)

class ProductResponse(ProductBase):
    id: int
    delivery_date: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)

class MealIngredient(BaseModel):
    product_id: int
    required_grams: float = Field(..., gt=0)

class MealBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    ingredients: List[MealIngredient]

class MealCreate(MealBase):
    pass

class MealUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    ingredients: Optional[List[MealIngredient]] = None

class MealResponse(MealBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class MealServingResponse(BaseModel):
    id: int
    meal_id: int
    meal_name: str
    user_id: int
    user_name: str
    served_at: datetime
    ingredients_used: List[Dict[str, Any]]
    success_status: bool

    model_config = ConfigDict(from_attributes=True)

class PortionEstimate(BaseModel):
    meal_id: int
    meal_name: str
    available_portions: int

class NotificationResponse(BaseModel):
    id: int
    message: str
    type: str
    is_read: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class MonthlyReportResponse(BaseModel):
    id: int
    month: int
    year: int
    total_served: int
    total_possible: int
    discrepancy_rate: float
    generated_at: datetime

    model_config = ConfigDict(from_attributes=True)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

def require_role(allowed_roles: List[str]):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user') or args[-1]  # Assume current_user is last parameter
            if hasattr(current_user, 'role') and current_user.role not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Utility functions
def log_inventory_transaction(db: Session, product_id: int, transaction_type: str, amount_grams: float, user_id: int, notes: str = None):
    transaction = InventoryTransaction(
        product_id=product_id,
        transaction_type=transaction_type,
        amount_grams=amount_grams,
        user_id=user_id,
        notes=notes
    )
    db.add(transaction)
    db.commit()

def create_notification(db: Session, user_id: int, message: str, notification_type: str):
    notification = Notification(
        user_id=user_id,
        message=message,
        type=notification_type
    )
    db.add(notification)
    db.commit()

def calculate_portion_estimates(db: Session) -> List[PortionEstimate]:
    meals = db.query(Meal).all()
    estimates = []

    for meal in meals:
        ingredients = json.loads(meal.ingredients_json)
        min_portions = float('inf')

        for ingredient in ingredients:
            product = db.query(Product).filter(Product.id == ingredient["product_id"]).first()
            if product:
                available_portions = int(product.current_weight_grams / ingredient["required_grams"])
                min_portions = min(min_portions, available_portions)
            else:
                min_portions = 0
                break

        if min_portions == float('inf'):
            min_portions = 0

        estimates.append(PortionEstimate(
            meal_id=meal.id,
            meal_name=meal.name,
            available_portions=max(0, min_portions)
        ))

    return estimates

# Celery Tasks
@celery_app.task
def check_low_inventory():
    db = SessionLocal()
    try:
        products = db.query(Product).filter(
            Product.current_weight_grams < Product.threshold_warning_grams
        ).all()

        admins = db.query(User).filter(User.role == "admin").all()

        for product in products:
            message = f"Low inventory warning: {product.name} has only {product.current_weight_grams}g remaining (threshold: {product.threshold_warning_grams}g)"
            for admin in admins:
                create_notification(db, admin.id, message, "low_inventory")

        logger.info(f"Checked inventory - {len(products)} products below threshold")
        return f"Processed {len(products)} low inventory alerts"
    finally:
        db.close()

@celery_app.task
def generate_monthly_reports():
    db = SessionLocal()
    try:
        current_date = datetime.now()
        month = current_date.month
        year = current_date.year

        # Check if report already exists
        existing_report = db.query(MonthlyReport).filter(
            MonthlyReport.month == month,
            MonthlyReport.year == year
        ).first()

        if existing_report:
            logger.info(f"Monthly report for {month}/{year} already exists")
            return f"Report for {month}/{year} already exists"

        # Calculate served meals this month
        start_of_month = datetime(year, month, 1)
        if month == 12:
            end_of_month = datetime(year + 1, 1, 1)
        else:
            end_of_month = datetime(year, month + 1, 1)

        total_served = db.query(MealServing).filter(
            MealServing.served_at >= start_of_month,
            MealServing.served_at < end_of_month,
            MealServing.success_status == True
        ).count()

        # Calculate theoretical maximum (simplified calculation)
        # This would need more complex logic in a real system
        total_possible = total_served + int(total_served * 0.2)  # Assuming 20% waste factor

        discrepancy_rate = ((total_possible - total_served) / total_possible * 100) if total_possible > 0 else 0

        report = MonthlyReport(
            month=month,
            year=year,
            total_served=total_served,
            total_possible=total_possible,
            discrepancy_rate=discrepancy_rate
        )

        db.add(report)
        db.commit()

        # Alert if discrepancy rate is high
        if discrepancy_rate > 15.0:
            admins = db.query(User).filter(User.role == "admin").all()
            message = f"High discrepancy rate detected for {month}/{year}: {discrepancy_rate:.2f}% - Potential ingredient misuse"
            for admin in admins:
                create_notification(db, admin.id, message, "misuse_alert")

        logger.info(f"Generated monthly report for {month}/{year}")
        return f"Generated report for {month}/{year} - Discrepancy: {discrepancy_rate:.2f}%"
    finally:
        db.close()

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    await initialize_sample_data()
    yield
    # Shutdown
    pass

app = FastAPI(
    title="Daycare Food Tracking System",
    description="A comprehensive backend system for managing kitchen inventory, food tracking, and daycare reporting",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample data initialization
async def initialize_sample_data():
    db = SessionLocal()
    try:
        # Check if data already exists
        if db.query(User).first():
            return

        # Create sample users
        users_data = [
            {"firstname": "Admin", "lastname": "User", "username": "admin", "password": "admin123", "role": "admin"},
            {"firstname": "John", "lastname": "Manager", "username": "manager1", "password": "manager123", "role": "manager"},
            {"firstname": "Jane", "lastname": "Manager", "username": "manager2", "password": "manager123", "role": "manager"},
            {"firstname": "Bob", "lastname": "Cook", "username": "cook1", "password": "cook123", "role": "cook"},
            {"firstname": "Alice", "lastname": "Cook", "username": "cook2", "password": "cook123", "role": "cook"},
        ]

        for user_data in users_data:
            user = User(
                firstname=user_data["firstname"],
                lastname=user_data["lastname"],
                username=user_data["username"],
                hashed_password=get_password_hash(user_data["password"]),
                role=user_data["role"]
            )
            db.add(user)

        # Create sample products
        products_data = [
            {"name": "Beef", "current_weight_grams": 5000.0, "threshold_warning_grams": 1000.0},
            {"name": "Chicken", "current_weight_grams": 4000.0, "threshold_warning_grams": 800.0},
            {"name": "Rice", "current_weight_grams": 10000.0, "threshold_warning_grams": 2000.0},
            {"name": "Potato", "current_weight_grams": 8000.0, "threshold_warning_grams": 1500.0},
            {"name": "Carrot", "current_weight_grams": 3000.0, "threshold_warning_grams": 500.0},
            {"name": "Onion", "current_weight_grams": 2000.0, "threshold_warning_grams": 400.0},
            {"name": "Salt", "current_weight_grams": 1000.0, "threshold_warning_grams": 200.0},
            {"name": "Oil", "current_weight_grams": 2000.0, "threshold_warning_grams": 300.0},
            {"name": "Tomato", "current_weight_grams": 2500.0, "threshold_warning_grams": 500.0},
            {"name": "Pasta", "current_weight_grams": 6000.0, "threshold_warning_grams": 1000.0},
        ]

        for product_data in products_data:
            product = Product(**product_data, delivery_date=datetime.now())
            db.add(product)

        db.commit()

        # Create sample meals
        meals_data = [
            {
                "name": "Beef Stew",
                "description": "Hearty beef stew with vegetables",
                "ingredients": [
                    {"product_id": 1, "required_grams": 200.0},  # Beef
                    {"product_id": 4, "required_grams": 150.0},  # Potato
                    {"product_id": 5, "required_grams": 50.0},   # Carrot
                    {"product_id": 6, "required_grams": 30.0},   # Onion
                    {"product_id": 7, "required_grams": 5.0},    # Salt
                ]
            },
            {
                "name": "Chicken Rice",
                "description": "Grilled chicken with steamed rice",
                "ingredients": [
                    {"product_id": 2, "required_grams": 180.0},  # Chicken
                    {"product_id": 3, "required_grams": 100.0},  # Rice
                    {"product_id": 8, "required_grams": 10.0},   # Oil
                    {"product_id": 7, "required_grams": 3.0},    # Salt
                ]
            },
            {
                "name": "Vegetable Pasta",
                "description": "Pasta with mixed vegetables",
                "ingredients": [
                    {"product_id": 10, "required_grams": 120.0}, # Pasta
                    {"product_id": 5, "required_grams": 40.0},   # Carrot
                    {"product_id": 9, "required_grams": 60.0},   # Tomato
                    {"product_id": 6, "required_grams": 25.0},   # Onion
                    {"product_id": 8, "required_grams": 15.0},   # Oil
                ]
            },
            {
                "name": "Chicken Soup",
                "description": "Nutritious chicken soup",
                "ingredients": [
                    {"product_id": 2, "required_grams": 150.0},  # Chicken
                    {"product_id": 4, "required_grams": 100.0},  # Potato
                    {"product_id": 5, "required_grams": 30.0},   # Carrot
                    {"product_id": 7, "required_grams": 4.0},    # Salt
                ]
            },
            {
                "name": "Beef Rice Bowl",
                "description": "Beef and rice with vegetables",
                "ingredients": [
                    {"product_id": 1, "required_grams": 160.0},  # Beef
                    {"product_id": 3, "required_grams": 120.0},  # Rice
                    {"product_id": 6, "required_grams": 20.0},   # Onion
                    {"product_id": 8, "required_grams": 12.0},   # Oil
                ]
            }
        ]

        for meal_data in meals_data:
            meal = Meal(
                name=meal_data["name"],
                description=meal_data["description"],
                ingredients_json=json.dumps(meal_data["ingredients"])
            )
            db.add(meal)

        db.commit()
        logger.info("Sample data initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing sample data: {e}")
        db.rollback()
    finally:
        db.close()

# API Endpoints

# Authentication Endpoints
@app.post("/auth/login", response_model=Token)
async def login(login_request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == login_request.username).first()
    if not user or not verify_password(login_request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/create", response_model=UserResponse)
async def create_user(
    user: UserCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can create users")

    # Check if username already exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = get_password_hash(user.password)
    db_user = User(
        firstname=user.firstname,
        lastname=user.lastname,
        username=user.username,
        hashed_password=hashed_password,
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Product Management Endpoints
@app.get("/products", response_model=List[ProductResponse])
async def get_products(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    products = db.query(Product).all()
    return products

@app.post("/products", response_model=ProductResponse)
async def create_product(
    product: ProductCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    db_product = Product(**product.dict(), delivery_date=datetime.now())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)

    # Log the inventory transaction
    log_inventory_transaction(
        db, db_product.id, "delivery", product.current_weight_grams,
        current_user.id, f"Initial product creation: {product.name}"
    )

    return db_product

@app.put("/products/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: int,
    product_update: ProductUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    db_product = db.query(Product).filter(Product.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")

    old_weight = db_product.current_weight_grams

    update_data = product_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_product, field, value)

    db_product.updated_at = datetime.now()
    db.commit()
    db.refresh(db_product)

    # Log weight changes
    if "current_weight_grams" in update_data:
        weight_diff = db_product.current_weight_grams - old_weight
        log_inventory_transaction(
            db, product_id, "adjustment", weight_diff,
            current_user.id, f"Manual inventory adjustment"
        )

    return db_product

@app.delete("/products/{product_id}")
async def delete_product(
    product_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can delete products")

    db_product = db.query(Product).filter(Product.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")

    db.delete(db_product)
    db.commit()
    return {"message": "Product deleted successfully"}

@app.post("/products/{product_id}/delivery")
async def record_delivery(
    product_id: int,
    delivery_amount: float = Query(..., gt=0),  # Use Query instead of Field
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    db_product = db.query(Product).filter(Product.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")

    db_product.current_weight_grams += delivery_amount
    db_product.delivery_date = datetime.now()
    db_product.updated_at = datetime.now()
    db.commit()

    # Log the delivery transaction
    log_inventory_transaction(
        db, product_id, "delivery", delivery_amount,
        current_user.id, f"New delivery recorded for {db_product.name}"
    )

    return {"message": f"Delivery recorded: {delivery_amount}g added to {db_product.name}"}

# Meal Management Endpoints
@app.get("/meals", response_model=List[MealResponse])
async def get_meals(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    meals = db.query(Meal).all()
    meal_responses = []

    for meal in meals:
        ingredients = json.loads(meal.ingredients_json)
        meal_data = {
            "id": meal.id,
            "name": meal.name,
            "description": meal.description,
            "ingredients": ingredients,
            "created_at": meal.created_at,
            "updated_at": meal.updated_at
        }
        meal_responses.append(MealResponse(**meal_data))

    return meal_responses

@app.post("/meals", response_model=MealResponse)
async def create_meal(
    meal: MealCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    print(current_user.role)
    if current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # Validate that all referenced products exist
    for ingredient in meal.ingredients:
        product = db.query(Product).filter(Product.id == ingredient.product_id).first()
        if not product:
            raise HTTPException(status_code=400, detail=f"Product with ID {ingredient.product_id} not found")

    db_meal = Meal(
        name=meal.name,
        description=meal.description,
        ingredients_json=json.dumps([ing.dict() for ing in meal.ingredients])
    )
    db.add(db_meal)
    db.commit()
    db.refresh(db_meal)

    # Convert back to response format
    ingredients = json.loads(db_meal.ingredients_json)
    meal_response = MealResponse(
        id=db_meal.id,
        name=db_meal.name,
        description=db_meal.description,
        ingredients=ingredients,
        created_at=db_meal.created_at,
        updated_at=db_meal.updated_at
    )

    return meal_response

@app.put("/meals/{meal_id}", response_model=MealResponse)
async def update_meal(
    meal_id: int,
    meal_update: MealUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    db_meal = db.query(Meal).filter(Meal.id == meal_id).first()
    if not db_meal:
        raise HTTPException(status_code=404, detail="Meal not found")

    update_data = meal_update.dict(exclude_unset=True)

    # Validate ingredients if provided
    if "ingredients" in update_data and update_data["ingredients"]:
        for ingredient in update_data["ingredients"]:
            product = db.query(Product).filter(Product.id == ingredient["product_id"]).first()
            if not product:
                raise HTTPException(status_code=400, detail=f"Product with ID {ingredient['product_id']} not found")
        update_data["ingredients_json"] = json.dumps([ing.dict() for ing in meal_update.ingredients])
        del update_data["ingredients"]

    for field, value in update_data.items():
        setattr(db_meal, field, value)

    db_meal.updated_at = datetime.now()
    db.commit()
    db.refresh(db_meal)

    # Convert back to response format
    ingredients = json.loads(db_meal.ingredients_json)
    meal_response = MealResponse(
        id=db_meal.id,
        name=db_meal.name,
        description=db_meal.description,
        ingredients=ingredients,
        created_at=db_meal.created_at,
        updated_at=db_meal.updated_at
    )

    return meal_response

@app.delete("/meals/{meal_id}")
async def delete_meal(
    meal_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    db_meal = db.query(Meal).filter(Meal.id == meal_id).first()
    if not db_meal:
        raise HTTPException(status_code=404, detail="Meal not found")

    db.delete(db_meal)
    db.commit()
    return {"message": "Meal deleted successfully"}

@app.get("/meals/{meal_id}/portions-available", response_model=PortionEstimate)
async def get_meal_portions_available(
    meal_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    meal = db.query(Meal).filter(Meal.id == meal_id).first()
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")

    ingredients = json.loads(meal.ingredients_json)
    min_portions = float('inf')

    for ingredient in ingredients:
        product = db.query(Product).filter(Product.id == ingredient["product_id"]).first()
        if product:
            available_portions = int(product.current_weight_grams / ingredient["required_grams"])
            min_portions = min(min_portions, available_portions)
        else:
            min_portions = 0
            break

    if min_portions == float('inf'):
        min_portions = 0

    return PortionEstimate(
        meal_id=meal.id,
        meal_name=meal.name,
        available_portions=max(0, min_portions)
    )

# Serving System Endpoints
@app.post("/meals/{meal_id}/serve")
async def serve_meal(
    meal_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    meal = db.query(Meal).filter(Meal.id == meal_id).first()
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")

    ingredients = json.loads(meal.ingredients_json)

    # Check if all ingredients are available
    insufficient_ingredients = []
    for ingredient in ingredients:
        product = db.query(Product).filter(Product.id == ingredient["product_id"]).first()
        if not product:
            raise HTTPException(status_code=400, detail=f"Product with ID {ingredient['product_id']} not found")

        if product.current_weight_grams < ingredient["required_grams"]:
            insufficient_ingredients.append({
                "product_name": product.name,
                "required": ingredient["required_grams"],
                "available": product.current_weight_grams,
                "deficit": ingredient["required_grams"] - product.current_weight_grams
            })

    if insufficient_ingredients:
        error_details = {
            "message": "Insufficient ingredients to serve this meal",
            "insufficient_ingredients": insufficient_ingredients
        }
        raise HTTPException(status_code=400, detail=error_details)

    # Deduct ingredients from inventory
    ingredients_used = []
    for ingredient in ingredients:
        product = db.query(Product).filter(Product.id == ingredient["product_id"]).first()
        product.current_weight_grams -= ingredient["required_grams"]
        product.updated_at = datetime.now()

        # Log usage transaction
        log_inventory_transaction(
            db, product.id, "usage", -ingredient["required_grams"],
            current_user.id, f"Used for serving {meal.name}"
        )

        ingredients_used.append({
            "product_id": ingredient["product_id"],
            "product_name": product.name,
            "used_grams": ingredient["required_grams"]
        })

    # Create meal serving record
    meal_serving = MealServing(
        meal_id=meal_id,
        user_id=current_user.id,
        ingredients_used_json=json.dumps(ingredients_used),
        success_status=True
    )
    db.add(meal_serving)
    db.commit()

    return {
        "message": f"Meal '{meal.name}' served successfully",
        "served_by": f"{current_user.firstname} {current_user.lastname}",
        "ingredients_used": ingredients_used,
        "served_at": meal_serving.served_at
    }

@app.get("/meals/portions-estimate", response_model=List[PortionEstimate])
async def get_portion_estimates(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return calculate_portion_estimates(db)

@app.get("/servings/history", response_model=List[MealServingResponse])
async def get_serving_history(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    servings = db.query(MealServing).join(Meal).join(User).order_by(MealServing.served_at.desc()).limit(limit).all()

    serving_responses = []
    for serving in servings:
        ingredients_used = json.loads(serving.ingredients_used_json)
        serving_response = MealServingResponse(
            id=serving.id,
            meal_id=serving.meal_id,
            meal_name=serving.meal.name,
            user_id=serving.user_id,
            user_name=f"{serving.user.firstname} {serving.user.lastname}",
            served_at=serving.served_at,
            ingredients_used=ingredients_used,
            success_status=serving.success_status
        )
        serving_responses.append(serving_response)

    return serving_responses

# Reporting Endpoints
@app.get("/reports/ingredient-usage")
async def get_ingredient_usage_report(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    start_date = datetime.now() - timedelta(days=days)

    transactions = db.query(InventoryTransaction).join(Product).join(User).filter(
        InventoryTransaction.timestamp >= start_date
    ).all()

    usage_report = {}
    for transaction in transactions:
        product_name = transaction.product.name
        if product_name not in usage_report:
            usage_report[product_name] = {
                "product_id": transaction.product_id,
                "deliveries": 0,
                "usage": 0,
                "adjustments": 0,
                "transactions": []
            }

        transaction_data = {
            "date": transaction.timestamp.isoformat(),
            "type": transaction.transaction_type,
            "amount_grams": transaction.amount_grams,
            "user": f"{transaction.user.firstname} {transaction.user.lastname}",
            "notes": transaction.notes
        }

        usage_report[product_name]["transactions"].append(transaction_data)

        if transaction.transaction_type == "delivery":
            usage_report[product_name]["deliveries"] += transaction.amount_grams
        elif transaction.transaction_type == "usage":
            usage_report[product_name]["usage"] += abs(transaction.amount_grams)
        elif transaction.transaction_type == "adjustment":
            usage_report[product_name]["adjustments"] += transaction.amount_grams

    return {
        "period_days": days,
        "start_date": start_date.isoformat(),
        "end_date": datetime.now().isoformat(),
        "ingredient_usage": usage_report
    }

@app.get("/reports/monthly-summary", response_model=List[MonthlyReportResponse])
async def get_monthly_summary_reports(
    months: int = 12,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    reports = db.query(MonthlyReport).order_by(MonthlyReport.year.desc(), MonthlyReport.month.desc()).limit(months).all()
    return reports

@app.get("/reports/user-activity")
async def get_user_activity_report(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    start_date = datetime.now() - timedelta(days=days)

    servings = db.query(MealServing).join(Meal).join(User).filter(
        MealServing.served_at >= start_date
    ).order_by(MealServing.served_at.desc()).all()

    user_activity = {}
    for serving in servings:
        user_name = f"{serving.user.firstname} {serving.user.lastname}"
        if user_name not in user_activity:
            user_activity[user_name] = {
                "user_id": serving.user_id,
                "total_meals_served": 0,
                "meals_by_type": {},
                "activity_log": []
            }

        user_activity[user_name]["total_meals_served"] += 1

        meal_name = serving.meal.name
        if meal_name not in user_activity[user_name]["meals_by_type"]:
            user_activity[user_name]["meals_by_type"][meal_name] = 0
        user_activity[user_name]["meals_by_type"][meal_name] += 1

        user_activity[user_name]["activity_log"].append({
            "meal_name": meal_name,
            "served_at": serving.served_at.isoformat(),
            "success": serving.success_status
        })

    return {
        "period_days": days,
        "start_date": start_date.isoformat(),
        "end_date": datetime.now().isoformat(),
        "user_activity": user_activity
    }

@app.get("/reports/discrepancy-analysis")
async def get_discrepancy_analysis(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # Get the latest monthly reports
    reports = db.query(MonthlyReport).order_by(MonthlyReport.year.desc(), MonthlyReport.month.desc()).limit(6).all()

    analysis = {
        "average_discrepancy_rate": 0,
        "trend": "stable",
        "high_risk_months": [],
        "monthly_data": []
    }

    if reports:
        total_discrepancy = sum(report.discrepancy_rate for report in reports)
        analysis["average_discrepancy_rate"] = total_discrepancy / len(reports)

        # Determine trend (simplified)
        if len(reports) >= 3:
            recent_avg = sum(report.discrepancy_rate for report in reports[:3]) / 3
            older_avg = sum(report.discrepancy_rate for report in reports[3:]) / (len(reports) - 3)

            if recent_avg > older_avg + 2:
                analysis["trend"] = "increasing"
            elif recent_avg < older_avg - 2:
                analysis["trend"] = "decreasing"

        for report in reports:
            report_data = {
                "month": report.month,
                "year": report.year,
                "discrepancy_rate": report.discrepancy_rate,
                "total_served": report.total_served,
                "total_possible": report.total_possible
            }
            analysis["monthly_data"].append(report_data)

            if report.discrepancy_rate > 15.0:
                analysis["high_risk_months"].append(f"{report.month}/{report.year}")

    return analysis

# Notification Endpoints
@app.get("/notifications", response_model=List[NotificationResponse])
async def get_notifications(
    unread_only: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Notification).filter(Notification.user_id == current_user.id)

    if unread_only:
        query = query.filter(Notification.is_read == False)

    notifications = query.order_by(Notification.created_at.desc()).all()
    return notifications

@app.post("/notifications/{notification_id}/mark-read")
async def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id
    ).first()

    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    notification.is_read = True
    db.commit()
    return {"message": "Notification marked as read"}

# Background Tasks Endpoints
@app.post("/tasks/check-inventory")
async def trigger_inventory_check(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    task = check_low_inventory.delay()
    return {"message": "Inventory check task started", "task_id": str(task.id)}

@app.post("/tasks/generate-monthly-report")
async def trigger_monthly_report(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can generate reports")

    task = generate_monthly_reports.delay()
    return {"message": "Monthly report generation task started", "task_id": str(task.id)}

# Health Check and System Information
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/system/info")
async def get_system_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    # Get system statistics
    total_users = db.query(User).count()
    total_products = db.query(Product).count()
    total_meals = db.query(Meal).count()
    total_servings = db.query(MealServing).count()

    # Get low inventory products
    low_inventory_products = db.query(Product).filter(
        Product.current_weight_grams < Product.threshold_warning_grams
    ).count()

    return {
        "system_statistics": {
            "total_users": total_users,
            "total_products": total_products,
            "total_meals": total_meals,
            "total_servings": total_servings,
            "low_inventory_products": low_inventory_products
        },
        "current_user": {
            "name": f"{current_user.firstname} {current_user.lastname}",
            "role": current_user.role
        },
        "system_info": {
            "database_url": DATABASE_URL,
            "redis_url": REDIS_URL,
            "version": "1.0.0"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)