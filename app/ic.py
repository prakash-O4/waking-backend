from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
import os
from supabase import create_client, Client

# Create router with the 'ic' prefix
ic_router = APIRouter(prefix="/ic", tags=["Appointment Booking"])

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Initialize Supabase client
def get_supabase() -> Client:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase


class AppointmentBase(BaseModel):
    user_id: int
    appointment_date: datetime
    service_type: str
    user_name: str
    phone: str
    area: str
    preferred_location: str
    gender: str
    notes: Optional[str] = None

class AppointmentCreate(AppointmentBase):
    pass

class Appointment(AppointmentBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True


# Models
class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    location: Optional[str] = None
    image_url: Optional[str] = None

class UserCreate(UserBase):
    password: str
    pass

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    location: Optional[str] = None
    image_url: Optional[str] = None

# Endpoints
@ic_router.post("/users/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, supabase: Client = Depends(get_supabase)):
    try:
        # Check if user already exists
        response = supabase.table("icusers").select("*").eq("email", user.email).execute()
        
        if response.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create new user
        user_data = user.model_dump()
        user_data["created_at"] = datetime.now().isoformat()
        
        response = supabase.table("icusers").insert(user_data).execute()
        return response.data[0]
    
    except Exception as e:
        print(f"Failed to create user: ${str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )
    
@ic_router.put("/users/{user_id}", response_model=User)
async def update_user(user_id: int, user_update: UserUpdate, supabase: Client = Depends(get_supabase)):
    try:
        # Check if user exists
        response = supabase.table("icusers").select("*").eq("id", str(user_id)).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Prepare updated data
        update_data = user_update.model_dump(exclude_unset=True)
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields provided for update"
            )

        # Perform update
        update_response = supabase.table("icusers").update(update_data).eq("id", str(user_id)).execute()
        
        if not update_response.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user"
            )
        
        return update_response.data[0]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}"
        )

@ic_router.post('/login/', response_model=User, status_code=status.HTTP_200_OK)
async def login_user(user: UserLogin, supabase: Client = Depends(get_supabase)):
    try:
        # Check if user exists
        response = supabase.table("icusers").select("*").eq("email", user.email).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify password (this is a placeholder, implement actual password verification)
        if response.data[0]["password"] != user.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        return response.data[0] 
    
    except HTTPException as e:
        print(f"Error while logging in: {str(e)}")
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to login: {str(e)}"
        )

@ic_router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int, supabase: Client = Depends(get_supabase)):
    try:
        response = supabase.table("icusers").select("*").eq("id", str(user_id)).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return response.data[0]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user: {str(e)}"
        )
from datetime import datetime

@ic_router.post("/appointments/", response_model=Appointment, status_code=status.HTTP_201_CREATED)
async def create_appointment(appointment: AppointmentCreate, supabase: Client = Depends(get_supabase)):
    try:
        # Verify user exists
        user_response = supabase.table("icusers").select("*").eq("id", str(appointment.user_id)).execute()
        
        if not user_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Create appointment
        appointment_data = appointment.model_dump()

        # Convert appointment_date to ISO if it's not None
        if appointment_data.get("appointment_date"):
            appointment_data["appointment_date"] = appointment_data["appointment_date"].isoformat()
        
        appointment_data["created_at"] = datetime.now().isoformat()
        appointment_data["user_id"] = appointment_data["user_id"]
        
        response = supabase.table("appointments").insert(appointment_data).execute()
        return response.data[0]
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to create appointment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create appointment: {str(e)}"
        )


@ic_router.get("/appointments/{appointment_id}", response_model=Appointment)
async def get_appointment(appointment_id: int, supabase: Client = Depends(get_supabase)):
    try:
        response = supabase.table("appointments").select("*").eq("id", str(appointment_id)).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Appointment not found"
            )
        
        return response.data[0]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve appointment: {str(e)}"
        )

@ic_router.get("/users/{user_id}/appointments/", response_model=list[Appointment])
async def get_user_appointments(user_id:int, supabase: Client = Depends(get_supabase)):
    try:
        # Verify user exists
        user_response = supabase.table("icusers").select("*").eq("id", str(user_id)).execute()
        
        if not user_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get appointments for user
        response = supabase.table("appointments").select("*").eq("user_id", str(user_id)).execute()
        return response.data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve appointments: {str(e)}"
        )

@ic_router.get('/health', status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "ok", "message": "IC Service is running."}