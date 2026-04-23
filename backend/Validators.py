from pydantic import BaseModel,Field, validator 

# class for file upload to backend to send to ai model
class FileUpload(BaseModel):
    file: bytes = Field(..., description="The file to be uploaded")

    @validator('file')
    def validate_file(cls, value):
        if not value:
            raise ValueError("File cannot be empty")
        return value


