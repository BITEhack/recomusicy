import uvicorn

# use this link in web http://127.0.0.1:8000/docs
if __name__ == "__main__":
    uvicorn.run("app.app:app", port=8000, reload=True)
