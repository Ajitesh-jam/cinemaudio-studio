# Docker Setup for Cinemaudio Backend

## Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)

## Quick Start

1. **Create a `.env` file in the backend directory** (or project root) with your environment variables:

   ```bash
   PORT=8000
   HOST=0.0.0.0
   GEMINI_API_KEY=your_api_key_here
   GOOGLE_API_KEY=your_api_key_here
   ```

2. **Build and run with Docker Compose:**

   ```bash
   docker-compose up --build
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Building the Docker Image

```bash
cd backend
docker build -t cinemaudio-backend .
```

## Running the Container

```bash
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  cinemaudio-backend
```

## GPU Support (Optional)

If you have NVIDIA GPU and want to use it for model inference:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Update `docker-compose.yml` to add GPU support:
   ```yaml
   services:
     backend:
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```

## Environment Variables

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `GEMINI_API_KEY`: Google Gemini API key for audio cue decision
- `GOOGLE_API_KEY`: Alternative Google API key
- `PYTHONPATH`: Python path (set to /app in container)

## Notes

- The first run will download model weights (TangoFlux, CLAP) which may take time
- Model cache is persisted in a Docker volume
- For development, the code is mounted as a volume for hot-reload
- For production, remove the volume mount in docker-compose.yml
