"""Model client for serving and querying LLMs via vLLM."""

import asyncio
import subprocess
import time
from dataclasses import dataclass

import httpx


@dataclass
class ModelConfig:
    """Configuration for model serving."""

    model_path: str
    port: int = 8000
    max_tokens: int = 512
    temperature: float = 0.7
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


class VLLMClient:
    """Client for interacting with a vLLM server."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        """Initialize the vLLM client.

        Args:
            base_url: Base URL of the vLLM server (e.g., http://localhost:8000)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def health_check(self) -> bool:
        """Check if the vLLM server is healthy."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion from the model.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["text"]

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate a chat completion.

        Args:
            messages: List of chat messages with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated assistant response
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class ModelServer:
    """Manages a vLLM server process."""

    def __init__(self, config: ModelConfig):
        """Initialize the model server manager.

        Args:
            config: Model serving configuration
        """
        self.config = config
        self._process: subprocess.Popen | None = None
        self._client: VLLMClient | None = None

    @property
    def base_url(self) -> str:
        """Get the server base URL."""
        return f"http://localhost:{self.config.port}"

    def start(self, wait_timeout: float = 300.0) -> VLLMClient:
        """Start the vLLM server.

        Args:
            wait_timeout: Maximum time to wait for server startup

        Returns:
            VLLMClient connected to the server

        Raises:
            TimeoutError: If server doesn't start within timeout
            RuntimeError: If server fails to start
            FileNotFoundError: If vllm is not installed
        """
        # Check if vllm is available
        import shutil

        if shutil.which("vllm") is None:
            raise FileNotFoundError(
                "vllm command not found. Install it with: uv add vllm\n"
                "Alternatively, start vLLM manually and use --vllm-url option."
            )

        cmd = [
            "vllm",
            "serve",
            self.config.model_path,
            "--port",
            str(self.config.port),
            "--tensor-parallel-size",
            str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        self._client = VLLMClient(self.base_url)
        start_time = time.time()

        while time.time() - start_time < wait_timeout:
            if self._process.poll() is not None:
                # Process exited
                _, stderr = self._process.communicate()
                raise RuntimeError(f"vLLM server failed to start: {stderr.decode()}")

            # Check health synchronously
            if asyncio.run(self._client.health_check()):
                return self._client

            time.sleep(2)

        raise TimeoutError(f"vLLM server did not start within {wait_timeout}s")

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self._client:
            asyncio.run(self._client.close())
            self._client = None

        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def __enter__(self) -> VLLMClient:
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def create_model_callback(
    base_url: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
):
    """Create a DeepTeam-compatible model callback.

    DeepTeam expects a sync or async callback. We use sync with requests
    to avoid async complexities.

    Args:
        base_url: Base URL of vLLM server (e.g., http://localhost:8000)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Callback function for DeepTeam
    """
    import requests

    def model_callback(input_text: str) -> str:
        """DeepTeam model callback (sync)."""
        # Use chat completions endpoint for better compatibility
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": input_text}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    return model_callback
