"""GAN Simulator Agent — Interface to Layer 1 TimeGAN generator."""

from typing import Any, Dict, Optional
import requests


class GANSimulatorAgent:
    """Calls the Layer 1 API to generate synthetic traffic scenarios."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url

    def generate_scenario(
        self, intersection_id: str, n_samples: int = 100, scenario: Optional[str] = None
    ) -> Dict[str, Any]:
        response = requests.post(
            f"{self.api_url}/generate",
            json={"intersection_id": intersection_id, "n_samples": n_samples, "scenario": scenario},
        )
        response.raise_for_status()
        return response.json()
