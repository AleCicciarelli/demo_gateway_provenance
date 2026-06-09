# Client for interacting with the AP Explanation API, which runs an explanation of a SQL query against provided CSV files and returns the result.

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from create_ap_template import build_ap_csv_template


class ExplanationClient:
    def __init__(
        self,
        base_url: str,
        post_endpoint: str,
        timeout: float = 300,
        poll_interval: float = 2,
        max_polls: int = 60,
        token: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.post_endpoint = post_endpoint
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_polls = max_polls
        self.token = token

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        return headers

    def run_explanation(
        self,
        sql_query: str,
        csv_files: List[str],
        delimiter: str = ",",
    ) -> Dict[str, Any]:
        payload = build_ap_csv_template(
            sql_query=sql_query,
            csv_files=csv_files,
            delimiter=delimiter,
        )

        post_url = f"{self.base_url}{self.post_endpoint}"

        response = requests.post(
            post_url,
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )

        if response.status_code not in {200, 201, 202}:
            raise RuntimeError(
                f"Explanation POST failed: {response.status_code} {response.text}"
            )

        task_data = response.json()

        task_id = task_data.get("task_id") or task_data.get("id")

        if not task_id:
            return task_data

        return self.poll_result(task_id)

    def poll_result(self, task_id: str) -> Dict[str, Any]:
        poll_url = f"{self.base_url}/api/v1/aps/explanation/{task_id}"

        for _ in range(self.max_polls):
            response = requests.get(
                poll_url,
                headers=self._headers(),
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Explanation polling failed: {response.status_code} {response.text}"
                )

            data = response.json()
            status = str(data.get("status", "")).lower()

            if status in {"success", "completed", "done"}:
                return data

            if status in {"failure", "failed", "error", "revoked"}:
                raise RuntimeError(f"Explanation task failed: {data}")

            time.sleep(self.poll_interval)

        raise TimeoutError(f"Explanation task did not finish. task_id={task_id}")