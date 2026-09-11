# Client for interacting with the AP Explanation API, which runs an explanation of a SQL query against provided CSV files and returns the result.

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from create_ap_template import build_ap_csv_template


logger = logging.getLogger(__name__)


class ExplanationClient:
    def __init__(
        self,
        base_url: str,
        post_endpoint: str,
        timeout: float = 300,
        poll_interval: float = 2,
        max_polls: Optional[int] = None,
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

        started_at = time.monotonic()
        deadline = started_at + self.timeout
        polls = 0
        last_status = "not polled"
        logger.info("Explanation polling started: task_id=%s timeout_seconds=%s", task_id, self.timeout)

        while self.max_polls is None or polls < self.max_polls:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            response = requests.get(
                poll_url,
                headers=self._headers(),
                timeout=remaining,
            )
            polls += 1

            if response.status_code != 200:
                raise RuntimeError(
                    f"Explanation polling failed: {response.status_code} {response.text}"
                )

            data = response.json()
            status = str(data.get("status", "")).strip().lower()
            if status != last_status:
                logger.info("Explanation task status: task_id=%s status=%s poll=%s", task_id, status, polls)
            last_status = status

            if status in {"success", "completed", "done"}:
                return data

            if status in {"failure", "failed", "error", "revoked"}:
                raise RuntimeError(f"Explanation task failed: {data}")

            if self.max_polls is not None and polls >= self.max_polls:
                break
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(self.poll_interval, remaining))

        elapsed = time.monotonic() - started_at
        message = (
            f"Explanation polling timed out after {elapsed:.1f}s and {polls} polls. "
            f"task_id={task_id}; last_status={last_status}. "
            "The service task may still complete; polling timeout does not cancel it."
        )
        logger.warning(message)
        raise TimeoutError(message)
