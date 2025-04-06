from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class Span:

    trace_id: str
    span_id: str
    parent_span_id: str  # 如果没有，则为None
    children_span_list: List['Span']

    start_time: datetime
    duration: float  # 单位为毫秒

    service_name: str
    anomaly: int = 0  # normal:0/anomaly:1
    status_code: str = None

    operation_name: str = None
    latency: int = None  # normal:None/anomaly:1
    structure: int = None  # normal:None/anomaly:1
    extra: dict = None


@dataclass
class Trace:

    trace_id: str
    root_span: Span
    span_count: int = 0
    anomaly_type: int = None  # normal:0/only_latency_anomaly:1/only_structural_anomaly:2/both_anomaly:3
    source: str = None
