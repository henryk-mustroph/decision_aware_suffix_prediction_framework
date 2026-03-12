from __future__ import annotations

import csv
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Configuration
# ============================================================

REQUESTERS = [
    ("Alice", "junior", "IT"),
    ("Bob", "senior", "IT"),
    ("Carol", "junior", "Finance"),
    ("David", "senior", "Finance"),
    ("Eva", "junior", "Operations"),
    ("Frank", "senior", "Operations"),
]

APPROVERS = {
    "IT": ["Manager_IT_1", "Manager_IT_2"],
    "Finance": ["Manager_FIN_1", "Manager_FIN_2"],
    "Operations": ["Manager_OPS_1", "Manager_OPS_2"],
}

APPROVAL_LIMITS = {
    "Manager_IT_1": 5000,
    "Manager_IT_2": 12000,
    "Manager_FIN_1": 7000,
    "Manager_FIN_2": 15000,
    "Manager_OPS_1": 6000,
    "Manager_OPS_2": 14000,
}

BUYERS = ["Buyer_1", "Buyer_2", "Buyer_3"]
RECEIVERS = ["Receiver_A", "Receiver_B"]
INVOICE_CLERKS = ["Clerk_1", "Clerk_2", "Clerk_3"]

SUPPLIERS = [
    ("Supplier_A", "preferred"),
    ("Supplier_B", "preferred"),
    ("Supplier_C", "standard"),
    ("Supplier_D", "risky"),
]

CATEGORIES = ["hardware", "software", "consulting", "office", "maintenance"]


# ============================================================
# Data structures
# ============================================================

@dataclass
class Event:
    case_id: str
    activity: str
    timestamp: datetime
    resource: str
    event_attributes: Dict[str, Any]


# ============================================================
# Simulator
# ============================================================

class ProcurementXESSimulator:
    """
    Procurement process with:
      - XOR 1: Approve requisition?  -> approved / rejected
      - Loop: rejected -> revise -> approve again
      - XOR 2: Invoice correct?      -> correct / incorrect

    The simulator exports:
      1. XES event log
      2. CSV event log (one row per event)
    """

    def __init__(
        self,
        seed: int = 42,
        max_revisions: int = 3,
        noise_probability: float = 0.02,
    ) -> None:
        self.rng = random.Random(seed)
        self.max_revisions = max_revisions
        self.noise_probability = noise_probability

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def simulate_case(self, case_id: str, start_time: datetime) -> List[Event]:
        requester, requester_seniority, department = self.rng.choice(REQUESTERS)

        case_payload: Dict[str, Any] = {
            "requester": requester,
            "requester_seniority": requester_seniority,
            "department": department,
            "category": self.rng.choice(CATEGORIES),
            "amount": round(self._sample_amount(department), 2),
            "budget_status": self._sample_budget_status(department),
            "revision_count": 0,
            "supplier": None,
            "supplier_type": None,
            "goods_match": None,
            "invoice_deviation_pct": None,
            "xor1_outcome": None,
            "xor2_outcome": None,
        }

        events: List[Event] = []
        current_time = start_time

        def add_event(activity: str, resource: str, extra_attrs: Optional[Dict[str, Any]] = None) -> None:
            nonlocal current_time

            attrs = {
                "case:concept:name": case_id,
                "concept:name": activity,
                "time:timestamp": current_time,
                "org:resource": resource,
                "requester": case_payload["requester"],
                "requester_seniority": case_payload["requester_seniority"],
                "department": case_payload["department"],
                "category": case_payload["category"],
                "amount": case_payload["amount"],
                "budget_status": case_payload["budget_status"],
                "revision_count": case_payload["revision_count"],
                "supplier": case_payload["supplier"],
                "supplier_type": case_payload["supplier_type"],
                "goods_match": case_payload["goods_match"],
                "invoice_deviation_pct": case_payload["invoice_deviation_pct"],
                "xor1_outcome": case_payload["xor1_outcome"],
                "xor2_outcome": case_payload["xor2_outcome"],
            }

            if extra_attrs:
                attrs.update(extra_attrs)

            events.append(
                Event(
                    case_id=case_id,
                    activity=activity,
                    timestamp=current_time,
                    resource=resource,
                    event_attributes=attrs,
                )
            )

            current_time += timedelta(hours=self.rng.randint(4, 36))

        # ----------------------------------------------------
        # Process execution
        # ----------------------------------------------------

        add_event("Create Purchase Requisition", requester)

        approved = False
        while not approved:
            approver = self._select_approver(case_payload["department"])
            xor1 = self._decide_approval(case_payload, approver)
            case_payload["xor1_outcome"] = "approved" if xor1 else "rejected"

            add_event(
                "Approve Requisition",
                approver,
                extra_attrs={
                    "decision_point": "xor_approval",
                    "taken_branch": case_payload["xor1_outcome"],
                },
            )

            if xor1:
                approved = True
            else:
                if case_payload["revision_count"] >= self.max_revisions:
                    self._revise_case(case_payload, forced=True)
                else:
                    self._revise_case(case_payload, forced=False)

                add_event(
                    "Revise Requisition",
                    requester,
                    extra_attrs={"loop_entry": True},
                )

        buyer = self.rng.choice(BUYERS)
        add_event("Collect Quotations", buyer)
        add_event("Evaluate Quotations", buyer)

        supplier, supplier_type = self._select_supplier(case_payload)
        case_payload["supplier"] = supplier
        case_payload["supplier_type"] = supplier_type
        add_event("Select Supplier", buyer)

        add_event("Create Purchase Order", buyer)
        add_event("Send Purchase Order", buyer)

        receiver = self.rng.choice(RECEIVERS)
        case_payload["goods_match"] = self._sample_goods_match(case_payload)
        add_event("Receive Goods", receiver)

        clerk = self.rng.choice(INVOICE_CLERKS)
        case_payload["invoice_deviation_pct"] = round(
            self._sample_invoice_deviation(case_payload), 4
        )

        xor2 = self._decide_invoice_correct(case_payload, clerk)
        case_payload["xor2_outcome"] = "correct" if xor2 else "incorrect"

        add_event(
            "Check Invoice",
            clerk,
            extra_attrs={
                "decision_point": "xor_invoice",
                "taken_branch": case_payload["xor2_outcome"],
            },
        )

        if xor2:
            add_event("Pay Invoice", clerk)
        else:
            add_event("Create Credit Note and Close Case", clerk)

        return events

    def simulate_log(
        self,
        n_cases: int = 100,
        start_time: Optional[datetime] = None,
    ) -> List[List[Event]]:
        if start_time is None:
            start_time = datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc)

        all_traces: List[List[Event]] = []
        current_start = start_time

        for i in range(1, n_cases + 1):
            case_id = f"C{i:05d}"
            trace = self.simulate_case(case_id, current_start)
            all_traces.append(trace)
            current_start += timedelta(minutes=self.rng.randint(15, 180))

        return all_traces

    def run_and_save(
        self,
        n_cases: int,
        xes_path: str,
        csv_path: str,
        start_time: Optional[datetime] = None,
    ) -> List[List[Event]]:
        traces = self.simulate_log(n_cases=n_cases, start_time=start_time)
        self.write_xes(traces, xes_path)
        self.write_event_csv(traces, csv_path)
        return traces

    # --------------------------------------------------------
    # CSV event log writer
    # --------------------------------------------------------

    def write_event_csv(self, traces: List[List[Event]], output_path: str) -> None:
        rows: List[Dict[str, Any]] = []

        for trace in traces:
            for event in trace:
                row = {
                    "case:concept:name": event.case_id,
                    "concept:name": event.activity,
                    "time:timestamp": self._format_xes_datetime(event.timestamp),
                    "org:resource": event.resource,
                }

                for key, value in event.event_attributes.items():
                    if key in row:
                        continue
                    if isinstance(value, datetime):
                        row[key] = self._format_xes_datetime(value)
                    else:
                        row[key] = value

                rows.append(row)

        if not rows:
            return

        # Stable and explicit column order
        preferred_columns = [
            "case:concept:name",
            "concept:name",
            "time:timestamp",
            "org:resource",
            "requester",
            "requester_seniority",
            "department",
            "category",
            "amount",
            "budget_status",
            "revision_count",
            "supplier",
            "supplier_type",
            "goods_match",
            "invoice_deviation_pct",
            "xor1_outcome",
            "xor2_outcome",
            "decision_point",
            "taken_branch",
            "loop_entry",
        ]

        all_columns = set()
        for row in rows:
            all_columns.update(row.keys())

        remaining_columns = [c for c in sorted(all_columns) if c not in preferred_columns]
        fieldnames = [c for c in preferred_columns if c in all_columns] + remaining_columns

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # --------------------------------------------------------
    # XES writer
    # --------------------------------------------------------

    def write_xes(self, traces: List[List[Event]], output_path: str) -> None:
        log = ET.Element(
            "log",
            {
                "xes.version": "1.0",
                "xes.features": "nested-attributes",
                "openxes.version": "1.0RC7",
                "xmlns": "http://www.xes-standard.org/",
            },
        )

        ET.SubElement(
            log,
            "extension",
            {"name": "Concept", "prefix": "concept", "uri": "http://www.xes-standard.org/concept.xesext"},
        )
        ET.SubElement(
            log,
            "extension",
            {"name": "Time", "prefix": "time", "uri": "http://www.xes-standard.org/time.xesext"},
        )
        ET.SubElement(
            log,
            "extension",
            {"name": "Organizational", "prefix": "org", "uri": "http://www.xes-standard.org/org.xesext"},
        )

        trace_global = ET.SubElement(log, "global", {"scope": "trace"})
        ET.SubElement(trace_global, "string", {"key": "concept:name", "value": ""})

        event_global = ET.SubElement(log, "global", {"scope": "event"})
        ET.SubElement(event_global, "string", {"key": "concept:name", "value": ""})
        ET.SubElement(event_global, "date", {"key": "time:timestamp", "value": "1970-01-01T00:00:00.000+00:00"})
        ET.SubElement(event_global, "string", {"key": "org:resource", "value": ""})

        ET.SubElement(log, "classifier", {"name": "Activity", "keys": "concept:name"})
        ET.SubElement(log, "classifier", {"name": "Activity Resource", "keys": "concept:name org:resource"})

        for trace_events in traces:
            trace_el = ET.SubElement(log, "trace")
            case_id = trace_events[0].case_id

            ET.SubElement(trace_el, "string", {"key": "concept:name", "value": case_id})
            ET.SubElement(trace_el, "string", {"key": "case:concept:name", "value": case_id})

            for event in trace_events:
                event_el = ET.SubElement(trace_el, "event")

                ET.SubElement(event_el, "string", {"key": "concept:name", "value": event.activity})
                ET.SubElement(
                    event_el,
                    "date",
                    {"key": "time:timestamp", "value": self._format_xes_datetime(event.timestamp)},
                )
                ET.SubElement(event_el, "string", {"key": "org:resource", "value": event.resource})

                for key, value in event.event_attributes.items():
                    if key in {"concept:name", "time:timestamp", "org:resource"}:
                        continue
                    self._append_xes_attribute(event_el, key, value)

        tree = ET.ElementTree(log)
        self._indent_xml(log)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    # --------------------------------------------------------
    # Decision logic
    # --------------------------------------------------------

    def _decide_approval(self, case_payload: Dict[str, Any], approver: str) -> bool:
        amount = float(case_payload["amount"])
        budget_status = str(case_payload["budget_status"])
        requester_seniority = str(case_payload["requester_seniority"])
        limit = APPROVAL_LIMITS[approver]

        approved = (
            budget_status == "approved"
            and amount <= limit
            and not (requester_seniority == "junior" and amount > 9000)
        )

        if self.rng.random() < self.noise_probability:
            approved = not approved

        return approved

    def _decide_invoice_correct(self, case_payload: Dict[str, Any], clerk: str) -> bool:
        supplier_type = str(case_payload["supplier_type"])
        goods_match = bool(case_payload["goods_match"])
        deviation = float(case_payload["invoice_deviation_pct"])

        clerk_tolerance = {
            "Clerk_1": 0.020,
            "Clerk_2": 0.035,
            "Clerk_3": 0.050,
        }[clerk]

        correct = (
            goods_match
            and deviation <= clerk_tolerance
            and supplier_type in {"preferred", "standard"}
        )

        if supplier_type == "risky" and deviation > 0.010:
            correct = False

        if self.rng.random() < self.noise_probability:
            correct = not correct

        return correct

    # --------------------------------------------------------
    # Sampling helpers
    # --------------------------------------------------------

    def _sample_amount(self, department: str) -> float:
        if department == "IT":
            return self.rng.uniform(1000, 18000)
        if department == "Finance":
            return self.rng.uniform(500, 15000)
        return self.rng.uniform(700, 16000)

    def _sample_budget_status(self, department: str) -> str:
        p_approved = {
            "IT": 0.72,
            "Finance": 0.82,
            "Operations": 0.68,
        }[department]
        return "approved" if self.rng.random() < p_approved else "pending"

    def _select_approver(self, department: str) -> str:
        return self.rng.choice(APPROVERS[department])

    def _revise_case(self, case_payload: Dict[str, Any], forced: bool = False) -> None:
        case_payload["revision_count"] += 1

        factor = self.rng.uniform(0.45, 0.70) if forced else self.rng.uniform(0.70, 0.90)
        case_payload["amount"] = round(float(case_payload["amount"]) * factor, 2)

        if forced or self.rng.random() < 0.75:
            case_payload["budget_status"] = "approved"

    def _select_supplier(self, case_payload: Dict[str, Any]) -> Tuple[str, str]:
        amount = float(case_payload["amount"])
        category = str(case_payload["category"])

        if category in {"hardware", "software"} and amount > 8000:
            candidates = [
                ("Supplier_A", "preferred"),
                ("Supplier_B", "preferred"),
                ("Supplier_C", "standard"),
            ]
        elif category == "consulting":
            candidates = [
                ("Supplier_B", "preferred"),
                ("Supplier_C", "standard"),
                ("Supplier_D", "risky"),
            ]
        else:
            candidates = SUPPLIERS

        return self.rng.choice(candidates)

    def _sample_goods_match(self, case_payload: Dict[str, Any]) -> bool:
        supplier_type = str(case_payload["supplier_type"])
        p_match = {
            "preferred": 0.95,
            "standard": 0.85,
            "risky": 0.65,
        }[supplier_type]
        return self.rng.random() < p_match

    def _sample_invoice_deviation(self, case_payload: Dict[str, Any]) -> float:
        supplier_type = str(case_payload["supplier_type"])

        if supplier_type == "preferred":
            return max(0.0, self.rng.gauss(0.010, 0.008))
        if supplier_type == "standard":
            return max(0.0, self.rng.gauss(0.025, 0.015))
        return max(0.0, self.rng.gauss(0.060, 0.025))

    # --------------------------------------------------------
    # XES helpers
    # --------------------------------------------------------

    def _append_xes_attribute(self, parent: ET.Element, key: str, value: Any) -> None:
        if value is None:
            return

        if isinstance(value, bool):
            ET.SubElement(parent, "boolean", {"key": key, "value": "true" if value else "false"})
        elif isinstance(value, int):
            ET.SubElement(parent, "int", {"key": key, "value": str(value)})
        elif isinstance(value, float):
            ET.SubElement(parent, "float", {"key": key, "value": str(value)})
        elif isinstance(value, datetime):
            ET.SubElement(parent, "date", {"key": key, "value": self._format_xes_datetime(value)})
        else:
            ET.SubElement(parent, "string", {"key": key, "value": str(value)})

    def _format_xes_datetime(self, dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat(timespec="milliseconds")

    def _indent_xml(self, elem: ET.Element, level: int = 0) -> None:
        indent = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            for child in elem:
                self._indent_xml(child, level + 1)
            if not elem[-1].tail or not elem[-1].tail.strip():
                elem[-1].tail = indent
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    simulator = ProcurementXESSimulator(
        seed=7,
        max_revisions=3,
        noise_probability=0.02,
    )

    traces = simulator.run_and_save(
        n_cases=10000,
        xes_path="procurement_log.xes",
        csv_path="procurement_event_log.csv",
        start_time=datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc),
    )

    print(f"Generated {len(traces)} traces.")
    print("Saved XES log to: procurement_log.xes")
    print("Saved event-log CSV to: procurement_event_log.csv")