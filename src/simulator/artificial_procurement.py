from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# Configuration

# The attribute values are chosen so that the decision-mining pipeline discovers human-readable guards that mix interval conditions (amount thresholds) and categorical conditions
# (department, supplier_type, priority, …).  Every XOR split in the process is governed by a deterministic rule (+ light noise) that can be recovered by a surrogate decision tree.
# created with the help of GenAI.

# Resources 
REQUESTERS = [
    ("Alice",   "junior", "IT"),
    ("Bob",     "senior", "IT"),
    ("Carol",   "junior", "Finance"),
    ("David",   "senior", "Finance"),
    ("Eva",     "junior", "Operations"),
    ("Frank",   "senior", "Operations"),
]

APPROVERS = {
    "IT":         ["Manager_IT_1",  "Manager_IT_2"],
    "Finance":    ["Manager_FIN_1", "Manager_FIN_2"],
    "Operations": ["Manager_OPS_1", "Manager_OPS_2"],
}

BUYERS    = ["Buyer_1", "Buyer_2", "Buyer_3"]
RECEIVERS = ["Receiver_A", "Receiver_B"]
INVOICE_CLERKS = ["Clerk_1", "Clerk_2", "Clerk_3"]

SUPPLIERS = [
    ("Supplier_A", "preferred"),
    ("Supplier_B", "preferred"),
    ("Supplier_C", "standard"),
    ("Supplier_D", "risky"),
]

CATEGORIES = ["hardware", "software", "consulting", "office", "maintenance"]

# Static case-level attribute (set once per case, never changes).
# Each case receives exactly one priority.
PRIORITIES = ["low", "medium", "high", "critical"]

# Data structures
@dataclass
class Event:
    case_id: str
    activity: str
    timestamp: datetime
    resource: str
    event_attributes: Dict[str, Any]


# Simulator
class ProcurementSimulator:
    """
    Artificial procurement process designed as a **running example**
    for decision-aware suffix prediction.

    Process structure (Petri-net friendly):
    ───────────────────────────────────────────────────────────────
    Create Purchase Requisition
      │
      ├─► XOR-1  Approve Requisition
      │     ├─ approved  ──────────────────────────────────────►
      │     └─ rejected  ─► Revise Requisition ─► (loop back)
      │
      ├─► Collect Quotations
      ├─► Evaluate Quotations
      ├─► Select Supplier
      ├─► Create Purchase Order
      ├─► Send Purchase Order
      ├─► Receive Goods
      │
      ├─► XOR-2  Check Invoice
      │     ├─ correct    ─► Pay Invoice
      │     └─ incorrect  ─► Request Credit Note
      │                        ├─► XOR-3
      │                        │     ├─ reorder  ─► Reorder Goods
      │                        │     │               └─► Close Case
      │                        │     └─ close    ─► Close Case
      │
      └─► Close Case  (only from Pay Invoice or terminal XOR-3)
    ───────────────────────────────────────────────────────────────

    Decision points and the ground-truth rules that govern them:

    **XOR-1  (Approve Requisition → approved | rejected)**
      approved iff:
        budget_status == "approved"
        AND amount <= manager_limit
        AND NOT (seniority == "junior" AND amount > 9000)

    **XOR-2  (Check Invoice → correct | incorrect)**
      correct iff:
        goods_match == True
        AND invoice_deviation_pct <= clerk_tolerance
        AND supplier_type in {"preferred", "standard"}
        (risky suppliers with deviation > 1 % are always incorrect)

    **XOR-3  (Request Credit Note → reorder | close)**
      reorder iff:
        priority in {"high", "critical"}
        AND category in {"hardware", "maintenance"}
        AND amount > 3000

    Design goals:
      • Each XOR rule uses a **mix of interval and categorical**
        conditions so that mined guards showcase both types.
      • Rules are deterministic up to a small noise flip, so
        CatBoost can learn them with near-perfect accuracy, and
        the surrogate tree extracts clean, readable guards.
      • Dynamic attributes (resource, amount, deviation, …)
        **change across events** within a case (e.g. after
        revision).  Static attributes (department, category,
        priority) are fixed per case.
      • Suffix length varies (revision loop, XOR-3 branch)
        giving the LSTM a non-trivial prediction problem.
      • The CSV output uses XES standard column names
        (case:concept:name, concept:name, time:timestamp,
        org:resource) so the existing data-loader / Petri-net-
        replay / decision-mining pipeline can consume it
        directly.
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

    # Internal: approval-limit look-up (kept module-level for
    # readability but accessed through the instance).
    _APPROVAL_LIMITS = {
        "Manager_IT_1":  5_000,
        "Manager_IT_2":  12_000,
        "Manager_FIN_1": 7_000,
        "Manager_FIN_2": 15_000,
        "Manager_OPS_1": 6_000,
        "Manager_OPS_2": 14_000,
    }

    _CLERK_TOLERANCE = {
        "Clerk_1": 0.020,
        "Clerk_2": 0.035,
        "Clerk_3": 0.050,
    }

    # Public API
    def simulate_case(self, case_id: str, start_time: datetime) -> List[Event]:
        requester, seniority, department = self.rng.choice(REQUESTERS)
        priority = self._sample_priority(department)

        case: Dict[str, Any] = {
            # static (case-level)
            "department":          department,
            "category":            self.rng.choice(CATEGORIES),
            "priority":            priority,
            # dynamic (change during execution)
            "requester":           requester,
            "requester_seniority": seniority,
            "amount":              round(self._sample_amount(department), 2),
            "budget_status":       self._sample_budget_status(department),
            "revision_count":      0,
            "supplier":            None,
            "supplier_type":       None,
            "goods_match":         None,
            "invoice_deviation_pct": None,
        }

        events: List[Event] = []
        t = start_time

        def emit(activity: str, resource: str) -> None:
            nonlocal t
            attrs = {
                "case:concept:name":    case_id,
                "concept:name":         activity,
                "time:timestamp":       t,
                "org:resource":         resource,
                # dynamic
                "requester_seniority":  case["requester_seniority"],
                "amount":               case["amount"],
                "budget_status":        case["budget_status"],
                "revision_count":       case["revision_count"],
                "supplier_type":        case["supplier_type"],
                "goods_match":          case["goods_match"],
                "invoice_deviation_pct": case["invoice_deviation_pct"],
                # static
                "department":           case["department"],
                "category":             case["category"],
                "priority":             case["priority"],
            }
            events.append(Event(case_id=case_id, activity=activity,
                                timestamp=t, resource=resource,
                                event_attributes=attrs))
            t += timedelta(hours=self.rng.randint(4, 36))

        # main flow

        emit("Create Purchase Requisition", requester)

        # XOR-1: approval loop
        approved = False
        while not approved:
            approver = self._select_approver(department)

            # After max_revisions forced revisions, deterministically approve
            if case["revision_count"] > self.max_revisions:
                emit("Approve Requisition", approver)
                approved = True
                continue

            xor1 = self._decide_approval(case, approver)

            if xor1:
                emit("Approve Requisition", approver)
                approved = True
            else:
                emit("Reject Requisition", approver)
                if case["revision_count"] >= self.max_revisions:
                    self._revise_case(case, forced=True)
                else:
                    self._revise_case(case, forced=False)
                emit("Revise Requisition", requester)

        # sourcing
        buyer = self.rng.choice(BUYERS)
        emit("Collect Quotations", buyer)
        emit("Evaluate Quotations", buyer)

        supplier, supplier_type = self._select_supplier(case)
        case["supplier"] = supplier
        case["supplier_type"] = supplier_type
        emit("Select Supplier", buyer)

        emit("Create Purchase Order", buyer)
        emit("Send Purchase Order", buyer)

        # receiving
        receiver = self.rng.choice(RECEIVERS)
        case["goods_match"] = self._sample_goods_match(case)
        emit("Receive Goods", receiver)

        # XOR-2: invoice check
        clerk = self.rng.choice(INVOICE_CLERKS)
        case["invoice_deviation_pct"] = round(
            self._sample_invoice_deviation(case), 4)

        xor2 = self._decide_invoice_correct(case, clerk)

        if xor2:
            emit("Approve Invoice", clerk)
            emit("Pay Invoice", clerk)
            emit("Close Case", clerk)
        else:
            emit("Flag Invoice Mismatch", clerk)
            emit("Request Credit Note", clerk)

            # XOR-3: reorder or close
            xor3 = self._decide_reorder(case)
            if xor3:
                emit("Reorder Goods", buyer)
            emit("Close Case", clerk)

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
        csv_path: str,
        start_time: Optional[datetime] = None,
    ) -> List[List[Event]]:
        traces = self.simulate_log(n_cases=n_cases, start_time=start_time)
        self.write_event_csv(traces, csv_path)
        return traces

    # Decision logic (deterministic rules + noise)
    
    def _decide_approval(self, case: Dict[str, Any], approver: str) -> bool:
        """
        XOR-1  ground-truth rule:
          approved iff budget_status == "approved"
                   AND amount <= approval_limit[approver]
                   AND NOT (seniority == "junior" AND amount > 9000)
        """
        amount   = float(case["amount"])
        budget   = str(case["budget_status"])
        senior   = str(case["requester_seniority"])
        limit    = self._APPROVAL_LIMITS[approver]

        approved = (
            budget == "approved"
            and amount <= limit
            and not (senior == "junior" and amount > 9000)
        )
        return self._flip(approved)

    def _decide_invoice_correct(self, case: Dict[str, Any], clerk: str) -> bool:
        """
        XOR-2  ground-truth rule:
          correct iff goods_match == True
                  AND invoice_deviation_pct <= clerk_tolerance
                  AND supplier_type in {"preferred", "standard"}
          (risky + deviation > 1 %  → always incorrect)
        """
        supplier_type = str(case["supplier_type"])
        goods_match   = bool(case["goods_match"])
        deviation     = float(case["invoice_deviation_pct"])
        tolerance     = self._CLERK_TOLERANCE[clerk]

        correct = (
            goods_match
            and deviation <= tolerance
            and supplier_type in {"preferred", "standard"}
        )
        if supplier_type == "risky" and deviation > 0.010:
            correct = False

        return self._flip(correct)

    def _decide_reorder(self, case: Dict[str, Any]) -> bool:
        """
        XOR-3  ground-truth rule:
          reorder iff priority in {"high", "critical"}
                  AND category in {"hardware", "maintenance"}
                  AND amount > 3000
        """
        reorder = (
            case["priority"] in {"high", "critical"}
            and case["category"] in {"hardware", "maintenance"}
            and float(case["amount"]) > 3000
        )
        return self._flip(reorder)

    # Sampling helpers
    
    def _flip(self, decision: bool) -> bool:
        """Apply noise: flip the decision with noise_probability."""
        if self.rng.random() < self.noise_probability:
            return not decision
        return decision

    def _sample_amount(self, department: str) -> float:
        mu = {"IT": 8_000, "Finance": 6_000, "Operations": 7_000}[department]
        return max(200.0, self.rng.gauss(mu, mu * 0.45))

    def _sample_budget_status(self, department: str) -> str:
        p = {"IT": 0.72, "Finance": 0.82, "Operations": 0.68}[department]
        return "approved" if self.rng.random() < p else "pending"

    def _sample_priority(self, department: str) -> str:
        weights = {
            "IT":         [0.20, 0.40, 0.25, 0.15],
            "Finance":    [0.35, 0.35, 0.20, 0.10],
            "Operations": [0.15, 0.30, 0.30, 0.25],
        }[department]
        return self.rng.choices(PRIORITIES, weights=weights, k=1)[0]

    def _select_approver(self, department: str) -> str:
        return self.rng.choice(APPROVERS[department])

    def _revise_case(self, case: Dict[str, Any], forced: bool = False) -> None:
        case["revision_count"] += 1
        factor = self.rng.uniform(0.45, 0.70) if forced else self.rng.uniform(0.70, 0.90)
        case["amount"] = round(float(case["amount"]) * factor, 2)
        if forced or self.rng.random() < 0.75:
            case["budget_status"] = "approved"

    def _select_supplier(self, case: Dict[str, Any]) -> Tuple[str, str]:
        amount   = float(case["amount"])
        category = str(case["category"])

        if category in {"hardware", "software"} and amount > 8000:
            candidates = [s for s in SUPPLIERS if s[1] != "risky"]
        elif category == "consulting":
            candidates = [s for s in SUPPLIERS if s[0] != "Supplier_A"]
        else:
            candidates = list(SUPPLIERS)
        return self.rng.choice(candidates)

    def _sample_goods_match(self, case: Dict[str, Any]) -> bool:
        p = {"preferred": 0.95, "standard": 0.85, "risky": 0.65}[case["supplier_type"]]
        return self.rng.random() < p

    def _sample_invoice_deviation(self, case: Dict[str, Any]) -> float:
        st = case["supplier_type"]
        if st == "preferred":
            return max(0.0, self.rng.gauss(0.010, 0.008))
        if st == "standard":
            return max(0.0, self.rng.gauss(0.025, 0.015))
        return max(0.0, self.rng.gauss(0.060, 0.025))

    # CSV event log writer

    def write_event_csv(self, traces: List[List[Event]], output_path: str) -> None:
        rows: List[Dict[str, Any]] = []

        for trace in traces:
            for event in trace:
                row = {
                    "case:concept:name": event.case_id,
                    "concept:name":      event.activity,
                    "time:timestamp":    self._format_datetime(event.timestamp),
                    "org:resource":      event.resource,
                }
                for key, value in event.event_attributes.items():
                    if key in row:
                        continue
                    if isinstance(value, datetime):
                        row[key] = self._format_datetime(value)
                    else:
                        row[key] = value
                rows.append(row)

        if not rows:
            return

        preferred_columns = [
            "case:concept:name",
            "concept:name",
            "time:timestamp",
            "org:resource",
            "requester_seniority",
            "department",
            "category",
            "priority",
            "amount",
            "budget_status",
            "revision_count",
            "supplier_type",
            "goods_match",
            "invoice_deviation_pct",
        ]

        all_columns = set()
        for row in rows:
            all_columns.update(row.keys())

        remaining = sorted(c for c in all_columns if c not in preferred_columns)
        fieldnames = [c for c in preferred_columns if c in all_columns] + remaining

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _format_datetime(self, dt: datetime) -> str:
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')


# Example usage
if __name__ == "__main__":
    simulator = ProcurementSimulator(
        seed=7,
        max_revisions=3,
        noise_probability=0.02,
    )

    traces = simulator.run_and_save(
        n_cases=10_000,
        csv_path="procurement_event_log.csv",
        start_time=datetime(2025, 1, 1, 8, 0, 0),
    )

    total_events = sum(len(tr) for tr in traces)
    print(f"Generated {len(traces)} traces  ({total_events} events).")
    print("Saved event-log CSV to: procurement_event_log.csv")