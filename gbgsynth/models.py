"""
Data models for synthetic population agents, households, and dwellings.

This module defines the core Agent (individual), Household, and Dwelling classes
with all necessary attributes for demographic and socioeconomic modeling.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import random


@dataclass
class Dwelling:
    """
    Represents a physical dwelling unit (apartment, house, etc.).

    Attributes:
        dwelling_id: Unique identifier
        floor_area: Floor area in square meters
        floor_area_range: Original SCB range (e.g., "61-70")
        house_type: Type of building ('detached_house', 'apartment', 'other')
        house_type_sv: Swedish name ('Småhus', 'Flerbostadshus', 'Övriga hus')
        building_id: Reference to building footprint (for spatial linking)
        household_id: ID of household occupying this dwelling (None if vacant)
        floor_number: Floor within building (0=ground, 1=first floor, etc.)
        centroid_x: X coordinate of building centroid (SWEREF99)
        centroid_y: Y coordinate of building centroid (SWEREF99)
        recommended_occupants: Suggested number of occupants based on floor area
    """

    dwelling_id: int
    floor_area: float
    floor_area_range: str = ""
    house_type: str = "apartment"
    house_type_sv: str = "Flerbostadshus"
    building_id: Optional[str] = None
    household_id: Optional[int] = None
    floor_number: Optional[int] = None
    centroid_x: Optional[float] = None
    centroid_y: Optional[float] = None

    def __post_init__(self):
        """Calculate recommended occupants based on floor area."""
        # Swedish housing norms: ~20m² per person minimum, ~40m² comfortable
        # We use a middle ground
        self._recommended = self._calc_recommended_occupants()

    def _calc_recommended_occupants(self) -> int:
        """Calculate recommended number of occupants."""
        if self.floor_area < 35:
            return 1
        elif self.floor_area < 55:
            return 1  # Could be 1-2
        elif self.floor_area < 75:
            return 2
        elif self.floor_area < 95:
            return 3
        elif self.floor_area < 120:
            return 4
        elif self.floor_area < 150:
            return 5
        else:
            return 6

    @property
    def recommended_occupants(self) -> int:
        """Get recommended number of occupants."""
        return self._recommended

    @property
    def min_occupants(self) -> int:
        """Minimum reasonable occupants."""
        return 1

    @property
    def max_occupants(self) -> int:
        """Maximum reasonable occupants based on floor area."""
        # ~15m² per person as absolute minimum
        return max(1, int(self.floor_area / 15))

    def is_vacant(self) -> bool:
        """Check if dwelling is unoccupied."""
        return self.household_id is None

    @property
    def is_occupied(self) -> bool:
        """Check if dwelling is occupied."""
        return self.household_id is not None

    def can_fit(self, num_people: int) -> bool:
        """
        Check if dwelling can reasonably accommodate a household.
        
        Args:
            num_people: Number of people in household
            
        Returns:
            True if dwelling size is appropriate for household
        """
        return self.min_occupants <= num_people <= self.max_occupants

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        status = "vacant" if self.is_vacant() else f"hh={self.household_id}"
        return f"Dwelling(id={self.dwelling_id}, {self.floor_area:.0f}m², {self.house_type_sv}, {status})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert dwelling to dictionary for export."""
        return {
            'dwelling_id': self.dwelling_id,
            'floor_area': self.floor_area,
            'floor_area_range': self.floor_area_range,
            'house_type': self.house_type,
            'house_type_sv': self.house_type_sv,
            'building_id': self.building_id,
            'household_id': self.household_id,
            'floor_number': self.floor_number,
            'centroid_x': self.centroid_x,
            'centroid_y': self.centroid_y,
            'recommended_occupants': self.recommended_occupants,
            'max_occupants': self.max_occupants,
            'is_vacant': self.is_vacant()
        }


@dataclass
class Agent:
    """
    Represents an individual person in the synthetic population.

    Attributes:
        agent_id: Unique identifier for the agent
        age: Age in years
        sex: Biological sex ('male' or 'female')
        hh_role: Household role ('single', 'cohabiting', 'child')
        status: Employment/life status (e.g., 'employed', 'student', 'retired')
        income: Annual income in SEK (can be None for children/students)
        income_decile: Income decile (1-10) for regional comparison
        education: Educational attainment level
        household_id: Reference to the household this agent belongs to
    """

    agent_id: int
    age: int
    sex: str
    hh_role: str = "single"
    status: Optional[str] = None
    income: Optional[float] = None
    income_decile: Optional[int] = None
    education: Optional[str] = None
    household_id: Optional[int] = None

    def __post_init__(self):
        """Validate agent attributes after initialization."""
        if self.sex not in ['male', 'female']:
            raise ValueError(f"Invalid sex: {self.sex}")
        
        if self.age < 0 or self.age > 120:
            raise ValueError(f"Invalid age: {self.age}")

        # Auto-assign some defaults based on age
        if self.age < 18:
            self.hh_role = "child"
            self.status = "child" if self.age < 16 else "student"
            self.income = 0
            self.income_decile = None
        elif self.age >= 65 and self.status is None:
            self.status = "retired"

    def is_adult(self) -> bool:
        """Check if agent is an adult (18+)."""
        return self.age >= 18

    def is_child(self) -> bool:
        """Check if agent is a child (<18)."""
        return self.age < 18

    def can_be_parent(self) -> bool:
        """Check if agent is old enough to be a parent."""
        return self.age >= 18

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        hh = f"hh={self.household_id}" if self.household_id else "unassigned"
        return f"Agent(id={self.agent_id}, {self.age}y {self.sex[0].upper()}, {self.hh_role}, {hh})"

    def can_be_partner_with(self, other: 'Agent', max_age_diff: int = 10) -> bool:
        """
        Check if this agent can be partnered with another agent.

        Args:
            other: Another agent to check compatibility with
            max_age_diff: Maximum age difference allowed

        Returns:
            True if partnership constraints are satisfied
        """
        # Both must be adults
        if not (self.is_adult() and other.is_adult()):
            return False

        # Age difference constraint
        if abs(self.age - other.age) > max_age_diff:
            return False

        # Different sex (can be modified for same-sex couples if data available)
        if self.sex == other.sex:
            return False

        return True

    def can_be_parent_of(self, child: 'Agent', min_age_gap: int = 18) -> bool:
        """
        Check if this agent can be the biological parent of a child.

        Args:
            child: Potential child agent
            min_age_gap: Minimum age gap for biological plausibility

        Returns:
            True if parent-child relationship is plausible
        """
        if not self.can_be_parent():
            return False

        if not child.is_child():
            return False

        # Biological age constraint
        if self.age - child.age < min_age_gap:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for export."""
        return {
            'agent_id': self.agent_id,
            'age': self.age,
            'sex': self.sex,
            'hh_role': self.hh_role,
            'status': self.status,
            'income': self.income,
            'income_decile': self.income_decile,
            'education': self.education,
            'household_id': self.household_id
        }


@dataclass
class Household:
    """
    Represents a household containing one or more agents.

    Attributes:
        household_id: Unique identifier
        size: Number of people in household
        house_type: Type of dwelling ('detached_house', 'apartment', 'special_housing')
        cars: Number of cars owned
        members: List of Agent objects in this household
        head_id: Agent ID of household head (primary adult)
        partner_id: Agent ID of partner (if couple household)
        child_ids: List of Agent IDs for children
        assigned_hustyp: Swedish house type category ('Småhus', 'Flerbostadshus', 'Specialbostad')
        building_id: Reference to building footprint (for spatial linking)
        dwelling_id: Reference to specific dwelling unit
        dwelling: Reference to Dwelling object (if assigned)
        floor_area: Floor area of assigned dwelling in m²
    """

    household_id: int
    size: int
    house_type: Optional[str] = None
    cars: int = 0
    members: List[Agent] = field(default_factory=list)
    head_id: Optional[int] = None
    partner_id: Optional[int] = None
    child_ids: List[int] = field(default_factory=list)
    assigned_hustyp: Optional[str] = None
    building_id: Optional[str] = None
    dwelling_id: Optional[int] = None
    dwelling: Optional[Dwelling] = field(default=None, repr=False)
    floor_area: Optional[float] = None

    def __post_init__(self):
        """Validate household attributes."""
        if self.size < 1:
            raise ValueError(f"Household size must be >= 1, got {self.size}")

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        members_str = f"{len(self.members)}/{self.size}"
        hustyp = self.assigned_hustyp or self.house_type or "unknown"
        return f"Household(id={self.household_id}, {members_str} members, {hustyp})"

    def add_member(self, agent: Agent) -> None:
        """
        Add an agent to this household.

        Args:
            agent: Agent to add
        """
        if len(self.members) >= self.size:
            raise ValueError(f"Household {self.household_id} is already full (size={self.size})")

        agent.household_id = self.household_id
        self.members.append(agent)

        # Auto-assign roles
        if agent.is_child():
            self.child_ids.append(agent.agent_id)
        elif self.head_id is None:
            self.head_id = agent.agent_id
        elif self.partner_id is None and len(self.members) > 1:
            self.partner_id = agent.agent_id

    def is_full(self) -> bool:
        """Check if household has reached its target size."""
        return len(self.members) == self.size

    @property
    def is_empty(self) -> bool:
        """Check if household has no members."""
        return len(self.members) == 0

    def can_fit(self, count: int = 1) -> bool:
        """Check if household can accommodate additional members."""
        return len(self.members) + count <= self.size

    @property
    def head(self) -> Optional[Agent]:
        """Get the household head agent."""
        if self.head_id:
            for member in self.members:
                if member.agent_id == self.head_id:
                    return member
        return None

    @property
    def partner(self) -> Optional[Agent]:
        """Get the partner agent."""
        if self.partner_id:
            for member in self.members:
                if member.agent_id == self.partner_id:
                    return member
        return None

    @property
    def children(self) -> List[Agent]:
        """Get all child agents."""
        return [m for m in self.members if m.agent_id in self.child_ids]

    @property
    def num_children(self) -> int:
        """Count of children (under 18) in household."""
        return sum(1 for m in self.members if m.age < 18)

    @property
    def num_adults(self) -> int:
        """Count of adults (18+) in household."""
        return sum(1 for m in self.members if m.age >= 18)

    def is_couple(self) -> bool:
        """Check if this is a couple household."""
        return self.head_id is not None and self.partner_id is not None

    def is_single_parent(self) -> bool:
        """Check if this is a single-parent household."""
        return len(self.child_ids) > 0 and not self.is_couple()

    def is_single(self) -> bool:
        """Check if this is a single-person household."""
        return self.size == 1

    @property
    def income(self) -> float:
        """Total household income."""
        return sum(m.income or 0 for m in self.members)

    def assign_dwelling(self, dwelling: Dwelling) -> None:
        """
        Assign this household to a dwelling.
        
        Args:
            dwelling: Dwelling to assign
        """
        self.dwelling = dwelling
        self.dwelling_id = dwelling.dwelling_id
        self.floor_area = dwelling.floor_area
        self.building_id = dwelling.building_id
        # Only set hustyp from dwelling if not already assigned by synthesizer
        # (Census assigns Specialbostad/Uppgift saknas which dwelling table doesn't have)
        if self.assigned_hustyp is None:
            self.assigned_hustyp = dwelling.house_type_sv
            self.house_type = dwelling.house_type
        # But always update house_type if dwelling has different physical type
        elif dwelling.house_type_sv in ('Småhus', 'Flerbostadshus'):
            # Keep census category for validation but note actual building type
            self.house_type = dwelling.house_type
        
        # Update dwelling's occupant reference
        dwelling.household_id = self.household_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert household to dictionary for export."""
        return {
            'household_id': self.household_id,
            'size': self.size,
            'actual_size': len(self.members),
            'house_type': self.house_type,
            'assigned_hustyp': self.assigned_hustyp,
            'building_id': self.building_id,
            'dwelling_id': self.dwelling_id,
            'floor_area': self.floor_area,
            'cars': self.cars,
            'head_id': self.head_id,
            'partner_id': self.partner_id,
            'child_ids': self.child_ids,
            'num_children': self.num_children,
            'num_adults': self.num_adults,
            'total_income': self.income,
            'is_couple': self.is_couple(),
            'is_single_parent': self.is_single_parent()
        }
