"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from gbgsynth.models import Agent, Household
from gbgsynth.config import Config


@pytest.fixture
def sample_population_data():
    """Create sample population data for testing."""
    data = []
    age_groups = ['0-5', '6-17', '18-29', '30-44', '45-64', '65-79', '80+']
    sexes = ['male', 'female']
    hh_roles = ['child', 'child', 'single', 'cohabiting', 'cohabiting', 'single', 'single']
    
    for i, age_group in enumerate(age_groups):
        for sex in sexes:
            data.append({
                'age_group': age_group,
                'sex': sex,
                'hh_role': hh_roles[i],
                'count': np.random.randint(50, 200)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_household_data():
    """Create sample household data for testing."""
    return pd.DataFrame({
        'hh_size': ['1 person', '2 persons', '3 persons', '4 persons', '5+ persons'],
        'count': [150, 200, 100, 80, 30]
    })


@pytest.fixture
def sample_income_data():
    """Create sample income data for testing."""
    return pd.DataFrame({
        'income_decile': list(range(1, 11)),
        'min_income': [0, 150000, 220000, 280000, 340000, 400000, 470000, 550000, 650000, 800000],
        'max_income': [150000, 220000, 280000, 340000, 400000, 470000, 550000, 650000, 800000, 2000000],
        'count': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    })


@pytest.fixture
def sample_agent():
    """Create a sample agent for testing."""
    return Agent(
        agent_id=1,
        age=35,
        sex='male',
        hh_role='cohabiting',
        status='employed',
        income=550000,
        income_decile=7
    )


@pytest.fixture
def sample_household():
    """Create a sample household for testing."""
    hh = Household(
        household_id=1,
        size=3,
        house_type='apartment',
        cars=1
    )
    
    # Add members
    father = Agent(agent_id=1, age=40, sex='male', income=600000)
    mother = Agent(agent_id=2, age=38, sex='female', income=450000)
    child = Agent(agent_id=3, age=10, sex='male')
    
    hh.add_member(father)
    hh.add_member(mother)
    hh.add_member(child)
    
    return hh


@pytest.fixture
def config():
    """Create a Config instance for testing."""
    return Config()


@pytest.fixture
def couple_household():
    """Create a couple household without children."""
    hh = Household(household_id=1, size=2)
    
    husband = Agent(agent_id=1, age=45, sex='male', hh_role='cohabiting', income=700000)
    wife = Agent(agent_id=2, age=43, sex='female', hh_role='cohabiting', income=550000)
    
    hh.add_member(husband)
    hh.add_member(wife)
    
    return hh


@pytest.fixture
def single_parent_household():
    """Create a single parent household."""
    hh = Household(household_id=1, size=3)
    
    parent = Agent(agent_id=1, age=38, sex='female', hh_role='single', income=480000)
    child1 = Agent(agent_id=2, age=12, sex='male')
    child2 = Agent(agent_id=3, age=8, sex='female')
    
    hh.add_member(parent)
    hh.add_member(child1)
    hh.add_member(child2)
    
    return hh


@pytest.fixture
def single_person_household():
    """Create a single person household."""
    hh = Household(household_id=1, size=1)
    agent = Agent(agent_id=1, age=55, sex='male', hh_role='single', income=520000)
    hh.add_member(agent)
    return hh
