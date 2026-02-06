"""
Tests for Iterative Proportional Fitting (IPF) module.
"""

import pytest
import numpy as np
import pandas as pd
from gbgsynth.ipf import IPFSynthesizer


class TestIPFSynthesizer:
    """Tests for the IPFSynthesizer class."""

    @pytest.fixture
    def ipf(self):
        """Create an IPF synthesizer instance."""
        return IPFSynthesizer(max_iterations=100, convergence_threshold=1e-6)

    def test_initialization(self, ipf):
        """Test IPF synthesizer initialization."""
        assert ipf.max_iterations == 100
        assert ipf.convergence_threshold == 1e-6
        assert ipf.fitted_weights is None

    def test_fit_2d_simple(self, ipf):
        """Test 2D IPF with simple marginals."""
        # Simple 2x2 case
        row_marginal = pd.Series([60, 40], index=['A', 'B'])
        col_marginal = pd.Series([50, 50], index=['X', 'Y'])
        
        result = ipf.fit_2d(row_marginal, col_marginal)
        
        # Check result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check marginals match (within tolerance)
        assert np.allclose(result.sum(axis=1).values, row_marginal.values, rtol=0.01)
        assert np.allclose(result.sum(axis=0).values, col_marginal.values, rtol=0.01)

    def test_fit_multidimensional(self, ipf):
        """Test multi-dimensional IPF."""
        marginals = {
            'age': pd.Series([100, 200, 100], index=['young', 'middle', 'old']),
            'sex': pd.Series([200, 200], index=['male', 'female']),
        }
        
        weights = ipf.fit(marginals)
        
        # Check shape
        assert weights.shape == (3, 2)
        
        # Check row marginals (age)
        assert np.allclose(weights.sum(axis=1), marginals['age'].values, rtol=0.01)
        
        # Check column marginals (sex)
        assert np.allclose(weights.sum(axis=0), marginals['sex'].values, rtol=0.01)

    def test_fit_with_seed(self, ipf):
        """Test IPF with custom seed matrix."""
        row_marginal = pd.Series([60, 40], index=['A', 'B'])
        col_marginal = pd.Series([50, 50], index=['X', 'Y'])
        
        # Non-uniform seed
        seed = np.array([[2, 1], [1, 2]], dtype=float)
        
        result = ipf.fit_2d(row_marginal, col_marginal, seed=seed)
        
        # Should still match marginals
        assert np.allclose(result.sum(axis=1).values, row_marginal.values, rtol=0.01)
        assert np.allclose(result.sum(axis=0).values, col_marginal.values, rtol=0.01)

    def test_convergence(self, ipf):
        """Test that IPF converges."""
        marginals = {
            'dim1': pd.Series([50, 50], index=['a', 'b']),
            'dim2': pd.Series([50, 50], index=['x', 'y']),
        }
        
        ipf.fit(marginals)
        
        # Should have convergence history
        assert len(ipf.convergence_history) > 0
        
        # Should converge (last change should be small)
        assert ipf.convergence_history[-1] < ipf.convergence_threshold

    def test_dimension_labels_stored(self, ipf):
        """Test that dimension labels are stored after fitting."""
        marginals = {
            'age': pd.Series([100, 200], index=['young', 'old']),
            'sex': pd.Series([150, 150], index=['male', 'female']),
        }
        
        ipf.fit(marginals)
        
        assert 'age' in ipf.dimension_labels
        assert 'sex' in ipf.dimension_labels
        assert ipf.dimension_labels['age'] == ['young', 'old']
        assert ipf.dimension_labels['sex'] == ['male', 'female']

    def test_unbalanced_marginals(self, ipf):
        """Test IPF with unbalanced marginals (different totals)."""
        # Marginals with different totals - IPF should normalize
        row_marginal = pd.Series([30, 20], index=['A', 'B'])  # Total: 50
        col_marginal = pd.Series([100, 100], index=['X', 'Y'])  # Total: 200
        
        result = ipf.fit_2d(row_marginal, col_marginal)
        
        # Result should match row marginal total (first provided)
        assert np.isclose(result.values.sum(), row_marginal.sum(), rtol=0.01)

    def test_empty_cells_handling(self, ipf):
        """Test that IPF handles near-zero cells gracefully."""
        # Seed with some very small values
        marginals = {
            'dim1': pd.Series([90, 10], index=['a', 'b']),
            'dim2': pd.Series([50, 50], index=['x', 'y']),
        }
        
        # Should not raise errors
        weights = ipf.fit(marginals)
        assert weights is not None
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))


class TestIPFSamplingIntegration:
    """Integration tests for IPF with sampling."""

    def test_sample_from_fitted_distribution(self):
        """Test that we can sample from IPF-fitted distribution."""
        ipf = IPFSynthesizer()
        
        marginals = {
            'age': pd.Series([100, 300, 100], index=['0-17', '18-64', '65+']),
            'sex': pd.Series([250, 250], index=['male', 'female']),
        }
        
        weights = ipf.fit(marginals)
        
        # Flatten and normalize to create probability distribution
        probs = weights.flatten() / weights.sum()
        
        # Should be valid probability distribution
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
        
        # Sample from distribution
        n_samples = 1000
        samples = np.random.choice(len(probs), size=n_samples, p=probs)
        
        assert len(samples) == n_samples
