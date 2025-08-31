import numpy as np
import pytest
from descriptive_stats import bowley_skewness, percentile_kurtosis

def test_bowley_zero_skew():
    # perfectly symmetric data => skew = 0
    data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert bowley_skewness(data) == pytest.approx(0.0)

def test_bowley_degenerate():
    # all values identical => denominator zero => nan
    data = np.ones(10)
    assert np.isnan(bowley_skewness(data))

def test_percentile_kurtosis_uniform():
    # uniform on [0,1]: (P90-P10)/(P75-P25) = (0.9-0.1)/(0.75-0.25) = 1.6
    data = np.linspace(0, 1, 1001)
    assert percentile_kurtosis(data) == pytest.approx(1.6, rel=1e-3)

def test_percentile_kurtosis_degenerate():
    # identical values => nan
    data = np.ones(20)
    assert np.isnan(percentile_kurtosis(data))

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])