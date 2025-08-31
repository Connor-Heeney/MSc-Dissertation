import numpy as np
import pytest
from descriptive_stats import robust_mad, OUTLIER_THRESHOLD

def test_median_and_mad_basic():
    # Toy dataset with known median and MAD
    data = np.array([0, 0, 1, 1, 2, 2, 100], dtype=float)
    med = np.median(data)
    
    # Compute robust_mad (scaled MAD)
    computed_mad = robust_mad(data)
    
    # Hand‐compute unscaled MAD:
    mad_unscaled = np.median(np.abs(data - med))
    expected_mad = mad_unscaled * 1.482602218505602
    
    assert computed_mad == pytest.approx(expected_mad, rel=1e-6)

def test_modified_z_outlier_flag():
    # Only the last point (100) should be flagged as an outlier
    data = np.array([0, 0, 1, 1, 2, 2, 100], dtype=float)
    med = np.median(data)
    mad = robust_mad(data)
    mz = 0.6745 * (data - med) / mad
    outliers = np.abs(mz) > OUTLIER_THRESHOLD
    
    assert outliers.sum() == 1
    # The index of the outlier should be the last element
    assert np.where(outliers)[0][0] == len(data) - 1

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
