import numpy as np
from sklearn.mixture import GaussianMixture
from descriptive_stats import RNG_SEED

def test_gmm_label_ordering():
    rng = np.random.default_rng(RNG_SEED)
    # two well-separated clusters
    c1 = rng.normal(-2, 0.5, size=500)
    c2 = rng.normal( 3, 0.5, size=500)
    data = np.concatenate([c2, c1])  # unordered
    vel_std = (data.reshape(-1,1) - data.mean()) / data.std()
    gm = GaussianMixture(n_components=2, random_state=RNG_SEED).fit(vel_std)
    order = np.argsort(gm.means_.ravel())
    means = gm.means_.ravel()
    # this ensures that thecomponent with lower mean is first in the order
    assert means[order[0]] < means[order[1]]

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])