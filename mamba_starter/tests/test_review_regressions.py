"""Student-facing boundary cases and evaluation reproducibility, reviewed in v1.1."""
import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split
from data import subset_indices, SequenceDataset, collate_sequences, image_sequences, Standardizer
from models import MambaClassifier
from evaluate import load_checkpoint, evaluation_data
from train import metrics


@pytest.mark.parametrize('n', [6, 114, 115, 119, 120])
def test_subset_size_class_coverage_and_no_duplicates(n):
    y = np.tile(np.arange(6), 20)
    chosen = subset_indices(y, n, seed=42)
    assert len(chosen) == len(np.unique(chosen)) == n
    assert set(y[chosen]) == set(y)
    np.testing.assert_array_equal(chosen, subset_indices(y, n, seed=42))


def test_subset_with_singleton_class_and_seed_compatibility():
    rare = np.r_[np.zeros(100, dtype=int), 1, 2]
    chosen = subset_indices(rare, 6)
    assert len(chosen) == 6 and set(rare[chosen]) == {0, 1, 2}
    ordinary = np.tile(np.arange(6), 100)
    expected, _ = train_test_split(np.arange(600), train_size=120, stratify=ordinary, random_state=123)
    np.testing.assert_array_equal(subset_indices(ordinary, 120, 123), np.sort(expected))


@pytest.mark.parametrize('selective', [True, False])
def test_elapsed_time_padding_matches_single_sequences(selective):
    torch.manual_seed(19)
    model = MambaClassifier(input_dim=3, d_model=8, n_layers=2, d_state=4,
                            selective=selective).double().set_delta_scale(2)
    rng = np.random.default_rng(31)
    data = SequenceDataset(rng.normal(size=(3, 29, 3)), np.array([0,1,2]),
                           keep_rate=0.45, gap_mode='delta')
    samples = [data[i] for i in range(3)]
    batch = collate_sequences(samples)
    singles = torch.cat([model(item['x'].double()[None], dt_mult=item['dt'][None]) for item in samples])
    batched = model(batch['x'].double(), dt_mult=batch['dt'], mask=batch['mask'])
    torch.testing.assert_close(singles, batched, rtol=1e-10, atol=1e-10)


def test_checkpoint_preserves_scale_and_validation_preprocessing(tmp_path, monkeypatch):
    """Evaluation must reuse the exact validation rows and training normalizer."""
    rng = np.random.default_rng(10)
    x = rng.normal(size=(12, 8, 3)).astype(np.float32)
    labels = np.arange(12) % 6
    norm = Standardizer().fit(x[:8])
    val_idx = [10, 8, 9, 11]
    kwargs = dict(name='mamba', input_dim=3, num_classes=6, d_model=8, n_layers=1, d_state=4)
    from models import build_model
    model = build_model(**kwargs).set_delta_scale(3.5).eval()
    checkpoint = dict(model_state=model.state_dict(), model_kwargs=kwargs,
                      config={'dataset':'har','data_root':'unused'}, normalizer=norm.state_dict(),
                      split={'val_indices':val_idx})
    path = tmp_path / 'checkpoint.pt'
    torch.save(checkpoint, path)
    restored, checkpoint = load_checkpoint(path)
    assert float(restored.layers[0].mixer.delta_scale) == 3.5
    monkeypatch.setattr('evaluate.load_har', lambda *a, **k: (x, labels, np.arange(12)))
    val_x, val_y = evaluation_data(checkpoint)
    np.testing.assert_array_equal(val_y, labels[val_idx])
    np.testing.assert_array_equal(val_x, norm.transform(x[val_idx]))
    with torch.no_grad():
        torch.testing.assert_close(restored(torch.tensor(val_x)), model(torch.tensor(val_x)))


def test_fashion_resolution_transfer_preserves_training_scale():
    rng = np.random.default_rng(3)
    images = rng.integers(0, 256, (4,28,28), dtype=np.uint8)
    cfg = dict(image_size=14, image_filter='blur', image_mode='pixel', scan='raster', permutation_seed=42)
    train = image_sequences(images, cfg)
    norm = Standardizer().fit(train)
    state = norm.state_dict()
    for method in ['naive', 'average', 'blur']:
        small = image_sequences(images, cfg, size=7, method=method)
        assert small.shape == (4,49,1)
        assert np.isfinite(norm.transform(small)).all()
    assert state == norm.state_dict()


def test_macro_f1_retains_missing_classes():
    result = metrics([0,0,1], [0,0,0], num_classes=3)
    assert result['accuracy'] == pytest.approx(2/3)
    assert result['macro_f1'] == pytest.approx(0.8/3)
    assert np.array(result['confusion_matrix']).shape == (3,3)


def test_evaluation_reports_observed_lengths_not_padding():
    from train import evaluate_model
    samples = [{'x':torch.ones(3,1),'y':0,'dt':None},
               {'x':torch.ones(7,1),'y':1,'dt':None}]
    model = MambaClassifier(input_dim=1,num_classes=2,d_model=4,n_layers=1,d_state=2)
    result = evaluate_model(model,[collate_sequences(samples)],'cpu',2)
    assert result['valid_length_min'] == 3
    assert result['valid_length_max'] == 7
    assert result['valid_length_mean'] == 5
