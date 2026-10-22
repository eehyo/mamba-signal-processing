import gzip
import struct
import numpy as np
import pytest
import torch
from signal_utils import downsample, upsample, alias_frequency, power_spectrum, ema, ema_features
from signal_utils import irregular_sample, interpolate_irregular
from data import subject_split, Standardizer, SequenceDataset, collate_sequences, load_har, load_fashion, _idx
from image_utils import resize_images, image_to_sequence


@pytest.mark.parametrize("m,expected",[(2,7),(4,5.5),(8,0.75)])
def test_tone_alias_and_antialias(m,expected):
    t=np.arange(6400)/50
    x=np.sin(2*np.pi*18*t)[:,None]
    naive=downsample(x,m,'naive');aa=downsample(x,m,'aa')
    f,p=power_spectrum(naive,50/m)
    assert alias_frequency(18,50/m)==expected
    assert f[p[:,0].argmax()]==pytest.approx(expected)
    assert np.mean(aa[32:-32]**2)<0.001*np.mean(naive[32:-32]**2)
    restored=upsample(naive,m,'poly')
    f,p=power_spectrum(restored,50)
    assert f[p[:,0].argmax()]==pytest.approx(expected)
    assert p[np.argmin(abs(f-18)),0] < p.max()*1e-4


def test_psd_normalization_and_spacing():
    # An integer-bin sinusoid gives integral power 1/2 under this convention.
    x=np.sin(2*np.pi*5*np.arange(1000)/50)[:,None]
    f,p=power_spectrum(x,50)
    assert f[1]-f[0]==pytest.approx(0.05)
    assert (p.sum()*(f[1]-f[0]))==pytest.approx(0.5,abs=1e-12)


def test_ema_impulse_and_features():
    x=np.zeros((1,16,2));x[:,0]=1
    for a in [0,0.5,0.99]:
        np.testing.assert_allclose(ema(x,a)[0,:,0],(1-a)*a**np.arange(16),atol=1e-15)
    features=ema_features(x,[0.5,0.9])
    assert features.shape==(1,12)
    filtered=ema(x,0.5)
    np.testing.assert_allclose(features[:,:2],filtered.mean(1))
    with pytest.raises(ValueError):ema(x,1)


def test_subject_split_disjoint_fixed_and_scaling():
    subjects=np.repeat(np.arange(1,11),12)
    tr,va=subject_split(subjects,split_seed=42)
    assert not set(subjects[tr]) & set(subjects[va])
    assert len(tr)+len(va)==len(subjects)
    np.testing.assert_array_equal(tr,subject_split(subjects,split_seed=42)[0])
    x=np.arange(24).reshape(3,4,2).astype(float)
    norm=Standardizer().fit(x)
    state=norm.state_dict()
    assert np.allclose(norm.transform(x).mean((0,1)),0,atol=1e-6)
    norm.transform(x+10000)
    assert state==norm.state_dict()


def test_har_loader_raw_channel_order_and_labels(tmp_path):
    from data import HAR_CHANNELS
    root=tmp_path/'UCI HAR Dataset'/'train'
    (root/'Inertial Signals').mkdir(parents=True)
    for c,name in enumerate(HAR_CHANNELS):
        np.savetxt(root/'Inertial Signals'/f'{name}_train.txt',np.full((6,128),c))
    np.savetxt(root/'y_train.txt',np.arange(1,7),fmt='%d')
    np.savetxt(root/'subject_train.txt',[1,1,2,2,3,3],fmt='%d')
    x,y,s=load_har(tmp_path)
    assert x.shape==(6,128,9)
    np.testing.assert_array_equal(x[0,0],np.arange(9))
    np.testing.assert_array_equal(y,np.arange(6))
    with pytest.raises(ValueError):load_har(tmp_path,split='test')
    with pytest.raises(ValueError):load_fashion(tmp_path,split='test')


def test_irregular_gaps_interpolation_collate():
    x=np.arange(128)[:,None].astype(np.float32)
    values,ticks,gaps=irregular_sample(x,0.4,np.random.default_rng(1))
    assert ticks[0]==0 and ticks[-1]==127 and gaps[0]==1
    np.testing.assert_array_equal(gaps[1:],np.diff(ticks))
    np.testing.assert_allclose(interpolate_irregular(values,ticks,128),x)
    xs=np.stack([x,x*2]);labels=np.array([0,1])
    ds=SequenceDataset(xs,labels,keep_rate=0.4,gap_mode='delta')
    batch=collate_sequences([ds[0],ds[1]])
    assert batch['mask'].dtype==torch.bool
    assert batch['dt'][~batch['mask']].count_nonzero()==0
    channel=SequenceDataset(xs,labels,keep_rate=0.4,gap_mode='channel')
    torch.testing.assert_close(channel[0]['x'][:,-1],ds[0]['dt'])
    # Same missingness by construction across all four strategies.
    ignore=SequenceDataset(xs,labels,keep_rate=0.4,gap_mode='ignore')
    torch.testing.assert_close(ignore[0]['x'],ds[0]['x'])


def test_image_scan_geometry_and_fixed_permutation():
    x=np.arange(9,dtype=np.float32).reshape(1,3,3)
    expected={'raster':[0,1,2,3,4,5,6,7,8], 'snake':[0,1,2,5,4,3,6,7,8],
              'column':[0,3,6,1,4,7,2,5,8]}
    for scan,order in expected.items():
        np.testing.assert_array_equal(image_to_sequence(x,scan)[0,:,0],order)
    r=image_to_sequence(np.repeat(x,2,axis=0),'random')
    np.testing.assert_array_equal(r[0],r[1])
    np.testing.assert_array_equal(np.sort(r[0,:,0]),np.arange(9))
    assert image_to_sequence(x,mode='row').shape==(1,3,3)


@pytest.mark.parametrize('method',['blur','average','naive'])
def test_image_resize(method):
    x=np.ones((2,28,28),dtype=np.float32)
    small=resize_images(x,14,method)
    assert small.shape==(2,14,14)
    np.testing.assert_allclose(small,1)
    assert image_to_sequence(small).shape==(2,196,1)
    with pytest.raises(ValueError):resize_images(x,13)


def test_idx_parser(tmp_path):
    path=tmp_path/'image.gz'
    x=np.arange(12,dtype=np.uint8).reshape(2,2,3)
    with gzip.open(path,'wb') as f:f.write(struct.pack('>IIII',2051,2,2,3)+x.tobytes())
    np.testing.assert_array_equal(_idx(path,True),x)
