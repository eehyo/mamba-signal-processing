import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from mamba_block import MambaBlock, causal_conv
from models import MambaClassifier, CNNClassifier, LSTMClassifier


def test_upstream_forward_and_parameter_gradients():
    """Actual vendored mamba-minimal, identical weights, independent implementation."""
    path = Path(__file__).parents[1] / "third_party" / "mamba_minimal.py"
    spec = importlib.util.spec_from_file_location("upstream_minimal", path)
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    torch.manual_seed(11)
    ref = upstream.MambaBlock(upstream.ModelArgs(d_model=8,n_layer=1,vocab_size=8,d_state=4))
    ours = MambaBlock(d_model=8,d_state=4)
    result = ours.load_state_dict(ref.state_dict(), strict=False)
    assert result.missing_keys == ["delta_scale"] and result.unexpected_keys == []
    x = torch.randn(2,13,8,requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    y, yy = ref(x), ours(xx)
    torch.testing.assert_close(y,yy,rtol=2e-5,atol=2e-6)
    y.square().sum().backward(); yy.square().sum().backward()
    torch.testing.assert_close(x.grad,xx.grad,rtol=2e-5,atol=2e-6)
    for name,p in ref.named_parameters():
        torch.testing.assert_close(p.grad,dict(ours.named_parameters())[name].grad,rtol=5e-5,atol=2e-6)


@pytest.mark.parametrize("length",[1,7,32])
@pytest.mark.parametrize("scale",[0.5,1.0,4.0])
def test_lti_recurrence_direct_fft(length,scale):
    torch.manual_seed(2)
    block=MambaBlock(d_model=3,d_state=4,selective=False).double().set_delta_scale(scale)
    u=torch.randn(2,length,6,dtype=torch.float64)
    kernel=block.lti_kernel(length)
    y=block.ssm(u)
    for method in ["direct","fft"]:
        torch.testing.assert_close(y,causal_conv(u,kernel,method),rtol=1e-11,atol=1e-11)


def test_lti_impulse_and_skip():
    block=MambaBlock(d_model=2,d_state=3,selective=False).double()
    u=torch.zeros(1,12,4,dtype=torch.float64);u[:,0]=1
    torch.testing.assert_close(block.ssm(u)[0],block.lti_kernel(12))
    no_skip=block.lti_kernel(12,include_skip=False)
    diff=block.lti_kernel(12)-no_skip
    torch.testing.assert_close(diff[0],block.D)
    assert diff[1:].count_nonzero()==0


def test_lti_dt_numpy_oracle():
    torch.manual_seed(5)
    block=MambaBlock(d_model=2,d_state=2,selective=False).double().set_delta_scale(2.5)
    u=torch.randn(1,5,4,dtype=torch.float64)
    gaps=torch.tensor([[1.,3.,1.,2.,4.]],dtype=torch.float64)
    A,B,C,D=[v.detach().numpy() for v in [block.A,block.B_const,block.C_const,block.D]]
    dt=torch.nn.functional.softplus(block.dt_bias).detach().numpy()*2.5
    state=np.zeros((4,2));oracle=[]
    for t in range(5):
        step=dt*gaps[0,t].item()
        state=np.exp(step[:,None]*A)*state + step[:,None]*B*u[0,t].numpy()[:,None]
        oracle.append((state*C).sum(-1)+D*u[0,t].numpy())
    np.testing.assert_allclose(block.ssm(u,dt_mult=gaps).detach().numpy()[0],oracle,atol=1e-12)


def test_fft_has_no_circular_wraparound():
    u=torch.zeros(1,8,1,dtype=torch.float64);u[0,-1]=1
    k=torch.ones(8,1,dtype=torch.float64)
    got=causal_conv(u,k,"fft")
    torch.testing.assert_close(got[:,:-1],torch.zeros_like(got[:,:-1]),atol=1e-12,rtol=0)
    circular=torch.fft.irfft(torch.fft.rfft(u,dim=1)*torch.fft.rfft(k,dim=0)[None],n=8,dim=1)
    assert circular[0,0,0] > 0.9


@pytest.mark.parametrize("cls",[MambaClassifier,CNNClassifier,LSTMClassifier])
def test_right_padding_logits_and_backward(cls):
    torch.manual_seed(7)
    model=cls(input_dim=3,num_classes=2,d_model=8,n_layers=2).double()
    a=torch.randn(1,7,3,dtype=torch.float64)
    b=torch.randn(1,11,3,dtype=torch.float64)
    padded=torch.cat([torch.cat([a,torch.full((1,4,3),float('nan'),dtype=torch.float64)],1),b],0)
    mask=torch.arange(11)[None] < torch.tensor([7,11])[:,None]
    individual=torch.cat([model(a),model(b)],0)
    batched=model(padded,mask=mask)
    torch.testing.assert_close(batched,individual,atol=1e-10,rtol=1e-10)
    batched.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_recording_freezes_state_and_masks_skip():
    block=MambaBlock(d_model=2,d_state=3).double().set_record(True,states=True)
    u=torch.randn(2,6,4,dtype=torch.float64)
    mask=torch.arange(6)[None] < torch.tensor([3,6])[:,None]
    u[0,3:]=float('nan')
    dt=torch.ones(2,6,dtype=torch.float64);dt[0,3:]=float('nan')
    y=block.ssm(u,dt_mult=dt,mask=mask)
    assert y[0,3:].count_nonzero()==0
    assert block.record['delta'][0,3:].count_nonzero()==0
    states=block.record['states']
    for t in [3,4,5]:
        torch.testing.assert_close(states[0,t],states[0,2])
    assert all(not t.requires_grad and t.device.type=='cpu' for t in block.record.values())
    block.set_record(False)
    assert block.record=={}


def test_causality():
    torch.manual_seed(1)
    model=MambaBlock(d_model=4).double()
    x=torch.randn(1,16,4,dtype=torch.float64)
    altered=x.clone();altered[:,8:]=torch.randn_like(altered[:,8:])*100
    torch.testing.assert_close(model(x)[:,:8],model(altered)[:,:8])


def test_scale_and_dt_multiply_delta_and_input_gain():
    model=MambaBlock(d_model=2,d_state=2).double().set_record(True)
    x=torch.randn(2,4,2,dtype=torch.float64)
    model(x);base=model.record['delta'].clone()
    model.set_delta_scale(4)(x,dt_mult=torch.tensor([1.,2.,3.,4.]))
    torch.testing.assert_close(model.record['delta'],base*4*torch.arange(1,5)[None,:,None])


def test_gradcheck_ssm():
    torch.manual_seed(4)
    block=MambaBlock(d_model=1,d_state=2).double()
    u=torch.randn(1,3,2,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(block.ssm,(u,),atol=1e-5,rtol=1e-3)


def test_invalid_masks_and_dt_rejected():
    block=MambaBlock(d_model=2)
    x=torch.ones(1,3,2)
    for mask in [torch.tensor([[1,0,1]],dtype=torch.bool),torch.zeros(1,3,dtype=torch.bool)]:
        with pytest.raises(ValueError):block(x,mask=mask)
    for dt in [0,-1,float('nan'),torch.ones(2,2)]:
        with pytest.raises(ValueError):block(x,dt_mult=dt)
    with pytest.raises(ValueError):block.lti_kernel(3)
    with pytest.raises(ValueError):block.set_delta_scale(0)


def test_tiny_batch_can_be_learned():
    torch.manual_seed(0)
    model=MambaClassifier(input_dim=2,num_classes=2,d_model=8,n_layers=1,d_state=4)
    y=torch.arange(12)%2
    x=(2*y.float()-1)[:,None,None].expand(-1,12,2) + torch.randn(12,12,2)*0.05
    optimizer=torch.optim.Adam(model.parameters(),lr=0.02)
    initial=float(torch.nn.functional.cross_entropy(model(x),y).detach())
    for _ in range(35):
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y)
        loss.backward();optimizer.step()
    final=float(torch.nn.functional.cross_entropy(model(x),y).detach())
    assert final < initial*0.2 and (model(x).argmax(-1)==y).all()
