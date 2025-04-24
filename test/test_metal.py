import unittest
import numpy as np
from tinygrad.device import CompileError, Device, Compiler
from tinygrad import Tensor
if Device.DEFAULT=="METAL":
  from tinygrad.runtime.ops_metal import MetalDevice, MetalCompiler, MetalProgram
@unittest.skipIf(Device.DEFAULT!="METAL", "Metal support required")
class TestMetal(unittest.TestCase):
  def test_alloc_oom(self):
    device = MetalDevice("metal")
    with self.assertRaises(MemoryError):
      device.allocator.alloc(10000000000000000000)

  def test_compile_error(self):
    compiler = MetalCompiler()
    with self.assertRaises(CompileError):
      compiler.compile("this is not valid metal")

  def test_compile_success(self):
    compiler = MetalCompiler()
    ret = compiler.compile("""
#include <metal_stdlib>
  using namespace metal;
  kernel void E_4n1(device int* data0, const device int* data1, const device int* data2,
          uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    int val0 = *(data1+0);
    int val1 = *(data1+1);
    int val2 = *(data1+2);
    int val3 = *(data1+3);
    int val4 = *(data2+0);
    int val5 = *(data2+1);
    int val6 = *(data2+2);
    int val7 = *(data2+3);
    *(data0+0) = (val0+val4);
    *(data0+1) = (val1+val5);
    *(data0+2) = (val2+val6);
    *(data0+3) = (val3+val7);
  }
""")
    assert ret is not None

  def test_failed_newLibraryWithData(self):
    device = MetalDevice("metal")
    compiler = MetalCompiler()
    compiled = compiler.compile("""
#include <metal_stdlib>
kernel void r_5(device int* data0, const device int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]){
  data0[0] = 0;
}
""")
    with self.assertRaises(RuntimeError):
      compiled = compiled[:40] # corrupt the compiled program
      MetalProgram(device, "r_5", compiled)

  def test_program_w_empty_compiler(self):
    device = MetalDevice("metal")
    compiler = Compiler(device)
    compiled = compiler.compile("""
#include <metal_stdlib>
kernel void r_5(device int* data0, const device int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]){
  data0[0] = 0;
}
""")
    MetalProgram(device, "r_5", compiled)

  def test_bad_program_w_empty_compiler(self):
    device = MetalDevice("metal")
    compiler = Compiler(device)
    # this does not raise
    compiled = compiler.compile("""
#include <metal_stdlib>
kernel void r_5(device int* data0, const device int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]){
  invalid codes;
}
""")
    with self.assertRaises(RuntimeError):
      MetalProgram(device, "r_5", compiled)

  def test_transfer_from_device0_to_all(self):
    NUM_DEVICES = 4
    devices = tuple(f"METAL:{i}" for i in range(NUM_DEVICES))

    # Create a tensor with a known value on device 0
    original = Tensor.full((10, 10), 42.0, device=devices[0])

    # Transfer to all other devices and verify
    for i in range(1, NUM_DEVICES):
      transfer = original.to(device=devices[i])
      transfer.realize()

      # Verify the transferred tensor has the correct value
      self.assertEqual(transfer.shape, original.shape)
      t_np = transfer.numpy()
      self.assertTrue(np.all(t_np == 42.0),
                    f"Transfer from {devices[0]} to {devices[i]} failed: "
                    f"Expected all values to be 42.0")

  def test_transfer_from_all_to_device0(self):
    NUM_DEVICES = 4
    devices = tuple(f"METAL:{i}" for i in range(NUM_DEVICES))

    # Create tensors with different known values on each device
    tensors = []
    for i in range(1, NUM_DEVICES):
      t = Tensor.full((10, 10), float(i*10), device=devices[i])
      tensors.append(t)

    # Transfer to device 0 and verify
    for i, original in enumerate(tensors):
      transfer = original.to(device=devices[0])
      transfer.realize()

      # Verify the transferred tensor has the correct value
      self.assertEqual(transfer.shape, original.shape)
      t_np = transfer.numpy()
      expected_value = float((i+1)*10)
      self.assertTrue(np.all(t_np == expected_value),
                    f"Transfer from {devices[i+1]} to {devices[0]} failed: "
                    f"Expected all values to be {expected_value}")

  def test_transfers_between_all_devices(self):
    NUM_DEVICES = 4
    devices = tuple(f"METAL:{i}" for i in range(NUM_DEVICES))

    tensors = []
    for i, device in enumerate(devices):
      t = Tensor.full((10, 10), float(i+1), device=device)
      tensors.append(t)

    for src_idx in range(NUM_DEVICES):
      for dst_idx in range(NUM_DEVICES):
        if src_idx != dst_idx:  # Skip same-device transfers
          original = tensors[src_idx]
          transfer = original.to(device=devices[dst_idx])
          transfer.realize()

          self.assertEqual(transfer.shape, original.shape)
          t_np = transfer.numpy()
          expected_value = float(src_idx+1)
          self.assertTrue(np.all(t_np == expected_value))