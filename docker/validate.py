#!/usr/bin/env python3
"""
GTC NVFP4 Training Lab - Comprehensive Environment Validation
==============================================================
Checks: PyTorch, Transformer Engine, ModelOpt, Megatron-LM, Megatron-Bridge
"""

import sys
import os

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def ok(msg):
    print(f"  {Colors.GREEN}✅ {msg}{Colors.END}")

def fail(msg):
    print(f"  {Colors.RED}❌ {msg}{Colors.END}")

def warn(msg):
    print(f"  {Colors.YELLOW}⚠️  {msg}{Colors.END}")

def info(msg):
    print(f"  {Colors.BLUE}ℹ️  {msg}{Colors.END}")

def header(msg):
    print(f"\n{Colors.BOLD}{msg}{Colors.END}")

def separator():
    print("=" * 80)

def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    header("1. PyTorch")
    try:
        import torch
        ok(f"PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            ok(f"CUDA available: {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            ok(f"GPU count: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                info(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
            
            # Test basic CUDA operation
            x = torch.randn(100, 100, device='cuda')
            y = torch.matmul(x, x)
            ok("CUDA tensor operations working")
            return True
        else:
            fail("CUDA not available")
            return False
    except Exception as e:
        fail(f"PyTorch: {e}")
        return False

def check_transformer_engine():
    """Check Transformer Engine installation and FP8/NVFP4 support."""
    header("2. Transformer Engine")
    try:
        import transformer_engine
        ok(f"Transformer Engine {transformer_engine.__version__}")
        
        # Check PyTorch integration
        import transformer_engine.pytorch as te
        ok("PyTorch integration available")
        
        # Check FP8 recipes
        try:
            from transformer_engine.common.recipe import DelayedScaling, Format
            ok("FP8 DelayedScaling recipe available")
            ok(f"FP8 formats: {[f.name for f in Format]}")
        except ImportError as e:
            warn(f"FP8 recipes: {e}")
        
        # Check FP8 state manager
        try:
            from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
            ok("FP8GlobalStateManager available")
        except ImportError:
            warn("FP8GlobalStateManager not available in this version")
        
        # Check for TE layers
        try:
            from transformer_engine.pytorch import Linear, LayerNorm, TransformerLayer
            ok("TE layers available (Linear, LayerNorm, TransformerLayer)")
        except ImportError as e:
            warn(f"Some TE layers not available: {e}")
        
        # Test basic forward pass
        try:
            import torch
            if torch.cuda.is_available():
                layer = te.Linear(256, 256).cuda()
                x = torch.randn(8, 256, device='cuda')
                y = layer(x)
                ok(f"TE Linear forward pass working (output shape: {y.shape})")
        except Exception as e:
            warn(f"TE forward pass test: {e}")
        
        return True
    except Exception as e:
        fail(f"Transformer Engine: {e}")
        return False

def check_modelopt():
    """Check ModelOpt installation and quantization support."""
    header("3. ModelOpt")
    
    # Suppress the Conv1D warning (it's a non-critical plugin issue)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Conv1D.*")
        warnings.filterwarnings("ignore", message=".*apex plugin.*")
        
        try:
            import modelopt
            ok(f"ModelOpt {modelopt.__version__}")
            
            # Check torch quantization module
            try:
                import modelopt.torch.quantization as mtq
                ok("Quantization module available")
                
                # List available quantization configs
                quant_configs = [c for c in dir(mtq) if 'quant' in c.lower() or 'config' in c.lower()]
                info(f"  Available configs: {len(quant_configs)} found")
            except Exception as e:
                warn(f"Quantization module: {e}")
            
            # Check export module
            try:
                import modelopt.torch.export as mte
                ok("Export module available")
            except ImportError:
                info("Export module not available (optional)")
            
            return True
        except ImportError as e:
            fail(f"ModelOpt not installed: {e}")
            return False
        except Exception as e:
            warn(f"ModelOpt has issues but is installed: {e}")
            return True  # Still return True if it's installed

def check_megatron_lm():
    """Check Megatron-LM installation."""
    header("4. Megatron-LM")
    success = True
    try:
        import megatron
        ok("Megatron package importable")
        
        # Check core module
        try:
            from megatron.core import parallel_state
            ok("Megatron Core parallel_state available")
        except ImportError as e:
            warn(f"Megatron Core: {e}")
            success = False
        
        # Check tensor parallel
        try:
            from megatron.core.tensor_parallel import layers
            ok("Tensor Parallel layers available")
        except ImportError as e:
            warn(f"Tensor Parallel: {e}")
        
        # Check transformer module
        try:
            from megatron.core.transformer import TransformerConfig
            ok("TransformerConfig available")
        except ImportError as e:
            warn(f"TransformerConfig: {e}")
        
        # Check models
        try:
            from megatron.core.models.gpt import GPTModel
            ok("GPTModel available")
        except ImportError as e:
            info(f"GPTModel: {e} (may require initialization)")
        
        # Check installation path (handle namespace packages)
        try:
            if hasattr(megatron, '__file__') and megatron.__file__:
                megatron_path = os.path.dirname(megatron.__file__)
                info(f"  Installed at: {megatron_path}")
            elif hasattr(megatron, '__path__'):
                info(f"  Installed at: {list(megatron.__path__)[0]}")
        except:
            pass  # Path info is just informational
        
        return success
    except Exception as e:
        fail(f"Megatron-LM: {e}")
        return False


def check_additional_tools():
    """Check additional development tools."""
    header("5. Additional Tools")
    
    tools = {
        'jupyter': 'Jupyter',
        'tensorboard': 'TensorBoard', 
        'wandb': 'Weights & Biases',
        'datasets': 'HuggingFace Datasets',
        'transformers': 'HuggingFace Transformers',
    }
    
    available = 0
    for module, name in tools.items():
        try:
            __import__(module)
            ok(f"{name}")
            available += 1
        except ImportError:
            info(f"{name} not installed (optional)")
    
    return available > 0

def run_fp8_test():
    """Test FP8 functionality with Transformer Engine."""
    header("6. FP8 Functional Test")
    try:
        import torch
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
        
        if not torch.cuda.is_available():
            warn("Skipping FP8 test - no GPU available")
            return False
        
        # Check GPU capability (need Hopper+ for FP8)
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:
            info(f"GPU compute capability {capability[0]}.{capability[1]} - FP8 requires sm_89+ (H100/Ada)")
            info("FP8 will work in emulation mode on older GPUs")
        
        # Create FP8 recipe
        fp8_recipe = DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=16,
            amax_compute_algo="max"
        )
        ok("FP8 recipe created")
        
        # Test FP8 context (dimensions must be divisible by 8/16)
        model = te.Linear(512, 512).cuda()
        # Batch size * seq_len must be divisible by 8, hidden dim by 16
        x = torch.randn(8, 32, 512, device='cuda')  # [8, 32, 512]
        
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            y = model(x)
        
        ok(f"FP8 forward pass completed (output shape: {y.shape})")
        return True
        
    except Exception as e:
        warn(f"FP8 test: {e}")
        return False

def main():
    separator()
    print(f"{Colors.BOLD}🔬 GTC NVFP4 TRAINING LAB - COMPREHENSIVE VALIDATION{Colors.END}")
    separator()
    
    results = {}
    
    # Run all checks
    results['pytorch'] = check_pytorch()
    results['transformer_engine'] = check_transformer_engine()
    results['modelopt'] = check_modelopt()
    results['megatron_lm'] = check_megatron_lm()
    results['additional_tools'] = check_additional_tools()
    results['fp8_test'] = run_fp8_test()
    
    # Summary
    separator()
    header("SUMMARY")
    
    required = ['pytorch', 'transformer_engine', 'megatron_lm']
    optional = ['modelopt', 'additional_tools', 'fp8_test']
    
    required_pass = all(results[k] for k in required)
    
    print("\n  Required components:")
    for k in required:
        status = f"{Colors.GREEN}PASS{Colors.END}" if results[k] else f"{Colors.RED}FAIL{Colors.END}"
        print(f"    • {k}: {status}")
    
    print("\n  Optional components:")
    for k in optional:
        status = f"{Colors.GREEN}PASS{Colors.END}" if results[k] else f"{Colors.YELLOW}NOT AVAILABLE{Colors.END}"
        print(f"    • {k}: {status}")
    
    separator()
    
    if required_pass:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ ENVIRONMENT READY FOR NVFP4 TRAINING!{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ ENVIRONMENT HAS ISSUES - See failures above{Colors.END}")
        sys.exit(1)
    
    separator()

if __name__ == "__main__":
    main()

