import torch
from slr_network import SLRModel
import warnings
warnings.filterwarnings('ignore')

def test_model():
    print("Testing SLRModel with Transformer Encoder...")
    
    # Initialize the model
    # SLRModel params: num_classes, c2d_type, conv_type
    model = SLRModel(num_classes=500, c2d_type='resnet18', conv_type=2, use_bn=False, hidden_size=1024)
    model.train() # Set to train mode
    
    print("Model initialized successfully.")
    
    # Create dummy data
    # Input shape: [B, T, C, H, W] for videos or [T, B, C] for frame-wise features. 
    # Let's provide frame-wise features to bypass the 2D CNN (resnet18) for speed, 
    # as we just want to test the temporal modeling.
    # Wait, the code expects [B, T, C, H, W] if len(x.shape) == 5, else frame-wise features.
    # However, frame-wise feature input is expected to be [B*T, C, H, W] in original code?
    # No, it says: `framewise = x` if not 5D. 
    # Let's see what `TemporalConv` expects.
    # TemporalConv takes `framewise` and `len_x`.
    # `framewise` should be `[B, C, T]` for `nn.Conv1d`? 
    # Wait, `TemporalConv` does `visual_feat = self.temporal_conv(frame_feat)`.
    # `nn.Conv1d` expects `[B, C, L]`. So frame_feat should be `[B, C, T]`.
    
    batch_size = 2
    temp = 16
    channels = 512 # resnet18 output channels
    
    # Let's pass frame-wise features [B, C, T]
    dummy_input = torch.randn(batch_size, channels, temp)
    len_x = torch.tensor([temp, temp-2])
    
    try:
        outputs = model(dummy_input, len_x)
        print("Forward pass completed successfully.")
        
        # Check shapes
        print(f"conv_logits shape: {outputs['conv_logits'].shape} (Expected: [T', B, num_classes])")
        print(f"sequence_logits shape: {outputs['sequence_logits'].shape} (Expected: [T', B, num_classes])")
        print(f"feat_len: {outputs['feat_len'].tolist()}")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_model()
