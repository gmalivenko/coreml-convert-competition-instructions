import torch
import numpy as np
import coremltools as ct

# Include your model here
from rtsrn import rtsrn


if __name__ == '__main__':
    # Initialize your model 
    model = rtsrn(scale=2)
    # model.load_state_dict(torch.load('./checkpoint.pt'))
    model.eval()

    # Specify input shapes
    input_channels = 3
    input_width = 640
    input_height = 360

    # Create random input and trace the model
    image_input = torch.rand(1, input_channels, input_height, input_width)
    traced_model = torch.jit.trace(model, image_input)

    # Convert model to CoreML format
    # It's important to use proper input and output names and shapes. 
    # Follow the competition instructions for input/output shapes.
    # If your model requires more precise compute, change FLOAT16 to FLOAT32.
    mlprogram = ct.convert(
        traced_model, 
        inputs=[
            ct.TensorType(shape=(1, input_channels, input_height, input_width), dtype=np.float16, name='input_image')
        ],
        outputs=[ct.TensorType(name='output_image', dtype=np.float16)],
        convert_to='mlprogram',
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    # Save model as "model.mlpackage"
    # hint -- use netron to check architecture (model.mlpackage/Data/com.apple.CoreML/model.mlmodel)
    mlprogram.save('model.mlpackage')

