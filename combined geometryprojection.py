import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow.python.ops import math_ops
import PIL.Image as Image
import tensorflow_addons as tfa

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow.python.ops import math_ops

# Essential utility functions for geometry projection
def softargmax(x, beta=100):
    """Softargmax function for differentiable argmax operations"""
    x_range = tf.range(x.shape.as_list()[-1], dtype=tf.float32)
    return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1, keep_dims=True)

def tf_shape(x, rank):
    """Get static and dynamic shape information"""
    static_shape = x.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(x), rank)
    return [s if s is not None else d for s,d in zip(static_shape, dynamic_shape)]

def safe_divide(numerator, denominator, name='safe_divide'):
    """Safe division operation"""
    return tf.where(math_ops.greater(denominator, 0), 
                   math_ops.divide(numerator, denominator), 
                   tf.zeros_like(numerator),
                   name=name)

def warp_pad_columns(x, n=1):
    """Pad columns for circular convolution"""
    out = tf.concat([x[:, :, -n:, :], x, x[:, :, :n, :]], axis=2)
    return tf.pad(out, [[0, 0], [n, n], [0, 0], [0, 0]])

# Essential image processing functions
def preprocess(image):
    """Convert [0, 1] to [-1, 1]"""
    with tf.name_scope("preprocess"):
        return image * 2 - 1

def deprocess(image):
    """Convert [-1, 1] to [0, 1]"""
    with tf.name_scope("deprocess"):
        return (image + 1) / 2

# Essential layer operations
def lrelu(x, a=0.2):
    """Leaky ReLU activation function"""
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(inputs):
    """Batch normalization"""
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, 
                                       momentum=0.1, training=True, 
                                       gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def gen_conv(batch_input, out_channels, separable_conv=False):
    """Convolutional layer for generator"""
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, 
                                        kernel_size=4, strides=(2, 2), 
                                        padding="same", 
                                        depthwise_initializer=initializer, 
                                        pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, 
                              kernel_size=4, strides=(2, 2), 
                              padding="same", 
                              kernel_initializer=initializer)

def gen_deconv(batch_input, out_channels, separable_conv=False):
    """Deconvolutional layer for generator"""
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], 
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, 
                                        kernel_size=4, strides=(1, 1), 
                                        padding="same", 
                                        depthwise_initializer=initializer, 
                                        pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, 
                                        kernel_size=4, strides=(2, 2), 
                                        padding="same", 
                                        kernel_initializer=initializer)
    
##projector py original
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import tf_shape

def over_composite(rgbas):
    """
    Combines a list of RGBA images using the over operation.
    Combines RGBA images from back to front with the over operation.
    The alpha image of the first image is ignored and assumed to be 1.0.

    Args:
        rgbas: A list of [batch, H, W, 4] RGBA images, combined from back to front.
    Returns:
        Composited RGB image.
    """
    for i in range(len(rgbas)):
        rgb = rgbas[i][:, :, :, 0:3]
        alpha = rgbas[i][:, :, :, 3:]
        if i == 0:
            output = rgb
        else:
            rgb_by_alpha = rgb * alpha
            output = rgb_by_alpha + output * (1.0 - alpha)
    return output

def mpi_render_grd_view(batch_rgbas, share_alpha=True):
    """
    Render ground-view MPI representation
    
    Args:
        batch_rgbas: Batch of RGBA images
        share_alpha: Whether to share alpha channel across planes
    Returns:
        Synthesized ground-view image
    """
    batch, height, width, channel = batch_rgbas.get_shape().as_list()

    if share_alpha:
        num_mpi_planes = int(channel/4)
        rgba_layers = tf.reshape(batch_rgbas, [-1, height, width, num_mpi_planes, 4])
        rgb = rgba_layers[..., :3]
        alpha = tf.expand_dims(rgba_layers[..., -1], axis=-1)
    else:
        num_mpi_planes = int(channel / 5)
        rgba_layers = tf.reshape(batch_rgbas, [-1, height, width, num_mpi_planes, 5])
        rgb = rgba_layers[..., :3]
        alpha = tf.expand_dims(rgba_layers[..., 4], axis=-1)

    alpha = (alpha + 1.)/2.
    rgba_layers = tf.transpose(tf.concat([rgb, alpha], axis=-1), [3, 0, 1, 2, 4])

    rgba_list = []
    for i in range(int(num_mpi_planes)):
        rgba_list.append(rgba_layers[i])

    synthesis_image = over_composite(rgba_list)
    return synthesis_image

def mpi_render_aer_view(batch_rgbas, share_alpha=True):
    """
    Render aerial-view MPI representation
    
    Args:
        batch_rgbas: Batch of RGBA images
        share_alpha: Whether to share alpha channel across planes
    Returns:
        Synthesized aerial-view image
    """
    batch, height, width, channel = batch_rgbas.get_shape().as_list()

    if share_alpha:
        num_mpi_planes = int(channel / 4)
        rgba_layers = tf.reshape(batch_rgbas, [-1, height, width, num_mpi_planes, 4])
        rgb = rgba_layers[..., :3]
        alpha = tf.expand_dims(rgba_layers[..., -1], axis=-1)
    else:
        num_mpi_planes = int(channel / 5)
        rgba_layers = tf.reshape(batch_rgbas, [-1, height, width, num_mpi_planes, 5])
        rgb = rgba_layers[..., :3]
        alpha = tf.expand_dims(rgba_layers[..., -1], axis=-1)

    alpha = (alpha + 1.) / 2.
    rgba_layers = tf.transpose(tf.concat([rgb, alpha], axis=-1), [1, 0, 2, 3, 4])

    rgba_list = []
    for i in range(int(height)):
        rgba_list.append(rgba_layers[i])

    rgba_list = rgba_list[::-1][:int(height//2)]
    synthesis_image = over_composite(rgba_list)
    return synthesis_image

def rtheta2uv(athetaimage, aer_size):
    """
    Convert r-theta coordinates to UV coordinates
    
    Args:
        athetaimage: Image in r-theta format [batch, width, PlaneNum, 3]
        aer_size: Size of aerial image
    Returns:
        Image in UV coordinates
    """
    batch, width, PlaneNum, channel = tf_shape(athetaimage, 4)
    
    # Create coordinate grid
    i = np.arange(aer_size)
    j = np.arange(aer_size)
    jj, ii = np.meshgrid(j, i)

    # Calculate center and angles
    center = aer_size / 2 - 0.5
    theta = np.arctan(-(jj - center) / (ii - center))
    theta[np.where(ii < center)] += np.pi
    theta[np.where((ii >= center) & (jj >= center))] += 2 * np.pi
    theta = theta/(2 * np.pi)*width

    # Calculate radius
    RadiusByPixel = np.sqrt((ii - center) ** 2 + (jj - center) ** 2)
    RadiusByPixel = (1-RadiusByPixel/aer_size*2)*PlaneNum

    # Create sampling grid
    uv = np.stack([RadiusByPixel, theta], axis=-1)
    uv = uv.astype(np.float32)
    warp = tf.stack([uv] * batch, axis=0)

    # Resample image
    sampler_output = tf.contrib.resampler.resampler(athetaimage, warp)
    return sampler_output
# geometry py oringial 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import *
import numpy as np
import tensorflow_addons as tfa

def geometry_transform(aer_imgs, estimated_height, target_height, target_width, mode, grd_height, max_height,
                      method='column', geoout_type='image', dataset='CVUSA'):
    """
    Main geometry transformation function
    
    Args:
        aer_imgs: Aerial/satellite images
        estimated_height: Height probability distribution
        target_height: Height of target street-view image
        target_width: Width of target street-view image
        mode: 'heightPlaneMethod' or 'radiusPlaneMethod'
        grd_height: Ground height parameter
        max_height: Maximum height to consider
        method: 'column' or 'point'
        geoout_type: 'volume' or 'image'
        dataset: Dataset name for scaling parameters
    """
    if mode == 'heightPlaneMethod':
        output = MultiPlaneImagesAer2Grd_height(aer_imgs, estimated_height, target_height, target_width, 
                                               grd_height, max_height, method, geoout_type, dataset)
    elif mode == 'radiusPlaneMethod':
        output = MultiPlaneImagesAer2Grd_radius(aer_imgs, estimated_height, target_height, target_width,
                                               grd_height, max_height, method, geoout_type, dataset)
    return output

def MultiPlaneImagesAer2Grd_height(signal, estimated_height, target_height, target_width, grd_height=-2, 
                                  max_height=30, method='column', geoout_type='image', dataset='CVUSA'):
    """
    Transform aerial images to street view using height-based multi-plane images
    """
    PlaneNum = estimated_height.get_shape().as_list()[-1]

    if method == 'column':
        estimated_height = tf.cumsum(estimated_height, axis=-1)

    batch, S, _, channel = tf_shape(signal, 4)
    H, W, C = signal.get_shape().as_list()[1:]
    assert (H == W)

    # Create coordinate grid
    i = np.arange(0, (target_height*2))
    j = np.arange(0, target_width)
    jj, ii = np.meshgrid(j, i)

    # Dataset-specific scaling factor
    if dataset == 'CVUSA':
        f = H/55
    elif dataset in ['CVACT', 'CVACThalf']:
        f = H/(50*206/256)
    elif dataset == 'CVACTunaligned':
        f = H/50
    elif dataset == 'OP':
        f = H/100

    tanii = np.tan(ii * np.pi / (target_height*2))

    images_list = []
    alphas_list = []

    # Process each plane
    for i in range(PlaneNum):
        z = grd_height + (max_height-grd_height) * i/PlaneNum

        u_dup = -1 * np.ones([(target_height*2), target_width])
        v_dup = -1 * np.ones([(target_height*2), target_width])
        m = target_height

        # Calculate projection coordinates
        v = S / 2. - f * z * tanii * np.sin(jj * 2 * np.pi / target_width)
        u = S / 2. + f * z * tanii * np.cos(jj * 2 * np.pi / target_width)

        # Handle projection based on height
        if z < 0:
            u_dup[-m:, :] = u[-m:, :]
            v_dup[-m:, :] = v[-m:, :]
        else:
            u_dup[0:m, :] = u[0:m, :]
            v_dup[0:m:, :] = v[0:m:, :]

        n = int(target_height/2)

        uv = np.stack([v_dup[n:-n,...], u_dup[n:-n,...]], axis=-1)
        uv = uv.astype(np.float32)
        warp = tf.stack([uv]*batch, axis=0)

        # Resample images and alphas
        images = tfa.image.resampler(signal, warp)
        alphas = tfa.image.resampler(estimated_height[..., i:i + 1], warp)
        images_list.append(images)
        alphas_list.append(alphas)

    # Return based on output type
    if geoout_type == 'volume':
        return tf.concat([images_list[i]*alphas_list[i] for i in range(PlaneNum)], axis=-1)
    elif geoout_type == 'image':
        # Composite images back-to-front
        for i in range(PlaneNum):
            rgb = images_list[i]
            a = alphas_list[i]
            if i == 0:
                output = rgb * a
            else:
                rgb_by_alpha = rgb * a
                output = rgb_by_alpha + output * (1 - a)
        return output

def MultiPlaneImagesAer2Grd_radius(signal, estimated_height, target_height, target_width, grd_height, max_height,
                                  method='column', geoout_type='image', dataset='CVUSA'):
    """
    Transform aerial images to street view using radius-based multi-plane images
    """
    PlaneNum = estimated_height.get_shape().as_list()[-1]
    batch, height, width, channel = tf_shape(signal, rank=4)

    if method == 'column':
        estimated_height = tf.cumsum(estimated_height, axis=-1)

    # Create voxel representation
    voxel = tf.transpose(tf.stack([signal]*PlaneNum, axis=-1), [0, 1, 2, 4, 3])
    voxel = tf.reshape(voxel, [batch, height, width, PlaneNum*channel])

    # Create cylinder coordinates
    S = signal.get_shape().as_list()[1]
    radius = int(S//4)
    azimuth = target_width

    i = np.arange(0, radius)
    j = np.arange(0, azimuth)
    jj, ii = np.meshgrid(j, i)

    # Calculate projection coordinates
    y = S / 2. - S / 2. / radius * (radius - 1 - ii) * np.sin(2 * np.pi * jj / azimuth)
    x = S / 2. + S / 2. / radius * (radius - 1 - ii) * np.cos(2 * np.pi * jj / azimuth)

    uv = np.stack([y, x], axis=-1)
    uv = uv.astype(np.float32)
    warp = tf.stack([uv] * batch, axis=0)

    # Project to cylindrical coordinates
    imgs = tfa.image.resampler(voxel, warp)
    imgs = tf.reshape(imgs, [batch, radius, azimuth, PlaneNum, channel])
    imgs = tf.transpose(imgs, [0, 3, 2, 1, 4])

    alpha = tfa.image.resampler(estimated_height, warp)
    alpha = tf.transpose(alpha, [0, 3, 2, 1])

    # Dataset-specific parameters
    if dataset == 'CVUSA':
        meters = 55
    elif dataset in ['CVACT', 'CVACThalf']:
        meters = (50 * 206 / 256)
    elif dataset == 'CVACTunaligned':
        meters = 50
    elif dataset == 'OP':
        meters = 100

    # Project to panorama coordinates
    if dataset in ['CVUSA', 'CVACThalf']:
        i = np.arange(0, target_height*2)
        j = np.arange(0, target_width)
        jj, ii = np.meshgrid(j, i)
        tanPhi = np.tan(ii / target_height / 2 * np.pi)
        tanPhi[np.where(tanPhi==0)] = 1e-16

        n = int(target_height//2)

        MetersPerRadius = meters / 2 / radius
        rgb_layers = []
        a_layers = []
        for r in range(0, radius):
            z = (radius-r-1)*MetersPerRadius/tanPhi[n:-n]
            z = (PlaneNum-1) - (z - grd_height)/(max_height - grd_height) * (PlaneNum-1)
            theta = jj[n:-n]
            uv = np.stack([theta, z], axis=-1)
            uv = uv.astype(np.float32)
            warp = tf.stack([uv] * batch, axis=0)
            rgb = tfa.image.resampler(imgs[..., r, :], warp)
            a = tfa.image.resampler(alpha[..., r:r+1], warp)
            rgb_layers.append(rgb)
            a_layers.append(a)
    else:
        # Similar process for other datasets with full height
        i = np.arange(0, target_height)
        j = np.arange(0, target_width)
        jj, ii = np.meshgrid(j, i)
        tanPhi = np.tan(ii / target_height * np.pi)
        tanPhi[np.where(tanPhi == 0)] = 1e-16

        MetersPerRadius = meters / 2 / radius
        rgb_layers = []
        a_layers = []
        for r in range(0, radius):
            z = (radius - r - 1) * MetersPerRadius / tanPhi
            z = (PlaneNum - 1) - (z - grd_height) / (max_height - grd_height) * (PlaneNum - 1)
            theta = jj
            uv = np.stack([theta, z], axis=-1)
            uv = uv.astype(np.float32)
            warp = tf.stack([uv] * batch, axis=0)
            rgb = tfa.image.resampler(imgs[..., r, :], warp)
            a = tfa.image.resampler(alpha[..., r:r + 1], warp)
            rgb_layers.append(rgb)
            a_layers.append(a)

    # Return based on output type
    if geoout_type == 'volume':
        return tf.concat([rgb_layers[i]*a_layers[i] for i in range(radius)], axis=-1)
    elif geoout_type == 'image':
        for i in range(radius):
            rgb = rgb_layers[i]
            a = a_layers[i]
            if i == 0:
                output = rgb * a
            else:
                rgb_by_alpha = rgb * a
                output = rgb_by_alpha + output * (1 - a)
        return output
#test geometry 
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import PIL.Image as Image
import tensorflow_addons as tfa

def test_geometry_projection(img_path, dataset='CVUSA'):
    """
    Test the geometry projection functionality
    
    Args:
        img_path: Path to test image
        dataset: Dataset name for scaling parameters
    """
    # Load and preprocess image
    img = np.asarray(Image.open(img_path), np.float32)[None, ...]  # [1, 256, 256, 3]
    signal = tf.constant(img)
    
    # Create dummy height map
    estimated_height = tf.concat([tf.zeros(signal.get_shape().as_list()[:-1] + [63]),
                                tf.ones(signal.get_shape().as_list()[:-1] + [1])], axis=-1)
    
    # Set projection parameters
    target_height = 512
    target_width = 512 * 2
    grd_height = -2 
    max_height = 32
    method = 'column'
    geoout_type = 'image'

    # Get dimensions
    PlaneNum = estimated_height.get_shape().as_list()[-1]
    batch, height, width, channel = signal.get_shape().as_list()

    # Cumulative sum for column method
    if method == 'column':
        estimated_height = tf.cumsum(estimated_height, axis=-1)

    # Create voxel representation
    voxel = tf.transpose(tf.stack([signal]*PlaneNum, axis=-1), [0, 1, 2, 4, 3])
    voxel = tf.reshape(voxel, [batch, height, width, PlaneNum*channel])

    # Setup cylindrical coordinates
    S = signal.get_shape().as_list()[1]
    radius = int(S//4)
    azimuth = target_width

    # Create coordinate grid
    i = np.arange(0, radius)
    j = np.arange(0, azimuth)
    jj, ii = np.meshgrid(j, i)

    # Calculate projection coordinates
    y = S / 2. - S / 2. / radius * (radius - 1 - ii) * np.sin(2 * np.pi * jj / azimuth)
    x = S / 2. + S / 2. / radius * (radius - 1 - ii) * np.cos(2 * np.pi * jj / azimuth)

    uv = np.stack([y, x], axis=-1)
    uv = uv.astype(np.float32)
    warp = tf.stack([uv] * batch, axis=0)

    # Project to cylindrical coordinates
    imgs = tfa.image.resampler(voxel, warp)
    imgs = tf.reshape(imgs, [batch, radius, azimuth, PlaneNum, channel])
    imgs = tf.transpose(imgs, [0, 3, 2, 1, 4])
    alpha = tfa.image.resampler(estimated_height, warp)
    alpha = tf.transpose(alpha, [0, 3, 2, 1])

    # Set dataset-specific parameters
    if dataset == 'CVUSA':
        meters = 55
    elif dataset in ['CVACT', 'CVACThalf']:
        meters = (50 * 206 / 256)
    elif dataset == 'CVACTunaligned':
        meters = 50
    elif dataset == 'OP':
        meters = 100

    # Project to panorama coordinates
    if dataset in ['CVUSA', 'CVACThalf']:
        i = np.arange(0, target_height*2)
        j = np.arange(0, target_width)
        jj, ii = np.meshgrid(j, i)
        tanPhi = np.tan(ii / target_height / 2 * np.pi)
        tanPhi[np.where(tanPhi==0)] = 1e-16

        n = int(target_height//2)
        MetersPerRadius = meters / 2 / radius
        rgb_layers = []
        a_layers = []
        
        for r in range(0, radius):
            z = (radius-r-1)*MetersPerRadius/tanPhi[n:-n]
            z = (PlaneNum-1) - (z - grd_height)/(max_height - grd_height) * (PlaneNum-1)
            theta = jj[n:-n]
            uv = np.stack([theta, z], axis=-1)
            uv = uv.astype(np.float32)
            warp = tf.stack([uv] * batch, axis=0)
            rgb = tfa.image.resampler(imgs[..., r, :], warp)
            a = tfa.image.resampler(alpha[..., r:r+1], warp)
            rgb_layers.append(rgb)
            a_layers.append(a)
    else:
        i = np.arange(0, target_height)
        j = np.arange(0, target_width)
        jj, ii = np.meshgrid(j, i)
        tanPhi = np.tan(ii / target_height * np.pi)
        tanPhi[np.where(tanPhi == 0)] = 1e-16

        MetersPerRadius = meters / 2 / radius
        rgb_layers = []
        a_layers = []
        
        for r in range(0, radius):
            z = (radius - r - 1) * MetersPerRadius / tanPhi
            z = (PlaneNum - 1) - (z - grd_height) / (max_height - grd_height) * (PlaneNum - 1)
            theta = jj
            uv = np.stack([theta, z], axis=-1)
            uv = uv.astype(np.float32)
            warp = tf.stack([uv] * batch, axis=0)
            rgb = tfa.image.resampler(imgs[..., r, :], warp)
            a = tfa.image.resampler(alpha[..., r:r + 1], warp)
            rgb_layers.append(rgb)
            a_layers.append(a)

    # Composite final image
    for i in range(radius):
        rgb = rgb_layers[i]
        a = a_layers[i]
        if i == 0:
            output = rgb * a
        else:
            rgb_by_alpha = rgb * a
            output = rgb_by_alpha + output * (1 - a)

    # Run session and save results
    with tf.Session() as sess:
        # Get intermediate results
        img0, alpha0 = sess.run([imgs, alpha])
        img1_list = sess.run(rgb_layers)
        alpha1_list = sess.run(a_layers)

        # Save intermediate stage
        iimg = Image.fromarray(img0[0,0].astype(np.uint8))
        iimg.save('stage1.png')
        
        # Save final output
        out_img = sess.run(output)
        iimg = Image.fromarray(out_img[0].astype(np.uint8))
        iimg.save('stage2.png')

if __name__ == "__main__":
    # Set CUDA device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Test with an example image
    test_img_path = "path/to/your/test/image.jpg"  # Replace with actual path
    test_geometry_projection(test_img_path, dataset='CVUSA')