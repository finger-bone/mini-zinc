__kernel void upsample_nearest(
    __global float* input,
    __global float* output,
    int batch_size,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w
) {
    int pos = get_global_id(0);
    int ow = pos % out_w;
    int oh = (pos / out_w) % out_h;
    int oc = (pos / (out_w * out_h)) % channels;
    int on = pos / (out_w * out_h * channels);
    if (on >= batch_size || oc >= channels || oh >= out_h || ow >= out_w) {
        return;
    }
    float scale_h = (float)out_h / (float)in_h;
    float scale_w = (float)out_w / (float)in_w;
    int ih = (int)floor((float)oh / scale_h);
    int iw = (int)floor((float)ow / scale_w);
    ih = min(ih, in_h - 1);
    iw = min(iw, in_w - 1);
    int input_idx = ((on * channels + oc) * in_h + ih) * in_w + iw;
    int output_idx = ((on * channels + oc) * out_h + oh) * out_w + ow;
    output[output_idx] = input[input_idx];
}