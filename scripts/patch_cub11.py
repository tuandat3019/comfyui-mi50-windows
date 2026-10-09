"""Backport CuMesh's in-place CUB scan calls to the CUDA 11 API."""
from pathlib import Path
import os
import re

root = Path(os.environ['MI50_RUNTIME']) / 'trellis/sources/CuMesh/src'
changed = 0
for path in [*root.rglob('*.cu'), *root.rglob('*.h')]:
    original = path.with_suffix(path.suffix + '.original-mi50')
    if not original.exists():
        original.write_bytes(path.read_bytes())
    text = original.read_text(encoding='utf-8')
    count = [0]
    def backport(match):
        args = match.group(2).split(',')
        if len(args) != 4:
            return match.group(0)
        args.insert(3, args[2])
        count[0] += 1
        return match.group(1) + '(' + ','.join(args) + ')'
    text = re.sub(r'(cub::DeviceScan::(?:ExclusiveSum|InclusiveSum))\(([^()]*)\)', backport, text)
    if path.name == 'clean_up.cu':
        text = re.sub(r'struct int3_decomposer\s*\{.*?\n\};', '''// CUDA 11 CUB has no custom-key decomposer. Stable scalar radix passes
// z, y, x produce the same lexicographic order without Thrust merge sort.
static __global__ void duplicate_face_key_kernel(
    const int3* faces, const int* indices, const size_t F, const int axis, int* keys
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    const int3 f = faces[indices[tid]];
    keys[tid] = axis == 0 ? f.x : (axis == 1 ? f.y : f.z);
}

static __global__ void duplicate_face_gather_kernel(
    const int3* faces, const int* indices, const size_t F, int3* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < F) output[tid] = faces[indices[tid]];
}''', text, flags=re.S)
        pattern = r'    CUDA_CHECK\(cub::DeviceRadixSort::SortPairs\(\s*nullptr, temp_storage_bytes,\s*cu_sorted_faces, cu_sorted_faces_output,.*?int3_decomposer\{\}\s*\)\);\s*this->cub_temp_storage.resize\(temp_storage_bytes\);\s*CUDA_CHECK\(cub::DeviceRadixSort::SortPairs\(.*?int3_decomposer\{\}\s*\)\);'
        text, sorts = re.subn(pattern, '''    int *cu_scalar_keys, *cu_scalar_keys_output;
    CUDA_CHECK(cudaMalloc(&cu_scalar_keys, F * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_scalar_keys_output, F * sizeof(int)));
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes, cu_scalar_keys, cu_scalar_keys_output,
        cu_sorted_face_indices, cu_sorted_indices_output, F
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    for (int axis = 2; axis >= 0; --axis) {
        duplicate_face_key_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            cu_sorted_faces, cu_sorted_face_indices, F, axis, cu_scalar_keys
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            this->cub_temp_storage.ptr, temp_storage_bytes,
            cu_scalar_keys, cu_scalar_keys_output,
            cu_sorted_face_indices, cu_sorted_indices_output, F
        ));
        if (axis > 0) CUDA_CHECK(cudaMemcpy(cu_sorted_face_indices, cu_sorted_indices_output,
                                           F * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    duplicate_face_gather_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_sorted_faces, cu_sorted_indices_output, F, cu_sorted_faces_output
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_scalar_keys));
    CUDA_CHECK(cudaFree(cu_scalar_keys_output));''', text, flags=re.S)
        assert sorts == 1, sorts
        count[0] += sorts
    if count[0]:
        encoded = text.encode('utf-8')
        if path.read_bytes() != encoded:
            path.write_bytes(encoded)
        print(path.name, count[0])
        changed += count[0]
print('CUB11_INPLACE_SCANS_BACKPORTED', changed)
