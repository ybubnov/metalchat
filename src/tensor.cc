// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metalchat/tensor/basic.h>


namespace metalchat {


void
basic_tensor::deduce_view_sizes(const int* begin, std::size_t count, std::size_t* result) const
{
    std::size_t this_numel = numel();
    std::size_t view_numel = 1;

    auto deduced_size = this_numel;
    auto deduced_dim = -1;

    for (std::size_t i = 0; i < count; i++) {
        auto size = *begin;

        if (size == -1) {
            if (deduced_dim >= 0) {
                throw std::invalid_argument(std::format(
                    "basic_tensor::view: only one size can be deduced, both sizes at positions {} "
                    "and {} are unspecified",
                    deduced_dim, i
                ));
            }
            deduced_dim = i;
        } else {
            result[i] = static_cast<std::size_t>(size);
            view_numel *= static_cast<std::size_t>(size);
            deduced_size = deduced_size / size;
        }
        begin++;
    }
    if (deduced_dim >= 0) {
        result[deduced_dim] = deduced_size;
        view_numel *= deduced_size;
    }

    if (view_numel != this_numel) {
        throw std::invalid_argument(std::format(
            "basic_tensor::view: resulting numel is not the same as tensor numel {} != {}",
            view_numel, this_numel
        ));
    }
}


void
basic_tensor::deduce_view_strides(const std::size_t* sizes, std::size_t count, std::size_t* strides)
    const
{
    std::size_t this_numel = 1;
    std::size_t view_numel = 1;

    int view_i = count - 1;
    auto base_stride = stride(dimensions() - 1);

    for (int i = dimensions() - 1; i >= 0; i--) {
        this_numel *= size(i);

        // When tensor stride is not equal to the "default" stride (which could happen
        // in case of slicing or narrowing a tensor), try computing new strides according
        // to the layout of the original tensor.
        //
        // A new shape of a view might break the contiguous layout of memory, in this
        // case throw an `invalid_argument` exception to the caller.
        if (i == 0 || stride(i - 1) != this_numel * base_stride) {
            while (view_i >= 0 && (view_numel < this_numel || sizes[view_i] == 1)) {
                strides[view_i] = view_numel * base_stride;
                view_numel *= sizes[view_i];
                view_i--;
            }

            if (view_numel != this_numel) {
                throw std::invalid_argument(std::format(
                    ("tensor::view: shape is invalid for input of size {}, "
                     "consider copying the tensor"),
                    numel()
                ));
            }

            if (i > 0) {
                base_stride = stride(i - 1);
                this_numel = 1;
                view_numel = 1;
            }
        }
    }
}


} // namespace metalchat
