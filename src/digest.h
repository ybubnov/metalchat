// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <concepts>
#include <iomanip>
#include <memory>
#include <sstream>

#include <openssl/evp.h>


namespace metalchat {
namespace runtime {


template <typename T>
concept string_convertible = requires(std::remove_reference_t<T> const t) {
    { t.string() } -> std::same_as<std::string>;
};


template <string_convertible T>
std::string
sha1(const T& t)
{
    std::shared_ptr<EVP_MD_CTX> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context) {
        throw std::runtime_error("failed allocating sha1");
    }

    if (auto ok = EVP_DigestInit_ex(context.get(), EVP_sha1(), nullptr); !ok) {
        throw std::runtime_error("failed initializing sha1");
    }

    const auto data = t.string();
    if (auto ok = EVP_DigestUpdate(context.get(), data.data(), data.size()); !ok) {
        throw std::runtime_error("failed updating sha1");
    }

    unsigned char digest[EVP_MAX_MD_SIZE]{0};
    unsigned int size = 0;

    if (auto ok = EVP_DigestFinal_ex(context.get(), digest, &size); !ok) {
        throw std::runtime_error("failed finalizing sha1");
    }

    std::stringstream hex;
    for (std::size_t i = 0; i < size; i++) {
        hex << std::hex << std::setw(2) << std::setfill('0');
        hex << static_cast<int>(digest[i]);
    }

    return hex.str();
}


} // namespace runtime
} // namespace metalchat
