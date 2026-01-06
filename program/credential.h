// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "command.h"


namespace metalchat {
namespace program {


struct credential {
    std::string protocol;
    std::string hostname;
    std::string username;
    std::string credential;
};


/// A credential repository that keeps secrets in Keychain Access.
///
/// The repository uses a configuration to store credentials parameters, like protocol,
/// hostname, etc. but stores credential secret in Keychain Access.
class credential_repository {
public:
    credential_repository();
    credential_repository(const std::string& package);

    /// Store the credential into the Keychain Access.
    ///
    /// The method queries OS user that launched a program to store the credential.
    /// The `OutputIt` is updated with the instances of \ref credential type.
    void
    store(const credential& cred) const;

    template <typename OutputIt, typename UnaryPred>
    void
    load_if(OutputIt output, UnaryPred pred) const
    {}

private:
    std::string
    username() const;

    std::string _M_package;
};


class credential_command : public basic_command {
public:
    credential_command(basic_command& parent);

    void
    add();

    void
    list();

    void
    remove();

private:
    parser_type _M_add;
    parser_type _M_list;
    parser_type _M_remove;

    credential _M_credential;
};


} // namespace program
} // namespace metalchat
