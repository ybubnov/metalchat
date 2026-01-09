// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include "command.h"


namespace metalchat {
namespace internal {


struct credential {
    std::string protocol;
    std::string hostname;
    std::string username;
    std::string secret;

    std::string
    url() const
    {
        return protocol + "://" + hostname;
    }
};


/// A credential repository that keeps secrets in Keychain Access.
///
/// The repository uses a configuration to store credentials parameters, like protocol,
/// hostname, etc. but stores credential secret in Keychain Access.
class keychain_provider {
public:
    keychain_provider();
    keychain_provider(const std::string& package);

    /// Store the credential into the Keychain Access.
    ///
    /// The method queries OS user that launched a program to store the credential.
    void
    store(const std::string& url, const std::string& secret) const;

    /// Load the secret from the Keychain Access.
    ///
    /// The method queries OS user that launched a program and load a secret from the
    /// Keychain Access repository for the specified URL.
    std::string
    load(const std::string& url) const;

    /// Remove the secret store in the Keychain Access.
    ///
    /// Method does not throw errors, when key is missing from the Keychain Access.
    void
    remove(const std::string& url) const;

private:
    std::string
    system_username() const;

    std::string _M_package;
};


class credential_command : public basic_command {
public:
    credential_command(basic_command& parent);

    void
    add(const command_context&);

    void
    list(const command_context&);

    void
    remove(const command_context&);

private:
    parser_type _M_add;
    parser_type _M_list;
    parser_type _M_remove;

    credential _M_credential;
};


} // namespace internal
} // namespace metalchat
