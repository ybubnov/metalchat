// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <format>
#include <iostream>
#include <unistd.h>

#include <keychain/keychain.h>

#include "credential.h"
#include "http.h"


namespace metalchat {
namespace runtime {


keychain_provider::keychain_provider(const std::string& package)
: _M_package(package)
{}


keychain_provider::keychain_provider()
: keychain_provider("org.metalchat.runtime")
{}


std::string
keychain_provider::system_username() const
{
    auto namesize = sysconf(_SC_LOGIN_NAME_MAX);
    if (namesize == -1) {
        throw std::runtime_error("credential: failed querying login name size");
    }

    auto name = std::make_unique<char[]>(namesize);
    const auto err = getlogin_r(name.get(), namesize);
    if (err != 0) {
        throw std::runtime_error(std::format("credential: failed getting login name ({})", err));
    }

    return std::string(name.get(), name.get() + namesize);
}


void
keychain_provider::store(const std::string& url, const std::string& secret) const
{
    keychain::Error kerr;
    keychain::setPassword(_M_package, url, system_username(), secret, kerr);

    if (kerr) {
        throw std::runtime_error(
            std::format("credential: failed saving credential, {}", kerr.message)
        );
    }
}


std::string
keychain_provider::load(const std::string& url) const
{
    keychain::Error kerr;
    auto secret = keychain::getPassword(_M_package, url, system_username(), kerr);
    if (kerr) {
        throw std::runtime_error(
            std::format("keychain: failed retrieving credential, {}", kerr.message)
        );
    }

    return secret;
}


void
keychain_provider::remove(const std::string& url) const
{
    // Ignore deletion errors.
    keychain::Error kerr;
    keychain::deletePassword(_M_package, url, system_username(), kerr);
}


credential_command::credential_command(basic_command& parent)
: basic_command("credential", parent),
  _M_add("add"),
  _M_list("list"),
  _M_remove("remove"),
  _M_credential()
{
    _M_command.add_description("retrieve and store user credentials");

    _M_add.add_description("add new credentials for a host");
    _M_add.add_argument("-p", "--protocol")
        .help("the protocol over which the credential will be used")
        .metavar("<protocol>")
        .choices("https")
        .default_value(std::string("https"))
        .nargs(1)
        .store_into(_M_credential.protocol);
    _M_add.add_argument("-H", "--hostname")
        .help("the remote hostname for a network credential")
        .metavar("<hostname>")
        .required()
        .store_into(_M_credential.hostname);
    _M_add.add_argument("-u", "--username")
        .help("the credential's username")
        .metavar("<username>")
        .required()
        .store_into(_M_credential.username);
    _M_add.add_argument("-s", "--secret")
        .help("the pre-encoded credential, suitable for protocol")
        .metavar("<secret>")
        .required()
        .store_into(_M_credential.secret);

    _M_list.add_description("list the available credentials");

    _M_remove.add_description("remove any stored matching credentials");
    _M_remove.add_argument("-p", "--protocol")
        .help("the protocol over which the credential will be used")
        .metavar("<protocol>")
        .store_into(_M_credential.protocol);
    _M_remove.add_argument("-H", "--hostname")
        .help("a remote hostname to matching the credentials")
        .metavar("<hostname>")
        .store_into(_M_credential.hostname);

    push_handler(_M_add, [&](const command_context& c) { add(c); });
    push_handler(_M_list, [&](const command_context& c) { list(c); });
    push_handler(_M_remove, [&](const command_context& c) { remove(c); });
}


void
credential_command::add(const command_context& context)
{
    credential_config credential = {.username = _M_credential.username, .provider = "@keychain"};

    keychain_provider provider;
    auto config = context.config_file.read();
    auto url = _M_credential.url();

    config.push_credential(url, credential);
    provider.store(url, _M_credential.secret);

    context.config_file.write(config);
}


void
credential_command::list(const command_context& context)
{
    auto config = context.config_file.read();
    if (!config.credential.has_value()) {
        return;
    }

    std::size_t url_size = 0;
    std::size_t username_size = 0;

    for (const auto& [url, c] : config.credential.value()) {
        url_size = std::max(url_size, url.size());
        username_size = std::max(username_size, c.username.size());
    }
    for (const auto& [url, c] : config.credential.value()) {
        std::cout << std::left;
        std::cout << std::setw(url_size) << url << '\t';
        std::cout << std::setw(username_size) << c.username << '\t';
        std::cout << c.provider << std::endl;
    }
}


void
credential_command::remove(const command_context& context)
{
    auto config = context.config_file.read();
    if (!config.credential.has_value()) {
        return;
    }

    std::vector<std::string> candidates;

    for (const auto& [cred_url, c] : config.credential.value()) {
        std::bitset<8> expect = 0;
        std::bitset<8> actual = 0;

        auto u = url(cred_url);
        if (!_M_credential.protocol.empty()) {
            expect |= 1;
            actual |= (u.protocol() == (_M_credential.protocol));
        }
        if (!_M_credential.hostname.empty()) {
            expect |= 1 << 1;
            actual |= ((u.host() == _M_credential.hostname) << 1);
        }

        if ((actual ^ expect).none()) {
            candidates.push_back(cred_url);
        }
    }

    if (!candidates.empty()) {
        keychain_provider provider;
        for (const auto& cred_url : candidates) {
            config.pop_credential(cred_url);
            provider.remove(cred_url);
        }
        context.config_file.write(config);
    }
}


} // namespace runtime
} // namespace metalchat
