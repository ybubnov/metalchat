// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <format>
#include <iostream>
#include <limits.h>
#include <regex>
#include <unistd.h>

#include <ada.h>
#include <keychain/keychain.h>

#include "credential.h"


namespace metalchat {
namespace program {


credential_repository::credential_repository(const std::string& package)
: _M_package(package)
{}


credential_repository::credential_repository()
: credential_repository("org.metalchat.program")
{}


std::string
credential_repository::username() const
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
credential_repository::store(const credential& cred) const
{
    keychain::Error kerr;
    keychain::setPassword(_M_package, cred.url(), username(), cred.credential, kerr);

    if (kerr) {
        throw std::runtime_error(
            std::format("credential: failed saving credential, {}", kerr.message)
        );
    }
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
        .default_value(std::string("https"))
        .choices("https")
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
    _M_add.add_argument("-c", "--credential")
        .help("the pre-encoded credential, suitable for protocol")
        .metavar("<credential>")
        .required()
        .store_into(_M_credential.credential);

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
    auto config = context.config_file.read();
    auto credential =
        credential_config{.username = _M_credential.username, .provider = "@keychain"};

    config.push_credential(_M_credential.url(), credential);
    context.config_file.write(config);
}


void
credential_command::list(const command_context& context)
{
    auto config = context.config_file.read();
    if (config.credential.has_value()) {
        std::size_t url_size = 0;
        std::size_t username_size = 0;

        for (const auto& [url, c] : config.credential.value()) {
            url_size = std::max(url_size, url.size());
            username_size = std::max(username_size, c.username.size());
        }
        for (const auto& [url, c] : config.credential.value()) {
            std::cout << std::left;
            std::cout << std::setw(url_size) << url << ' ';
            std::cout << std::setw(username_size) << c.username << ' ';
            std::cout << c.provider << std::endl;
        }
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

        auto url = ada::parse(cred_url);
        if (!_M_credential.protocol.empty()) {
            expect |= 1;
            actual |= (url->get_protocol() == (_M_credential.protocol + ":"));
        }
        if (!_M_credential.hostname.empty()) {
            expect |= 1 << 1;
            actual |= ((url->get_hostname() == _M_credential.hostname) << 1);
        }

        if ((actual ^ expect).none()) {
            candidates.push_back(cred_url);
        }
    }

    if (!candidates.empty()) {
        for (const auto& cred_url : candidates) {
            config.pop_credential(cred_url);
        }
        context.config_file.write(config);
    }
}


} // namespace program
} // namespace metalchat
