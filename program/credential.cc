// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <format>
#include <iostream>
#include <limits.h>
#include <unistd.h>

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
    const auto service = cred.protocol + "://" + cred.hostname;

    keychain::Error kerr;
    keychain::setPassword(_M_package, service, username(), cred.credential, kerr);

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
        .metavar("PROTOCOL")
        .default_value(std::string("https"))
        .choices("https")
        .store_into(_M_credential.protocol);
    _M_add.add_argument("-H", "--hostname")
        .help("the remote hostname for a network credential")
        .metavar("HOSTNAME")
        .required()
        .store_into(_M_credential.hostname);
    _M_add.add_argument("-u", "--username")
        .help("the credential's username")
        .metavar("USERNAME")
        .required()
        .store_into(_M_credential.username);
    _M_add.add_argument("-c", "--credential")
        .help("the pre-encoded credential, suitable for protocol")
        .metavar("CREDENTIAL")
        .required()
        .store_into(_M_credential.credential);

    _M_list.add_description("list the available credentials");

    _M_remove.add_description("remove any stored matching credentials");
    _M_remove.add_argument("-p", "--protocol")
        .help("the protocol over which the credential will be used")
        .metavar("PROTOCOL")
        .store_into(_M_credential.protocol);
    _M_remove.add_argument("-H", "--hostname")
        .help("a remote hostname to matching the credentials")
        .metavar("HOSTNAME")
        .store_into(_M_credential.hostname);

    push_handler(_M_add, [&] { add(); });
    push_handler(_M_list, [&] { list(); });
    push_handler(_M_remove, [&] { remove(); });
}


void
credential_command::add()
{
    credential_repository repository;

    std::cout << "add: hostname=" << _M_credential.hostname;
    std::cout << ", token=" << _M_credential.credential << std::endl;
}


void
credential_command::list()
{}


void
credential_command::remove()
{}


} // namespace program
} // namespace metalchat
