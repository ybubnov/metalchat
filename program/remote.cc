// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <format>
#include <iostream>
#include <limits.h>
#include <unistd.h>

#include <keychain/keychain.h>

#include "remote.h"


namespace metalchat {
namespace program {


remote_command::remote_command(CLI::App& app)
: _M_add_options(),
  _M_remove_options()
{
    auto remote = app.add_subcommand("remote", "Manage list of remote servers");
    auto remote_add = remote->add_subcommand("add", "Add new remote servers");
    remote_add->add_option("hostname", _M_add_options.hostname, "Remote host name")->required();
    remote_add->add_option("--token", _M_add_options.token, "Remote trust token");
    remote_add->callback([&]() { add(); });

    auto remote_list = remote->add_subcommand("list", "List the available remotes");
    remote_list->callback([&]() { list(); });

    auto remote_remove = remote->add_subcommand("remove", "Remove remotes");
    remote_remove->add_option("hostname", _M_remove_options.hostname, "Remote host name to remove")
        ->required();
    remote_remove->callback([&]() { remove(); });
}


void
remote_command::add()
{
    std::cout << "add: hostname=" << _M_add_options.hostname;
    std::cout << ", token=" << _M_add_options.token.value_or("*****") << std::endl;
}


void
remote_command::list()
{}


void
remote_command::remove()
{}


remote::remote(const std::string& host)
: _M_host(host)
{}


void
remote::add(const std::string& token)
{
    auto namesize = sysconf(_SC_LOGIN_NAME_MAX);
    if (namesize == -1) {
        throw std::runtime_error("remote: failed querying login name size");
    }

    auto name = std::make_unique<char[]>(namesize);
    const auto err = getlogin_r(name.get(), namesize);
    if (err != 0) {
        throw std::runtime_error(std::format("remote: failed getting login name ({})", err));
    }

    const std::string package("org.metalchat.program");
    const std::string service(_M_host);
    const std::string username(name.get());

    keychain::Error kerr;
    keychain::setPassword(package, service, username, token, kerr);

    if (kerr) {
        throw std::runtime_error(std::format("remote: failed saving access token, {}", kerr.message)
        );
    }
}


} // namespace program
} // namespace metalchat
