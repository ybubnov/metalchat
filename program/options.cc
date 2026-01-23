// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "options.h"
#include "manifest.h"
#include "model.h"


namespace metalchat {
namespace runtime {


options_command::options_command(basic_command& parent)
: basic_command("options", parent),
  _M_get("get"),
  _M_set("set"),
  _M_unset("unset")
{
    _M_command.add_description("manage model run options");

    _M_get.add_description("query model run options");
    _M_get.add_argument("id").help("a model identifier").required().store_into(_M_id);
    _M_get.add_argument("name")
        .help("name of the option to query")
        .metavar("<name>")
        .store_into(_M_name)
        .required()
        .nargs(1);
    push_handler(_M_get, [&](const command_context& c) { get(c); });

    _M_set.add_description("change model run options");
    _M_set.add_argument("id").help("a model identifier").required().store_into(_M_id);
    _M_set.add_argument("name")
        .help("name of the option to change")
        .metavar("<name>")
        .store_into(_M_name)
        .required()
        .nargs(1);
    _M_set.add_argument("value")
        .help("value of the target option")
        .metavar("<value>")
        .store_into(_M_value)
        .required()
        .nargs(1);
    push_handler(_M_set, [&](const command_context& c) { set(c); });

    _M_unset.add_description("unset model run options");
    _M_unset.add_argument("id").help("a model identifier").required().store_into(_M_id);
    _M_unset.add_argument("name")
        .help("name of the option to remove")
        .metavar("<name>")
        .store_into(_M_name)
        .required()
        .nargs(1);
    push_handler(_M_unset, [&](const command_context& c) { unset(c); });
}


void
options_command::get(const command_context& context) const
{
    model_provider models(context.root_path);
    auto model = models.find(_M_id);

    if (auto value = model.manifest.get_option(_M_name); value) {
        std::cout << (*value) << std::endl;
        return;
    }

    // Throw an exception with an empty error string, so that the program only
    // returns a non-zero status code without printing any error information.
    throw std::invalid_argument("");
}


void
options_command::set(const command_context& context) const
{
    model_provider models(context.root_path);

    // TODO: ensure that option is supported by the model.
    auto model = models.find(_M_id);
    model.manifest.set_option(_M_name, _M_value);

    models.update(model);
}

void
options_command::unset(const command_context& context) const
{
    model_provider models(context.root_path);

    auto model = models.find(_M_id);
    model.manifest.unset_option(_M_name);

    models.update(model);
}


} // namespace runtime
} // namespace metalchat
