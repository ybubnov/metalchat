// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <algorithm>
#include <iterator>

#include <metalchat/huggingface/llama.h>
#include <metalchat/repository.h>

#include "iterator.h"
#include "manifest.h"
#include "model.h"
#include "options.h"


namespace metalchat {
namespace runtime {


std::string optionkind::integer = "int";
std::string optionkind::boolean = "bool";
std::string optionkind::floating = "float";
std::string optionkind::string = "str";


bool
stob(const std::string& s)
{
    bool value{};
    std::istringstream(s) >> std::boolalpha >> value;
    return value;
}


options_command::options_command(basic_command& parent)
: basic_command("options", parent),
  _M_get("get"),
  _M_set("set"),
  _M_unset("unset"),
  _M_list("list"),
  _M_name(),
  _M_value(),
  _M_type(),
  _M_option_getters(),
  _M_option_listers()
{
    constexpr auto arch_size = std::tuple_size_v<decltype(supported_architectures)>;
    register_supported_architechtures(std::make_index_sequence<arch_size>{});

    add_scope_arguments(_M_command);
    _M_command.add_description("manage model run options");

    _M_get.add_description("query model run options");
    _M_get.add_argument("name")
        .help("name of the option to query")
        .store_into(_M_name)
        .required()
        .nargs(1);
    push_handler(_M_get, [&](const command_context& c) { get(c); });

    _M_set.add_description("change model run options");
    _M_set.add_argument("name")
        .help("name of the option to change")
        .store_into(_M_name)
        .required()
        .nargs(1);
    _M_set.add_argument("value")
        .help("value of the target option")
        .store_into(_M_value)
        .required()
        .nargs(1);
    _M_set.add_argument("-t", "--type")
        .help("type of the target option")
        .metavar("<type>")
        .choices(optionkind::boolean, optionkind::integer, optionkind::floating, optionkind::string)
        .store_into(_M_type)
        .required()
        .nargs(1);
    push_handler(_M_set, [&](const command_context& c) { set(c); });

    _M_unset.add_description("unset model run options");
    _M_unset.add_argument("name")
        .help("name of the option to remove")
        .store_into(_M_name)
        .required()
        .nargs(1);
    push_handler(_M_unset, [&](const command_context& c) { unset(c); });

    _M_list.add_description("list model run options");
    _M_list.add_argument("--show-scope")
        .help(("augment the output of all queried options with\n"
               "the scope of that value (global, local, model)"))
        .flag();
    push_handler(_M_list, [&](const command_context& c) { list(c); });
}


void
options_command::get(const command_context& context) const
{
    model_provider models(context.root_path);

    auto manifest = resolve_manifest(context, _M_command).read();
    auto model = models.find(manifest.id());
    model_info scoped_model{.manifest = manifest, .path = model.path};

    auto& option_getter = _M_option_getters.at(manifest.model.architecture);
    auto option_value = option_getter(scoped_model, _M_name);

    if (option_value) {
        std::cout << option_value.value() << std::endl;
        return;
    }

    // Throw an exception with an empty error string, so that the program only
    // returns a non-zero status code without printing any error information.
    throw std::invalid_argument("");
}


void
options_command::set(const command_context& context) const
{
    using K = std::string;
    using V = std::function<option_value(const std::string&)>;

    auto converters = std::unordered_map<K, V>({
        {optionkind::boolean, [](const std::string& s) { return stob(s); }},
        {optionkind::integer, [](const std::string& s) { return std::stoi(s); }},
        {optionkind::floating, [](const std::string& s) { return std::stof(s); }},
        {optionkind::string, [](const std::string& s) { return s; }},
    });

    auto& from_string = converters[_M_type];
    auto value = from_string(_M_value);

    // TODO: ensure that option is supported by the model.
    auto manifest_file = resolve_manifest(context, _M_command);
    auto manifest = manifest_file.read();

    manifest.set_option(_M_name, value);
    manifest_file.write(manifest);
}

void
options_command::unset(const command_context& context) const
{
    auto manifest_file = resolve_manifest(context, _M_command);
    auto manifest = manifest_file.read();

    manifest.unset_option(_M_name);
    manifest_file.write(manifest);
}


void
options_command::list(const command_context& context) const
{
    model_provider models(context.root_path);

    auto manifest = resolve_manifest(context, _M_command).read();
    auto model = models.find(manifest.id());
    auto scope = resolve_scope(_M_command);

    model_info scoped_model{.manifest = manifest, .path = model.path};
    auto& option_lister = _M_option_listers.at(manifest.model.architecture);

    std::vector<option> runtime_options;
    option_lister(scoped_model, scope, runtime_options);

    auto less = [](option o1, option o2) {
        if (o1.scope == o2.scope) {
            return o1.name < o2.name;
        }
        return o1.scope < o2.scope;
    };

    std::sort(runtime_options.begin(), runtime_options.end(), less);

    bool use_show_scope = _M_list.get<bool>("--show-scope");
    for (const auto& [scope, key, value] : runtime_options) {
        if (use_show_scope) {
            std::cout << scope << "  ";
        }
        std::cout << key << "=" << value << std::endl;
    }
}


} // namespace runtime
} // namespace metalchat
