// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <algorithm>
#include <iterator>

#include <metalchat/huggingface.h>
#include <metalchat/repository.h>

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
  _M_type()
{
    _M_command.add_description("manage model run options");
    _M_command.add_argument("--local").help("use a current working directory manifest").flag();
    _M_command.add_argument("--global").help("use a global manifest").flag();

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
        .help(("augment the output of all queried options with"
               "the scope of that value (local, model)"))
        .flag();
    push_handler(_M_list, [&](const command_context& c) { list(c); });
}


tomlfile<manifest>
options_command::resolve_manifest(const command_context& context) const
{
    return context.resolve_manifest(resolve_scope(_M_command));
}


void
options_command::get(const command_context& context) const
{
    auto manifest_file = resolve_manifest(context);
    auto manifest = manifest_file.read();

    if (auto option = manifest.get_option(_M_name); option) {
        std::visit([](auto&& value) {
            std::cout << std::boolalpha << value << std::endl;
        }, option.value());
        return;
    }

    // Throw an exception with an empty error string, so that the program only
    // returns a non-zero status code without printing any error information.
    throw std::invalid_argument("");
}


void
options_command::set(const command_context& context) const
{
    using option_value = manifest::option_value;
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
    auto manifest_file = resolve_manifest(context);
    auto manifest = manifest_file.read();

    manifest.set_option(_M_name, value);
    manifest_file.write(manifest);
}

void
options_command::unset(const command_context& context) const
{
    auto manifest_file = resolve_manifest(context);
    auto manifest = manifest_file.read();

    manifest.unset_option(_M_name);
    manifest_file.write(manifest);
}


void
options_command::list(const command_context& context) const
{
    using Transformer = huggingface::llama3;
    using TransformerTraits = transformer_traits<Transformer>;

    auto manifest_file = resolve_manifest(context);
    auto scope = resolve_scope(_M_command);
    auto scope_manifest = manifest_file.read();
    auto scope_options = scope_manifest.options.value_or(manifest::options_section());

    model_provider models(context.root_path);
    auto model = models.find(scope_manifest.id());

    auto repository = filesystem_repository<Transformer>(model.path);
    auto options = repository.retrieve_options();

    if (!scope_options.empty()) {
        auto first = scope_options.begin();
        auto last = scope_options.end();
        options = TransformerTraits::merge_options(first, last, options);
    }

    using options_type = std::pair<std::string, std::string>;
    std::vector<options_type> runtime_options;
    TransformerTraits::iter_options(options, std::back_inserter(runtime_options));

    auto less = [](options_type o1, options_type o2) { return o1.first < o2.first; };
    std::sort(runtime_options.begin(), runtime_options.end(), less);

    bool use_show_scope = _M_list.get<bool>("--show-scope");
    for (const auto& [key, value] : runtime_options) {
        if (use_show_scope) {
            auto scope_name = context_scope::string(scope);
            if (!scope_options.contains(key)) {
                scope_name = context_scope::string(context_scope::model);
            }
            std::cout << scope_name << "  ";
        }
        std::cout << key << "=" << value << std::endl;
    }
}


} // namespace runtime
} // namespace metalchat
