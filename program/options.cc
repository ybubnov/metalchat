// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <algorithm>
#include <iterator>

#include <metalchat/huggingface.h>
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
  _M_type()
{
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

    using Transformer = huggingface::llama3;
    using TransformerTraits = transformer_traits<Transformer>;

    std::optional<std::string> option_value;
    auto option_iterator = function_output_iterator([&](auto option) {
        if (option.first == _M_name) {
            option_value = option.second;
        }
    });

    scoped_repository_adapter<Transformer> repo(model.path, manifest);
    TransformerTraits::iter_options(repo.retrieve_options(), option_iterator);

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

    using options_type = std::tuple<std::string, std::string, std::string>;
    std::vector<options_type> runtime_options;

    // Insert the options into the runtime options container, so that it is possible
    // to sort the values in a container and print options sorted by scope.
    auto back_inserter = function_output_iterator([&](auto option) {
        auto option_scope_name = context_scope::string(scope);
        if (!manifest.get_option(option.first)) {
            option_scope_name = context_scope::string(context_scope::model);
        }
        runtime_options.emplace_back(option_scope_name, option.first, option.second);
    });

    using Transformer = huggingface::llama3;
    using TransformerTraits = transformer_traits<Transformer>;

    scoped_repository_adapter<Transformer> repo(model.path, manifest);
    TransformerTraits::iter_options(repo.retrieve_options(), back_inserter);

    auto less = [](options_type o1, options_type o2) {
        const auto& [scope1, key1, value1] = o1;
        const auto& [scope2, key2, value2] = o2;
        if (scope1 == scope2) {
            return key1 < key2;
        }
        return scope1 < scope2;
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
