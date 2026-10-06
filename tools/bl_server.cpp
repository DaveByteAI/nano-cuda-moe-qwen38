// An OpenAI-compatible HTTP server (one model, one request at a time):
//   GET  /v1/models
//   POST /v1/chat/completions   messages, stream, temperature, top_p, top_k, max_tokens / max_completion_tokens, seed,
//                               reasoning_effort, chat_template_kwargs.enable_thinking; non-standard: spec (MTP drafts)
//   GET  /health
// Thinking is off unless the request asks (enable_thinking: true or a reasoning_effort); the thinking comes back as
// reasoning_content. The engine keeps the conversation between requests: a request that continues it (the same
// messages plus new ones) only feeds the new part.
//   bl-server --model SHARD1.gguf [--host 127.0.0.1] [--port 8080] [--ctx 32768] [--name NAME] [--web DIR|none]
//             [--expert-cache FILE] [--gpus 0,1 [--layer-split K|auto]]
// The chat page (web/index.html) is served at /. With --expert-cache, the expert cache (which experts sit in VRAM) is
// saved after every reply and restored at the next start. Off by default: on new text it did not raise the hit rate
// (88.4% restored vs 89.0% from the profile, 2026-10-05).
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>

#include "bl/chat.h"
#include "httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

namespace {

std::string content_text(const json & c) {   // a message's content: a string, or parts of which the text ones count
    if (c.is_string()) return c.get<std::string>();
    std::string s;
    if (c.is_array())
        for (const auto & part : c)
            if (part.is_object() && part.value("type", "") == "text") s += part.value("text", "");
    return s;
}

json error_json(const std::string & msg, const std::string & type = "invalid_request_error") {
    return json{{"error", {{"message", msg}, {"type", type}}}};
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model, host = "127.0.0.1", name = "qwen3.8-flash-iq3_xxs", web, cache_file, gpus, layer_split;
    int port = 8080, ctx = 32768;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") model = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::stoi(next());
        else if (a == "--ctx") ctx = std::stoi(next());
        else if (a == "--name") name = next();
        else if (a == "--web") web = next();   // the chat page's directory (default: ../web next to the executable)
        else if (a == "--expert-cache") cache_file = next();
        else if (a == "--gpus") gpus = next();                 // several GPUs: the layers split between them
        else if (a == "--layer-split") layer_split = next();   // the first layer of each later GPU, or auto
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model.empty()) { std::fprintf(stderr, "usage: %s --model SHARD1.gguf [--host H] [--port P] [--ctx N]\n", argv[0]); return 2; }

    std::fprintf(stderr, "loading %s ...\n", model.c_str());
    bl::Text text(model);
    bl::GpuSplit split;
    try { split = bl::parse_gpu_split(gpus, layer_split); }
    catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 2; }
    bl::Engine eng(model, ctx, cache_file, split);
    auto save_cache = [&] {   // after a reply (under `busy`): the next start begins with this cache
        if (cache_file.empty()) return;
        try { eng.save_cache(cache_file); } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); }
    };
    bl::Chat chat(eng, text, ctx);
    std::mutex busy;   // one generation at a time
    std::atomic<long> seq{0};

    httplib::Server srv;
    if (web.empty()) {   // ../web relative to this executable (build/bl-server -> web/)
        char exe[4096];
        const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        if (n > 0) {
            std::string d(exe, static_cast<size_t>(n));
            d = d.substr(0, d.rfind('/'));
            web = d.substr(0, d.rfind('/')) + "/web";
        }
    }
    if (web != "none" && srv.set_mount_point("/", web)) std::fprintf(stderr, "chat page: %s\n", web.c_str());
    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"object", "list"}, {"data", json::array({{{"id", name}, {"object", "model"}, {"owned_by", "boundless"}}})}}.dump(),
                        "application/json");
    });
    srv.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { res.status = 400; res.set_content(error_json("invalid JSON").dump(), "application/json"); return; }
        std::vector<bl::Text::Message> msgs;
        bl::ChatParams cp;
        try {
            for (const auto & m : body.at("messages"))
                msgs.push_back({m.at("role").get<std::string>(), content_text(m.value("content", json(""))), m.value("reasoning_content", std::string())});
            cp.sampling.temperature = body.value("temperature", 0.7f);
            cp.sampling.top_p = body.value("top_p", 0.8f);
            cp.sampling.top_k = body.value("top_k", 20);
            cp.sampling.seed = body.value("seed", 0ull);
            cp.max_new = body.value("max_completion_tokens", body.value("max_tokens", 4096));
            cp.spec = body.value("spec", 3);
            const bool think = body.contains("chat_template_kwargs") && body["chat_template_kwargs"].value("enable_thinking", false);
            std::string effort = body.value("reasoning_effort", std::string());
            if (effort == "high") effort = "xhigh";
            cp.think = !effort.empty() ? effort : think ? "xhigh" : "";
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_json(e.what()).dump(), "application/json");
            return;
        }
        const bool stream = body.value("stream", false);
        const std::string id = "chatcmpl-" + std::to_string(++seq);
        const long created = static_cast<long>(std::time(nullptr));
        auto usage = [](const bl::ChatResult & r) {
            return json{{"prompt_tokens", r.n_prompt + r.n_reused}, {"completion_tokens", r.n_generated},
                        {"total_tokens", r.n_prompt + r.n_reused + r.n_generated},
                        {"prompt_tokens_details", {{"cached_tokens", r.n_reused}}}};
        };
        if (!stream) {
            std::lock_guard<std::mutex> lk(busy);
            try {
                const bl::ChatResult r = chat.reply(msgs, cp);
                json msg{{"role", "assistant"}, {"content", r.content}};
                if (!r.reasoning.empty()) msg["reasoning_content"] = r.reasoning;
                res.set_content(json{{"id", id}, {"object", "chat.completion"}, {"created", created}, {"model", name},
                                     {"choices", json::array({{{"index", 0}, {"message", msg}, {"finish_reason", r.stopped_eog ? "stop" : "length"}}})},
                                     {"usage", usage(r)}}.dump(), "application/json");
                std::fprintf(stderr, "%s: %d prompt (+%d reused) in %.2f s, %d tokens at %.1f tok/s\n", id.c_str(), r.n_prompt, r.n_reused,
                             r.prompt_s, r.n_generated, r.gen_s > 0 ? (r.n_generated - 1) / r.gen_s : 0.0);
                save_cache();
            } catch (const std::exception & e) {
                res.status = 400;
                res.set_content(error_json(e.what()).dump(), "application/json");
            }
            return;
        }
        res.set_chunked_content_provider("text/event-stream", [&, msgs, cp, id, created](size_t, httplib::DataSink & sink) {
            std::lock_guard<std::mutex> lk(busy);
            auto send = [&](const json & j) {
                const std::string s = "data: " + j.dump() + "\n\n";
                return sink.write(s.data(), s.size());
            };
            auto chunk = [&](const json & delta, const json & finish) {
                return json{{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", name},
                            {"choices", json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}};
            };
            send(chunk({{"role", "assistant"}, {"content", ""}}, nullptr));
            try {
                const bl::ChatResult r = chat.reply(msgs, cp, [&](const std::string & s, bool reasoning) {
                    return send(chunk(json{{reasoning ? "reasoning_content" : "content", s}}, nullptr));   // false: the client left
                });
                json last = chunk(json::object(), r.stopped_eog ? "stop" : "length");
                last["usage"] = usage(r);
                send(last);
                std::fprintf(stderr, "%s (stream): %d prompt (+%d reused) in %.2f s, %d tokens at %.1f tok/s\n", id.c_str(), r.n_prompt,
                             r.n_reused, r.prompt_s, r.n_generated, r.gen_s > 0 ? (r.n_generated - 1) / r.gen_s : 0.0);
                save_cache();
            } catch (const std::exception & e) {
                send(error_json(e.what()));
            }
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    });
    std::fprintf(stderr, "listening on http://%s:%d (OpenAI API: /v1/chat/completions)\n", host.c_str(), port);
    if (!srv.listen(host, port)) { std::fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port); return 1; }
    return 0;
}
