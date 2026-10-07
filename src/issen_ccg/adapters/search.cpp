// The search library is independent of the neural inference runtime.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

namespace {
struct Rule { int parent, id, head, kind; };
struct Node {
    int category, begin, end, left, right, rule, head;
    float inside;
    int state;
};
struct Entry {
    double bound;
    int node;
    bool operator<(const Entry& other) const {
        if (bound != other.bound) return bound < other.bound;
        return node > other.node;
    }
};
uint64_t pair_key(int left, int right) {
    return (uint64_t(uint32_t(left)) << 32) | uint32_t(right);
}
struct Cell {
    std::unordered_map<int, int> best;
    std::vector<int> finished;
    std::unordered_map<int, bool> closed;
};
struct Engine {
    int categories;
    std::vector<std::vector<Rule>> unary;
    std::unordered_map<uint64_t, std::vector<Rule>> binary;
    std::vector<bool> roots;
    std::vector<Node> nodes;
    std::vector<int> output;
    int root = -1, status = 1, expanded = 0, pushed = 0;
    float score = -std::numeric_limits<float>::infinity();

    int emit(int index) {
        const Node& n = nodes[index];
        int left = n.left < 0 ? -1 : emit(n.left);
        int right = n.right < 0 ? -1 : emit(n.right);
        int result = int(output.size() / 7);
        output.insert(output.end(), {n.category, n.begin, n.end, left, right, n.rule, n.head});
        return result;
    }

    void parse(int length, const float* tags, const float* arcs, float arc_weight,
               int max_nodes, int timeout_ms, float unary_cost) {
        nodes.clear(); output.clear(); root = -1; status = 1; expanded = pushed = 0;
        score = -std::numeric_limits<float>::infinity();
        if (length <= 0) return;
        std::vector<double> tag_prefix(length + 1), arc_prefix(length + 1);
        for (int i = 0; i < length; ++i) {
            double tag = -std::numeric_limits<double>::infinity();
            for (int c = 0; c < categories; ++c) tag = std::max(tag, double(tags[i*categories+c]));
            if (!std::isfinite(tag)) return;
            double arc = 0;
            if (i > 0) {
                arc = -std::numeric_limits<double>::infinity();
                for (int j = 0; j < i; ++j) arc = std::max(arc, double(arcs[i*length+j]) * arc_weight);
                if (!std::isfinite(arc)) return;
            }
            tag_prefix[i+1] = tag_prefix[i] + tag;
            arc_prefix[i+1] = arc_prefix[i] + arc;
        }
        std::vector<Cell> chart((length+1)*(length+1));
        auto cell = [&](int b, int e) -> Cell& { return chart[b*(length+1)+e]; };
        std::priority_queue<Entry> agenda;
        bool limited = false;
        auto push = [&](Node node) {
            if (limited || !std::isfinite(node.inside)) return;
            Cell& target = cell(node.begin, node.end);
            int key = node.category*3 + node.state;
            if (target.closed.count(key)) return;
            auto old = target.best.find(key);
            if (old != target.best.end() && nodes[old->second].inside >= node.inside) return;
            if (int(nodes.size()) >= max_nodes) { limited = true; return; }
            int index = int(nodes.size());
            nodes.push_back(node);
            target.best[key] = index;
            double outside = tag_prefix[length] - (tag_prefix[node.end] - tag_prefix[node.begin])
                + arc_prefix[length] - (arc_prefix[node.end] - arc_prefix[node.begin+1]);
            agenda.push({double(node.inside) + outside, index});
            ++pushed;
        };
        for (int i = 0; i < length; ++i)
            for (int c = 0; c < categories; ++c)
                if (std::isfinite(tags[i*categories+c]))
                    push({c, i, i+1, -1, -1, -1, 0, tags[i*categories+c], 0});
        auto combine = [&](int left_index, int right_index) {
            // Copy nodes: push can reallocate the arena.
            Node left = nodes[left_index], right = nodes[right_index];
            auto found = binary.find(pair_key(left.category, right.category));
            if (found == binary.end()) return;
            float value = left.inside + right.inside + arc_weight*arcs[right.begin*length+left.begin];
            for (const Rule& rule : found->second) {
                // Type raising cannot feed argument positions or conjunction introduction.
                if ((rule.kind == 1 && right.state == 2) ||
                    (rule.kind == 2 && left.state == 2) ||
                    (rule.kind == 4 && right.state == 2)) continue;
                push({rule.parent, left.begin, right.end, left_index, right_index, rule.id, rule.head, value, 0});
            }
        };
        auto started = std::chrono::steady_clock::now();
        while (!agenda.empty() && !limited) {
            int index = agenda.top().node; agenda.pop();
            Node node = nodes[index];
            Cell& target = cell(node.begin, node.end);
            int key = node.category*3 + node.state;
            if (target.closed.count(key) || target.best[key] != index) continue;
            target.closed[key] = true;
            target.finished.push_back(index);
            ++expanded;
            if (node.begin == 0 && node.end == length && roots[node.category]) {
                score = node.inside; root = emit(index); status = 0; return;
            }
            for (const Rule& rule : unary[node.category]) {
                int next_state = rule.kind == 2 ? 2 : 1;
                if (node.state >= next_state) continue;
                push({rule.parent, node.begin, node.end, index, -1, rule.id, 0, node.inside-unary_cost, next_state});
            }
            for (int begin = 0; begin < node.begin; ++begin)
                for (int left : cell(begin, node.begin).finished) combine(left, index);
            for (int end = node.end+1; end <= length; ++end)
                for (int right : cell(node.end, end).finished) combine(index, right);
            if ((expanded & 255) == 0 && timeout_ms > 0 &&
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now()-started).count() >= timeout_ms) {
                status = 3; return;
            }
        }
        status = limited ? 2 : 1;
    }
};
}

extern "C" {
void* issen_create(int categories, int nu, const int* unary, int nb, const int* binary,
                   int nr, const int* roots) {
    try {
        auto* engine = new Engine;
        engine->categories = categories;
        engine->unary.resize(categories);
        engine->roots.resize(categories, false);
        for (int i=0; i<nu; ++i) engine->unary[unary[3*i]].push_back({unary[3*i+1], i, 0, unary[3*i+2]});
        for (int i=0; i<nb; ++i) engine->binary[pair_key(binary[5*i],binary[5*i+1])].push_back(
            {binary[5*i+2], nu+i, binary[5*i+3], binary[5*i+4]});
        for (int i=0; i<nr; ++i) engine->roots[roots[i]] = true;
        return engine;
    } catch (...) { return nullptr; }
}
void issen_destroy(void* pointer) { delete static_cast<Engine*>(pointer); }
int issen_parse(void* pointer, int length, const float* tags, const float* arcs,
                float arc_weight, int max_nodes, int timeout_ms, float unary_cost,
                int* metadata, float* score) {
    try {
        auto& engine = *static_cast<Engine*>(pointer);
        engine.parse(length,tags,arcs,arc_weight,max_nodes,timeout_ms,unary_cost);
        metadata[0] = engine.root; metadata[1] = int(engine.output.size()/7);
        metadata[2] = engine.expanded; metadata[3] = engine.pushed;
        *score = engine.score;
        return engine.status;
    } catch (...) { return 4; }
}
const int* issen_nodes(void* pointer) { return static_cast<Engine*>(pointer)->output.data(); }
}
