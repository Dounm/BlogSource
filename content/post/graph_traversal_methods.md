---
title: "图遍历方法：ForEachNode + DFS Vs. TopoForEachNode"
date: 2019-06-17T18:10:30+08:00
lastmod: 2019-06-17T18:10:30+08:00
keywords: []
description: ""
tags: [DFS, TOPO]
categories: [图论]
author: ""
---

在图遍历领域，有一类常见问题：**在DAG图上，以某一类节点为起点，找到并标记其下游的符合某些条件的节点**。

## 1 思路一：ForEachNode + DFS

对于此类问题来说，最常见的思路是：`ForEachNode + DFS`，即循环遍历图中每个节点，判断图中节点是否满足起始条件，满足则以其为起始节点，进行DFS遍历。

粗略来看，这个解法本质上是**循环+DFS**，时间复杂度应该是`o(n^2)`。但如果跳过那些被访问过的节点，那么考虑到每个节点只会被遍历到一次，其时间复杂度就下降为`o(n)`。

### 1.1 代码

``` cpp
template<typename Key, typename T, typename Hash = std::hash<Key>>
using HashMap = std::unordered_map<Key, T, Hash>;

template<typename Key, typename Hash = std::hash<Key>>
using HashSet = std::unordered_set<Key, Hash>;

class Node {
 public:
  void ForEachNodeOnOutEdge(std::function<void(Node*)>);
  void ForEachNodeOnInEdge(std::function<void(Node*)>);
  ... 
};

class Graph {
 public:
  const std::vector<Node*>& nodes();
  const std::vector<Node*>& start_nodes();
  ... 
};

template<typename MapT, typename KeyT>
bool IsKeyFound(const MatT& m, const KeyT& k) { return m.find(k) != m.end(); }


void DfsGraphTraversal(
    const Graph& graph, const std::vector<Node*>& starts,
    std::function<void(Node*, std::function<void(Node*)>)> ForEachNextNode,
    std::function<bool(Node*)> IsNodeTraversable, std::function<void(Node*)> Handler) {
  HashSet<Node*> visited;
  std::stack<Node*> stack;
  for (Node* start : starts) { stack.push(start); }
  while (!stack.empty()) {
    Node* cur_node = stack.top();
    stack.pop();

    if (IsNodeTraversable(cur_node) && !IsKeyFound(visited, cur_node)) {
      Handler(cur_node);
      CHECK(visited.insert(cur_node).second);
      ForEachNextNode(cur_node, [&stack](Node* next) { stack.push(next); });-
    }
  }
}

// ForEachNode + DFS
void GraphTraversal1(const Graph& graph, std::function<bool(Node*)> IsStartNode,-
    std::function<bool(Node*)> IsNodeTraversable,
    std::function<void(Node*)> NodeHandler) {
  HashSet<Node*> visited;
  for (const Node* node : graph.nodes()) {
    if (IsKeyFound(visited, node)) { continue; }
    if (IsStartNode(node)) {
      DfsGraphTraversal(graph, std::vector<Node*>{node},
          &Node::ForEachNodeOnOutEdge,
          [&](Node* node) {
            return !IsKeyFound(visited, node) && IsNodeTraversable(node);
          },
          [&](Node* node) {
            CHECK(visited.insert(node).second);
            NodeHandler(node);
          }); 
    } 
  }
}
```

### 1.2 代码解析

如上代码所示，对于思路1来说，其蕴含了三个判断条件

- `IsStartNode()`：判断该节点是否能作为DFS的起始节点
- `IsNodeTraversable()`：如果为`true`，则表明该节点应该被标记；否则跳过该节点，不再遍历其下游节点。
- **隐藏条件**：因为是在图上遍历，所以DFS过程中每个被遍历到的节点，其父节点都得满足`IsStartNode() || IsNodeTraversable()`的条件。

## 2 思路二：TopoDFS遍历

思路一的代码里，引入了一个变量`visited`来保存被访问过的节点，然后跳过被访问的节点来保证时间复杂度为`o(n)`。

问题在于，我们需要小心的维护`visited`，来保证其不会出错，因此导致代码比较凌乱。

因此思路二使用了Topo遍历的方法，利用Topo序来自然而然的保证每个节点只用遍历一次。

### 2.1 代码

``` cpp
void DfsTopoGraphTraversal(
    const Graph& graph, const std::vector<Node*>& starts,
    std::function<void(Node*, const std::function<void(Node*)>&)> ForEachInNode,
    std::function<void(Node*, const std::function<void(Node*)>&)> ForEachOutNode,
    std::function<void(Node*)> Handler) const {
  HashMap<Node*, bool> be_visited;
  std::stack<Node*> stack;
  for (Node* start : starts) {
    stack.push(start);
    ForEachInNode(start, [&](Node*) { assert(false); }); 
  }
  while (!stack.empty()) {
    Node* cur_node = stack.top();
    stack.pop();
    Handler(cur_node);
    be_visited[cur_node] = true;
    ForEachOutNode(cur_node, [&](Node* out) {
      bool is_ready = true;
      ForEachInNode(out, [&](Node* in) {
        if (is_ready && !be_visited[in]) { is_ready = false; }
      }); 
      if (is_ready && !be_visited[out]) { stack.push(out); }
    }); 
  }
}

// TopoDfs
void GraphTraversal2(const Graph& graph, std::function<bool(Node*)> IsFatherNodeSatisfied,
    std::function<bool(Node*)> IsCurNodeSatisfied,
    std::function<void(Node*)> NodeHandler) {
  std::function<void(Node*)> HandlerWithCondition = [](Node* node) {
    if (IsCurNodeSatisfied(node)) {
      bool is_one_father_of_node_satisfied = false;
      node->ForEachNodeOnInEdge([&](Node* father_node) {
        if (IsFatherNodeSatisfied(father_node)) { is_one_father_of_node_satisfied = true; }
      }); 
      if (is_one_father_of_node_satisfied) { NodeHandler(node); }
    } 
  };
  DfsTopoGraphTraversal(graph, graph.start_nodes(),
      &Node::ForEachNodeOnInEdge, &Node::ForEachNodeOnOutEdge,
      HandlerWithCondition);
}
```

### 2.2 代码解析

思路二的代码中，其实也蕴含了两个判断条件：

- `IsCurNodeSatisfied()`：当前节点是否满足被标记的条件

- `IsFatherNodeSatisfied()`：当前节点的父亲节点是否满足条件。

### 2.3 证明：思路一与思路二等价

前面提到，思路一有三个条件

- $A$条件：`IsStartNode()`
- $B$条件：`IsNodeTraversable()`
- $C$条件：DFS被遍历节点的父节点满足`IsStartNode() || IsNodeTraversable()`

而思路二有两个条件：

- $D$条件：`IsCurNodeSatisfied()`

- $E$条件：`IsFatherNodeSatisfied()`



仔细对比思路一二，可得

1. $B$ 和 $D$ 条件都是对当前被遍历到的节点是否满足被标记条件的描述，所以在描述能力上 $B=D​$
2. $C$ 和 $E$ 条件都是描述当前被遍历节点的父节点所需满足的条件的，所以描述能力上，同样有 $C=E$
3. 而对于 $A$ 条件来说，其被蕴含在了 $C$ 条件中，而又因为描述能力上 $C=E$，所以 $A$ 也被蕴含在 $E$ 中。

所以，思路二的两个条件足以表达思路一的三个条件，再考虑到有了Topo序就无需临时变量`visited`来跳过重复遍历的节点，所以思路二更优。



## 3 总结

总的来说，两种思路 `TopoDFS`和`ForEachNode+DFS`二者在对问题的描述能力上是等价的，而且算法时间复杂度都是`o(n)`。但考虑到代码实现的简单和健壮性，所以还是推荐使用`TopoDFS`的方法。