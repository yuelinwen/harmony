#ifndef HARMONY_NODE_NODE_H
#define HARMONY_NODE_NODE_H

// Node: abstract base class for MasterNode and WorkerNode.

namespace harmony {

class Node {
public:
    Node(int id) {
        id_ = id;
    }

    virtual ~Node() {
    }

    // Pure virtual function. Each subclass writes its own run().
    virtual int run() = 0;

protected:
    // 0 = master, 1,2,3... = worker. A worker derives its column and the rank
    // of its row's first worker from it.
    int id_;
};

}  // namespace harmony

#endif  // HARMONY_NODE_NODE_H
