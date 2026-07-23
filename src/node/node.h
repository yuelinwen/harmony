#ifndef HARMONY_NODE_NODE_H
#define HARMONY_NODE_NODE_H

// Node: abstract base class for MasterNode and WorkerNode.

namespace harmony {

class Node {
public:
    Node(int id) {
        id_ = id;
        running_ = false;
    }

    virtual ~Node() {
    }

    // Pure virtual function. Each subclass writes its own run().
    virtual int run() = 0;

    int getId() {
        return id_;
    }

    bool isMaster() {
        return id_ == 0;
    }

protected:
    int id_;        // 0 = master, 1,2,3... = worker
    bool running_;  // main loop flag
};

}  // namespace harmony

#endif  // HARMONY_NODE_NODE_H
