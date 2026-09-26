#ifndef HARMONY_COMM_MESSAGES_H
#define HARMONY_COMM_MESSAGES_H

#include <mpi.h>

// What master and workers send each other. Both sides include this file, so a
// tag can never disagree between them -- a mismatch would not fail to compile,
// it would hang in MPI_Recv.
//
// Rank 0 is the master, 1..N are the workers. Partial results go straight from
// one worker to the next and never through rank 0 (paper Fig. 5b).

namespace harmony {

const int MASTER_RANK = 0;

// startup
// int[9]: myDim, nClusters (this row), bDim, batch, sendSlots, bVec, block,
// pipeline, vectorPruning.
//
// The last four are the shape of the schedule. With them a worker rebuilds
// the group order itself and walks the batch without being told anything
// about it -- which query group reaches it at which stage, where that group
// starts, how it splits into blocks, and what tag each block carries. The
// master sends only the thresholds.
//
// sendSlots is how many blocks the master can have in flight, and the worker
// sizes its outgoing buffer pool from it. That is an invariant, not a
// convenience: a worker must wait for a buffer's send to complete before
// reusing it, blocks enter a row at rotating columns so two workers can be
// sending to each other, and a worker that has run out of buffers has not yet
// drained its job queue to post the matching receive. Two workers in that
// state wait for each other forever.
//
// The pool used to be sized locally as 2 * bDim + 2, which was >= the master's
// in-flight count only because --block defaulted to 4. Raising that default
// made the pool too small and about one run in six deadlocked, at 100% CPU on
// both sides because MPI busy-waits -- it looked like a slow run, not a hang.
// One formula, on the master (maxBlocksInFlight()), shipped here.
const int TAG_SETUP   = 1;
const int TAG_CLUSTER = 2;   // int[2]: clusterId, nIds
const int TAG_IDS     = 3;   // int[nIds]
const int TAG_DATA    = 4;   // float[nIds * myDim]
const int TAG_ORDER   = 13;  // int[3 * bDim]: this worker's rows of the
                             // chain table -- next, prev, stage -- one entry
                             // per item. See engine/search_order.h.

// per batch of queries
const int TAG_JOB       = 5;   // int[5], see below
const int TAG_QUERY     = 6;   // float[batch * myDim]: this worker's slices
const int TAG_PROBES    = 9;   // int[count * nprobe]: every query's clusters,
                               // nearest first. Sent once per batch. A worker
                               // needs them to lay out a block's buffer the
                               // same way its neighbours in the chain do.
// float[gLen]: tau^2 for one query group, sent when the row it is on is
// about to start it. Nothing announces it: after JOB_QUERY there are exactly
// bVec stages, and a worker knows from the group order which group each of
// them brings. This is the message the vector-level pipeline runs on -- a
// group's next partition starts from a threshold its last one tightened
// (Fig. 5a) -- and it is the only thing the master sends during a batch.
const int TAG_THRESHOLD = 7;
const int TAG_STATS     = 10;  // long[bDim]: survivors per chain position
const int TAG_TIMES     = 12;  // double[WORKER_TIMES]: worker's wall time

// How many numbers a worker reports about where its time went, in this order:
//
//   total  idle  recv  compute  send  jobs  setup  poll  admin  bytes
//
// Both ends of TAG_TIMES size their buffer from this, because a mismatch
// would not fail -- MPI would deliver the shorter count and the master would
// read whatever was next in the array as a timing.
const int WORKER_TIMES = 10;

// Partial sums and top-K answers get a tag of their own per block in flight.
//
// They have to, because a worker does not process blocks in the order they
// were handed over: one whose upstream has arrived overtakes one still
// waiting. Two blocks travelling between the same pair of ranks would then be
// matched by arrival order rather than by which is which, and a worker would
// add its slice to the wrong running totals. MPI matches on the tag instead,
// so the order stops mattering.
//
// The number is blockTag() below -- a pure function of where the block sits
// in the batch's schedule, so the master and the workers arrive at it
// separately and always agree. It used to be an index into a pool the master
// allocated from, which nobody else could derive, and that is what forced a
// message per block. (The sample uses the same idea:
// tag = groupId * blockCount + blockId.)
const int TAG_CHAIN_BASE = 100;

// Which block this is, out of everything one batch will produce: query group
// g, visiting vector partition number `stage` of the bVec it must visit, and
// the b-th of the group's `block` query blocks. Unique across a batch, and a
// batch is fully drained before the next begins.
inline int blockTag(int g, int stage, int b, int bVec, int block) {
    return (g * bVec + stage) * block + b;
}

// float[m * n]: running partial distances, one worker to the next.
inline int tagSums(int slot) { return TAG_CHAIN_BASE + 2 * slot; }

// Candidate[m * kSend]: the chain tail's answer, straight to the master. The
// last worker is the only one that ever sees a full distance, so it does not
// forward totals at all -- it keeps the k nearest per query and sends those
// (paper §4.3, only the last reports back). k is around 100 against a
// cluster's few thousand vectors, which is why this is the hop worth
// shrinking.
//
// The ids are global vector ids: the tail reads them straight out of the
// cluster block it holds, so the master pushes them into the heap as they
// arrive with nothing to map back. There is no de-duplication to do either --
// prewarm keeps only a threshold, so the heap it hands over is empty.
inline int tagTopk(int slot) { return TAG_CHAIN_BASE + 2 * slot + 1; }

// A pruned candidate is marked by setting its running sum to this, rather
// than carrying a separate alive flag: it is larger than any real squared
// distance and any threshold, so downstream workers skip it on their own.
//
// Two assumptions ride on the value, and both hold for every dataset here:
//   - a real squared distance never reaches it, or a live candidate would be
//     read as pruned. Sift1M distances run to about 1e5.
//   - it is above TopKHeap::worst()'s not-yet-full sentinel (1e30), so a
//     threshold from an empty heap prunes nothing rather than everything.
const float PRUNED = 1e38f;

// TAG_JOB carries int[5] = {what, ...}, the rest reading per `what`. Only
// between batches, or to open one: the search itself sends no jobs at all.
const int JOB_QUERY    = -1;   // the batch's query slices follow
const int JOB_SHUTDOWN = -2;   // stop and exit
// Zero the pruning counters. With --loop the query set is run several times;
// only the last pass is counted, so the earlier ones have to be forgotten or
// the survivor counts would not match the candidates the master saw.
const int JOB_RESET    = -3;
// Report the counters so far and keep going, which shutdown cannot do because
// it ends the process. --nprobes runs several searches against one
// distribution and each needs its own numbers.
const int JOB_STATS    = -4;
// A new chain table follows, same int[3 * bDim] as TAG_ORDER at setup. Sent
// between batches, when nothing is on the chain: a block carries only its
// item, and both ends of a hop have to agree on what that item means.
const int JOB_ORDER    = -6;
// How a worker knows what to do during a batch, given that nothing tells it.
//
// A batch is bVec stages. At stage s, the query group reaching row r is
// groupOrder_.stepsOf(r)[s] -- a rotation both sides build from bVec alone,
// so at every stage the mapping group -> row is a permutation and no row is
// idle (paper §4.2.2, Fig. 4b). The group's queries are the contiguous run
// [ceil(g*count/bVec), ceil((g+1)*count/bVec)), the same split the master
// makes. It is cut into `block` query blocks by the same formula, block b
// enters the row at column b % bDim so no worker is always the first stop --
// the one that can prune nothing (paper §4.3) -- and carries blockTag(g, s,
// b). A block this partition holds nothing for adds up to zero on both sides
// and simply does not happen.
//
// All of which is why the search sends no jobs. The master dispatched a
// message per block once, then a message per group; now the only thing it
// sends during a batch is the thresholds. At 1x8 with --block 128 that is
// 1040 messages a batch down to 8.

// The largest chain tag this layout will use has to be one MPI accepts. The
// standard only promises 32767, and the tags here stay far below that, so
// this is a guard against a future layout rather than a live worry. The
// sample checks the same attribute inside its tag generator.
inline bool tagsFitMpi(int maxSlots) {
    int* ub = nullptr;
    int found = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &found);
    if (!found || ub == nullptr) {
        return true;
    }
    return tagTopk(maxSlots - 1) <= *ub;
}

}  // namespace harmony

#endif  // HARMONY_COMM_MESSAGES_H
