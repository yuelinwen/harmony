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
const int TAG_SETUP   = 1;   // int[3]: myDim, nClusters (this row), bDim
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
const int TAG_THRESHOLD = 7;   // float[m]: tau^2 for each of them
const int TAG_STATS     = 10;  // long[bDim]: survivors per chain position
const int TAG_TIMES     = 12;  // double[6]: where a worker's wall time went

// Partial sums and top-K answers get a tag of their own per in-flight
// cluster, taken from the master's slot for it.
//
// They have to, because a worker does not process clusters in the order the
// master handed them over: one whose upstream has arrived overtakes one still
// waiting. Two clusters travelling between the same pair of ranks would then
// be matched by arrival order rather than by which is which, and a worker
// would add its slice to the wrong running totals. MPI matches on the tag
// instead, so the order stops mattering. (The sample does the same thing with
// tag = groupId * blockCount + blockId.)
//
// A slot is reused only once its cluster has fully reported, so slot numbers
// are unique among everything in flight. Values stay small -- one per worker
// plus a spare -- and tagsFitMpi() confirms the largest is one MPI will
// accept.
const int TAG_CHAIN_BASE = 100;

// float[m * n]: running partial distances, one worker to the next.
inline int tagSums(int slot) { return TAG_CHAIN_BASE + 2 * slot; }

// Candidate[m * kSend]: the chain tail's answer, straight to the master. The
// last worker is the only one that ever sees a full distance, so it does not
// forward totals at all -- it keeps the k nearest per query and sends those
// (paper §4.3, only the last reports back). k is around 100 against a
// cluster's few thousand vectors, which is why this is the hop worth
// shrinking.
//
// The ids are positions within the cluster, not global vector ids -- the
// master maps them back, and that lets it apply the same prewarm
// de-duplication it did when it received raw totals.
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

// TAG_JOB carries int[5] = {what, firstQuery, blockLen, item, slot}.
//
// A job is one block of queries against one vector partition (paper §4.2.2,
// Fig. 4b: a query mapped to a partition is split by dimension and the pieces
// routed to the machines holding them). firstQuery and blockLen name a
// contiguous run of the batch; the worker walks those queries' probe lists,
// keeps the clusters its own partition holds, and that walk is the buffer
// layout. Every worker in the row derives the same one from the same probe
// lists, so the running totals line up without anyone sending an index.
//
// `item` picks a row out of the chain table the worker was given at setup.
// Different items run the row in different orders, so no worker is always the
// first stop, which is the one that can prune nothing (paper §4.3). slot is
// the master's slot for this block and names its chain tags above.
const int JOB_BLOCK    = -5;   // what: a block of queries, fields as above
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
