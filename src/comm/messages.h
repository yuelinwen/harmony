#ifndef HARMONY_COMM_MESSAGES_H
#define HARMONY_COMM_MESSAGES_H

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

// per batch of queries
const int TAG_JOB       = 5;   // int[4], see below
const int TAG_QUERY     = 6;   // float[batch * myDim]: this worker's slices
const int TAG_QIDX      = 9;   // int[m]: which queries of the batch take part
const int TAG_THRESHOLD = 7;   // float[m]: tau^2 for each of them
const int TAG_SUMS      = 8;   // float[m * n]: running partial distances
const int TAG_STATS     = 10;  // long[bDim]: survivors per chain position
const int TAG_TOPK      = 11;  // Candidate[m * kSend]: the chain tail's answer

// TAG_SUMS carries running totals from one worker to the next, one float per
// candidate, because the next worker needs every candidate's total to add to.
// The last worker in the chain is the only one that ever sees a full distance,
// so it does not forward totals at all: it keeps the k nearest per query and
// sends those as TAG_TOPK (paper §4.3, only the last reports back). k is
// around 100 against a cluster's few thousand vectors, which is why this is
// the hop worth shrinking.
//
// The ids in TAG_TOPK are positions within the cluster, not global vector
// ids -- the master maps them back, and that lets it apply the same prewarm
// de-duplication it did when it received raw totals.

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

// TAG_JOB carries int[4] = {what, n, startCol, m}. `what` >= 0 is a cluster
// id, n is how many vectors it holds, and m is how many queries of the batch
// probed it -- usually only part of the batch, so TAG_QIDX names which.
//
// startCol is where this cluster's chain begins in the row. Clusters start at
// different columns so no worker is always the first stop, which is the one
// that can prune nothing (paper §4.3).
const int JOB_QUERY    = -1;   // the batch's query slices follow
const int JOB_SHUTDOWN = -2;   // stop, report stats, exit

}  // namespace harmony

#endif  // HARMONY_COMM_MESSAGES_H
