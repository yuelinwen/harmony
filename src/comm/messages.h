#ifndef HARMONY_COMM_MESSAGES_H
#define HARMONY_COMM_MESSAGES_H

// What master and workers send each other. MPI moves bytes; the two sides
// have to agree on what those bytes mean, and this file is that agreement.
// Both include it, so a tag can never disagree between them -- a mismatch
// would not fail to compile, it would just hang forever in MPI_Recv.
//
// Rank 0 is the master, 1..N are the workers. Partial results go straight
// from one worker to the next; only the last in a chain reports back, so they
// never pass through rank 0 (paper Fig. 5b).
//
//   startup   master --> worker      the worker's slice of every cluster
//   query     master --> worker      the worker's slice of the query
//   job       master --> worker      which cluster to work on, and the threshold
//   sums      worker --> worker      running totals down the chain
//             worker --> master      from the last worker in the chain

namespace harmony {

const int MASTER_RANK = 0;

// startup
const int TAG_SETUP   = 1;   // int[3]: myDim, nClusters (this row), bDim
const int TAG_CLUSTER = 2;   // int[2]: clusterId, nIds
const int TAG_IDS     = 3;   // int[nIds]
const int TAG_DATA    = 4;   // float[nIds * myDim]

// per query
const int TAG_JOB       = 5;   // int[4], see below
const int TAG_QUERY     = 6;   // float[myDim]: this worker's slice of the query
const int TAG_THRESHOLD = 7;   // float: tau^2
const int TAG_SUMS      = 8;   // float[n]: running partial distances
const int TAG_STATS     = 10;  // long[bDim]: survivors per chain position

// A pruned candidate is marked by setting its running sum to this instead of
// carrying a separate alive flag: it is larger than any real squared distance
// and larger than any threshold, so downstream workers skip it on their own
// and only one array has to travel between workers.
const float PRUNED = 1e38f;

// TAG_JOB carries int[4] = {what, n, skip, startCol}.
// `what` >= 0 is a cluster id to work on; n is how many vectors it holds,
// skip is how many leading ones the master already handled itself, and
// startCol is the column of the row where this cluster's chain begins.
// Clusters start at different columns so that no worker is always the first
// stop -- the first stop can prune nothing and so does the most work
// (paper Section 4.3, Load Balancing Strategies).
const int JOB_QUERY    = -1;   // a new query slice follows
const int JOB_SHUTDOWN = -2;   // stop, report stats, exit

}  // namespace harmony

#endif  // HARMONY_COMM_MESSAGES_H
