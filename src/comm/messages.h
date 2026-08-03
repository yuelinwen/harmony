#ifndef HARMONY_COMM_MESSAGES_H
#define HARMONY_COMM_MESSAGES_H

// MPI message tags and the small protocol between master and workers.
//
// Ranks: 0 is the master, 1..N are the workers. A worker forwards its partial
// results straight to the next worker and only the last one reports back to
// the master, so intermediate results never pass through rank 0
// (paper Fig. 5b).

namespace harmony {

const int MASTER_RANK = 0;

// startup
const int TAG_SETUP   = 1;   // int[3]: myDim, nClusters (this row), bDim
const int TAG_CLUSTER = 2;   // int[2]: clusterId, nIds
const int TAG_IDS     = 3;   // int[nIds]
const int TAG_DATA    = 4;   // float[nIds * myDim]

// per query
const int TAG_JOB       = 5;   // int[3], see below
const int TAG_QUERY     = 6;   // float[myDim]: this worker's slice of the query
const int TAG_THRESHOLD = 7;   // float: tau^2
const int TAG_SUMS      = 8;   // float[n]: running partial distances
const int TAG_ALIVE     = 9;   // char[n]: 0 = pruned
const int TAG_STATS     = 10;  // long: how many survived this worker, in total

// TAG_JOB carries int[3] = {what, n, skip}.
// `what` >= 0 is a cluster id to work on, and then n is how many vectors it
// holds and skip is how many leading ones the master already handled itself.
const int JOB_QUERY    = -1;   // a new query slice follows
const int JOB_SHUTDOWN = -2;   // stop, report stats, exit

}  // namespace harmony

#endif  // HARMONY_COMM_MESSAGES_H
