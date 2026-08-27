# Raw measurement rows

One file per run, exactly as the harness wrote it, with the framing rule in the
filename and the full framing in the commit that added it.

These exist because a headline number that lives only in a chat message and a
`/tmp` directory on a box is not a measurement anyone can check. On 2026-08-27 an
independent review grepped the tree for "+2.28% on the production two-node step"
and correctly reported that it did not exist -- the run was real, the artifacts
were in `/tmp/pipe-swzab` on spark2, and `/tmp` is cleared. The number was one
reboot from being an anecdote.

A row here is not a conclusion. It is what the harness printed, so that anyone
can recompute the median, check the spread, and see how many reps actually
landed rather than how many were requested.
