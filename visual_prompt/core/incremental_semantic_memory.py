"""Append support keys without overwriting existing target directions or keys."""
import copy
from collections import Counter

import torch
import torch.nn.functional as F


def append_memory_entries(memory, positive_keys, positive_sources, negative_keys):
    """Return new data memory and the old-to-new channel map for reader.expanded.

    Keys must come from the same frozen encoder/normalization protocol. The caller
    owns label/provenance checks and a storage budget policy. This append-only
    primitive does not claim bounded memory or continual-learning performance.
    """
    keys = memory['keys'].detach().cpu()
    positive_keys = positive_keys.detach().cpu().float()
    negative_keys = negative_keys.detach().cpu().float()
    dim = keys.shape[1]
    for values in (positive_keys, negative_keys):
        if values.ndim != 2 or values.shape[1] != dim or not torch.isfinite(values).all():
            raise ValueError('New memory keys must be finite [N,C] matching old keys')
        if len(values) and (values.norm(dim=1) <= 1e-8).any():
            raise ValueError('Zero memory keys cannot be normalized')
    if len(positive_sources) != len(positive_keys):
        raise ValueError('Each positive key requires a support source identifier')
    old_count = memory['positive_count']
    old_sources = [r['sample_id'] for r in memory['audit']]
    if len(old_sources) != old_count:
        raise ValueError('Old positive provenance must match memory slots')
    new = copy.deepcopy(memory)
    new['keys'] = torch.cat((keys[:2+old_count], F.normalize(positive_keys,dim=1),
                             keys[2+old_count:], F.normalize(negative_keys,dim=1)),dim=0)
    new['positive_count'] = old_count + len(positive_keys)
    new['negative_count'] = memory['negative_count'] + len(negative_keys)
    sources = old_sources + list(positive_sources)
    counts = Counter(sources)
    new['positive_weights'] = [1/(len(counts)*counts[s]) for s in sources]
    new['audit'] += [dict(sample_id=s, added_incrementally=True) for s in positive_sources]
    new['support_ids'] = list(dict.fromkeys(memory['support_ids']+list(positive_sources)))
    mapping = list(range(2+old_count)) + [i+len(positive_keys) for i in range(2+old_count,len(keys))]
    assert torch.equal(new['keys'][mapping],keys)
    return new, mapping
