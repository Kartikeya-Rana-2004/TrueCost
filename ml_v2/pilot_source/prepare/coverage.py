from collections import Counter

def coverage(tasks):
    return {key:dict(Counter(t.get(key,'unknown') for t in tasks))
            for key in ('family','source_id','complexity','review_status','size_band')}
