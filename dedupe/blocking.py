#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import logging
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from typing import Generator, Tuple, Iterable, Dict, List, Union
from dedupe._typing import Record, RecordID, Data

import dedupe.predicates
from dedupe.core import chunked

logger = logging.getLogger(__name__)

Docs = Union[Iterable[str], Iterable[Iterable[str]]]

# Assuming a worst case of 10 KBs per record,
# BATCH_SIZE = 20_000 means 200 MBs per process on RAM,
# plus 200 MBs * num_cores for the main process.
BATCH_SIZE = 20_000


def index_list():
    return defaultdict(list)


def _batch_call_fingerprinter(predicates, record_batch, target):
    blocking_map_rows = []

    for record in record_batch:
        record_id, instance = record

        for pred_id, predicate in predicates:
            block_keys = predicate(instance, target=target)
            for block_key in block_keys:
                blocking_map_rows.append((block_key + pred_id, record_id))
    return blocking_map_rows


def _batch_call_fingerprinter_and_log(start_time, iteration, predicates, record_batch, target):
    blocking_map_rows = _batch_call_fingerprinter(predicates, record_batch, target)
    # if iteration:
    #     logger.info('%(iteration)d, %(elapsed)f2 seconds',
    #                 {'iteration': iteration,
    #                  'elapsed': time.perf_counter() - start_time})
    return blocking_map_rows


class Fingerprinter(object):
    '''Takes in a record and returns all blocks that record belongs to'''

    def __init__(self, predicates: Iterable[dedupe.predicates.Predicate], num_cores: int) -> None:

        self.predicates = predicates
        self.num_cores = num_cores

        self.index_fields: Dict[str,
                                Dict[str,
                                     List[dedupe.predicates.IndexPredicate]]]
        self.index_fields = defaultdict(index_list)
        '''
        A dictionary of all the fingerprinter methods that use an
        index of data field values. The keys are the field names,
        which can be useful to know for indexing the data.
        '''

        self.index_predicates = []

        for full_predicate in predicates:
            for predicate in full_predicate:
                if hasattr(predicate, 'index'):
                    self.index_fields[predicate.field][predicate.type].append(
                        predicate)
                    self.index_predicates.append(predicate)

    def _separate_index_predicates(self, predicates: Iterable[Tuple[str, dedupe.predicates.Predicate]]):
        full_predicates_with_index = []
        full_predicates_without_index = []

        for (pred_id, full_predicate) in predicates:
            for predicate in full_predicate:
                if predicate in self.index_predicates:
                    full_predicates_with_index.append((pred_id, full_predicate))
                else:
                    full_predicates_without_index.append((pred_id, full_predicate))
        
        return full_predicates_with_index, full_predicates_without_index

    def __call__(self,
                 records: Iterable[Record],
                 target: bool = False) -> Generator[Tuple[str, RecordID], None, None]:
        '''
        Generate the predicates for records. Yields tuples of (predicate,
        record_id).

        Args:
            records: A sequence of tuples of (record_id,
                  record_dict). Can often be created by
                  `data_dict.items()`.
            target: Indicates whether the data should be treated as
                    the target data. This effects the behavior of
                    search predicates. If `target` is set to
                    `True`, an search predicate will return the
                    value itself. If `target` is set to `False` the
                    search predicate will return all possible
                    values within the specified search distance.

                    Let's say we have a
                    `LevenshteinSearchPredicate` with an associated
                    distance of `1` on a `"name"` field; and we
                    have a record like `{"name": "thomas"}`. If the
                    `target` is set to `True` then the predicate
                    will return `"thomas"`.  If `target` is set to
                    `False`, then the blocker could return
                    `"thomas"`, `"tomas"`, and `"thoms"`. By using
                    the `target` argument on one of your datasets,
                    you will dramatically reduce the total number
                    of comparisons without a loss of accuracy.

        .. code:: python

           > data = [(1, {'name' : 'bob'}), (2, {'name' : 'suzanne'})]
           > blocked_ids = deduper.fingerprinter(data)
           > print list(blocked_ids)
           [('foo:1', 1), ..., ('bar:1', 100)]

        '''

        predicates = [(':' + str(i), predicate)
                      for i, predicate
                      in enumerate(self.predicates)]

        if self.num_cores == 1:
            yield from self._serial_call(records, predicates, target)
        else:
            (
                full_predicates_with_index,
                full_predicates_without_index
            ) = self._separate_index_predicates(predicates)

            if full_predicates_without_index:
                yield from self._parallel_call(records, full_predicates_with_index, full_predicates_without_index, target)
            else:
                # If all predicates have indexes, we can't run anything in parallel,
                # so fallback to _serial_call.
                # Predicates with index can't run in multiple processes because
                # each process would need it's own copy of the indexes.
                # This can easily blow memory up.
                yield from self._serial_call(records, predicates)

    def _serial_call(self,
                     records: Iterable[Record],
                     predicates: Iterable[Tuple[str, dedupe.predicates.Predicate]],
                     target: bool) -> Generator[Tuple[str, RecordID], None, None]:
        start_time = time.perf_counter()
        for i, record in enumerate(records):
            record_id, instance = record

            for pred_id, predicate in predicates:
                block_keys = predicate(instance, target=target)
                for block_key in block_keys:
                    yield block_key + pred_id, record_id

            if i and i % 10000 == 0:
                logger.info('%(iteration)d, %(elapsed)f2 seconds',
                            {'iteration': i,
                             'elapsed': time.perf_counter() - start_time})

    def _parallel_call(self,
                       records: Iterable[Record],
                       full_predicates_with_index: Iterable[Tuple[str, dedupe.predicates.Predicate]],
                       full_predicates_without_index: Iterable[Tuple[str, dedupe.predicates.Predicate]],
                       target: bool) -> Generator[Tuple[str, RecordID], None, None]:
        start_time = time.perf_counter()
        future_set = set()

        with ProcessPoolExecutor(max_workers=self.num_cores - 1) as executor:
            i = 0
            record_batch_for_index_predicates = []

            for record_batch in chunked(records, BATCH_SIZE):
                i += len(record_batch)
                if full_predicates_with_index:
                    record_batch_for_index_predicates.extend(record_batch)

                future = executor.submit(
                    _batch_call_fingerprinter_and_log,
                    start_time  if i % 10000 == 0 else None,
                    i if i % 10000 == 0 else None,
                    full_predicates_without_index,
                    record_batch,
                    target)
                future_set.add(future)

                if len(future_set) >= (self.num_cores - 1) * 2:
                    # Now that enough tasks are scheduled,
                    # do the full_predicates_with_index work here.
                    if full_predicates_with_index:
                        yield from _batch_call_fingerprinter(
                            full_predicates_with_index,
                            record_batch_for_index_predicates,
                            target)
                        record_batch_for_index_predicates.clear()

                    # Don't leave many results into memory, yield them:
                    done_future_set, future_set = wait(
                        future_set,
                        return_when=FIRST_COMPLETED)
                    for future in done_future_set:
                        yield from future.result()

            # Do and yield the final work
            if full_predicates_with_index:
                yield from _batch_call_fingerprinter(
                    full_predicates_with_index,
                    record_batch_for_index_predicates,
                    target)
            for future in future_set:
                yield from future.result()

    def reset_indices(self) -> None:
        '''
        Fingeprinter indicdes can take up a lot of memory. If you are
        done with blocking, the method will reset the indices to free up.
        If you need to block again, the data will need to be re-indexed.
        '''
        for predicate in self.index_predicates:
            predicate.reset()

    def index(self,
              docs: Docs,
              field: str) -> None:
        '''
        Add docs to the indices used by fingerprinters.

        Some fingerprinter methods depend upon having an index of
        values that a field may have in the data. This method adds
        those values to the index. If you don't have any fingerprinter
        methods that use an index, this method will do nothing.

        Args:
            docs: an iterator of values from your data to index. While
                  not required, it is recommended that docs be a unique
                  set of of those values. Indexing can be an expensive
                  operation.
            field: fieldname or key associated with the values you are
                   indexing

        '''
        indices = extractIndices(self.index_fields[field])

        for doc in docs:
            if doc:
                for _, index, preprocess in indices:
                    index.index(preprocess(doc))

        for index_type, index, _ in indices:

            index.initSearch()

            for predicate in self.index_fields[field][index_type]:
                logger.debug("Canopy: %s", str(predicate))
                predicate.index = index
                predicate.bust_cache()

    def unindex(self, docs: Docs, field: str) -> None:
        '''Remove docs from indices used by fingerprinters

        Args:
            docs: an iterator of values from your data to remove. While
                  not required, it is recommended that docs be a unique
                  set of of those values. Indexing can be an expensive
                  operation.
            field: fieldname or key associated with the values you are
                   unindexing
        '''

        indices = extractIndices(self.index_fields[field])

        for doc in docs:
            if doc:
                for _, index, preprocess in indices:
                    try:
                        index.unindex(preprocess(doc))
                    except KeyError:
                        pass

        for index_type, index, _ in indices:

            index._index.initSearch()

            for predicate in self.index_fields[field][index_type]:
                logger.debug("Canopy: %s", str(predicate))
                predicate.index = index
                predicate.bust_cache()

    def index_all(self, data: Data):
        for field in self.index_fields:
            unique_fields = {record[field]
                             for record
                             in data.values()
                             if record[field]}
            self.index(unique_fields, field)


def extractIndices(index_fields):

    indices = []
    for index_type, predicates in index_fields.items():
        predicate = predicates[0]
        index = predicate.index
        preprocess = predicate.preprocess
        if predicate.index is None:
            index = predicate.initIndex()
        indices.append((index_type, index, preprocess))

    return indices
