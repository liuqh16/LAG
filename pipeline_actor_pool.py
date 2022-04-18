import ray
from typing import List, Callable, Optional, Tuple
from copy import deepcopy
from collections import deque, OrderedDict
import time
import traceback
import logging


class PipelineActorPool:

    def __init__(self, actors: List[ray.actor.ActorHandle], resource_providers: List['PipelineActorPool']):
        """
        Nomenclature:
        - pending task: task for which not all the resources are available
        - queued task: task for which all the resources are available, but still
            waiting for idle actor
        """
        self._idle_actors = deque(actors)
        self._future_to_actor = {}
        self._index_to_future = {}
        self._index_to_product_key = {}
        self._product_values = {}
        self._next_task_index = 0
        self._resource_providers = resource_providers
        for provider in resource_providers:
            provider.add_consumer(self)
        self._resource_available = OrderedDict()
        self._resource_to_pending_task_index = {}
        self._pending_tasks = {}
        self._queued_tasks = deque()
        self._consumers = []

        self.timeout = 400      # detect dead actors

    def map(self,
            fn: Callable[[ray.actor.ActorHandle, List, object], ray.ObjectID],
            resource_specs: List[List[Tuple]],
            additional_values: List,
            product_keys: Optional[List] = None) -> None:
        """
        :param fn: (actor, resource values, additional value) -> object ID for
            product value
        :param resource_specs: each element is the list of resource keys,
            and the resource values corresponding to these keys will be passed
            to `fn` when they are available from the resource providers.
            Each resource key is a tuple (provider index, product key).
            e.g., [[(0, 1), (0, 2)], [(0, 1), (1, 1)]] means the first function
            call takes product 1 and product 2 from provider 0, and the second
            function call takes product 1 from provider 0 and product 1 from
            provider 1.
        :param additional_values: each element is an additional value to pass to
            `fn`.
        :param product_keys: keys for the products; defaults to the task counter.
        """
        assert len(resource_specs) == len(additional_values)
        if product_keys is not None:
            assert len(resource_specs) == len(product_keys)
        else:
            product_keys = list(range(self._next_task_index, self._next_task_index + len(resource_specs)))

        for r, v, p in zip(resource_specs, additional_values, product_keys):
            self.submit(fn, r, v, p)

    def submit(self,
               fn: Callable[[ray.actor.ActorHandle, List, object], ray.ObjectID],
               resource_spec: List[Tuple],
               additional_value,
               product_key,
               res_timeout=1.0) -> None:
        task_index = self._next_task_index
        self._next_task_index += 1

        for resource_key in resource_spec:
            if resource_key not in self._resource_available:
                self._resource_available[resource_key] = False

        self._update_resource_status(res_timeout)

        self._pending_tasks[task_index] = (fn, resource_spec, additional_value)
        assert product_key not in self._index_to_product_key.values()
        self._index_to_product_key[task_index] = product_key

        if all(self._resource_available[k] for k in resource_spec):
            self._queue(task_index)
            self._try_start_queued()
        else:
            for k in resource_spec:
                if k not in self._resource_to_pending_task_index:
                    self._resource_to_pending_task_index[k] = []
                self._resource_to_pending_task_index[k].append(task_index)

    def has_next(self):
        return bool(self._future_to_actor) or bool(self._pending_tasks) \
            or bool(self._queued_tasks)

    def wait(self, timeout=1.0, res_timeout=1.0):
        if not self.has_next():
            raise StopIteration("No more results to get")

        # check status for pending tasks
        self._update_resource_status(res_timeout)

        # wait for tasks already started
        new_products = []
        if self._future_to_actor:
            ready_ids, _ = ray.wait(list(self._future_to_actor.keys()),
                                    num_returns=len(self._future_to_actor.keys()),
                                    timeout=timeout)
            for future in ready_ids:
                i, a, _ = self._future_to_actor.pop(future)
                try:
                    status_code, value = ray.get(future)
                except (ray.exceptions.ObjectReconstructionFailedError, ray.exceptions.RayActorError):
                    traceback.print_exc()
                    status_code, value = 1, None
                # blacklist failed actors
                if status_code == 0:
                    self._return_actor(a)
                del self._index_to_future[i]
                product_key = self._index_to_product_key[i]
                self._product_values[product_key] = value
                new_products.append(product_key)
                running_keys = [self._index_to_product_key[self._future_to_actor[future][0]] for future in self._future_to_actor.keys()]
                logging.debug(f"Task {product_key} done, remaining: {len(self._future_to_actor)} running ({running_keys}), {len(self._queued_tasks)} queued, {len(self._pending_tasks)} pending")

            for future, (i, a, t) in deepcopy(self._future_to_actor).items():
                if time.time() - t > self.timeout:
                    self._future_to_actor.pop(future)
                    del self._index_to_future[i]
                    product_key = self._index_to_product_key[i]
                    self._product_values[product_key] = None
                    new_products.append(product_key)
                    running_keys = [self._index_to_product_key[self._future_to_actor[future][0]] for future in self._future_to_actor.keys()]
                    logging.debug(f"Task {product_key} aborted, remaining: {len(self._future_to_actor)} running ({running_keys}), {len(self._queued_tasks)} queued, {len(self._pending_tasks)} pending")

        # check status for pending tasks again
        self._update_resource_status(res_timeout)

        # start queued tasks if there are idle actors
        self._try_start_queued()

        return new_products

    def wait_generator(self, timeout_per_loop=1.0, res_timeout=1.0):
        while True:
            try:
                new_products = self.wait(timeout_per_loop, res_timeout)
                for product_key in new_products:
                    yield product_key, self._product_values[product_key]
            except StopIteration:
                return

    def _try_start_queued(self):
        while self._idle_actors and self._queued_tasks:
            actor = self._idle_actors.popleft()
            task_index, fn, resource_spec, additional_value = self._queued_tasks.popleft()
            future = fn(actor, [self._resource_providers[p].get_product(k) for p, k in resource_spec], additional_value)
            self._future_to_actor[future] = (task_index, actor, time.time())
            self._index_to_future[task_index] = future
            logging.debug(f"Task {self._index_to_product_key[task_index]} started")

    def _update_resource_status(self, timeout=1.0):
        for provider in self._resource_providers:
            try:
                provider.wait(timeout, timeout)
            except StopIteration:
                pass

        for resource_key in self._resource_available.keys():
            status_old = self._resource_available[resource_key]
            p, k = resource_key
            self._resource_available[resource_key] = \
                self._resource_providers[p].check_product_available(k)
            if (not status_old) and self._resource_available[resource_key]:
                # newly arrived resource
                logging.debug(f"Resource {resource_key} available")
                if resource_key in self._resource_to_pending_task_index:
                    for i in deepcopy(self._resource_to_pending_task_index[resource_key]):
                        # for each pending task, check whether all the required
                        # resources are available
                        if all(self._resource_available[k_] for k_ in self._pending_tasks[i][1]):
                            self._queue(i)

    def _queue(self, task_index):
        f, r, v = self._pending_tasks.pop(task_index)
        self._queued_tasks.append((task_index, f, r, v))
        logging.debug(f"Task {self._index_to_product_key[task_index]} queued")
        for k in r:
            if k in self._resource_to_pending_task_index:
                self._resource_to_pending_task_index[k].remove(task_index)

    def add_consumer(self, consumer: 'PipelineActorPool'):
        self._consumers.append(consumer)

    def _return_actor(self, actor: ray.actor.ActorHandle):
        self._idle_actors.append(actor)
        self._try_start_queued()

    def get_product(self, key):
        return self._product_values[key]

    def check_product_available(self, key):
        return (key in self._product_values.keys())

    def flush(self, flush_resources=False):
        assert not self.has_next()
        for consumer in self._consumers:
            consumer.flush(flush_resources=True)
        self._product_values = {}
        self._index_to_product_key = {}
        if flush_resources:
            self._resource_available = OrderedDict()
            self._resource_to_pending_task_index = {}