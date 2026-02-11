import sys
import os
sys.path.append(os.getcwd())
from repo.VarParser.varparser.cache import VarCache
from repo.VarParser.varparser.prompt import llm_querying
from repo.VarParser.varparser.utils import correct_message, parse_args
from repo.VarParser.varparser.eval import evaluator
from repo.VarParser.running import VarParser

import time
from queue import Queue
from threading import Thread
from tqdm import tqdm
import pandas as pd
import os


class RealStreamSimulator:

    def __init__(self, dataset, data_type, model, shot, arrival_interval=0.01):
        self.dataset = dataset
        self.data_type = data_type
        self.model = model
        self.shot = shot
        self.arrival_interval = arrival_interval

        self.queue = Queue(maxsize=10000)

        self.parser = VarParser(dataset, data_type, model, shot)
        self.varcache = VarCache(dataset=dataset, shot=shot)

        sampled_df = self.parser.read_sampled_data(dataset)
        self.varcache.extract_vars(sampled_df)

        log_df = self.parser.read_data()
        self.logs = log_df["Content"].tolist()

        self.metrics = {
            "arrival_time": [],
            "start_time": [],
            "end_time": [],
            "latency": [],
            "cache_hit": [],
            "llm_invoked": [],
        }

    def producer(self):
        for log in self.logs:
            arrival = time.time()
            self.queue.put((log, arrival))
            time.sleep(self.arrival_interval)

    def consumer(self):
        processed = 0
        total_logs = len(self.logs)
        for _ in tqdm(range(total_logs), desc="Streaming"):
            log, arrival_time = self.queue.get()
            start_time = time.time()

            # cache match
            index, hit = self.varcache.match(log)
            llm_invoked = False

            if not hit:
                llm_invoked = True
                var_prompt = self.varcache.adaptive_var_selection(log)
                reference = self.varcache.get_reference(log)
                response, _ = llm_querying(
                    query_log=log,
                    var_prompt=var_prompt,
                    reference=reference,
                    model=self.model
                )
                index = self.varcache.insert(log, response)

            end_time = time.time()
            latency = end_time - arrival_time

            # save metrics
            self.metrics["arrival_time"].append(arrival_time)
            self.metrics["start_time"].append(start_time)
            self.metrics["end_time"].append(end_time)
            self.metrics["latency"].append(latency)
            self.metrics["cache_hit"].append(hit)
            self.metrics["llm_invoked"].append(llm_invoked)

            processed += 1

    def run(self):
        producer_thread = Thread(target=self.producer)
        consumer_thread = Thread(target=self.consumer)

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        metrics_df = pd.DataFrame(self.metrics)
        os.makedirs("parsed", exist_ok=True)
        metrics_csv = f"parsed/stream_real_{self.dataset}.csv"
        metrics_df.to_csv(metrics_csv, index=False)

        # summary
        avg_latency = metrics_df["latency"].mean()
        max_latency = metrics_df["latency"].max()
        cache_hit_rate = sum(metrics_df["cache_hit"]) / len(metrics_df)
        processing_times = metrics_df["end_time"] - metrics_df["start_time"]
        throughput = len(metrics_df) / processing_times.sum()
        print("=" * 50)
        print(f"Dataset: {self.dataset}")
        print(
            f"Arrival interval: {self.arrival_interval:.3f}s ({1/self.arrival_interval:.1f} logs/sec)")
        print(f"Avg latency: {avg_latency*1000:.2f} ms")
        print(f"Max latency: {max_latency*1000:.2f} ms")
        print(f"Cache hit rate: {cache_hit_rate:.4f}")
        print(f"Throughput: {throughput:.2f} logs/sec")
        print(f"Metrics saved to: {metrics_csv}")
        print("=" * 50)
        return metrics_df


if __name__ == "__main__":
    args = parse_args()

    simulator = RealStreamSimulator(
        dataset='Apache',
        data_type=args.data_type,
        model=args.model,
        shot=args.shot,
        arrival_interval=0.001
    )
    simulator.run()
