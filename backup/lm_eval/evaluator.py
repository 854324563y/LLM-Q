import collections
import itertools
import numpy as np
import random
import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
from lm_eval.utils import positional_deprecated, run_task_tests
import fnmatch
import torch
import gc
from contextlib import contextmanager

@contextmanager
def gpu_memory_manager():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

def clean_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

@positional_deprecated
def simple_evaluate(
    lm,
    tasks,
    model_args=None,
    num_fewshot=0,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    decontamination_ngrams_path=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int, optional
        Batch size for model
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :return
        Dictionary of results
    """
    try:
        with gpu_memory_manager():
            random.seed(1234)
            np.random.seed(1234)
            
            if tasks is None:
                raise ValueError("Please specify a task to run")
            else:
                task_names = pattern_match(tasks.split(","), lm_eval.tasks.ALL_TASKS)
            
            assert tasks != [], "No tasks specified"
            print(f"Selected Tasks: {task_names}")
            
            task_dict = lm_eval.tasks.get_task_dict(task_names)
            
            results = evaluate(
                lm=lm,
                task_dict=task_dict,
                num_fewshot=num_fewshot,
                limit=limit,
                bootstrap_iters=bootstrap_iters,
                description_dict=description_dict,
                decontamination_ngrams_path=decontamination_ngrams_path,
            )
            
            # add info about the model and few shot config
            results["config"] = {
                # "model": lm,
                "model_args": model_args,
                "num_fewshot": num_fewshot,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "description_dict": description_dict,
            }
            
            return results
            
    except Exception as e:
        clean_gpu_memory()
        raise e


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    provide_description=None,
    num_fewshot=0,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    decontamination_ngrams_path=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param provide_description: bool
        Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :return
        Dictionary of results
    """
    try:
        with gpu_memory_manager():
            assert not provide_description
            if provide_description is not None:
                print(
                    "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
                )

            decontaminate = decontamination_ngrams_path is not None

            task_dict_items = [
                (name, task)
                for name, task in task_dict.items()
                if (task.has_validation_docs() or task.has_test_docs())
            ]
            
            results = collections.defaultdict(dict)
            versions = collections.defaultdict(dict)

            requests = collections.defaultdict(list)
            requests_origin = collections.defaultdict(list)

            overlaps = collections.defaultdict(list)
            docs = {}
            docs_for_decontamination = collections.defaultdict(list)

            # 处理任务和文档
            for task_name, task in task_dict_items:
                with gpu_memory_manager():  # 每个任务都确保内存清理
                    versions[task_name] = task.VERSION
                    
                    if task.has_test_docs():
                        task_doc_func = task.test_docs
                        task_set = "test"
                    elif task.has_validation_docs():
                        task_set = "val"
                        task_doc_func = task.validation_docs
                    else:
                        raise RuntimeError("Task has neither test_docs nor validation_docs")

                    task_docs = list(task_doc_func())
                    rnd = random.Random()
                    rnd.seed(42)
                    rnd.shuffle(task_docs)

                    description = (
                        description_dict[task_name]
                        if description_dict and task_name in description_dict
                        else ""
                    )

                    for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
                        if decontaminate and task.should_decontaminate():
                            docs_for_decontamination[(task_name, task_set)].append(
                                task.doc_to_decontamination_query(doc)
                            )
                        docs[(task_name, doc_id)] = doc
                        ctx = task.fewshot_context(
                            doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
                        )
                        reqs = task.construct_requests(doc, ctx)

                        if not isinstance(reqs, (list, tuple)):
                            reqs = [reqs]
                        for j, req in enumerate(reqs):
                            requests[req.request_type].append(req)
                            requests_origin[req.request_type].append((j, task_name, doc, doc_id))
                    
                    clean_gpu_memory()  # 每个任务处理完后清理内存

            if decontaminate:
                from lm_eval.decontamination.decontaminate import get_train_overlap
                print("Finding train/test overlap, please wait...")
                overlaps = get_train_overlap(
                    docs_for_decontamination, decontamination_ngrams_path, limit
                )

            # 处理请求和响应
            process_res_queue = collections.defaultdict(list)
            
            for reqtype, reqs in requests.items():
                print("Running", reqtype, "requests")
                
                with torch.no_grad():  # 使用no_grad减少内存使用
                    resps = getattr(lm, reqtype)([req.args for req in reqs])
                    resps = [
                        x if req.index is None else x[req.index] 
                        for x, req in zip(resps, reqs)
                    ]

                for resp, (j, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
                    process_res_queue[(task_name, doc_id)].append((j, resp))
                
                clean_gpu_memory()  # 每种请求处理完后清理内存

            # 处理结果
            vals = collections.defaultdict(list)
            
            for (task_name, doc_id), requests in process_res_queue.items():
                requests.sort(key=lambda x: x[0])
                requests = [x[1] for x in requests]

                task = task_dict[task_name]
                doc = docs[(task_name, doc_id)]

                metrics = task.process_results(doc, requests)
                for metric, value in metrics.items():
                    vals[(task_name, metric)].append(value)

                    if decontaminate and task_name in overlaps:
                        if doc_id not in overlaps[task_name]:
                            vals[(task_name, metric + decontaminate_suffix)].append(value)

            # 聚合结果
            for (task_name, metric), items in vals.items():
                task = task_dict[task_name]
                real_metric = metric
                if metric.endswith(decontaminate_suffix):
                    real_metric = metric.replace(decontaminate_suffix, "")
                
                results[task_name][metric] = task.aggregation()[real_metric](items)

                stderr = lm_eval.metrics.stderr_for_metric(
                    metric=task.aggregation()[real_metric],
                    bootstrap_iters=min(bootstrap_iters, 1000)
                    if metric in ["bleu", "chrf", "ter"]
                    else bootstrap_iters,
                )

                if stderr is not None:
                    results[task_name][metric + "_stderr"] = stderr(items)

            return {"results": dict(results), "versions": dict(versions)}
            
    except Exception as e:
        clean_gpu_memory()
        raise e


def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
            else:
                values.append([k, version, m, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()
