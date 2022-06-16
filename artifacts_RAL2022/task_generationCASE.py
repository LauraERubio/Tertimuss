"""
Experiments for WCET corrections
Work in progress
added: restrict worst_case_execution_time per task such that its value is not comparable to the cost of preemptions
"""


import json
import os
import time
import random

from drs import drs
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Dict

from tertimuss.simulation_lib.simulator import RawSimulationResult
from tertimuss.simulation_lib.system_definition import Job

from tertimuss.tasks_generator.deadline_generator import UniformIntegerDeadlineGenerator
from dataclasses_serialization.json import JSONSerializer

from tertimuss.tasks_generator.periodic_tasks import PeriodicGeneratedTask
from tertimuss.simulation_lib.math_utils import list_float_lcm


@dataclass
class _GeneratedPeriodicTask:
    worst_case_execution_time: int  # Worst case execution time in cycles
    deadline: int  # Deadline in seconds (for dataset 1, it should be a float)


@dataclass
class _ImplicitDeadlineTask:
    identifier: int
    worst_case_execution_time: int
    relative_deadline: float


@dataclass
class _GeneratedExperiment:
    periodic_task_set: List[_GeneratedPeriodicTask]
    number_of_cpus: int
    experiment_identification: int


@dataclass
class _GeneratedExperimentList:
    generated_experiments: List[_GeneratedExperiment]


@dataclass
class _SimulationExtendedResult:
    number_of_cpus: int
    number_of_tasks: int
    experiment_identification: int
    simulation_result: RawSimulationResult
    jobs: List[Job]
    clustering_obtained: Optional[List[int]]


@dataclass
class _SimulationMeasurements:
    scheduler_have_fail: bool
    preemption_number_by_job: Dict[int, int]
    migration_number_by_job: Dict[int, int]
    tasks_job_list: Dict[int, Set[int]]
    cpus_frequencies: int


@dataclass
class _ResultsWCETAdjustment:
    have_converged: bool
    scheduler_have_fail: bool
    resulted_utilization: float
    resulted_frequency: float
    iterations: int
    tasks_overhead: Dict[int, int]
    adjusted_tasks: List[_ImplicitDeadlineTask]


@dataclass
class _ExtendedResultsWCETAdjustment:
    number_of_cpus: int
    number_of_tasks: int
    experiment_identification: int
    results: _ResultsWCETAdjustment


def __get_number_divisors(number: int):
    return [actual_divisor for actual_divisor in range(1, number + 1) if number % actual_divisor == 0]


def __drs_constrained_deadlines(utilization: float, major_cycle: int, processor_frequency: int, number_of_tasks: int,
                                min_wcet: int) \
            -> List[PeriodicGeneratedTask]:
    """
    Generate a list of periodic tasks

    :param utilization: utilization of the task set
    :param major_cycle: hiper-period of the tasks set in seconds
    :param processor_frequency: frequency used to calculate the worst case execution time of each task
    :param min_wcet: minimum value for worst case execution times
    """

    total_cycles = round(major_cycle * utilization * processor_frequency)
    iter_drs = 0
    # low_bound = [0.0175] * number_of_tasks
    low_bound = [(min_wcet/(processor_frequency*major_cycle))] * number_of_tasks

    if sum(low_bound) < utilization:
        u_i = drs(n=number_of_tasks, sumu=utilization, lower_bounds=low_bound)
        #u_i = drs(n=number_of_tasks, sumu=utilization)
        while any([i for i in u_i if i > 1]):
            iter_drs += 1
            print(f"Bad task set{iter_drs}")
            u_i = drs(n=number_of_tasks, sumu=utilization, lower_bounds=low_bound)
        u_tot = sum(u_i)
    else:
        return [PeriodicGeneratedTask(worst_case_execution_time=0, deadline=0) for _ in range(number_of_tasks)]

    major_cycle_divisors = __get_number_divisors(major_cycle)
    min_di = [min_wcet / (processor_frequency * ui) for ui in u_i]
    possible_deadlines = [[xi for xi in major_cycle_divisors if xi >= d_min] for d_min in min_di]

    if any([len(i) == 0 for i in possible_deadlines]):
        print("Error: no valid min_di")

    tasks_deadlines = [random.choice(i) for i in possible_deadlines]

    cci = [round(i*t*processor_frequency) for (i, t) in zip(u_i, tasks_deadlines)]
    cycles = sum([ci * major_cycle//ti for (ci, ti) in zip(cci, tasks_deadlines)])

    while cycles > total_cycles:
        extra_cycles = total_cycles - round(cycles)
        index = tasks_deadlines.index(max(tasks_deadlines))
        remove = round(extra_cycles*tasks_deadlines[index]/major_cycle)
        if remove == 0:
            cci[index] = cci[index] + extra_cycles
        else:
            cci[index] = cci[index] + remove
        cycles = sum([ci * major_cycle // ti for (ci, ti) in zip(cci, tasks_deadlines)])

    return [PeriodicGeneratedTask(worst_case_execution_time=ci, deadline=t) for (ci, t) in
            zip(cci, tasks_deadlines)]


def __drs_deadlines_random(utilization: float, tasks_deadlines: List[int], processor_frequency: int,
                 min_wcet: int) \
            -> List[PeriodicGeneratedTask]:
    """
    Generate a list of periodic tasks

    :param utilization: utilization of the task set
    :param tasks_deadlines: deadline of the tasks in seconds
    :param processor_frequency: frequency used to calculate the worst case execution time of each task
    :param min_wcet: minimum value for worst case execution times
    """

    major_cycle = round(list_float_lcm(tasks_deadlines))
    total_cycles = round(major_cycle * utilization * processor_frequency)
    number_of_tasks = len(tasks_deadlines)

    low_bound = [min_wcet/(ti*processor_frequency) for ti in tasks_deadlines]
    u_low_bound = sum(low_bound)

    u_i = drs(n=number_of_tasks, sumu=utilization, lower_bounds=low_bound)

    negative = any([i <= 0 for i in u_i])
    u_tot = sum(u_i)

    cci = [round(i*t*processor_frequency) for (i, t) in zip(u_i, tasks_deadlines)]
    cycles = sum([ci * major_cycle/ti for (ci, ti) in zip(cci, tasks_deadlines)])

    while cycles > total_cycles:
        extra_cycles = total_cycles - round(cycles)
        index = tasks_deadlines.index(max(tasks_deadlines))
        remove = round(extra_cycles*tasks_deadlines[index]/major_cycle)
        if remove == 0:
            cci[index] = cci[index] + extra_cycles
        else:
            cci[index] = cci[index] + remove
        cycles = sum([ci * major_cycle / ti for (ci, ti) in zip(cci, tasks_deadlines)])

    return [PeriodicGeneratedTask(worst_case_execution_time=ci, deadline=t) for (ci, t) in
            zip(cci, tasks_deadlines)]


def __generate_task_set(identification: int, number_of_cpus: int, number_of_tasks: int, processor_frequency: int,
                        major_cycle_local: int, cpu_utilization: float, min_wcet: int) -> _GeneratedExperiment:
    """
    Generate a task set
    """
    di = UniformIntegerDeadlineGenerator.generate(number_of_tasks, min_deadline=1,
                                                  max_deadline=major_cycle_local, major_cycle=major_cycle_local)
    deadlines = [int(d) for d in di]

    if cpu_utilization <= number_of_cpus:
        # tasks = __drs_deadlines_random(cpu_utilization, deadlines, processor_frequency, min_wcet)
        tasks = __drs_constrained_deadlines(cpu_utilization, major_cycle_local, processor_frequency, number_of_tasks,
                                            min_wcet)
    else:
        #tasks = __drs_deadlines_random(number_of_cpus, deadlines, processor_frequency, min_wcet)
        tasks = __drs_constrained_deadlines(cpu_utilization, major_cycle_local, processor_frequency, number_of_tasks,
                                            min_wcet)

    # u = sum([i.worst_case_execution_time/(i.deadline*processor_frequency) for i in tasks])

    # print(f"U= {u}, experiment:{identification}")
    return _GeneratedExperiment(number_of_cpus=number_of_cpus, experiment_identification=identification,
                                periodic_task_set=[_GeneratedPeriodicTask(
                                                   worst_case_execution_time=i.worst_case_execution_time,
                                                   deadline=int(i.deadline)) for i in tasks])


if __name__ == '__main__':

    # Task generation configuration

    execute_task_generation = True
    save_generated_tasks_as_json = True
    M = [2, 4]
    R = list(range(4, 52, 4))
    configurations = [(m, m * r) for m in M for r in R]
    data_folder_name = "data"

    R = list(range(0, 12)) # for file names
    generated_tasks_path = [f"{data_folder_name}/generated_tasks_M{m}_R{r}.json" for m in M
                            for r in R]

    # Total number of task sets per configuration
    total_number_of_experiments = 5

    _major_cycle = 60
    cpus_frequency = 1000
    _available_frequencies = set(range(cpus_frequency, cpus_frequency * 2, 20))

    _preemption_cost = 10
    _migration_cost = 20
    _min_wcet = 350

    if save_generated_tasks_as_json:
        if not os.path.exists(data_folder_name):
            os.makedirs(data_folder_name)

    start = time.time()
    for current, generated_tasks_path_i in enumerate(generated_tasks_path):
        task_number_processors: Tuple[int, int] = configurations[current]
        m = task_number_processors[0]
        n = task_number_processors[1]
        print("Generating:", total_number_of_experiments, "task sets")
        generated_tasks_list_aux: List[_GeneratedExperiment] = [__generate_task_set(identification=k,
                                                                                    number_of_cpus=m,
                                                                                    number_of_tasks=n,
                                                                                    processor_frequency=cpus_frequency,
                                                                                    major_cycle_local=_major_cycle,
                                                                                    cpu_utilization=m,
                                                                                    min_wcet=_min_wcet)
                                                                for k in
                                                                range(total_number_of_experiments)]
        generated_tasks_list = _GeneratedExperimentList(generated_tasks_list_aux)
        if save_generated_tasks_as_json:
            serialized_experiments = JSONSerializer.serialize(generated_tasks_list)
            with open(generated_tasks_path_i, "w") as text_file:
                json.dump(serialized_experiments, text_file)

    end = time.time()
    print("Task generation time:", end - start)
