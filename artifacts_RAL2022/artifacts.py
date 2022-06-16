"""
Experiments for WCET corrections
Work in progress
added: restrict worst_case_execution_time per task such that its value is not comparable to the cost of preemptions
"""


import json
import pickle
import time
import random
import argparse

from drs import drs
from dataclasses import dataclass
from multiprocessing import Pool
from typing import List, Tuple, Set, Optional, Dict

from tertimuss.analysis import obtain_preemptions_migrations_analysis
from tertimuss.schedulers.calecs import SCALECS
from tertimuss.simulation_lib.schedulers_definition import CentralizedScheduler
from tertimuss.simulation_lib.simulator import SimulationConfiguration, execute_scheduler_simulation_simple, \
    RawSimulationResult
from tertimuss.simulation_lib.system_definition import TaskSet, PeriodicTask, PreemptiveExecution, Criticality, Job
from tertimuss.simulation_lib.system_definition.utils import default_environment_specification, generate_default_cpu

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
    total_preemptions: int
    total_migrations: int
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
    max_preemptions_by_task: Dict[int, Dict[int, int]]
    max_migrations_by_task: Dict[int, Dict[int, int]]
    preemptions_per_job: Dict[int, float]
    migrations_per_job: Dict[int, float]


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


def __generate_task_set(identification: int, number_of_cpus: int, number_of_tasks: int, processor_frequency: int,
                        major_cycle_local: int, cpu_utilization: float, min_wcet: int) -> _GeneratedExperiment:
    """
    Generate a task set
    """
    if cpu_utilization <= number_of_cpus:
        tasks = __drs_constrained_deadlines(cpu_utilization, major_cycle_local, processor_frequency, number_of_tasks,
                                            min_wcet)
    else:
        tasks = __drs_constrained_deadlines(cpu_utilization, major_cycle_local, processor_frequency, number_of_tasks,
                                            min_wcet)

    return _GeneratedExperiment(number_of_cpus=number_of_cpus, experiment_identification=identification,
                                periodic_task_set=[_GeneratedPeriodicTask(
                                                   worst_case_execution_time=i.worst_case_execution_time,
                                                   deadline=int(i.deadline)) for i in tasks])


def _get_task_set(generated_experiment: _GeneratedExperiment) -> TaskSet:
    """
    Get a TaskSet from a _GeneratedExperiment
    """
    periodic_tasks = [PeriodicTask(
        identifier=i,
        worst_case_execution_time=l.worst_case_execution_time,
        relative_deadline=l.deadline,
        best_case_execution_time=None,
        execution_time_distribution=None,
        memory_footprint=None,
        priority=None,
        preemptive_execution=PreemptiveExecution.FULLY_PREEMPTIVE,
        deadline_criteria=Criticality.HARD,
        energy_consumption=None,
        phase=None,
        period=l.deadline
    ) for i, l in enumerate(generated_experiment.periodic_task_set)]

    task_set = TaskSet(
        periodic_tasks=periodic_tasks,
        aperiodic_tasks=[],
        sporadic_tasks=[]
    )
    return task_set


def __get_task_set_aux(periodic_task_set: List[_ImplicitDeadlineTask]) -> TaskSet:
    """
    Get a TaskSet from a _GeneratedExperiment
    """
    periodic_tasks = [PeriodicTask(
        identifier=i.identifier,
        worst_case_execution_time=i.worst_case_execution_time,
        relative_deadline=i.relative_deadline,
        best_case_execution_time=None,
        execution_time_distribution=None,
        memory_footprint=None,
        priority=None,
        preemptive_execution=PreemptiveExecution.FULLY_PREEMPTIVE,
        deadline_criteria=Criticality.HARD,
        energy_consumption=None,
        phase=None,
        period=i.relative_deadline
    ) for i in periodic_task_set]

    task_set = TaskSet(
        periodic_tasks=periodic_tasks,
        aperiodic_tasks=[],
        sporadic_tasks=[]
    )
    return task_set


def __execute_simulation_and_obtain_measurements(periodic_task_set: List[_ImplicitDeadlineTask],
                                                 number_of_cpus: int,
                                                 available_frequencies: Set[int],
                                                 scheduler: CentralizedScheduler,
                                                 iteration_number: int) \
        -> Tuple[_SimulationMeasurements, RawSimulationResult]:
    """
    Execute a simulation
    """
    task_set = __get_task_set_aux(periodic_task_set)

    simulation_result, periodic_jobs, _ = execute_scheduler_simulation_simple(
        tasks=task_set,
        aperiodic_tasks_jobs=[],
        sporadic_tasks_jobs=[],
        processor_definition=generate_default_cpu(number_of_cpus, available_frequencies),
        environment_specification=default_environment_specification(),
        simulation_options=SimulationConfiguration(id_debug=True),
        scheduler=scheduler
    )


    preemption_migration_analysis = obtain_preemptions_migrations_analysis(jobs=periodic_jobs,
                                                                           task_set=task_set,
                                                                           schedule_result=simulation_result)

    job_task_association: Dict[int, Set[int]] = {i.identifier: set() for i in task_set.periodic_tasks}
    for i in periodic_jobs:
        job_task_association[i.task.identifier].add(i.identifier)

    return _SimulationMeasurements(
        scheduler_have_fail=not simulation_result.have_been_scheduled
                            or simulation_result.hard_real_time_deadline_missed_stack_trace is not None,
        preemption_number_by_job=preemption_migration_analysis.number_of_preemptions_by_job,
        migration_number_by_job=preemption_migration_analysis.number_of_migrations_by_job,
        total_preemptions=preemption_migration_analysis.number_of_preemptions,
        total_migrations=preemption_migration_analysis.number_of_migrations,
        tasks_job_list=job_task_association,
        cpus_frequencies=simulation_result.cpus_frequencies[0].__getitem__(0).frequency_used), simulation_result


def __adjust_wcet(generated_experiment: _GeneratedExperiment,
                  available_frequencies: Set[int], preemption_cost,
                  migration_cost) -> _ExtendedResultsWCETAdjustment:
    # Experiment used variables
    _have_converged = False

    _scheduler_have_fail = False

    _iteration_number = 0

    task_set = _get_task_set(generated_experiment)
    original_tasks = task_set.periodic_tasks

    number_of_cpus = generated_experiment.number_of_cpus
    _tasks_ids: Set[int] = {i.identifier for i in original_tasks}

    _old_tasks_overhead: Dict[int, int] = {i: 0 for i in _tasks_ids}

    _adjusted_tasks = [_ImplicitDeadlineTask(
        worst_case_execution_time=i.worst_case_execution_time + _old_tasks_overhead[i.identifier],
        relative_deadline=i.relative_deadline,
        identifier=i.identifier) for i in original_tasks]

    # Cpu utilization
    cpu_frequency = sorted(available_frequencies)[0]  # just for initial setup
    cpu_utilization = sum((i.worst_case_execution_time / cpu_frequency) / i.relative_deadline
                          for i in original_tasks)
    _resultant_cpu_utilization = 0
    new_cpu_frequency = 0

    raw_results: RawSimulationResult

    _max_preemptions_task = {}
    _max_migrations_task = {}
    _preemptions_per_job = {}
    _migrations_per_job = {}

    while not _have_converged and not _scheduler_have_fail:
        _iteration_number += 1

        # Calculate adjusted tasks
        _adjusted_tasks: List[_ImplicitDeadlineTask] = [_ImplicitDeadlineTask(
            worst_case_execution_time=i.worst_case_execution_time + _old_tasks_overhead[i.identifier],
            relative_deadline=i.relative_deadline,
            identifier=i.identifier) for i in original_tasks]

        # Do scheduling
        measures, raw_results = \
            __execute_simulation_and_obtain_measurements(periodic_task_set=_adjusted_tasks,
                                                         number_of_cpus=number_of_cpus,
                                                         available_frequencies=available_frequencies,
                                                         scheduler=SCALECS(activate_debug=False,
                                                                           store_clusters_obtained=False),
                                                         iteration_number=_iteration_number)

        if not measures.scheduler_have_fail:

            new_cpu_frequency = measures.cpus_frequencies
            _resultant_cpu_utilization = sum((i.worst_case_execution_time / new_cpu_frequency) / i.relative_deadline
                                             for i in _adjusted_tasks)

            _job_to_task: Dict[int, int] = {k: i for i, j in measures.tasks_job_list.items() for k in j}

            # _jobs_exclusion_overhead: Dict[int, int] = {}

            # Overhead that must be introduced in each job
            _jobs_overhead = {i: measures.preemption_number_by_job.get(i, 0) * preemption_cost +
                                 measures.migration_number_by_job.get(i, 0) * migration_cost
                              for i in _job_to_task.keys()}

            # Overhead that must be introduced in each task
            _tasks_overhead: Dict[int, int] = {i: max(_jobs_overhead[k] for k in j) for i, j in
                                               measures.tasks_job_list.items()}

            # Check if overhead has increased
            _has_overhead_increased = any(_old_tasks_overhead[i] < _tasks_overhead[i] for i in _tasks_ids)

            # Maximum of migrations and preemptions in this iteration per job in task
            _task_max_preemptions: Dict[int, int] = {i: max(measures.preemption_number_by_job.get(k, 0) for k in j)
                                                     for i, j in
                                                     measures.tasks_job_list.items()}
            _task_max_mig: Dict[int, int] = {i: max(measures.migration_number_by_job.get(k, 0) for k in j)
                                             for i, j in
                                             measures.tasks_job_list.items()}

            # Update iteration values
            _number_jobs = len(measures.preemption_number_by_job)
            _max_preemptions_task[_iteration_number] = _task_max_preemptions
            _max_migrations_task[_iteration_number] = _task_max_mig

            # Preemption and migration analysis of current iteration
            _preemptions_per_job[_iteration_number] = (measures.total_preemptions / _number_jobs) - 1
            _migrations_per_job[_iteration_number] = measures.total_migrations / _number_jobs

            if _has_overhead_increased:
                _new_task_overhead = {i: max(_tasks_overhead[i], _old_tasks_overhead[i]) for i in _tasks_ids}
                _old_tasks_overhead = _new_task_overhead
            else:
                _have_converged = True
        else:
            _scheduler_have_fail = True

    return _ExtendedResultsWCETAdjustment(results=_ResultsWCETAdjustment(have_converged=_have_converged,
                                                                         scheduler_have_fail=_scheduler_have_fail,
                                                                         resulted_utilization=_resultant_cpu_utilization,
                                                                         resulted_frequency=new_cpu_frequency,
                                                                         iterations=_iteration_number,
                                                                         tasks_overhead=_old_tasks_overhead,
                                                                         adjusted_tasks=_adjusted_tasks,
                                                                         max_preemptions_by_task=_max_preemptions_task,
                                                                         max_migrations_by_task=_max_migrations_task,
                                                                         preemptions_per_job=_preemptions_per_job,
                                                                         migrations_per_job=_migrations_per_job
                                                                         ),
                                          number_of_cpus=generated_experiment.number_of_cpus,
                                          experiment_identification=generated_experiment.experiment_identification,
                                          number_of_tasks=len(generated_experiment.periodic_task_set))


def _caller_function(arguments: Tuple[_GeneratedExperiment, Set[int], int, int]) \
        -> _ExtendedResultsWCETAdjustment:
    return __adjust_wcet(arguments[0], arguments[1], arguments[2], arguments[3])


def input_data():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store',
                        dest='input_task_file',
                        type=str,
                        default="task_set.json",
                        help='Provide tasks file name with .json extension')

    parser.add_argument('-o', action='store',
                        dest='simulation_file',
                        type=str,
                        default="simulation_results",
                        help='Provide name')

    parser.add_argument('-g', action='store_false',
                        default=True,
                        dest='task_gen',
                        help='No generate tasks, should supply an input task file')

    parser.add_argument('-p', action='store',
                        default=1,
                        type=int,
                        dest='parallel_l',
                        help='Parallelization level, this must be equal to the number of CPUS '
                             'that will be used in the simulation')

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    """
    With this script we will execute several wcet adjustments on tasks
    This script requires tertimuss and dataclasses-serialization dependencies
    """
    # experiments configuration input data
    _arguments = input_data()
    # Simulation execution options
    execute_task_generation = _arguments.task_gen
    save_generated_tasks_as_json = True
    generated_tasks_path = _arguments.input_task_file

    execute_scheduler_simulation = True
    save_scheduler_simulation_result = True
    generated_scheduler_solution_path = _arguments.simulation_file + ".pickle"

    activate_debug = False

    # Parallelization level (this must be equal to the number of CPUS that will be used in the simulation)
    parallel_level = _arguments.parallel_l

    # Experiments configuration
    _major_cycle = 2 * 2 * 5  
    cpus_frequency = 1000
    _available_frequencies = set(range(cpus_frequency, cpus_frequency * 2, 20))

    _preemption_cost = 10
    _migration_cost = 20
    _min_wcet = 350

    if execute_task_generation:
        start = time.time()
        # Total number of experiments to RUN
        total_number_of_experiments = 100
        M = [2, 4]
        R = list(range(4, 48, 4))
        task_number_processors: List[Tuple[int, int]] = [(m, m * r) for m in M for r in R]
        print("Generating:", len(task_number_processors) * total_number_of_experiments, "task sets")
        generated_tasks_list_aux: List[_GeneratedExperiment] = [__generate_task_set(identification=k,
                                                                                    number_of_cpus=m,
                                                                                    number_of_tasks=n,
                                                                                    processor_frequency=cpus_frequency,
                                                                                    major_cycle_local=_major_cycle,
                                                                                    cpu_utilization=m,
                                                                                    min_wcet=_min_wcet)
                                                                for m, n in task_number_processors for k in
                                                                range(total_number_of_experiments)]

        generated_tasks_list = _GeneratedExperimentList(generated_tasks_list_aux)

        if save_generated_tasks_as_json:
            serialized_experiments = JSONSerializer.serialize(generated_tasks_list)
            with open(generated_tasks_path, "w") as text_file:
                json.dump(serialized_experiments, text_file)

        end = time.time()
        print("Task generation time:", end - start)
    else:
        start = time.time()
        with open(generated_tasks_path, "r") as text_file:
            serialized_experiments = json.load(text_file)
        generated_tasks_list = JSONSerializer.deserialize(_GeneratedExperimentList, serialized_experiments)

        end = time.time()
        print("Task set load time:", end - start)

    if execute_scheduler_simulation:
        start = time.time()
        with Pool(parallel_level) as p:
            print("Executing Experiment")

            simulations_results = p.map(_caller_function,
                                        [(i, _available_frequencies, _preemption_cost,
                                          _migration_cost)
                                         for i in generated_tasks_list.generated_experiments])

            end_exp = time.time()
            print("Algo execution time:", end_exp - start)

        if save_scheduler_simulation_result:
            start = time.time()
            with open(generated_scheduler_solution_path, "wb") as text_file:
                pickle.dump(simulations_results, text_file)

            end = time.time()
            print("Schedulers solution save time:", end - start)
    else:
        start = time.time()
        with open(generated_scheduler_solution_path, "rb") as text_file:
            simulations_results = pickle.load(text_file)

        end = time.time()
        print("Schedulers solution load time:", end - start)
