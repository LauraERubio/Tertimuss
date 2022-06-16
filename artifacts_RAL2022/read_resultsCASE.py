"""
Experiments for WCET corrections
Work in progress
added: restrict worst_case_execution_time per task such that its value is not comparable to the cost of preemptions
"""

import csv
import json
import math
import os
import pickle
import statistics
import time
import argparse

import scipy.stats
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Dict

from matplotlib import pyplot
from tertimuss.simulation_lib.simulator import RawSimulationResult
from tertimuss.simulation_lib.system_definition import TaskSet, PeriodicTask, PreemptiveExecution, Criticality, Job

from dataclasses_serialization.json import JSONSerializer


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


def _wcet_increase(tasks_overhead: Dict[int, int], generated_experiment: _GeneratedExperiment) \
        -> Tuple[Dict[int, float], float]:

    task_set = _get_task_set(generated_experiment)
    original_tasks = task_set.periodic_tasks

    _increase: Dict[int, float] = {i.identifier: round(overhead/i.worst_case_execution_time * 100, 2)
                                   for i, overhead in zip(original_tasks, tasks_overhead.values())}

    return _increase, max(_increase.values())


def _do_boxplot(categories_names: List[str], data_to_plot: List[List[int]], name_of_plot: str,
                save_path: Optional[str], plot_limits: Tuple[float, float], plot_title: Optional[str]):
    fig1, ax1 = pyplot.subplots()

    ax1.set_ylim(plot_limits)
    if plot_title is not None:
        ax1.set_title(plot_title)
    ax1.set_ylabel(f"{name_of_plot}")
    ax1.set_xlabel("# of cores / # of tasks")
    bp = ax1.boxplot(data_to_plot)

    ax1.set_xticklabels(categories_names)

    pyplot.setp(bp['boxes'], color='black')
    pyplot.setp(bp['whiskers'], color='black')
    pyplot.setp(bp['fliers'], color='green')
    pyplot.setp(bp['medians'], color='blue')

    if save_path is not None:
        h = 431
        w = 869
        wi, hi = fig1.get_size_inches()
        fig1.set_size_inches(hi * (w / h), hi)
        fig1.savefig(save_path, bbox_inches='tight', dpi=h/hi)
    else:
        pyplot.show()


def _run_anderson_darling(x) -> float:
    # Obtained from
    # https://stats.stackexchange.com/questions/350443/how-do-i-get-the-p-value-of-ad-test-using-the-results-of-scipy
    # -stats-anderson
    ad, crit, sig = scipy.stats.anderson(x, dist='norm')
    ad = ad * (1 + (.75 / 50) + 2.25 / (50 ** 2))
    if ad >= .6:
        p_value = math.exp(1.2937 - 5.709 * ad - .0186 * (ad ** 2))
    elif ad >= .34:
        p_value = math.exp(.9177 - 4.279 * ad - 1.38 * (ad ** 2))
    elif ad > .2:
        p_value = 1 - math.exp(-8.318 + 42.796 * ad - 59.938 * (ad ** 2))
    else:
        p_value = 1 - math.exp(-13.436 + 101.14 * ad - 223.73 * (ad ** 2))
    return p_value


def _do_statistic_analysis(categories_configuration: List[Tuple[int, int]], data: List[List[int]],
                           name_of_scheduler: str) \
        -> List[Tuple[str, int, int, float, float, float, float, float, float, float, float, float, float]]:
    data_to_return = []

    for (number_of_cpus, number_of_tasks), data_local in zip(categories_configuration, data):
        data_mean = statistics.mean(data_local)
        data_standard_deviation = statistics.pstdev(data_local)
        data_q1, data_q2, data_q3 = statistics.quantiles(data_local)
        data_min = min(data_local)
        data_max = max(data_local)

        _, p_shapiro = scipy.stats.shapiro(data_local)
        _, p_agostino = scipy.stats.normaltest(data_local)
        p_darling = _run_anderson_darling(data_local)

        data_to_return.append(
            (name_of_scheduler, number_of_cpus, number_of_tasks, data_mean, data_standard_deviation, data_min, data_max,
             data_q1, data_q2, data_q3, p_shapiro, p_agostino, p_darling))

    return data_to_return


def input_data():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', action='store',
                        default=2,
                        type=int,
                        dest='cpu',
                        help='Indicate in which configuration compute resulst m=2 or m=4')

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    """
    With this script we will read the results from several experiments on wcet adjustments on tasks
    This script requires tertimuss and dataclasses-serialization dependencies
    """
    _arguments = input_data()
    # Simulation execution options
    M = [_arguments.cpu]
    R = list(range(0, 12))

    generated_tasks_path = [f"data/generated_tasks_M{m}_R{r}.json" for m in M
                            for r in R]

    generated_scheduler_solution_path = [f"solution/generated_solution_M{m}_R{r}.pickle"
                                         for m in M
                                         for r in R]

    execute_analysis = True
    max_preemption_migration_analysis = True
    generate_statistics = True
    generate_boxplots = True
    save_boxplots = True
    analysis_report_path = f".analysis{M[0]}"

    activate_debug = False

    # Experiments configuration
    _major_cycle = 60
    cpus_frequency = 1000
    _available_frequencies = set(range(cpus_frequency, cpus_frequency * 2, 20))

    # Load Task Sets

    generated_tasks_list_read = []

    start = time.time()
    for task_set_path in generated_tasks_path:
        with open(task_set_path, "r") as text_file:
            serialized_experiments = json.load(text_file)
        generated_tasks_list_read.append(JSONSerializer.deserialize(_GeneratedExperimentList, serialized_experiments))

    generated_tasks_list_aux = [tset for exp in generated_tasks_list_read for tset in exp.generated_experiments]
    generated_tasks_list = _GeneratedExperimentList(generated_tasks_list_aux)
    end = time.time()
    print("Task set load time:", end - start)

    # Load Experiment results
    simulations_results_aux = []
    start = time.time()
    for simulation_file in generated_scheduler_solution_path:
        with open(simulation_file, "rb") as text_file:
            simulations_results_aux.append(pickle.load(text_file))

    simulations_results = [sol_exp for sol in simulations_results_aux for sol_exp in sol]
    end = time.time()
    print("Experiments solution load time:", end - start)

    if execute_analysis:
        if not os.path.exists(analysis_report_path):
            os.makedirs(analysis_report_path)

        results_to_analyze = simulations_results

        configurations_analyzed_set: Set[Tuple[int, int]] = {(i.number_of_cpus, len(i.periodic_task_set)) for i in
                                                             generated_tasks_list.generated_experiments}
        max_number_of_tasks = max(i[1] for i in configurations_analyzed_set)

        configurations_analyzed = sorted(configurations_analyzed_set,
                                         key=lambda x: x[0] * (max_number_of_tasks + 1 + x[1]))

        configurations_names = [f"{i[0]}/{i[1]}" for i in configurations_analyzed]

        # Plot limits configuration
        plot_limits_configuration = (0, 50)

        # Headers for the statistic analysis table
        statistic_analysis_headers = ["Scheduler", "Number of CPUs", "Number of tasks", "Mean", "Standard deviation",
                                      "Min", "Max", "Q1", "Q2", "Q3", "Shapiro-Wilk p-value",
                                      "D’Agostino’s K^2 p-value", "Anderson-Darling p-value"]

        start = time.time()
        has_any_not_converged = any([i.results.have_converged is not True for i in results_to_analyze])

        if has_any_not_converged:
            print(f"\t At least one set of tasks failed in cpu experiments")

        iterations = {i: list() for i in configurations_analyzed}
        resulted_frequency = {i: list() for i in configurations_analyzed}
        increment_wcet = {i: list() for i in configurations_analyzed}

        for i, j in zip(results_to_analyze, generated_tasks_list.generated_experiments):
            iterations[
                (j.number_of_cpus, len(j.periodic_task_set))].append(i.results.iterations)
            resulted_frequency[
                (j.number_of_cpus,
                 len(j.periodic_task_set))].append(100*(i.results.resulted_frequency-cpus_frequency)/cpus_frequency)
            increase, max_increase = _wcet_increase(i.results.tasks_overhead, j)
            increment_wcet[(j.number_of_cpus, len(j.periodic_task_set))].append(max_increase)

        # only the values on number of iterations
        list_iterations = [iterations[i] for i in configurations_analyzed]

        # only the values of frequencies
        list_frequency = [resulted_frequency[i] for i in configurations_analyzed]

        list_increment_wcet = [increment_wcet[i] for i in configurations_analyzed]


        if generate_boxplots:
            _do_boxplot(categories_names=configurations_names,
                        data_to_plot=list_iterations,
                        name_of_plot="Algorithm iterations",
                        save_path=f"{analysis_report_path}/boxplot_total_iterations.png"
                        if save_boxplots else None,
                        plot_limits=(0, 40),
                        plot_title=f"Iteration analysis")

            _do_boxplot(categories_names=configurations_names,
                        data_to_plot=list_frequency,
                        name_of_plot="% Frequency increment",
                        save_path=f"{analysis_report_path}/boxplot_resulted_frequency.png"
                        if save_boxplots else None,
                        plot_limits=(0, 25),
                        plot_title=f"Frequency increment analysis")

            _do_boxplot(categories_names=configurations_names,
                        data_to_plot=list_increment_wcet,
                        name_of_plot="% WCET_increment",
                        save_path=f"{analysis_report_path}/boxplot_WCET_increment.png"
                        if save_boxplots else None,
                        plot_limits=(0, 50),
                        plot_title=f"WCET increment analysis")

        if generate_statistics:
            number_iteration_analysis = _do_statistic_analysis(categories_configuration=configurations_analyzed,
                                                               data=list_iterations,
                                                               name_of_scheduler="Iterations")

            increment_frequency_analysis = _do_statistic_analysis(categories_configuration=configurations_analyzed,
                                                                  data=list_frequency,
                                                                  name_of_scheduler="Frequency")

            increment_wcet_analysis = _do_statistic_analysis(categories_configuration=configurations_analyzed,
                                                             data=list_increment_wcet,
                                                             name_of_scheduler="%_WCET")

            data_to_file = []

            for i, j in zip(results_to_analyze, generated_tasks_list.generated_experiments):
                n = len(j.periodic_task_set)
                t_i = [t.deadline for t in j.periodic_task_set]
                c_i = [t.worst_case_execution_time for t in j.periodic_task_set]
                jobs = sum([_major_cycle / ti for ti in t_i])
                increase, max_increase = _wcet_increase(i.results.tasks_overhead, j)
                data_to_file.append([j.number_of_cpus, n, t_i, c_i, list(i.results.tasks_overhead.values()),
                                     jobs, list(increase.values()),
                                     max_increase, i.results.iterations, i.results.resulted_frequency])

            data_to_file_headers = ["Number of CPUs", "Number of tasks", "Tasks deadlines", "Tasks WCET",
                                    "Task's overhead", "# jobs", "% of increment WCET", "Max increase WCET",
                                    "# iterations", "Frequency"]

            with open(f"{analysis_report_path}/experiment_data_jobs.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data_to_file_headers)
                for j in data_to_file:
                    writer.writerow(j)

            with open(f"{analysis_report_path}/iteration_analysis.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(statistic_analysis_headers)
                for j in number_iteration_analysis:
                    writer.writerow(j)

            with open(f"{analysis_report_path}/frequency_analysis.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(statistic_analysis_headers)
                for j in increment_frequency_analysis:
                    writer.writerow(j)

            with open(f"{analysis_report_path}/wcet_increment_analysis.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(statistic_analysis_headers)
                for j in increment_wcet_analysis:
                    writer.writerow(j)

        if max_preemption_migration_analysis:

            max_preemptions_by_task = {i: list() for i in configurations_analyzed}
            max_migrations_by_task = {i: list() for i in configurations_analyzed}

            # preemptions and migrations analysis
            preemptions_per_job = {i: list() for i in configurations_analyzed}
            migrations_per_job = {i: list() for i in configurations_analyzed}

            for i, j in zip(results_to_analyze, generated_tasks_list.generated_experiments):
                max_preemptions_by_task[
                    (j.number_of_cpus, len(j.periodic_task_set))].append(
                    max(i.results.max_preemptions_by_task[i.results.iterations].values()))
                max_migrations_by_task[
                    (j.number_of_cpus, len(j.periodic_task_set))].append(
                    max(i.results.max_migrations_by_task[i.results.iterations].values()))
                preemptions_per_job[
                    (j.number_of_cpus, len(j.periodic_task_set))].append(
                    i.results.preemptions_per_job[i.results.iterations])
                migrations_per_job[
                    (j.number_of_cpus, len(j.periodic_task_set))].append(
                    i.results.migrations_per_job[i.results.iterations])

            list_max_preemptions_by_task = [max_preemptions_by_task[i] for i in configurations_analyzed]
            list_max_migrations_by_task = [max_migrations_by_task[i] for i in configurations_analyzed]

            list_preemptions_per_job = [preemptions_per_job[i] for i in configurations_analyzed]
            list_migrations_per_job = [migrations_per_job[i] for i in configurations_analyzed]

            _do_boxplot(categories_names=configurations_names,
                        data_to_plot=list_max_preemptions_by_task,
                        name_of_plot="max #preemptions by a job",
                        save_path=f"{analysis_report_path}/boxplot_max_preemption_by_job.png"
                        if save_boxplots else None,
                        plot_limits=(0, 20),
                        plot_title=f"Max #preemptions by a job in convergence iteration")

            _do_boxplot(categories_names=configurations_names,
                        data_to_plot=list_max_migrations_by_task,
                        name_of_plot="max #migrations by a job",
                        save_path=f"{analysis_report_path}/boxplot_max_migration_by_job.png"
                        if save_boxplots else None,
                        plot_limits=(0, 20),
                        plot_title=f"Max #migrations by a job in convergence iteration")

            _do_boxplot(categories_names=configurations_names,
                        data_to_plot=list_preemptions_per_job,
                        name_of_plot="#preemptions/# jobs",
                        save_path=f"{analysis_report_path}/boxplot_preemption_per_job.png"
                        if save_boxplots else None,
                        plot_limits=(0, 2),
                        plot_title=f"#preemptions/# jobs of convergence iteration")

            _do_boxplot(categories_names=configurations_names,
                        data_to_plot=list_migrations_per_job,
                        name_of_plot="#migrations/#jobs",
                        save_path=f"{analysis_report_path}/boxplot_migration_per_job.png"
                        if save_boxplots else None,
                        plot_limits=(0, 2),
                        plot_title=f"#migrations/# jobs of convergence iteration")

            max_preemptions_by_task1 = {i: list() for i in configurations_analyzed}
            max_migrations_by_task1 = {i: list() for i in configurations_analyzed}
            # preemptions and migrations analysis
            preemptions_per_job1 = {i: list() for i in configurations_analyzed}
            migrations_per_job1 = {i: list() for i in configurations_analyzed}

            for i, j in zip(results_to_analyze, generated_tasks_list.generated_experiments):
                max_preemptions_by_task1[
                    (j.number_of_cpus, len(j.periodic_task_set))].append(
                    max(i.results.max_preemptions_by_task[1].values()))
                max_migrations_by_task1[
                    (j.number_of_cpus, len(j.periodic_task_set))].append(
                    max(i.results.max_migrations_by_task[1].values()))
                preemptions_per_job1[
                    (j.number_of_cpus, len(j.periodic_task_set))].append(
                    i.results.preemptions_per_job[1])
                migrations_per_job1[
                    (j.number_of_cpus, len(j.periodic_task_set))].append(
                    i.results.migrations_per_job[1])

            list_max_preemptions_by_task = [max_preemptions_by_task1[i] for i in configurations_analyzed]
            list_max_migrations_by_task = [max_migrations_by_task1[i] for i in configurations_analyzed]

            list_preemptions_per_job1 = [preemptions_per_job1[i] for i in configurations_analyzed]
            list_migrations_per_job1 = [migrations_per_job1[i] for i in configurations_analyzed]

            _do_boxplot(categories_names=configurations_names,
                        data_to_plot=list_max_preemptions_by_task,
                        name_of_plot="max #preemptions by a job",
                        save_path=f"{analysis_report_path}/boxplot_max_preemption_by_job_iter1.png"
                        if save_boxplots else None,
                        plot_limits=(0, 20),
                        plot_title=f"Max #preemptions by a job in 1st iteration")

            _do_boxplot(categories_names=configurations_names,
                        data_to_plot=list_max_migrations_by_task,
                        name_of_plot="max #migrations by a job",
                        save_path=f"{analysis_report_path}/boxplot_max_migration_by_job_iter1.png"
                        if save_boxplots else None,
                        plot_limits=(0, 20),
                        plot_title=f"Max #migrations by a job in 1st iteration")

            _do_boxplot(categories_names=configurations_names,
                        data_to_plot=list_preemptions_per_job1,
                        name_of_plot="#preemptions/#jobs",
                        save_path=f"{analysis_report_path}/boxplot_preemption_per_job_iter1.png"
                        if save_boxplots else None,
                        plot_limits=(0, 2),
                        plot_title=f"#preemptions/#jobs in 1st iteration")

            _do_boxplot(categories_names=configurations_names,
                        data_to_plot=list_migrations_per_job1,
                        name_of_plot="#migrations/#jobs",
                        save_path=f"{analysis_report_path}/boxplot_migration_per_job_iter1.png"
                        if save_boxplots else None,
                        plot_limits=(0, 2),
                        plot_title=f"#migrations/#jobs in 1st iteration")



        end = time.time()
        print("Analysis time:", end - start)
        print(f"See results on folder {analysis_report_path}")
