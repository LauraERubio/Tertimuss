from dataclasses import dataclass
from typing import Dict, Set, List

import numpy
from matplotlib import pyplot, cm, colors, patches
from matplotlib.figure import Figure

from tertimuss_simulation_lib.simulator import RawSimulationResult, JobSectionExecution
from tertimuss_simulation_lib.system_definition import TaskSet, Job, Task, PreemptiveExecution, Criticality


def generate_task_assignation_plot(task_set: TaskSet, jobs: List[Job],
                                   schedule_result: RawSimulationResult) -> Figure:
    num_colors = len(task_set.periodic_tasks) + len(task_set.aperiodic_tasks) + len(task_set.sporadic_tasks)

    periodic_tasks_color_id: Dict[int, int] = {j.identification: i for i, j in enumerate(task_set.periodic_tasks)}
    previous_len = len(periodic_tasks_color_id)
    aperiodic_tasks_color_id: Dict[int, int] = {j.identification: i + previous_len for i, j in
                                                enumerate(task_set.aperiodic_tasks)}
    previous_len = len(periodic_tasks_color_id) + len(aperiodic_tasks_color_id)
    sporadic_tasks_color_id: Dict[int, int] = {j.identification: i + previous_len for i, j in
                                               enumerate(task_set.sporadic_tasks)}
    tasks_color_id: Dict[int, int] = {**periodic_tasks_color_id, **aperiodic_tasks_color_id, **sporadic_tasks_color_id}

    color_map = pyplot.get_cmap('nipy_spectral')
    cNorm = colors.Normalize(vmin=0, vmax=num_colors - 1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=color_map)

    fig, ax = pyplot.subplots()

    for cpu_id, sections_list in schedule_result.job_sections_execution.items():
        for section in sections_list:
            ax.barh(cpu_id, width=section.execution_end_time - section.execution_start_time,
                    left=section.execution_start_time, color=scalarMap.to_rgba(tasks_color_id[section.task_id]))

    ax.set_yticks(range(len(schedule_result.job_sections_execution)))
    ax.set_yticklabels([f'CPU {i + 1}' for i in schedule_result.job_sections_execution.keys()])

    periodic_tasks_legend = [patches.Patch(color=scalarMap.to_rgba(j), label=f'P.Task {i}') for i, j in
                             sorted(periodic_tasks_color_id.items(), key=lambda k: k[0])]

    aperiodic_tasks_legend = [patches.Patch(color=scalarMap.to_rgba(j), label=f'A.Task {i}') for i, j in
                              sorted(aperiodic_tasks_color_id.items(), key=lambda k: k[0])]

    sporadic_tasks_legend = [patches.Patch(color=scalarMap.to_rgba(j), label=f'S.Task {i}') for i, j in
                             sorted(sporadic_tasks_color_id.items(), key=lambda k: k[0])]

    ax.legend(handles=periodic_tasks_legend + aperiodic_tasks_legend + sporadic_tasks_legend, loc="upper right")

    return fig
