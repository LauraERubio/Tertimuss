import unittest

from main.core.schedulers.templates.abstract_scheduler.AbstractScheduler import AbstractScheduler
from main.core.schedulers.implementations.OLDTFS import GlobalThermalAwareScheduler
from tests.core.schedulers.scheduler_abstract_test import SchedulerAbstractTest


class RtTcpnScheduler(SchedulerAbstractTest):

    @staticmethod
    def get_global_variables() -> [AbstractScheduler, str]:
        # Scheduler
        scheduler = GlobalThermalAwareScheduler()

        # Result base name
        scheduler_name = "global_thermal_aware"

        return scheduler, scheduler_name

    def test_with_thermal(self):
        scheduler, scheduler_name = self.get_global_variables()
        self.run_test_hash_based(scheduler, True, False, "e4e64c425b65322ad36a55a5a6bcf9bf")
        # self.save_plot_outputs_result(scheduler, True, False, "out/" + scheduler_name + "_thermal")
        # self.save_matlab_result(scheduler, True, False, "out/" + scheduler_name + "_thermal")

    def test_without_thermal(self):
        scheduler, scheduler_name = self.get_global_variables()
        self.run_test_hash_based(scheduler, False, False, "e4e64c425b65322ad36a55a5a6bcf9bf")
        # self.save_plot_outputs_result(scheduler, False, False, "out/" + scheduler_name + "_no_thermal")
        # self.save_matlab_result(scheduler, False, False, "out/" + scheduler_name + "_no_thermal")

    def test_with_thermal_and_aperiodic(self):
        scheduler, scheduler_name = self.get_global_variables()
        self.run_test_hash_based(scheduler, True, True, "4317a9610c218106d0b8b9f41c6e017e")
        # self.save_plot_outputs_result(scheduler, True, True, "out/" + scheduler_name + "_thermal_and_aperiodic")
        # self.save_matlab_result(scheduler, True, True, "out/" + scheduler_name + "_thermal_and_aperiodic")


if __name__ == '__main__':
    unittest.main()
