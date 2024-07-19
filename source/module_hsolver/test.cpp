if (nks%kpar != 0) {
        ModuleBase::WARNING("Parallel_K2D::set_para_env",
                                 "nks is not divisible by kpar.");
        std::cout << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    "%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        std::cout << " Warning: nks (" << nks << ") is not divisible by kpar ("
                  << kpar << ")." << std::endl;
        std::cout << " This may lead to poor load balance. It is strongly suggested to" << std::endl;
        std::cout << " set nks to be divisible by kpar, but if this is really what" << std::endl;
        std::cout << " you want, please ignore this warning." << std::endl;
        std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                         "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                         "%%%%%%%%%%%%\n";
    }