# PINNs_multi_turbine_wake
# A PINN for Multi-Turbine Turbulent Wake Simulations

A Physics-informed neural network model to simulate turbulent flows on a grid of wind turbine.

- run pip install -r requirements.txt file before executing the python scripts
---------------------------------------------------------------------------
## 3D Turbulent wake on a grid of wind turbine
    - Rectangular and Diamond grids

- Plots are saved in ^_grid/Plots/line_plots folder (^ - Rectangular, Diamond)
- Trained models are saved in ^_grid/Pre_trained  (^ - Rectangular, Diamond)
- Data for generating contour plots are saved in ^_grid/Plots/csv_data (^ - Rectangular, Diamond)

    First run the below python script
    1. python3 ^_grid/Run_multiturbine_model.py (^ - Rectangular, Diamond)
       
       To simulate different inlet velocities: Change the u_ref value in line no 23 of Run_multiturbine_model.py
       
            - 23   u_ref=11.      #reference velocity/target velocity
 
    Run all cells in the below python notebook to get plots
    1. ^_case/Post_Processing.ipynb (^ - Rectangular, Diamond)
 
    For transfer learning, run the below python script
    1. python3 Transfer_learning/Run_multiturbine_model.py
       
       Change the values of u_ref (target) and Pre_trained_model_velocity (starting model velocity) in line no 23 and 28, respectively
       
           - 23    u_ref=11.      #reference velocity/target velocity
           - 28    Pre_trained_model_velocity = 9  #starting from a pre-trained model for velocity
---------------------------------------------------------------------------


