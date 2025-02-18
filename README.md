# AI
Artificial Intelligence - Genetic Algorithms Travelling Salesman Problem (TSP)

- To clone this project simply press code -> copy the link -> open your directory in the terminal where you want the file and type:
'git clone directory-link'

- Install the requirements with 'pip install -r requirements.txt' in the terminal
- Run the files in the jupyter notebook, ensuring you run the imports section
- Run each function cell
- Call the genetic algorithm and using the main, define the file name and the parameter settings
- There is also a test file that constructs simple unit tests validating that they function correctly on a smaller level

### Parameter setting
- To set your own parameters for testing, find the parameter_sets and insert your values with the order being population size, crossover rates, mutation rates
  i.e (500, 0.8, 0.1).
- You can also set the other parameters such as generations, tournament size
- To change crossover and mutation mechanism, navigate to the genetic algorithms() cell and replace the current mechanism
- If adding more mechanisms you must create the functionality and run the cell before adding to GA() and then run GA()
